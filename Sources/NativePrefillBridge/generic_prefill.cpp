// generic_prefill.cpp — Model-agnostic native prefill bridge
//
// Shared infrastructure for any MLX model. Model-specific layer forward
// passes are dispatched by model_type. Currently supports:
//   - gemma4_text (Gemma 4 E2B/A4B with shared KV, PLE, sliding/full attention)
//   - minimax_m2 (MiniMax M2.7 with 256-expert SwitchGLU MoE)
//
// Adding a new model: implement a LayerForward function and register it.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "generic_prefill.h"

using namespace mlx::core;

// ============================================================================
// Shared primitives
// ============================================================================

static constexpr float DEFAULT_RMS_EPS = 1e-6f;

struct QuantizedLinear {
    array weight = array(0.0f);
    array scales = array(0.0f);
    array biases = array(0.0f);
    int group_size = 64;
    int bits = 4;
    bool quantized = true;

    array operator()(const array& x) const {
        if (quantized) {
            return quantized_matmul(x, weight, scales, biases,
                true, group_size, bits, "affine");
        }
        // Unquantized: simple matmul (weight is [out, in], need transpose)
        return matmul(x, transpose(weight));
    }
};

struct Norm {
    array weight = array(0.0f);
    float eps = 1e-6f;
    bool has_weight = true;

    array operator()(const array& x) const {
        if (has_weight) return fast::rms_norm(x, weight, eps);
        return fast::rms_norm(x, std::nullopt, eps);
    }
};

struct Embedding {
    array weight = array(0.0f);
    array scales = array(0.0f);
    array biases = array(0.0f);
    int group_size = 64;
    int bits = 4;
    bool quantized = true;

    array operator()(const array& indices) const {
        if (quantized) {
            auto w = take(weight, indices, 0);
            auto s = take(scales, indices, 0);
            auto b = take(biases, indices, 0);
            return dequantize(w, s, b, group_size, bits, "affine");
        }
        return take(weight, indices, 0);
    }
};

// ============================================================================
// Model config — parsed from JSON
// ============================================================================

struct ModelConfig {
    std::string model_type;
    int hidden_size = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int rotary_dim = 0;  // 0 = full head_dim
    bool tie_word_embeddings = false;
    bool use_qk_norm = false;

    // MoE params
    int num_local_experts = 0;
    int num_experts_per_tok = 0;
    std::string scoring_func = "softmax";

    // Gemma4 params
    int sliding_window = 0;
    int sliding_pattern = 0;
    int num_kv_shared_layers = 0;
    int hidden_per_layer = 0;
    int total_layers = 0;
    float partial_rotary_factor = 1.0f;
    float global_rope_theta = 1000000.0f;
    int global_head_dim = 0;

    // Quantization
    int quant_bits = 4;
    int quant_group_size = 64;

    bool is_moe() const { return num_local_experts > 0; }
};

// ============================================================================
// KV Cache (same as decode bridge)
// ============================================================================

struct KVCache {
    array keys = array(0.0f);
    array values = array(0.0f);
    int offset = 0;
    bool has_data = false;
};

// ============================================================================
// Generic attention (works for all models)
// ============================================================================

static array generic_attention(
    const array& x,
    const QuantizedLinear& q_proj,
    const QuantizedLinear& k_proj,
    const QuantizedLinear& v_proj,
    const QuantizedLinear& o_proj,
    const Norm* q_norm,
    const Norm* k_norm,
    int num_heads, int num_kv_heads, int head_dim,
    bool qk_norm_before_reshape,  // MiniMax: true, Gemma4: false
    float rope_theta, int rotary_dim,
    const array* rope_freqs,  // for proportional RoPE (Gemma4 full layers)
    KVCache& cache
) {
    int B = x.shape(0);
    int S = x.shape(1);

    auto q = q_proj(x);
    auto k = k_proj(x);
    auto v = v_proj(x);

    // QK norm: before or after reshape depending on model
    if (q_norm && qk_norm_before_reshape) {
        q = (*q_norm)(q);
        k = (*k_norm)(k);
    }

    q = transpose(reshape(q, {B, S, num_heads, head_dim}), {0, 2, 1, 3});
    k = transpose(reshape(k, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});
    v = transpose(reshape(v, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});

    // QK norm after reshape (per-head, e.g. Gemma4)
    if (q_norm && !qk_norm_before_reshape) {
        q = (*q_norm)(q);
        k = (*k_norm)(k);
    }

    // V norm (Gemma4 only — no weight)
    // TODO: make this configurable

    // RoPE
    int actual_rotary = rotary_dim > 0 ? rotary_dim : head_dim;
    if (rope_freqs && rope_freqs->ndim() == 1) {
        // Proportional RoPE with precomputed frequencies
        q = fast::rope(q, head_dim, false, std::nullopt, 1.0f, 0, *rope_freqs);
        k = fast::rope(k, head_dim, false, std::nullopt, 1.0f, 0, *rope_freqs);
    } else {
        q = fast::rope(q, actual_rotary, false, rope_theta, 1.0f, 0);
        k = fast::rope(k, actual_rotary, false, rope_theta, 1.0f, 0);
    }

    // Store as bf16
    k = astype(k, bfloat16);
    v = astype(v, bfloat16);

    // Update cache
    cache.keys = k;
    cache.values = v;
    cache.offset = S;
    cache.has_data = true;

    // SDPA
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn = fast::scaled_dot_product_attention(q, k, v, scale, "causal");

    attn = reshape(transpose(attn, {0, 2, 1, 3}), {B, S, num_heads * head_dim});
    return o_proj(attn);
}

// ============================================================================
// MoE routing (MiniMax: sigmoid + top-k + correction bias)
// ============================================================================

static array moe_forward(
    const array& x,
    const QuantizedLinear& gate,
    const array& correction_bias,
    // SwitchGLU expert weights: [num_experts, hidden, dim] stacked
    const array& gate_proj_w, const array& gate_proj_s, const array& gate_proj_b,
    const array& up_proj_w, const array& up_proj_s, const array& up_proj_b,
    const array& down_proj_w, const array& down_proj_s, const array& down_proj_b,
    int num_experts_per_tok,
    int group_size,
    int gate_bits, int up_bits, int down_bits,
    const std::string& scoring_func
) {
    // Route
    auto gates = gate(astype(x, float32));
    array scores = array(0.0f);
    if (scoring_func == "sigmoid") {
        scores = sigmoid(gates);
    } else {
        scores = softmax(gates, -1);
    }
    auto orig_scores = scores;
    auto corrected = scores + correction_bias;

    int k = num_experts_per_tok;
    auto inds = argpartition(negative(corrected), k - 1, -1);
    inds = slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto sel_scores = take_along_axis(orig_scores, inds, -1);

    // Normalize
    sel_scores = sel_scores / (sum(sel_scores, -1, true) + 1e-20f);
    sel_scores = astype(sel_scores, x.dtype());

    // SwitchGLU via gather_qmm with expert-sorted dispatch
    // Sorting tokens by expert index improves GPU cache locality (38-48% at T≥1024)
    auto B = x.shape(0);
    auto S = x.shape(1);

    // Sort by expert for cache-friendly dispatch (matches Swift's SwitchGLU)
    bool do_sort = true;  // always sort — expert locality is always beneficial

    auto x_exp = expand_dims(x, {-2, -3});  // [B, S, 1, 1, D]
    array use_x = x_exp;
    array use_inds = inds;
    array inv_order = array(0);

    if (do_sort) {
        // Flatten indices [B, S, K] → [B*S*K]
        auto flat_inds = flatten(inds);
        auto order = argsort(flat_inds);
        inv_order = argsort(order);

        // Reorder x: flatten to [B*S, 1, 1, D], gather by order/K
        auto x_flat = flatten(x_exp, 0, 1);  // [B*S, 1, 1, D]
        auto token_idx = astype(order / array(k), uint32);  // which token each slot maps to
        use_x = take(x_flat, token_idx, 0);  // [B*S*K, 1, 1, D]
        use_inds = reshape(take(flat_inds, order, 0), {-1, 1});  // [B*S*K, 1] sorted
    }

    auto gate_out = gather_qmm(
        use_x, gate_proj_w, gate_proj_s, gate_proj_b,
        std::nullopt, use_inds,
        true, group_size, gate_bits, "affine", do_sort);

    auto up_out = gather_qmm(
        use_x, up_proj_w, up_proj_s, up_proj_b,
        std::nullopt, use_inds,
        true, group_size, up_bits, "affine", do_sort);

    // SiLU activation on gate, multiply with up
    auto hidden_act = (gate_out * sigmoid(gate_out)) * up_out;

    auto down_out = gather_qmm(
        hidden_act, down_proj_w, down_proj_s, down_proj_b,
        std::nullopt, use_inds,
        true, group_size, down_bits, "affine", do_sort);

    // Squeeze extra dim from gather_qmm output
    down_out = squeeze(down_out, {-2});  // [N, D] where N=B*S*K (sorted) or [B,S,K,D]

    if (do_sort) {
        // Unsort back to original token order and reshape
        down_out = take(down_out, inv_order, 0);  // [B*S*K, D]
        down_out = reshape(down_out, {B, S, k, -1});  // [B, S, K, D]
    }

    auto y = sum(down_out * expand_dims(sel_scores, -1), -2);

    return y;
}

// ============================================================================
// Standard MLP (GeGLU for Gemma4, SiLU+gate for others)
// ============================================================================

static array gelu_approx(const array& x) {
    auto dt = x.dtype();
    auto x3 = power(x, array(3, dt));
    auto inner = array(0.7978845608028654f, dt) * (x + array(0.044715f, dt) * x3);
    return array(0.5f, dt) * x * (array(1.0f, dt) + tanh(inner));
}

static array mlp_forward(
    const array& x,
    const QuantizedLinear& gate_proj,
    const QuantizedLinear& up_proj,
    const QuantizedLinear& down_proj,
    const std::string& activation = "silu"
) {
    auto gate = gate_proj(x);
    auto up = up_proj(x);
    if (activation == "gelu") {
        return down_proj(gelu_approx(gate) * up);
    }
    return down_proj((gate * sigmoid(gate)) * up);
}

// ============================================================================
// Generic layer
// ============================================================================

struct Layer {
    Norm input_norm;
    Norm post_attn_norm;

    // Attention
    QuantizedLinear q_proj, k_proj, v_proj, o_proj;
    Norm q_norm, k_norm;
    bool has_qk_norm = false;
    bool qk_norm_before_reshape = false;
    float rope_theta = 10000.0f;
    int rotary_dim = 0;
    array rope_freqs = array(0.0f);  // for proportional RoPE

    // MLP (standard)
    QuantizedLinear gate_proj, up_proj, down_proj;

    // MoE
    bool is_moe = false;
    QuantizedLinear moe_gate;
    array correction_bias = array(0.0f);
    array switch_gate_w = array(0.0f), switch_gate_s = array(0.0f), switch_gate_b = array(0.0f);
    array switch_up_w = array(0.0f), switch_up_s = array(0.0f), switch_up_b = array(0.0f);
    array switch_down_w = array(0.0f), switch_down_s = array(0.0f), switch_down_b = array(0.0f);
    int num_experts_per_tok = 0;
    int moe_gate_bits = 4;  // bits for gate_proj MoE weights (per-layer)
    int moe_up_bits = 4;    // bits for up_proj MoE weights
    int moe_down_bits = 4;  // bits for down_proj MoE weights
    std::string scoring_func = "softmax";

    // Gemma4 extras
    QuantizedLinear per_layer_gate, per_layer_proj;
    Norm post_per_layer_norm;
    array layer_scalar = array(0.0f);
    bool has_ple = false;

    int num_heads, num_kv_heads, head_dim;
    int group_size = 64, bits = 4;

    KVCache cache;
};

// ============================================================================
// Model
// ============================================================================

struct GenericModel {
    ModelConfig config;
    Embedding embed_tokens;
    std::vector<Layer> layers;
    Norm final_norm;

    // Gemma4 PLE
    Embedding embed_per_layer;
    QuantizedLinear per_layer_projection;
    Norm per_layer_projection_norm;
    bool has_ple = false;
    float embed_scale = 1.0f;
    float pl_embed_scale = 1.0f;
    float pl_input_scale = 0.707f;

    array forward(const array& token_ids) {
        int B = token_ids.shape(0);
        int S = token_ids.shape(1);

        auto h = embed_tokens(token_ids);
        if (embed_scale != 1.0f) {
            h = h * array(embed_scale, bfloat16);
        }

        // Gemma4 PLE setup
        array combined_per_layer = array(0.0f);
        if (has_ple) {
            auto pli = embed_per_layer(token_ids) * array(pl_embed_scale, bfloat16);
            pli = reshape(pli, {B, S, config.total_layers, config.hidden_per_layer});
            auto plp = per_layer_projection(h);
            plp = reshape(plp, {B, S, config.total_layers, config.hidden_per_layer});
            plp = per_layer_projection_norm(plp);
            combined_per_layer = (plp + pli) * array(pl_input_scale, bfloat16);
        }

        // Layer loop — eval periodically to prevent resource exhaustion
        // (62 MoE layers × 256 experts creates massive lazy graphs)
        for (int i = 0; i < (int)layers.size(); i++) {
            auto& layer = layers[i];
            auto residual = h;

            // Attention
            auto normed = layer.input_norm(h);
            auto attn_out = generic_attention(
                normed,
                layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.has_qk_norm ? &layer.q_norm : nullptr,
                layer.has_qk_norm ? &layer.k_norm : nullptr,
                layer.num_heads, layer.num_kv_heads, layer.head_dim,
                layer.qk_norm_before_reshape,
                layer.rope_theta, layer.rotary_dim,
                layer.rope_freqs.ndim() == 1 ? &layer.rope_freqs : nullptr,
                layer.cache
            );
            // post_attn_norm is the pre-MLP norm, NOT applied to attn output
            // (matches MiniMax: r = x + self_attn(input_layernorm(x)))
            h = residual + attn_out;

            // MLP or MoE
            residual = h;
            if (layer.is_moe) {
                auto normed_ff = layer.post_attn_norm(h);  // reuse or separate norm
                h = residual + moe_forward(
                    normed_ff, layer.moe_gate, layer.correction_bias,
                    layer.switch_gate_w, layer.switch_gate_s, layer.switch_gate_b,
                    layer.switch_up_w, layer.switch_up_s, layer.switch_up_b,
                    layer.switch_down_w, layer.switch_down_s, layer.switch_down_b,
                    layer.num_experts_per_tok, layer.group_size,
                    layer.moe_gate_bits, layer.moe_up_bits, layer.moe_down_bits,
                    layer.scoring_func
                );
            } else {
                auto normed_ff = layer.post_attn_norm(h);
                h = residual + mlp_forward(normed_ff, layer.gate_proj, layer.up_proj, layer.down_proj);
            }

                // With shared allocator (SPM target), no per-layer eval needed.
            // The full lazy graph is evaluated once at the end in gp_run().

            // PLE (Gemma4)
            if (layer.has_ple) {
                residual = h;
                auto pli = slice(combined_per_layer,
                    {0, 0, i, 0}, {B, S, i + 1, config.hidden_per_layer});
                pli = reshape(pli, {B, S, config.hidden_per_layer});
                auto gate = layer.per_layer_gate(h);
                gate = gelu_approx(gate) * pli;
                gate = layer.per_layer_proj(gate);
                gate = layer.post_per_layer_norm(gate);
                h = residual + gate;
                h = h * layer.layer_scalar;
            }
        }

        return final_norm(h);
    }
};

// ============================================================================
// Global state
// ============================================================================

static std::unique_ptr<GenericModel> g_model;
static std::unordered_map<std::string, array> g_weights;
static ModelConfig g_config;
static bool g_initialized = false;
static bool g_finalized = false;

// ============================================================================
// Weight helpers
// ============================================================================

static array extract_array(void* ptr) {
    return *static_cast<mlx::core::array*>(ptr);
}

static array get_w(const std::string& key) {
    auto it = g_weights.find(key);
    if (it != g_weights.end()) return it->second;
    throw std::runtime_error("Missing weight: " + key);
}

static bool has_w(const std::string& key) {
    return g_weights.count(key) > 0;
}

static QuantizedLinear make_qlinear(const std::string& prefix, int gs = 0, int b = 0) {
    if (gs == 0) gs = g_config.quant_group_size;
    // Check if weight is quantized (has scales)
    if (has_w(prefix + ".scales")) {
        auto w = get_w(prefix + ".weight");
        auto s = get_w(prefix + ".scales");
        auto bi = get_w(prefix + ".biases");
        // Auto-detect bits from weight/scales shape ratio
        // ratio = weight.shape(-1) / scales.shape(-1) = bits * group_size / 32
        int w_last = w.shape(-1);
        int s_last = s.shape(-1);
        if (s_last > 0 && b == 0) {
            int ratio = w_last / s_last;
            b = ratio * 32 / gs;  // bits = ratio * 32 / group_size
        }
        if (b == 0) b = g_config.quant_bits;
        return {w, s, bi, gs, b, true};
    }
    // Unquantized linear
    if (b == 0) b = g_config.quant_bits;
    return {get_w(prefix + ".weight"), array(0.0f), array(0.0f), gs, b, false};
}

static int detect_bits(const std::string& weight_key, const std::string& scales_key, int gs = 64) {
    if (!has_w(weight_key) || !has_w(scales_key)) return g_config.quant_bits;
    auto w = get_w(weight_key);
    auto s = get_w(scales_key);
    int w_last = w.shape(-1);
    int s_last = s.shape(-1);
    if (s_last <= 0) return g_config.quant_bits;
    return (w_last * 32) / (s_last * gs);
}

static Norm make_norm(const std::string& prefix) {
    return {get_w(prefix + ".weight"), g_config.rms_norm_eps, true};
}

static Embedding make_embedding(const std::string& prefix) {
    bool q = has_w(prefix + ".scales");
    if (q) {
        int gs = g_config.quant_group_size;
        int b = detect_bits(prefix + ".weight", prefix + ".scales", gs);
        return {get_w(prefix + ".weight"), get_w(prefix + ".scales"),
                get_w(prefix + ".biases"), gs, b, true};
    }
    return {get_w(prefix + ".weight"), array(0.0f), array(0.0f), 64, 4, false};
}

// ============================================================================
// Config parsing (minimal JSON — just extract what we need)
// ============================================================================

static std::string json_string(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    auto start = json.find("\"", pos + 1);
    auto end = json.find("\"", start + 1);
    return json.substr(start + 1, end - start - 1);
}

static int json_int(const std::string& json, const std::string& key, int def = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(":", pos);
    auto start = json.find_first_of("-0123456789", pos + 1);
    auto end = json.find_first_not_of("-0123456789", start);
    return std::stoi(json.substr(start, end - start));
}

static float json_float(const std::string& json, const std::string& key, float def = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(":", pos);
    auto start = json.find_first_of("-0123456789.", pos + 1);
    auto end = json.find_first_not_of("-0123456789.eE+", start);
    return std::stof(json.substr(start, end - start));
}

static bool json_bool(const std::string& json, const std::string& key, bool def = false) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(":", pos);
    auto val_start = json.find_first_not_of(" \t\n", pos + 1);
    return json.substr(val_start, 4) == "true";
}

// ============================================================================
// Build model from config + weights
// ============================================================================

static void build_minimax_layer(Layer& layer, int idx) {
    std::string p = "model.layers." + std::to_string(idx);

    layer.input_norm = make_norm(p + ".input_layernorm");
    layer.post_attn_norm = make_norm(p + ".post_attention_layernorm");

    layer.q_proj = make_qlinear(p + ".self_attn.q_proj");
    layer.k_proj = make_qlinear(p + ".self_attn.k_proj");
    layer.v_proj = make_qlinear(p + ".self_attn.v_proj");
    layer.o_proj = make_qlinear(p + ".self_attn.o_proj");

    if (g_config.use_qk_norm) {
        layer.q_norm = make_norm(p + ".self_attn.q_norm");
        layer.k_norm = make_norm(p + ".self_attn.k_norm");
        layer.has_qk_norm = true;
        layer.qk_norm_before_reshape = true;  // MiniMax: norm before reshape
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    // Auto-detect per-layer attention bits from weight shapes
    layer.bits = layer.q_proj.bits;

    // MoE
    layer.is_moe = true;
    layer.moe_gate = make_qlinear(p + ".block_sparse_moe.gate");
    layer.correction_bias = get_w(p + ".block_sparse_moe.e_score_correction_bias");
    layer.switch_gate_w = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.weight");
    layer.switch_gate_s = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.scales");
    layer.switch_gate_b = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.biases");
    layer.switch_up_w = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.weight");
    layer.switch_up_s = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.scales");
    layer.switch_up_b = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.biases");
    layer.switch_down_w = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.weight");
    layer.switch_down_s = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.scales");
    layer.switch_down_b = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.biases");
    // Auto-detect per-layer MoE bits from weight shapes
    std::string moe_p = p + ".block_sparse_moe.switch_mlp";
    layer.moe_gate_bits = detect_bits(moe_p + ".gate_proj.weight", moe_p + ".gate_proj.scales");
    layer.moe_up_bits = detect_bits(moe_p + ".up_proj.weight", moe_p + ".up_proj.scales");
    layer.moe_down_bits = detect_bits(moe_p + ".down_proj.weight", moe_p + ".down_proj.scales");
    layer.num_experts_per_tok = g_config.num_experts_per_tok;
    layer.scoring_func = g_config.scoring_func;
}

static GenericModel build_model() {
    GenericModel m;
    m.config = g_config;
    m.embed_tokens = make_embedding("model.embed_tokens");
    m.final_norm = make_norm("model.norm");

    if (g_config.model_type == "gemma4_text" || g_config.model_type == "gemma4") {
        m.embed_scale = std::sqrt(static_cast<float>(g_config.hidden_size));
        // TODO: add Gemma4-specific layer building (PLE, shared KV, etc.)
        // For now, fall through to generic
    }

    int num_layers = g_config.num_layers;
    m.layers.resize(num_layers);

    for (int i = 0; i < num_layers; i++) {
        if (g_config.model_type == "minimax_m2") {
            build_minimax_layer(m.layers[i], i);
        } else {
            // Generic transformer layer
            std::string p = "model.layers." + std::to_string(i);
            m.layers[i].input_norm = make_norm(p + ".input_layernorm");
            m.layers[i].post_attn_norm = make_norm(p + ".post_attention_layernorm");
            m.layers[i].q_proj = make_qlinear(p + ".self_attn.q_proj");
            m.layers[i].k_proj = make_qlinear(p + ".self_attn.k_proj");
            m.layers[i].v_proj = make_qlinear(p + ".self_attn.v_proj");
            m.layers[i].o_proj = make_qlinear(p + ".self_attn.o_proj");
            m.layers[i].gate_proj = make_qlinear(p + ".mlp.gate_proj");
            m.layers[i].up_proj = make_qlinear(p + ".mlp.up_proj");
            m.layers[i].down_proj = make_qlinear(p + ".mlp.down_proj");
            m.layers[i].num_heads = g_config.num_heads;
            m.layers[i].num_kv_heads = g_config.num_kv_heads;
            m.layers[i].head_dim = g_config.head_dim;
            m.layers[i].rope_theta = g_config.rope_theta;
            m.layers[i].rotary_dim = g_config.rotary_dim;
        }
        fprintf(stderr, "[gp] Layer %d/%d built\n", i + 1, num_layers);
    }

    return m;
}

// ============================================================================
// C ABI
// ============================================================================

extern "C" {

int gp_init(const char* config_json) {
    try {
        std::string json(config_json);
        g_config.model_type = json_string(json, "model_type");
        g_config.hidden_size = json_int(json, "hidden_size", 3072);
        g_config.num_layers = json_int(json, "num_hidden_layers", 62);
        g_config.num_heads = json_int(json, "num_attention_heads", 48);
        g_config.num_kv_heads = json_int(json, "num_key_value_heads", 8);
        g_config.head_dim = json_int(json, "head_dim", 128);
        g_config.intermediate_size = json_int(json, "intermediate_size", 1536);
        g_config.vocab_size = json_int(json, "vocab_size", 200064);
        g_config.rms_norm_eps = json_float(json, "rms_norm_eps", 1e-6f);
        g_config.rope_theta = json_float(json, "rope_theta", 5000000.0f);
        g_config.rotary_dim = json_int(json, "rotary_dim", 64);
        g_config.tie_word_embeddings = json_bool(json, "tie_word_embeddings", false);
        g_config.use_qk_norm = json_bool(json, "use_qk_norm", true);
        g_config.num_local_experts = json_int(json, "num_local_experts", 0);
        g_config.num_experts_per_tok = json_int(json, "num_experts_per_tok", 0);
        g_config.scoring_func = json_string(json, "scoring_func");
        if (g_config.scoring_func.empty()) g_config.scoring_func = "softmax";

        g_weights.clear();
        g_model.reset();
        g_initialized = true;
        g_finalized = false;

        fprintf(stderr, "[gp] Init: model_type=%s layers=%d hidden=%d heads=%d/%d experts=%d\n",
            g_config.model_type.c_str(), g_config.num_layers,
            g_config.hidden_size, g_config.num_heads, g_config.num_kv_heads,
            g_config.num_local_experts);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] init error: %s\n", e.what());
        return -1;
    }
}

int gp_set_weight(const char* key, void* arr_ptr) {
    if (!g_initialized || g_finalized) return -1;
    try {
        g_weights.insert_or_assign(std::string(key), extract_array(arr_ptr));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] set_weight error for '%s': %s\n", key, e.what());
        return -1;
    }
}

int gp_finalize(void) {
    if (!g_initialized || g_finalized) return -1;
    try {
        fprintf(stderr, "[gp] Finalizing with %zu weights\n", g_weights.size());
        g_model = std::make_unique<GenericModel>(build_model());
        g_finalized = true;
        g_weights.clear();

        fprintf(stderr, "[gp] Model built (%d layers)\n", g_config.num_layers);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] finalize error: %s\n", e.what());
        return -1;
    }
}

int gp_run(void* token_array_ptr, double* out_elapsed_ms) {
    if (!g_model) return -1;
    try {
        auto& tokens = *static_cast<mlx::core::array*>(token_array_ptr);
        auto t0 = std::chrono::high_resolution_clock::now();

        auto output = g_model->forward(tokens);

        // Eval all KV caches
        std::vector<array> to_eval;
        for (auto& layer : g_model->layers) {
            if (layer.cache.has_data) {
                to_eval.push_back(layer.cache.keys);
                to_eval.push_back(layer.cache.values);
            }
        }
        eval(to_eval);

        // Free intermediate activation buffers (critical for MoE with 256 experts)
        mlx::core::clear_cache();

        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_elapsed_ms) {
            *out_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] run error: %s\n", e.what());
        return -1;
    }
}

int gp_num_cache_layers(void) {
    return g_model ? (int)g_model->layers.size() : 0;
}

void* gp_get_k_ptr(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return nullptr;
    return new mlx::core::array(g_model->layers[layer_idx].cache.keys);
}

void* gp_get_v_ptr(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return nullptr;
    return new mlx::core::array(g_model->layers[layer_idx].cache.values);
}

int gp_kv_shape(int layer_idx, int* kv_heads, int* seq_len, int* head_dim) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return -1;
    auto& k = g_model->layers[layer_idx].cache.keys;
    if (!g_model->layers[layer_idx].cache.has_data) return -2;
    *kv_heads = k.shape(1);
    *seq_len = k.shape(2);
    *head_dim = k.shape(3);
    return 0;
}

void gp_cleanup(void) {
    g_model.reset();
    g_weights.clear();
    g_initialized = false;
    g_finalized = false;
}

} // extern "C"
