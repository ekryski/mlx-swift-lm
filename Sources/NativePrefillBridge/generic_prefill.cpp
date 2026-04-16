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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "generic_prefill.h"

using namespace mlx::core;

// ============================================================================
// Reliable observer — NEVER use raw sum(abs(x)) on float16.
// MLX's float16 reduction kernel produces wrong results for large arrays.
// ============================================================================

static float safe_abssum(const array& a) {
    // Cast to float32 before reducing to avoid float16 reduction bug
    auto f32 = astype(a, float32);
    auto result = sum(abs(f32));
    eval({result});
    return result.item<float>();
}

// CPU-side checksum: read first N float16 elements, hash them
static uint64_t cpu_hash(const array& a, int n = 256) {
    auto* raw = static_cast<uint16_t*>(
        const_cast<allocator::Buffer&>(a.buffer()).raw_ptr());
    uint64_t h = 0;
    for (int i = 0; i < std::min(n, (int)a.size()); i++) {
        h = h * 31 + raw[i];
    }
    return h;
}

// ============================================================================
// Shared primitives
// ============================================================================

static constexpr float DEFAULT_RMS_EPS = 1e-6f;

struct QuantizedLinear {
    std::optional<array> weight;
    std::optional<array> scales;
    std::optional<array> biases;  // quantization biases
    std::optional<array> linear_bias;  // linear layer bias (e.g., Qwen2 attention)
    int group_size = 64;
    int bits = 4;
    bool quantized = true;

    array operator()(const array& x) const {
        if (quantized) {
            auto out = quantized_matmul(x, *weight, *scales, *biases,
                true, group_size, bits, "affine");
            if (linear_bias) out = out + *linear_bias;
            return out;
        } else {
            auto out = matmul(x, transpose(*weight));
            if (linear_bias) out = out + *linear_bias;
            return out;
        }
    }
};

struct Norm {
    std::optional<array> weight;
    float eps = 1e-6f;

    array operator()(const array& x) const {
        if (weight) return fast::rms_norm(x, *weight, eps);
        return fast::rms_norm(x, std::nullopt, eps);
    }
};

struct Embedding {
    std::optional<array> weight;
    std::optional<array> scales;
    std::optional<array> biases;
    int group_size = 64;
    int bits = 4;
    bool quantized = false;

    array operator()(const array& indices) const {
        if (quantized) {
            auto w = take(*weight, indices, 0);
            auto s = take(*scales, indices, 0);
            auto b = take(*biases, indices, 0);
            return dequantize(w, s, b, group_size, bits, "affine");
        }
        return take(*weight, indices, 0);
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

    // YaRN RoPE params
    std::string rope_type = "";  // "", "yarn", "linear", "llama3"
    float yarn_factor = 1.0f;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    int yarn_original_max_pos = 4096;
    float yarn_mscale = 1.0f;
    float yarn_mscale_all_dim = 0.0f;

    // Quantization
    int quant_bits = 4;
    int quant_group_size = 64;

    bool is_moe() const { return num_local_experts > 0; }
};

// ============================================================================
// KV Cache (same as decode bridge)
// ============================================================================

struct KVCache {
    std::optional<array> keys;
    std::optional<array> values;
    int offset = 0;

    bool has_data() const { return keys.has_value(); }

    void append(const array& k, const array& v, int seq_len) {
        if (keys) {
            keys = concatenate({*keys, k}, 2);
            values = concatenate({*values, v}, 2);
        } else {
            keys = k;
            values = v;
        }
        offset += seq_len;
    }

    void reset() {
        keys.reset();
        values.reset();
        offset = 0;
    }
};

// ============================================================================
// YaRN RoPE frequency computation
// ============================================================================

// Compute YaRN-corrected RoPE frequencies.
// Returns: [dims/2] frequency tensor suitable for fast::rope(..., freqs)
// Also returns the mscale factor to apply to Q/K before RoPE.
static std::pair<array, float> compute_yarn_freqs(
    int dims, float base, float factor,
    int original_max_pos, float beta_fast, float beta_slow,
    float mscale_val, float mscale_all_dim) {

    // yarnFindCorrectionDim: find the dimension index for a given rotation count
    auto find_correction_dim = [&](float num_rotations) -> float {
        return (float)dims
            * std::log((float)original_max_pos / (num_rotations * 2.0f * M_PI))
            / (2.0f * std::log(base));
    };

    // yarnFindCorrectionRange
    int low = std::max(0, (int)std::floor(find_correction_dim(beta_fast)));
    int high = std::min(dims - 1, (int)std::ceil(find_correction_dim(beta_slow)));

    // yarnGetMscale
    auto get_mscale = [](float scale, float ms) -> float {
        if (scale <= 1.0f) return 1.0f;
        return 0.1f * ms * std::log(scale) + 1.0f;
    };

    float mscale = get_mscale(factor, mscale_val) / get_mscale(factor, mscale_all_dim);

    // Compute base frequencies: base^(i / dims) for i in [0, 2, 4, ..., dims-2]
    int half = dims / 2;
    auto indices = astype(arange(0, dims, 2), float32);
    auto base_freqs = power(array(base, float32), indices / (float)dims);

    // freqExtra = base_freqs (unscaled)
    // freqInter = factor * base_freqs (interpolated)
    auto freq_extra = base_freqs;
    auto freq_inter = array(factor, float32) * base_freqs;

    // yarnLinearRampMask: linear interpolation mask between low and high
    float max_val = (low == high) ? (float)high + 0.001f : (float)high;
    auto ramp = (astype(arange(0, half), float32) - (float)low) / (max_val - (float)low);
    ramp = clip(ramp, array(0.0f), array(1.0f));
    auto freq_mask = array(1.0f, float32) - ramp;

    // Final frequency combination (matches Swift exactly):
    // freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    auto freqs = (freq_inter * freq_extra)
        / (freq_inter * freq_mask + freq_extra * (array(1.0f, float32) - freq_mask));

    eval({freqs});
    return {freqs, mscale};
}

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
    KVCache& cache,
    int rope_offset = 0,  // position offset for chunked prefill
    float custom_scale = 0.0f,  // 0 = use 1/sqrt(head_dim)
    float yarn_mscale = 1.0f,  // YaRN mscale for Q/K before RoPE
    const std::optional<array>& sinks = {}  // attention sinks (GPT-OSS)
) {
    int B = x.shape(0);
    int S = x.shape(1);


    auto q = q_proj(x);
    auto k = k_proj(x);
    auto v = v_proj(x);

    if (q_norm && qk_norm_before_reshape) {
        q = (*q_norm)(q); k = (*k_norm)(k);
    }

    q = transpose(reshape(q, {B, S, num_heads, head_dim}), {0, 2, 1, 3});
    k = transpose(reshape(k, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});
    v = transpose(reshape(v, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});

    if (q_norm && !qk_norm_before_reshape) {
        q = (*q_norm)(q); k = (*k_norm)(k);
    }

    if (yarn_mscale != 1.0f) {
        q = q * array(yarn_mscale, q.dtype());
        k = k * array(yarn_mscale, k.dtype());
    }


    int actual_rotary = rotary_dim > 0 ? rotary_dim : head_dim;
    if (rope_freqs && rope_freqs->ndim() == 1) {
        q = fast::rope(q, head_dim, false, std::nullopt, 1.0f, rope_offset, *rope_freqs);
        k = fast::rope(k, head_dim, false, std::nullopt, 1.0f, rope_offset, *rope_freqs);
    } else {
        q = fast::rope(q, actual_rotary, false, rope_theta, 1.0f, rope_offset);
        k = fast::rope(k, actual_rotary, false, rope_theta, 1.0f, rope_offset);
    }

    auto kv_dtype = k_proj.scales ? k_proj.scales->dtype() : float16;
    k = astype(k, kv_dtype);
    v = astype(v, kv_dtype);

    cache.append(k, v, S);

    float scale = custom_scale > 0.0f ? custom_scale
                  : 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto attn = fast::scaled_dot_product_attention(
        q, *cache.keys, *cache.values, scale, "causal", std::nullopt, sinks);

    attn = reshape(transpose(attn, {0, 2, 1, 3}), {B, S, num_heads * head_dim});

    auto out = o_proj(attn);

    return out;
}

// ============================================================================
// MoE routing (MiniMax: sigmoid + top-k + correction bias)
// ============================================================================

static array moe_forward(
    const array& x,
    const QuantizedLinear& gate,
    const array& correction_bias,
    // SwitchGLU expert weights: [num_experts, hidden, dim] stacked
    const array& gate_proj_w, const array& gate_proj_s, const std::optional<array>& gate_proj_b,
    const array& up_proj_w, const array& up_proj_s, const std::optional<array>& up_proj_b,
    const array& down_proj_w, const array& down_proj_s, const std::optional<array>& down_proj_b,
    int num_experts_per_tok,
    int group_size,
    int gate_bits, int up_bits, int down_bits,
    const std::string& scoring_func
) {
    // Route: compute gate scores and select top-k experts
    auto gates = gate(astype(x, float32));
    array scores = array(0.0f);
    if (scoring_func == "sigmoid") {
        scores = sigmoid(gates);
    } else {
        scores = softmax(gates, -1);
    }

    int k = num_experts_per_tok;
    // Select top-k experts. Use raw gates for argpartition (monotonic with softmax)
    // to match Swift's implementation and avoid unnecessary negative(softmax) copies.
    // For sigmoid routing with correction_bias (MiniMax), apply bias before selection.
    array sort_keys = array(0.0f);
    if (scoring_func == "sigmoid" && correction_bias.size() > 1) {
        sort_keys = negative(scores + correction_bias);
    } else {
        sort_keys = negative(gates);  // Match Swift: argPartition(-gates)
    }
    auto inds = argpartition(sort_keys, k - 1, -1);
    inds = slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto sel_scores = take_along_axis(scores, inds, -1);

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

    // Detect quantization mode from biases presence and scales dtype
    // affine: has biases (float16/bfloat16 scales)
    // mxfp4: no biases (uint8 scales)
    auto detect_mode = [](const std::optional<array>& b, const array& s) -> std::string {
        if (!b || b->size() <= 1) {
            // No biases — check scales dtype for MXFP variants
            if (s.dtype() == uint8) return "mxfp4";
            return "affine";  // fallback
        }
        return "affine";
    };

    auto gate_out = gather_qmm(
        use_x, gate_proj_w, gate_proj_s, gate_proj_b,
        std::nullopt, use_inds,
        true, group_size, gate_bits, detect_mode(gate_proj_b, gate_proj_s), do_sort);

    auto up_out = gather_qmm(
        use_x, up_proj_w, up_proj_s, up_proj_b,
        std::nullopt, use_inds,
        true, group_size, up_bits, detect_mode(up_proj_b, up_proj_s), do_sort);

    // SiLU activation on gate, multiply with up
    auto hidden_act = (gate_out * sigmoid(gate_out)) * up_out;

    auto down_out = gather_qmm(
        hidden_act, down_proj_w, down_proj_s, down_proj_b,
        std::nullopt, use_inds,
        true, group_size, down_bits, detect_mode(down_proj_b, down_proj_s), do_sort);

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

struct MoEWeights {
    QuantizedLinear gate;
    std::optional<array> correction_bias;
    std::optional<array> switch_gate_w, switch_gate_s, switch_gate_b;
    std::optional<array> switch_up_w, switch_up_s, switch_up_b;
    std::optional<array> switch_down_w, switch_down_s, switch_down_b;
    int num_experts_per_tok = 0;
    int gate_bits = 4;
    int up_bits = 4;
    int down_bits = 4;
    std::string scoring_func = "softmax";
};

struct PLEWeights {
    QuantizedLinear gate, proj;
    Norm post_norm;
    std::optional<array> layer_scalar;
};

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
    std::optional<array> rope_freqs;

    // Custom attention scale (Gemma3 uses queryPreAttnScalar^(-0.5))
    float attn_scale = 0.0f;  // 0 = use default 1/sqrt(head_dim)

    // YaRN mscale: applied to Q/K before RoPE (1.0 = no scaling)
    float yarn_mscale = 1.0f;

    // Attention sinks (GPT-OSS)
    std::optional<array> sinks;

    // MLP (standard dense path)
    QuantizedLinear gate_proj, up_proj, down_proj;
    std::string mlp_activation = "silu";  // "silu" or "gelu"

    // Gemma3 extra norms (pre/post feedforward)
    Norm pre_ff_norm, post_ff_norm;
    bool has_gemma3_norms = false;
    bool is_sliding = false;  // sliding window attention layer

    // MoE (only allocated for MoE models)
    bool is_moe = false;
    std::optional<MoEWeights> moe;

    // Gemma4 PLE (only allocated for Gemma4)
    bool has_ple = false;
    std::optional<PLEWeights> ple;

    int num_heads = 0, num_kv_heads = 0, head_dim = 0;
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

    static constexpr int CHUNK_SIZE = 4096;

    // Eval barrier cadence: every EVAL_CADENCE layers to prevent
    // lazy-graph pressure from reclaiming weight Metal buffers.
    static constexpr int EVAL_CADENCE = 8;


    array forward(const array& token_ids) {
        int B = token_ids.shape(0);
        int S = token_ids.shape(1);

        auto all_h = embed_tokens(token_ids);
        if (embed_scale != 1.0f) {
            all_h = all_h * array(embed_scale, all_h.dtype());
        }

        // Gemma4 PLE setup (over full sequence)
        array combined_per_layer = array(0.0f);
        if (has_ple) {
            auto pli = embed_per_layer(token_ids) * array(pl_embed_scale, bfloat16);
            pli = reshape(pli, {B, S, config.total_layers, config.hidden_per_layer});
            auto plp = per_layer_projection(all_h);
            plp = reshape(plp, {B, S, config.total_layers, config.hidden_per_layer});
            plp = per_layer_projection_norm(plp);
            combined_per_layer = (plp + pli) * array(pl_input_scale, bfloat16);
        }

        int num_chunks = (S + CHUNK_SIZE - 1) / CHUNK_SIZE;
        array h = array(0.0f);

        for (int c = 0; c < num_chunks; c++) {
            int chunk_start = c * CHUNK_SIZE;
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, S);
            int chunk_len = chunk_end - chunk_start;

            h = slice(all_h, {0, chunk_start, 0}, {B, chunk_end, config.hidden_size});


            for (int i = 0; i < (int)layers.size(); i++) {
                auto& layer = layers[i];
                auto residual = h;

                auto normed = layer.input_norm(h);

                auto attn_out = generic_attention(
                    normed,
                    layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                    layer.has_qk_norm ? &layer.q_norm : nullptr,
                    layer.has_qk_norm ? &layer.k_norm : nullptr,
                    layer.num_heads, layer.num_kv_heads, layer.head_dim,
                    layer.qk_norm_before_reshape,
                    layer.rope_theta, layer.rotary_dim,
                    layer.rope_freqs && layer.rope_freqs->ndim() == 1 ? &*layer.rope_freqs : nullptr,
                    layer.cache,
                    chunk_start,
                    layer.attn_scale,
                    layer.yarn_mscale,
                    layer.sinks
                );

                if (layer.has_gemma3_norms) {
                    // Gemma3: post_attn_norm on attention output, then add residual
                    h = residual + layer.post_attn_norm(attn_out);
                } else {
                    h = residual + attn_out;
                }


                residual = h;
                if (layer.is_moe) {
                    auto normed_ff = layer.post_attn_norm(h);
                    h = residual + moe_forward(
                        normed_ff, layer.moe->gate, *layer.moe->correction_bias,
                        *layer.moe->switch_gate_w, *layer.moe->switch_gate_s, layer.moe->switch_gate_b,
                        *layer.moe->switch_up_w, *layer.moe->switch_up_s, layer.moe->switch_up_b,
                        *layer.moe->switch_down_w, *layer.moe->switch_down_s, layer.moe->switch_down_b,
                        layer.moe->num_experts_per_tok, layer.group_size,
                        layer.moe->gate_bits, layer.moe->up_bits, layer.moe->down_bits,
                        layer.moe->scoring_func
                    );
                } else if (layer.has_gemma3_norms) {
                    // Gemma3: pre_ff_norm → MLP → post_ff_norm → add residual
                    auto pre_ff = layer.pre_ff_norm(h);
                    auto mlp_out = mlp_forward(pre_ff, layer.gate_proj, layer.up_proj, layer.down_proj, layer.mlp_activation);
                    h = residual + layer.post_ff_norm(mlp_out);
                } else {
                    auto normed_ff = layer.post_attn_norm(h);
                    h = residual + mlp_forward(normed_ff, layer.gate_proj, layer.up_proj, layer.down_proj, layer.mlp_activation);
                }

                // PLE (Gemma4)
                if (layer.has_ple) {
                    residual = h;
                    auto pli = slice(combined_per_layer,
                        {0, chunk_start, i, 0}, {B, chunk_end, i + 1, config.hidden_per_layer});
                    pli = reshape(pli, {B, chunk_len, config.hidden_per_layer});
                    auto gate = layer.ple->gate(h);
                    gate = gelu_approx(gate) * pli;
                    gate = layer.ple->proj(gate);
                    gate = layer.ple->post_norm(gate);
                    h = residual + gate;
                    h = h * *layer.ple->layer_scalar;
                }

                if (c == 0 && (i == 0 || i == (int)layers.size() - 1)) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "L%d post_layer", i);
                }

                // Periodic eval barrier to prevent lazy-graph pressure
                // from reclaiming weight Metal buffers.
                // Include embedding weight/scales/biases to keep them pinned.
                if ((i + 1) % EVAL_CADENCE == 0 || i == (int)layers.size() - 1) {
                    std::vector<array> to_sync = {h};
                    if (layer.cache.has_data()) {
                        to_sync.push_back(*layer.cache.keys);
                        to_sync.push_back(*layer.cache.values);
                    }
                    if (embed_tokens.weight) to_sync.push_back(*embed_tokens.weight);
                    if (embed_tokens.scales) to_sync.push_back(*embed_tokens.scales);
                    if (embed_tokens.biases) to_sync.push_back(*embed_tokens.biases);
                    eval(to_sync);
                }
            }

            // Final chunk eval — sync all KV caches
            std::vector<array> to_eval = {h};
            for (auto& layer : layers) {
                if (layer.cache.has_data()) {
                    to_eval.push_back(*layer.cache.keys);
                    to_eval.push_back(*layer.cache.values);
                }
            }
            eval(to_eval);

            if (num_chunks > 1) {
            }
        }

        auto out = final_norm(h);
        return out;
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
    if (it == g_weights.end()) throw std::runtime_error("Missing weight: " + key);
    return it->second;
}

static bool has_w(const std::string& key) {
    return g_weights.count(key) > 0;
}

// Split a fused [N, D] quantized weight into two halves along dim 0
// Used for fused qkv_proj → q_proj + kv_proj, or gate_up_proj → gate + up
// Returns two QuantizedLinears via out params to avoid std::pair construction issues
static void split_fused_qlinear(QuantizedLinear& out1, QuantizedLinear& out2,
    const std::string& prefix, int split_at, int gs = 0, int b = 0) {
    if (gs == 0) gs = g_config.quant_group_size;
    auto w = get_w(prefix + ".weight");
    auto s = get_w(prefix + ".scales");
    // Quantization biases are optional (not present in symmetric/MXFP quant)
    bool has_qbiases = has_w(prefix + ".biases");
    array bi = has_qbiases ? get_w(prefix + ".biases") : array(0.0f);
    int w_last = w.shape(-1);
    int s_last = s.shape(-1);
    if (s_last > 0 && b == 0) {
        int ratio = w_last / s_last;
        b = ratio * 32 / gs;
    }
    if (b == 0) b = g_config.quant_bits;

    // split_at is in OUTPUT dimension (dim 0 of weight)
    // For quantized weights, the packed dim is dim 1, so we split dim 0
    int scale_split = split_at / gs;  // scales have output_dim / group_size cols... no
    // Actually for quantized: weight=[out, packed_in], scales=[out, groups]
    // Split along dim 0 (output dimension)
    auto w1 = slice(w, {0, 0}, {split_at, w.shape(1)});
    auto w2 = slice(w, {split_at, 0}, {w.shape(0), w.shape(1)});
    auto s1 = slice(s, {0, 0}, {split_at, s.shape(1)});
    auto s2 = slice(s, {split_at, 0}, {s.shape(0), s.shape(1)});

    // Split quantization biases if present
    std::optional<array> b1_opt, b2_opt;
    if (has_qbiases) {
        b1_opt = slice(bi, {0, 0}, {split_at, bi.shape(1)});
        b2_opt = slice(bi, {split_at, 0}, {bi.shape(0), bi.shape(1)});
    }

    // Split linear bias if present
    std::optional<array> lb1, lb2;
    if (has_w(prefix + ".bias")) {
        auto lbias = get_w(prefix + ".bias");
        lb1 = slice(lbias, {0}, {split_at});
        lb2 = slice(lbias, {split_at}, {lbias.shape(0)});
    }

    out1.weight = w1; out1.scales = s1; out1.biases = b1_opt.value_or(array(0.0f));
    out1.linear_bias = lb1; out1.group_size = gs; out1.bits = b; out1.quantized = true;
    out2.weight = w2; out2.scales = s2; out2.biases = b2_opt.value_or(array(0.0f));
    out2.linear_bias = lb2; out2.group_size = gs; out2.bits = b; out2.quantized = true;
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
        // Check for linear bias (e.g., Qwen2 attention has bias: true)
        std::optional<array> lb;
        if (has_w(prefix + ".bias")) {
            lb = get_w(prefix + ".bias");
        }
        return {w, s, bi, lb, gs, b, true};
    }
    // Unquantized linear
    if (b == 0) b = g_config.quant_bits;
    std::optional<array> lb;
    if (has_w(prefix + ".bias")) lb = get_w(prefix + ".bias");
    return {get_w(prefix + ".weight"), array(0.0f), array(0.0f), lb, gs, b, false};
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
    return {get_w(prefix + ".weight"), g_config.rms_norm_eps};
}

static Embedding make_embedding(const std::string& prefix) {
    bool q = has_w(prefix + ".scales");
    Embedding emb;
    // Use direct weight reference. Swift already evaluated these arrays,
    // so they have pinned GPU buffers. The add-zero copy is unnecessary.
    emb.weight = get_w(prefix + ".weight");
    eval({*emb.weight});
    if (q) {
        emb.scales = get_w(prefix + ".scales");
        emb.biases = get_w(prefix + ".biases");
        eval({*emb.scales, *emb.biases});
        emb.group_size = g_config.quant_group_size;
        emb.bits = detect_bits(prefix + ".weight", prefix + ".scales", g_config.quant_group_size);
        emb.quantized = true;
    }
    return emb;
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
    auto end = json.find_first_not_of("-0123456789.eE+-", start);
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
    layer.moe.emplace();
    layer.moe->gate = make_qlinear(p + ".block_sparse_moe.gate");
    layer.moe->correction_bias = get_w(p + ".block_sparse_moe.e_score_correction_bias");
    layer.moe->switch_gate_w = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.weight");
    layer.moe->switch_gate_s = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.scales");
    layer.moe->switch_gate_b = get_w(p + ".block_sparse_moe.switch_mlp.gate_proj.biases");
    layer.moe->switch_up_w = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.weight");
    layer.moe->switch_up_s = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.scales");
    layer.moe->switch_up_b = get_w(p + ".block_sparse_moe.switch_mlp.up_proj.biases");
    layer.moe->switch_down_w = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.weight");
    layer.moe->switch_down_s = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.scales");
    layer.moe->switch_down_b = get_w(p + ".block_sparse_moe.switch_mlp.down_proj.biases");
    // Auto-detect per-layer MoE bits from weight shapes
    std::string moe_p = p + ".block_sparse_moe.switch_mlp";
    layer.moe->gate_bits = detect_bits(moe_p + ".gate_proj.weight", moe_p + ".gate_proj.scales");
    layer.moe->up_bits = detect_bits(moe_p + ".up_proj.weight", moe_p + ".up_proj.scales");
    layer.moe->down_bits = detect_bits(moe_p + ".down_proj.weight", moe_p + ".down_proj.scales");
    layer.moe->num_experts_per_tok = g_config.num_experts_per_tok;
    layer.moe->scoring_func = g_config.scoring_func;
}

static void build_qwen3_moe_layer(Layer& layer, int idx) {
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
        layer.qk_norm_before_reshape = false;  // Qwen3: norm AFTER reshape (per-head)
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    layer.bits = layer.q_proj.bits;

    // MoE -- Qwen3 uses mlp.gate and mlp.switch_mlp (not block_sparse_moe)
    layer.is_moe = true;
    layer.moe.emplace();
    layer.moe->gate = make_qlinear(p + ".mlp.gate");
    // Qwen3 uses softmax routing, no correction_bias
    layer.moe->correction_bias = array(0.0f);
    layer.moe->switch_gate_w = get_w(p + ".mlp.switch_mlp.gate_proj.weight");
    layer.moe->switch_gate_s = get_w(p + ".mlp.switch_mlp.gate_proj.scales");
    layer.moe->switch_gate_b = get_w(p + ".mlp.switch_mlp.gate_proj.biases");
    layer.moe->switch_up_w = get_w(p + ".mlp.switch_mlp.up_proj.weight");
    layer.moe->switch_up_s = get_w(p + ".mlp.switch_mlp.up_proj.scales");
    layer.moe->switch_up_b = get_w(p + ".mlp.switch_mlp.up_proj.biases");
    layer.moe->switch_down_w = get_w(p + ".mlp.switch_mlp.down_proj.weight");
    layer.moe->switch_down_s = get_w(p + ".mlp.switch_mlp.down_proj.scales");
    layer.moe->switch_down_b = get_w(p + ".mlp.switch_mlp.down_proj.biases");
    std::string moe_p = p + ".mlp.switch_mlp";
    layer.moe->gate_bits = detect_bits(moe_p + ".gate_proj.weight", moe_p + ".gate_proj.scales");
    layer.moe->up_bits = detect_bits(moe_p + ".up_proj.weight", moe_p + ".up_proj.scales");
    layer.moe->down_bits = detect_bits(moe_p + ".down_proj.weight", moe_p + ".down_proj.scales");
    layer.moe->num_experts_per_tok = g_config.num_experts_per_tok;
    layer.moe->scoring_func = "softmax";  // Qwen3 always uses softmax
}

// Qwen2/Qwen3 dense layer builder -- handles attention bias, QK norm
static void build_qwen_dense_layer(Layer& layer, int idx) {
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
        layer.qk_norm_before_reshape = false;  // Qwen: norm after reshape
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    layer.bits = layer.q_proj.bits;

    // Dense MLP (SiLU-gated)
    layer.is_moe = false;
    layer.gate_proj = make_qlinear(p + ".mlp.gate_proj");
    layer.up_proj = make_qlinear(p + ".mlp.up_proj");
    layer.down_proj = make_qlinear(p + ".mlp.down_proj");

    // Force-eval all layer weights to pin GPU buffers.
    // Without this, the Metal allocator reclaims weight buffers when
    // subsequent layers are constructed (hundreds of array(0.0f) defaults).
    std::vector<array> layer_weights = {
        *layer.q_proj.weight, *layer.q_proj.scales,
        *layer.k_proj.weight, *layer.k_proj.scales,
        *layer.v_proj.weight, *layer.v_proj.scales,
        *layer.o_proj.weight, *layer.o_proj.scales,
        *layer.gate_proj.weight, *layer.gate_proj.scales,
        *layer.up_proj.weight, *layer.up_proj.scales,
        *layer.down_proj.weight, *layer.down_proj.scales,
        *layer.input_norm.weight, *layer.post_attn_norm.weight
    };
    if (layer.q_proj.linear_bias) {
        layer_weights.push_back(*layer.q_proj.linear_bias);
        layer_weights.push_back(*layer.k_proj.linear_bias);
        layer_weights.push_back(*layer.v_proj.linear_bias);
    }
    eval(layer_weights);
}

// Phi3 layer builder — handles fused qkv_proj and gate_up_proj
static void build_phi3_layer(Layer& layer, int idx) {
    std::string p = "model.layers." + std::to_string(idx);

    layer.input_norm = make_norm(p + ".input_layernorm");
    layer.post_attn_norm = make_norm(p + ".post_attention_layernorm");

    // Fused QKV: split into Q, K, V
    // qkv_proj output dim = (num_heads + 2*num_kv_heads) * head_dim
    int q_dim = g_config.num_heads * g_config.head_dim;
    int kv_dim = g_config.num_kv_heads * g_config.head_dim;
    {
        // Split: [q_dim, kv_dim, kv_dim] along output dim 0
        auto qkv_w = get_w(p + ".self_attn.qkv_proj.weight");
        auto qkv_s = get_w(p + ".self_attn.qkv_proj.scales");
        auto qkv_b = get_w(p + ".self_attn.qkv_proj.biases");
        int gs = g_config.quant_group_size;
        int w_last = qkv_w.shape(-1);
        int s_last = qkv_s.shape(-1);
        int bits = (s_last > 0) ? (w_last * 32) / (s_last * gs) : g_config.quant_bits;

        auto make_split = [&](int start, int end) -> QuantizedLinear {
            auto w = slice(qkv_w, {start, 0}, {end, qkv_w.shape(1)});
            auto s = slice(qkv_s, {start, 0}, {end, qkv_s.shape(1)});
            auto b = slice(qkv_b, {start, 0}, {end, qkv_b.shape(1)});
            return {w, s, b, std::nullopt, gs, bits, true};
        };
        layer.q_proj = make_split(0, q_dim);
        layer.k_proj = make_split(q_dim, q_dim + kv_dim);
        layer.v_proj = make_split(q_dim + kv_dim, q_dim + 2 * kv_dim);
    }
    layer.o_proj = make_qlinear(p + ".self_attn.o_proj");

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    layer.bits = layer.q_proj.bits;

    // Fused gate_up_proj: split into gate and up (equal halves)
    {
        auto gu_w = get_w(p + ".mlp.gate_up_proj.weight");
        int half = gu_w.shape(0) / 2;
        split_fused_qlinear(layer.gate_proj, layer.up_proj, p + ".mlp.gate_up_proj", half);
    }
    layer.down_proj = make_qlinear(p + ".mlp.down_proj");
    layer.is_moe = false;
}

// GPT-OSS layer builder — standard attention + MoE with fused gate_up SwitchGLU
static void build_gptoss_layer(Layer& layer, int idx) {
    std::string p = "model.layers." + std::to_string(idx);

    layer.input_norm = make_norm(p + ".input_layernorm");
    layer.post_attn_norm = make_norm(p + ".post_attention_layernorm");

    layer.q_proj = make_qlinear(p + ".self_attn.q_proj");
    layer.k_proj = make_qlinear(p + ".self_attn.k_proj");
    layer.v_proj = make_qlinear(p + ".self_attn.v_proj");
    layer.o_proj = make_qlinear(p + ".self_attn.o_proj");

    // Attention sinks
    if (has_w(p + ".self_attn.sinks")) {
        auto s = get_w(p + ".self_attn.sinks");
        // Only use sinks if they're non-zero
        eval({s});
        float smax = max(abs(s)).item<float>();
        if (smax > 0.0f) {
            layer.sinks = s;
        }
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = 0;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    layer.bits = layer.q_proj.bits;

    // YaRN RoPE: compute corrected frequencies if configured
    if (g_config.rope_type == "yarn" && g_config.yarn_factor > 1.0f) {
        auto [freqs, mscale] = compute_yarn_freqs(
            g_config.head_dim, g_config.rope_theta, g_config.yarn_factor,
            g_config.yarn_original_max_pos, g_config.yarn_beta_fast, g_config.yarn_beta_slow,
            g_config.yarn_mscale, g_config.yarn_mscale_all_dim);
        layer.rope_freqs = freqs;
        layer.yarn_mscale = mscale;
        if (idx == 0) {
        }
    }

    // GPT-OSS MLP uses FusedGateUpSwitchGLU with stacked expert weights [E, out, in]
    // The gate_up_proj is fused: first half is gate, second half is up
    // Route through MoE path with gather_qmm
    layer.is_moe = true;
    layer.moe.emplace();
    layer.moe->gate = make_qlinear(p + ".mlp.router");
    layer.moe->correction_bias = array(0.0f);  // GPT-OSS uses softmax routing, no correction

    // Fused gate_up_proj: [E, gate_dim+up_dim, packed_in]
    // Need to split into separate gate and up for gather_qmm
    auto gu_w = get_w(p + ".mlp.experts.gate_up_proj.weight");
    auto gu_s = get_w(p + ".mlp.experts.gate_up_proj.scales");
    int E = gu_w.shape(0);
    int fused_out = gu_w.shape(1);
    int half = fused_out / 2;

    // Split along dim 1 (output dim): [E, half, in] each
    layer.moe->switch_gate_w = slice(gu_w, {0, 0, 0}, {E, half, gu_w.shape(2)});
    layer.moe->switch_gate_s = slice(gu_s, {0, 0, 0}, {E, half, gu_s.shape(2)});
    layer.moe->switch_up_w = slice(gu_w, {0, half, 0}, {E, fused_out, gu_w.shape(2)});
    layer.moe->switch_up_s = slice(gu_s, {0, half, 0}, {E, fused_out, gu_s.shape(2)});

    // Quantization biases: optional (MXFP4 doesn't have them)
    if (has_w(p + ".mlp.experts.gate_up_proj.biases")) {
        auto gu_b = get_w(p + ".mlp.experts.gate_up_proj.biases");
        layer.moe->switch_gate_b = slice(gu_b, {0, 0, 0}, {E, half, gu_b.shape(2)});
        layer.moe->switch_up_b = slice(gu_b, {0, half, 0}, {E, fused_out, gu_b.shape(2)});
    }
    // else: leave as std::nullopt — gather_qmm handles absent biases

    // down_proj: [E, out, in]
    layer.moe->switch_down_w = get_w(p + ".mlp.experts.down_proj.weight");
    layer.moe->switch_down_s = get_w(p + ".mlp.experts.down_proj.scales");
    if (has_w(p + ".mlp.experts.down_proj.biases")) {
        layer.moe->switch_down_b = get_w(p + ".mlp.experts.down_proj.biases");
    }

    // Bits detection: MXFP4 uses uint8 scales, so detect_bits formula doesn't apply
    // Use config quant_bits (default 4) for MXFP4 models
    std::string moe_p = p + ".mlp.experts";
    auto gs_scales = get_w(moe_p + ".gate_up_proj.scales");
    if (gs_scales.dtype() == uint8) {
        // MXFP4: bits=4, group_size derived from weight/scales shapes
        // w=[E, out, packed_in] (uint32), s=[E, out, groups] (uint8)
        // in_dim = packed_in * 32 / bits, group_size = in_dim / groups
        auto w0 = get_w(moe_p + ".gate_up_proj.weight");
        int packed_in = w0.shape(2);
        int groups = gs_scales.shape(2);
        int in_dim = packed_in * 32 / 4;  // uint32 packing: 8 values per uint32
        layer.group_size = in_dim / groups;
        layer.moe->gate_bits = 4;
        layer.moe->up_bits = 4;
        layer.moe->down_bits = 4;
    } else {
        layer.moe->gate_bits = detect_bits(moe_p + ".gate_up_proj.weight", moe_p + ".gate_up_proj.scales");
        layer.moe->up_bits = layer.moe->gate_bits;
        layer.moe->down_bits = detect_bits(moe_p + ".down_proj.weight", moe_p + ".down_proj.scales");
    }
    layer.moe->num_experts_per_tok = g_config.num_experts_per_tok;
    layer.moe->scoring_func = "softmax";
}

// Gemma3 clip residual: clamp the updated residual to prevent overflow
static array clip_residual(const array& x, const array& r) {
    // Gemma uses soft capping: tanh(residual / 50) * 50
    // But in practice the Swift code just does x + r without explicit clipping
    // for the text-only path. Let's match the Swift behavior.
    return x + r;
}

// Gemma3 layer builder — handles 4 norms, QK norm, GELU, sliding/global RoPE
static void build_gemma3_layer(Layer& layer, int idx) {
    std::string p = "model.layers." + std::to_string(idx);

    layer.input_norm = make_norm(p + ".input_layernorm");
    layer.post_attn_norm = make_norm(p + ".post_attention_layernorm");
    layer.pre_ff_norm = make_norm(p + ".pre_feedforward_layernorm");
    layer.post_ff_norm = make_norm(p + ".post_feedforward_layernorm");
    layer.has_gemma3_norms = true;

    layer.q_proj = make_qlinear(p + ".self_attn.q_proj");
    layer.k_proj = make_qlinear(p + ".self_attn.k_proj");
    layer.v_proj = make_qlinear(p + ".self_attn.v_proj");
    layer.o_proj = make_qlinear(p + ".self_attn.o_proj");

    layer.q_norm = make_norm(p + ".self_attn.q_norm");
    layer.k_norm = make_norm(p + ".self_attn.k_norm");
    layer.has_qk_norm = true;
    layer.qk_norm_before_reshape = false;  // Gemma3: per-head norm

    // Sliding window pattern: most layers are sliding, every Nth is global
    int pattern = g_config.sliding_pattern > 0 ? g_config.sliding_pattern : 6;
    layer.is_sliding = ((idx + 1) % pattern != 0);

    // Different RoPE for sliding vs global layers
    if (layer.is_sliding) {
        // Sliding layers use local base freq (default 10000)
        layer.rope_theta = 10000.0f;  // TODO: parse rope_local_base_freq from config
    } else {
        // Global layers use the main rope_theta with scaling
        layer.rope_theta = g_config.rope_theta;
    }
    layer.rotary_dim = 0;  // full head_dim
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;
    layer.bits = layer.q_proj.bits;

    // Custom attention scale: queryPreAttnScalar^(-0.5)
    // For gemma3-4b, queryPreAttnScalar = head_dim = 256, so scale = 1/16
    float query_pre_attn_scalar = static_cast<float>(g_config.head_dim);  // default
    layer.attn_scale = 1.0f / std::sqrt(query_pre_attn_scalar);

    // MLP uses GELU activation
    layer.gate_proj = make_qlinear(p + ".mlp.gate_proj");
    layer.up_proj = make_qlinear(p + ".mlp.up_proj");
    layer.down_proj = make_qlinear(p + ".mlp.down_proj");
    layer.mlp_activation = "gelu";
    layer.is_moe = false;
}

static GenericModel build_model() {
    GenericModel m;
    m.config = g_config;
    // NOTE: embedding and final_norm are built AFTER layers.
    // layers.resize() creates hundreds of default array(0.0f) objects
    // which can cause the Metal allocator to reclaim GPU buffers
    // backing previously-assigned arrays (embedding scales, etc.).

    if (g_config.model_type == "gemma4_text" || g_config.model_type == "gemma4" ||
        g_config.model_type == "gemma3" || g_config.model_type == "gemma3_text") {
        m.embed_scale = std::sqrt(static_cast<float>(g_config.hidden_size));
    }

    // Build embedding and norm FIRST (before layer loop).
    // Layer default-construction creates many arrays which can trigger
    // MLX Metal buffer reclamation. Building embedding first and verifying
    // it survives the loop is the key test.
    m.embed_tokens = make_embedding("model.embed_tokens");
    m.final_norm = make_norm("model.norm");

    int num_layers = g_config.num_layers;
    m.layers.reserve(num_layers);

    for (int i = 0; i < num_layers; i++) {
        m.layers.emplace_back();

        if (g_config.model_type == "minimax_m2") {
            build_minimax_layer(m.layers[i], i);
        } else if (g_config.model_type == "qwen3_moe") {
            build_qwen3_moe_layer(m.layers[i], i);
        } else if (g_config.model_type == "qwen2" || g_config.model_type == "qwen3") {
            build_qwen_dense_layer(m.layers[i], i);
        } else if (g_config.model_type == "phi3") {
            build_phi3_layer(m.layers[i], i);
        } else if (g_config.model_type == "gemma3" || g_config.model_type == "gemma3_text") {
            build_gemma3_layer(m.layers[i], i);
        } else if (g_config.model_type == "gpt_oss") {
            build_gptoss_layer(m.layers[i], i);
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
        g_config.sliding_window = json_int(json, "sliding_window", 0);
        g_config.sliding_pattern = json_int(json, "sliding_pattern", 0);

        // YaRN RoPE params
        g_config.rope_type = json_string(json, "rope_type");
        g_config.yarn_factor = json_float(json, "yarn_factor", 1.0f);
        g_config.yarn_beta_fast = json_float(json, "yarn_beta_fast", 32.0f);
        g_config.yarn_beta_slow = json_float(json, "yarn_beta_slow", 1.0f);
        g_config.yarn_original_max_pos = json_int(json, "yarn_original_max_pos", 4096);
        g_config.yarn_mscale = json_float(json, "yarn_mscale", 1.0f);
        g_config.yarn_mscale_all_dim = json_float(json, "yarn_mscale_all_dim", 0.0f);

        g_weights.clear();
        g_model.reset();
        g_initialized = true;
        g_finalized = false;

        fprintf(stderr, "[gp] Init: model_type=%s layers=%d hidden=%d heads=%d/%d experts=%d rope_theta=%.0f rms_eps=%.1e head_dim=%d\n",
            g_config.model_type.c_str(), g_config.num_layers,
            g_config.hidden_size, g_config.num_heads, g_config.num_kv_heads,
            g_config.num_local_experts, g_config.rope_theta, g_config.rms_norm_eps,
            g_config.head_dim);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] init error: %s\n", e.what());
        return -1;
    }
}

int gp_set_weight(const char* key, void* arr_ptr) {
    if (!g_initialized || g_finalized) return -1;
    try {
        auto arr = extract_array(arr_ptr);
        std::string k(key);
        g_weights.insert_or_assign(k, arr);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[gp] set_weight error for '%s': %s\n", key, e.what());
        return -1;
    }
}

int gp_finalize(void) {
    if (!g_initialized || g_finalized) return -1;
    try {

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

        // Reset KV caches before forward
        for (auto& layer : g_model->layers) {
            layer.cache.reset();
        }

        auto output = g_model->forward(tokens);

        // Eval all KV caches
        std::vector<array> to_eval;
        for (auto& layer : g_model->layers) {
            if (layer.cache.has_data()) {
                to_eval.push_back(*layer.cache.keys);
                to_eval.push_back(*layer.cache.values);
            }
        }
        eval(to_eval);

        // Free intermediate activation buffers (critical for MoE with 256 experts)
        // DO NOT clear_cache -- reclaims weight buffers

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
    auto& c = g_model->layers[layer_idx].cache;
    if (!c.keys) return nullptr;
    return new mlx::core::array(*c.keys);
}

void* gp_get_v_ptr(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return nullptr;
    auto& c = g_model->layers[layer_idx].cache;
    if (!c.values) return nullptr;
    return new mlx::core::array(*c.values);
}

int gp_kv_shape(int layer_idx, int* kv_heads, int* seq_len, int* head_dim) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return -1;
    auto& c = g_model->layers[layer_idx].cache;
    if (!c.has_data()) return -2;
    *kv_heads = c.keys->shape(1);
    *seq_len = c.keys->shape(2);
    *head_dim = c.keys->shape(3);
    return 0;
}

void gp_cleanup(void) {
    g_model.reset();
    g_weights.clear();
    g_initialized = false;
    g_finalized = false;
}

} // extern "C"
