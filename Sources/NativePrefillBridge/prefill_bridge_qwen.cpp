// prefill_bridge_qwen.cpp — Qwen-family native prefill bridge
//
// Dedicated bridge for Qwen2, Qwen3 (dense), and Qwen3 MoE.
// Key differences from generic_prefill.cpp:
//   - Qwen-specific eval barrier tuning
//   - MoE routing optimized for softmax-only (no sigmoid/correction_bias)
//   - Single forward path per architecture variant (no runtime dispatch)
//   - Independent from other model families — changes here don't affect
//     Mistral, Phi-4, GPT-OSS, etc.
//
// Architecture support:
//   - qwen2: dense with optional attention bias, no QK norm
//   - qwen3: dense with QK norm (per-head, after reshape)
//   - qwen3_moe: MoE with SwitchGLU, softmax routing, QK norm

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "prefill_bridge_qwen.h"

using namespace mlx::core;

// ============================================================================
// Per-stage profiler — enabled by QWEN_PROFILE=1
//
// Forces eval+sync after each stage to get true GPU wall-clock time.
// This DESTROYS graph fusion, so numbers will be ~2-3x slower than
// production. The point is relative attribution, not absolute speed.
// ============================================================================

struct QwenProfiler {
    bool enabled = false;

    // Accumulate across all layers
    double embed_ms = 0, input_norm_ms = 0;
    double qkv_proj_ms = 0, qk_norm_ms = 0, rope_ms = 0;
    double kv_append_ms = 0, sdpa_ms = 0, o_proj_ms = 0;
    double post_norm_ms = 0;
    double moe_routing_ms = 0, moe_sort_ms = 0;
    double moe_gqmm_gate_up_ms = 0, moe_silu_ms = 0, moe_gqmm_down_ms = 0;
    double moe_combine_ms = 0;
    double dense_mlp_ms = 0;
    double eval_barrier_ms = 0, final_norm_ms = 0;
    double residual_ms = 0;  // residual adds
    int layer_count = 0;

    using clock = std::chrono::high_resolution_clock;
    clock::time_point tp;

    void tick() { if (enabled) tp = clock::now(); }

    double tock_ms() {
        if (!enabled) return 0;
        auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - tp).count();
    }

    // eval + sync + return elapsed since last tick
    double sync_tock(const array& a) {
        if (!enabled) return 0;
        eval({a});
        return tock_ms();
    }
    double sync_tock(const std::vector<array>& v) {
        if (!enabled) return 0;
        eval(v);
        return tock_ms();
    }

    void reset() {
        embed_ms = input_norm_ms = 0;
        qkv_proj_ms = qk_norm_ms = rope_ms = 0;
        kv_append_ms = sdpa_ms = o_proj_ms = 0;
        post_norm_ms = 0;
        moe_routing_ms = moe_sort_ms = 0;
        moe_gqmm_gate_up_ms = moe_silu_ms = moe_gqmm_down_ms = 0;
        moe_combine_ms = 0;
        dense_mlp_ms = 0;
        eval_barrier_ms = final_norm_ms = 0;
        residual_ms = 0;
        layer_count = 0;
    }

    void report(int seq_len) {
        if (!enabled) return;
        double moe_gqmm_ms = moe_gqmm_gate_up_ms + moe_silu_ms + moe_gqmm_down_ms;
        double total = embed_ms + input_norm_ms + qkv_proj_ms + qk_norm_ms
            + rope_ms + kv_append_ms + sdpa_ms + o_proj_ms + post_norm_ms
            + moe_routing_ms + moe_sort_ms + moe_gqmm_ms + moe_combine_ms
            + dense_mlp_ms + eval_barrier_ms + final_norm_ms + residual_ms;

        auto pct = [&](double v) { return total > 0 ? 100.0 * v / total : 0; };

        fprintf(stderr, "\n[qwen-profile] === %d tokens, %d layers ===\n", seq_len, layer_count);
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "embedding",      embed_ms,       pct(embed_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "input_norm",     input_norm_ms,  pct(input_norm_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "qkv_proj",       qkv_proj_ms,    pct(qkv_proj_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "qk_norm",        qk_norm_ms,     pct(qk_norm_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "rope",           rope_ms,        pct(rope_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "kv_append",      kv_append_ms,   pct(kv_append_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "sdpa",           sdpa_ms,        pct(sdpa_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "o_proj",         o_proj_ms,      pct(o_proj_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "residual_add",   residual_ms,    pct(residual_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "post_norm",      post_norm_ms,   pct(post_norm_ms));
        if (moe_routing_ms > 0 || moe_gqmm_ms > 0) {
            fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "moe_routing",    moe_routing_ms, pct(moe_routing_ms));
            fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "moe_sort",       moe_sort_ms,    pct(moe_sort_ms));
            fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "moe_gqmm_total", moe_gqmm_ms,   pct(moe_gqmm_ms));
            fprintf(stderr, "[qwen-profile]   %-18s %8.1fms (%5.1f%%)\n", "gate+up",      moe_gqmm_gate_up_ms, pct(moe_gqmm_gate_up_ms));
            fprintf(stderr, "[qwen-profile]   %-18s %8.1fms (%5.1f%%)\n", "silu*up",      moe_silu_ms,    pct(moe_silu_ms));
            fprintf(stderr, "[qwen-profile]   %-18s %8.1fms (%5.1f%%)\n", "down",         moe_gqmm_down_ms, pct(moe_gqmm_down_ms));
            fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "moe_combine",   moe_combine_ms, pct(moe_combine_ms));
        }
        if (dense_mlp_ms > 0)
            fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "dense_mlp",     dense_mlp_ms,   pct(dense_mlp_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "eval_barrier",   eval_barrier_ms, pct(eval_barrier_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms (%5.1f%%)\n", "final_norm",     final_norm_ms,  pct(final_norm_ms));
        fprintf(stderr, "[qwen-profile] %-20s %8.1fms\n",           "TOTAL",          total);
        fprintf(stderr, "[qwen-profile] %-20s %8.0f tok/s (profiled, ~2-3x slower than production)\n",
            "throughput", seq_len / (total / 1000.0));
    }
};

static QwenProfiler g_prof;

// ============================================================================
// Primitives (same as generic but self-contained)
// ============================================================================

struct QuantizedLinear {
    std::optional<array> weight, scales, biases, linear_bias;
    int group_size = 64, bits = 4;
    bool quantized = true;

    array operator()(const array& x) const {
        if (quantized) {
            auto out = quantized_matmul(x, *weight, *scales, *biases,
                true, group_size, bits, "affine");
            if (linear_bias) out = out + *linear_bias;
            return out;
        }
        auto out = matmul(x, transpose(*weight));
        if (linear_bias) out = out + *linear_bias;
        return out;
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
    std::optional<array> weight, scales, biases;
    int group_size = 64, bits = 4;
    bool quantized = false;

    array operator()(const array& indices) const {
        if (quantized) {
            return dequantize(
                take(*weight, indices, 0),
                take(*scales, indices, 0),
                take(*biases, indices, 0),
                group_size, bits, "affine");
        }
        return take(*weight, indices, 0);
    }
};

struct KVCache {
    std::optional<array> keys, values;
    int offset = 0;
    bool has_data() const { return keys.has_value(); }
    void append(const array& k, const array& v, int S) {
        if (keys) {
            keys = concatenate({*keys, k}, 2);
            values = concatenate({*values, v}, 2);
        } else {
            keys = k; values = v;
        }
        offset += S;
    }
    void reset() { keys.reset(); values.reset(); offset = 0; }
};

// ============================================================================
// Config
// ============================================================================

struct QwenConfig {
    std::string model_type;  // "qwen2", "qwen3", "qwen3_moe"
    int hidden_size = 0, num_layers = 0, num_heads = 0, num_kv_heads = 0;
    int head_dim = 0, intermediate_size = 0, vocab_size = 0;
    float rms_norm_eps = 1e-6f, rope_theta = 1000000.0f;
    int rotary_dim = 0;
    bool tie_word_embeddings = false, use_qk_norm = false;
    int num_local_experts = 0, num_experts_per_tok = 0;
    int quant_bits = 4, quant_group_size = 64;
    bool is_moe() const { return num_local_experts > 0; }
};

// ============================================================================
// MoE forward (Qwen3 softmax-only, no correction bias)
// ============================================================================

static array qwen_moe_forward(
    const array& x,
    const QuantizedLinear& gate,
    const array& gate_w, const array& gate_s, const std::optional<array>& gate_b,
    const array& up_w, const array& up_s, const std::optional<array>& up_b,
    const array& down_w, const array& down_s, const std::optional<array>& down_b,
    int num_experts_per_tok, int group_size,
    int gate_bits, int up_bits, int down_bits
) {
    // Softmax routing (Qwen3 MoE only uses softmax, no sigmoid/correction)
    // Match Swift: no float32 cast on gate input, use precise softmax instead
    auto gates = gate(x);
    auto scores = softmax(gates, -1, true);  // precise=true matches Swift
    int k = num_experts_per_tok;

    auto inds = argpartition(negative(gates), k - 1, -1);
    inds = slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto sel_scores = take_along_axis(scores, inds, -1);
    sel_scores = sel_scores / (sum(sel_scores, -1, true) + 1e-20f);
    sel_scores = astype(sel_scores, x.dtype());

    int B = x.shape(0), S = x.shape(1);

    // Expert-sorted dispatch (matches Swift gatherSort/scatterUnsort pattern)
    auto x_exp = expand_dims(x, {-2, -3});
    auto flat_inds = flatten(inds);
    auto order = argsort(flat_inds);
    auto inv_order = argsort(order);
    auto x_sorted = take(flatten(x_exp, 0, -3), floor_divide(order, array(k)), 0);
    auto sorted_inds = take(flat_inds, order, 0);

    auto gate_out = gather_qmm(x_sorted, gate_w, gate_s, gate_b,
        std::nullopt, sorted_inds, true, group_size, gate_bits, "affine", true);
    auto up_out = gather_qmm(x_sorted, up_w, up_s, up_b,
        std::nullopt, sorted_inds, true, group_size, up_bits, "affine", true);

    auto hidden = (gate_out * sigmoid(gate_out)) * up_out;

    auto down_out = gather_qmm(hidden, down_w, down_s, down_b,
        std::nullopt, sorted_inds, true, group_size, down_bits, "affine", true);

    down_out = squeeze(down_out, {-2});
    down_out = take(down_out, inv_order, 0);
    down_out = reshape(down_out, {inds.shape(0), inds.shape(1), k, -1});

    return sum(down_out * expand_dims(sel_scores, -1), -2);
}

// Fused gate+up variant: single gather_qmm for gate+up, then split output
static array qwen_moe_forward_fused(
    const array& x,
    const QuantizedLinear& gate,
    const array& fused_w, const array& fused_s, const std::optional<array>& fused_b,
    int gate_up_split,
    const array& down_w, const array& down_s, const std::optional<array>& down_b,
    int num_experts_per_tok, int group_size,
    int fused_bits, int down_bits
) {
    auto gates = gate(x);
    auto scores = softmax(gates, -1, true);
    int k = num_experts_per_tok;

    auto inds = argpartition(negative(gates), k - 1, -1);
    inds = slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto sel_scores = take_along_axis(scores, inds, -1);
    sel_scores = sel_scores / (sum(sel_scores, -1, true) + 1e-20f);
    sel_scores = astype(sel_scores, x.dtype());

    int B = x.shape(0), S = x.shape(1);

    auto x_exp = expand_dims(x, {-2, -3});
    auto flat_inds = flatten(inds);
    auto order = argsort(flat_inds);
    auto inv_order = argsort(order);
    auto x_sorted = take(flatten(x_exp, 0, -3), floor_divide(order, array(k)), 0);
    auto sorted_inds = take(flat_inds, order, 0);

    // Single gather_qmm for gate+up (fused along output dim)
    auto fused_out = gather_qmm(x_sorted, fused_w, fused_s, fused_b,
        std::nullopt, sorted_inds, true, group_size, fused_bits, "affine", true);

    auto gate_out = slice(fused_out, {0, 0, 0, 0},
        {fused_out.shape(0), fused_out.shape(1), fused_out.shape(2), gate_up_split});
    auto up_out = slice(fused_out, {0, 0, 0, gate_up_split},
        {fused_out.shape(0), fused_out.shape(1), fused_out.shape(2), fused_out.shape(3)});

    auto hidden = (gate_out * sigmoid(gate_out)) * up_out;

    auto down_out = gather_qmm(hidden, down_w, down_s, down_b,
        std::nullopt, sorted_inds, true, group_size, down_bits, "affine", true);

    down_out = squeeze(down_out, {-2});
    down_out = take(down_out, inv_order, 0);
    down_out = reshape(down_out, {inds.shape(0), inds.shape(1), k, -1});

    return sum(down_out * expand_dims(sel_scores, -1), -2);
}

// ============================================================================
// Dense MLP
// ============================================================================

static array qwen_mlp_forward(
    const array& x,
    const QuantizedLinear& gate_proj,
    const QuantizedLinear& up_proj,
    const QuantizedLinear& down_proj
) {
    auto gate = gate_proj(x);
    auto up = up_proj(x);
    return down_proj((gate * sigmoid(gate)) * up);
}

// ============================================================================
// Attention
// ============================================================================

static array qwen_attention(
    const array& x,
    const QuantizedLinear& q_proj, const QuantizedLinear& k_proj,
    const QuantizedLinear& v_proj, const QuantizedLinear& o_proj,
    const Norm* q_norm, const Norm* k_norm,
    int num_heads, int num_kv_heads, int head_dim,
    float rope_theta, int rotary_dim,
    KVCache& cache, int rope_offset
) {
    int B = x.shape(0), S = x.shape(1);
    auto q = q_proj(x), k = k_proj(x), v = v_proj(x);

    q = reshape(q, {B, S, num_heads, head_dim});
    k = reshape(k, {B, S, num_kv_heads, head_dim});
    v = reshape(v, {B, S, num_kv_heads, head_dim});

    // QK norm applied BEFORE transpose (matches Swift — norm on contiguous layout)
    if (q_norm) { q = (*q_norm)(q); k = (*k_norm)(k); }

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    int rot = rotary_dim > 0 ? rotary_dim : head_dim;
    q = fast::rope(q, rot, false, rope_theta, 1.0f, rope_offset);
    k = fast::rope(k, rot, false, rope_theta, 1.0f, rope_offset);

    // KV dtype matches model's quantization scales
    auto kv_dtype = k_proj.scales ? k_proj.scales->dtype() : float16;
    k = astype(k, kv_dtype);
    v = astype(v, kv_dtype);
    cache.append(k, v, S);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn = fast::scaled_dot_product_attention(
        q, *cache.keys, *cache.values, scale, "causal");

    attn = reshape(transpose(attn, {0, 2, 1, 3}), {B, S, num_heads * head_dim});
    return o_proj(attn);
}

// ============================================================================
// Layer
// ============================================================================

struct MoEWeights {
    QuantizedLinear gate;
    // Original per-projection weights (kept for reference/fallback)
    std::optional<array> gate_w, gate_s, gate_b;
    std::optional<array> up_w, up_s, up_b;
    std::optional<array> down_w, down_s, down_b;
    // Fused gate+up weights (concatenated along output dim for single gather_qmm)
    std::optional<array> fused_gate_up_w, fused_gate_up_s, fused_gate_up_b;
    int gate_up_split = 0;  // output dim to split at (= intermediate_size / pack_factor)
    int num_experts_per_tok = 0;
    int gate_bits = 4, up_bits = 4, down_bits = 4;
    bool use_fused_gate_up = false;
};

struct QwenLayer {
    Norm input_norm, post_attn_norm;
    QuantizedLinear q_proj, k_proj, v_proj, o_proj;
    Norm q_norm, k_norm;
    bool has_qk_norm = false;

    // Dense MLP (qwen2/qwen3 dense)
    QuantizedLinear gate_proj, up_proj, down_proj;

    // MoE (qwen3_moe)
    bool is_moe = false;
    std::optional<MoEWeights> moe;

    float rope_theta = 1000000.0f;
    int rotary_dim = 0;
    int num_heads = 0, num_kv_heads = 0, head_dim = 0;
    int group_size = 64;

    KVCache cache;
};

// ============================================================================
// Model
// ============================================================================

struct QwenModel {
    QwenConfig config;
    Embedding embed_tokens;
    std::vector<QwenLayer> layers;
    Norm final_norm;

    // Tunable: how many layers per eval barrier.
    // Set via QWEN_EVAL_CADENCE env var for experimentation.
    int eval_cadence = 8;

    // Production forward — no profiling overhead
    array forward(const array& token_ids) {
        int B = token_ids.shape(0), S = token_ids.shape(1);
        auto h = embed_tokens(token_ids);

        for (int i = 0; i < (int)layers.size(); i++) {
            auto& L = layers[i];
            auto residual = h;

            auto normed = L.input_norm(h);
            auto attn_out = qwen_attention(normed,
                L.q_proj, L.k_proj, L.v_proj, L.o_proj,
                L.has_qk_norm ? &L.q_norm : nullptr,
                L.has_qk_norm ? &L.k_norm : nullptr,
                L.num_heads, L.num_kv_heads, L.head_dim,
                L.rope_theta, L.rotary_dim,
                L.cache, 0);
            h = residual + attn_out;

            residual = h;
            if (L.is_moe) {
                auto normed_ff = L.post_attn_norm(h);
                if (L.moe->use_fused_gate_up) {
                    h = residual + qwen_moe_forward_fused(normed_ff,
                        L.moe->gate,
                        *L.moe->fused_gate_up_w, *L.moe->fused_gate_up_s, L.moe->fused_gate_up_b,
                        L.moe->gate_up_split,
                        *L.moe->down_w, *L.moe->down_s, L.moe->down_b,
                        L.moe->num_experts_per_tok, L.group_size,
                        L.moe->gate_bits, L.moe->down_bits);
                } else {
                    h = residual + qwen_moe_forward(normed_ff,
                        L.moe->gate,
                        *L.moe->gate_w, *L.moe->gate_s, L.moe->gate_b,
                        *L.moe->up_w, *L.moe->up_s, L.moe->up_b,
                        *L.moe->down_w, *L.moe->down_s, L.moe->down_b,
                        L.moe->num_experts_per_tok, L.group_size,
                        L.moe->gate_bits, L.moe->up_bits, L.moe->down_bits);
                }
            } else {
                auto normed_ff = L.post_attn_norm(h);
                h = residual + qwen_mlp_forward(normed_ff,
                    L.gate_proj, L.up_proj, L.down_proj);
            }

            // No intermediate barriers — single eval at the end in qwen_run
            // matches Swift's pattern of building one massive lazy graph
        }

        return final_norm(h);
    }

    // Profiled forward — eval+sync after every stage for accurate attribution.
    // ~2-3x slower than production due to killed graph fusion.
    array forward_profiled(const array& token_ids) {
        auto& P = g_prof;
        P.reset();
        int B = token_ids.shape(0), S = token_ids.shape(1);

        // Embedding
        P.tick();
        auto h = embed_tokens(token_ids);
        P.embed_ms += P.sync_tock(h);

        for (int i = 0; i < (int)layers.size(); i++) {
            auto& L = layers[i];
            P.layer_count++;

            // Input norm
            P.tick();
            auto normed = L.input_norm(h);
            P.input_norm_ms += P.sync_tock(normed);

            // Q/K/V projections
            P.tick();
            auto q = L.q_proj(normed), k = L.k_proj(normed), v = L.v_proj(normed);
            P.qkv_proj_ms += P.sync_tock({q, k, v});

            // Reshape (norm before transpose — matches Swift contiguous layout)
            q = reshape(q, {B, S, L.num_heads, L.head_dim});
            k = reshape(k, {B, S, L.num_kv_heads, L.head_dim});
            v = reshape(v, {B, S, L.num_kv_heads, L.head_dim});

            // QK norm (before transpose — on contiguous layout)
            if (L.has_qk_norm) {
                P.tick();
                q = L.q_norm(q); k = L.k_norm(k);
                P.qk_norm_ms += P.sync_tock({q, k});
            }

            q = transpose(q, {0, 2, 1, 3});
            k = transpose(k, {0, 2, 1, 3});
            v = transpose(v, {0, 2, 1, 3});

            // RoPE
            P.tick();
            int rot = L.rotary_dim > 0 ? L.rotary_dim : L.head_dim;
            q = fast::rope(q, rot, false, L.rope_theta, 1.0f, 0);
            k = fast::rope(k, rot, false, L.rope_theta, 1.0f, 0);
            P.rope_ms += P.sync_tock({q, k});

            // KV cache append
            P.tick();
            auto kv_dtype = L.k_proj.scales ? L.k_proj.scales->dtype() : float16;
            k = astype(k, kv_dtype);
            v = astype(v, kv_dtype);
            L.cache.append(k, v, S);
            P.kv_append_ms += P.sync_tock({*L.cache.keys, *L.cache.values});

            // SDPA
            P.tick();
            float scale = 1.0f / std::sqrt(static_cast<float>(L.head_dim));
            auto attn = fast::scaled_dot_product_attention(
                q, *L.cache.keys, *L.cache.values, scale, "causal");
            P.sdpa_ms += P.sync_tock(attn);

            // O proj
            P.tick();
            attn = reshape(transpose(attn, {0, 2, 1, 3}), {B, S, L.num_heads * L.head_dim});
            auto attn_out = L.o_proj(attn);
            P.o_proj_ms += P.sync_tock(attn_out);

            // Residual add
            P.tick();
            h = h + attn_out;
            P.residual_ms += P.sync_tock(h);

            // Post-attention norm
            P.tick();
            auto normed_ff = L.post_attn_norm(h);
            P.post_norm_ms += P.sync_tock(normed_ff);

            // MoE or dense MLP
            if (L.is_moe) {
                // MoE routing (gate + softmax + topk + score normalize)
                P.tick();
                auto gates = L.moe->gate(normed_ff);
                auto scores = softmax(gates, -1, true);
                int kk = L.moe->num_experts_per_tok;
                auto inds = argpartition(negative(gates), kk - 1, -1);
                inds = slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), kk});
                auto sel_scores = take_along_axis(scores, inds, -1);
                sel_scores = sel_scores / (sum(sel_scores, -1, true) + 1e-20f);
                sel_scores = astype(sel_scores, normed_ff.dtype());
                P.moe_routing_ms += P.sync_tock({inds, sel_scores});

                // Expert sort + dispatch prep
                P.tick();
                auto x_exp = expand_dims(normed_ff, {-2, -3});
                auto flat_inds = flatten(inds);
                auto order = argsort(flat_inds);
                auto inv_order = argsort(order);
                auto x_sorted = take(flatten(x_exp, 0, -3), floor_divide(order, array(kk)), 0);
                auto sorted_inds = take(flat_inds, order, 0);
                P.moe_sort_ms += P.sync_tock({x_sorted, sorted_inds, inv_order});

                // gather_qmm: gate + up (fused or separate)
                P.tick();
                std::optional<array> gate_out_opt, up_out_opt;
                if (L.moe->use_fused_gate_up) {
                    auto fused_out = gather_qmm(x_sorted,
                        *L.moe->fused_gate_up_w, *L.moe->fused_gate_up_s, L.moe->fused_gate_up_b,
                        std::nullopt, sorted_inds, true, L.group_size, L.moe->gate_bits, "affine", true);
                    int sp = L.moe->gate_up_split;
                    gate_out_opt = slice(fused_out, {0, 0, 0, 0},
                        {fused_out.shape(0), fused_out.shape(1), fused_out.shape(2), sp});
                    up_out_opt = slice(fused_out, {0, 0, 0, sp},
                        {fused_out.shape(0), fused_out.shape(1), fused_out.shape(2), fused_out.shape(3)});
                } else {
                    gate_out_opt = gather_qmm(x_sorted, *L.moe->gate_w, *L.moe->gate_s, L.moe->gate_b,
                        std::nullopt, sorted_inds, true, L.group_size, L.moe->gate_bits, "affine", true);
                    up_out_opt = gather_qmm(x_sorted, *L.moe->up_w, *L.moe->up_s, L.moe->up_b,
                        std::nullopt, sorted_inds, true, L.group_size, L.moe->up_bits, "affine", true);
                }
                auto& gate_out = *gate_out_opt;
                auto& up_out = *up_out_opt;
                P.moe_gqmm_gate_up_ms += P.sync_tock({gate_out, up_out});

                // SiLU activation + elementwise multiply
                P.tick();
                auto hidden = (gate_out * sigmoid(gate_out)) * up_out;
                P.moe_silu_ms += P.sync_tock(hidden);

                // gather_qmm: down projection
                P.tick();
                auto down_out = gather_qmm(hidden, *L.moe->down_w, *L.moe->down_s, L.moe->down_b,
                    std::nullopt, sorted_inds, true, L.group_size, L.moe->down_bits, "affine", true);
                P.moe_gqmm_down_ms += P.sync_tock(down_out);

                // Combine (unsort + weighted sum + residual)
                P.tick();
                down_out = squeeze(down_out, {-2});
                down_out = take(down_out, inv_order, 0);
                down_out = reshape(down_out, {inds.shape(0), inds.shape(1), kk, -1});
                auto moe_out = sum(down_out * expand_dims(sel_scores, -1), -2);
                h = h + moe_out;
                P.moe_combine_ms += P.sync_tock(h);
            } else {
                P.tick();
                auto mlp_out = qwen_mlp_forward(normed_ff,
                    L.gate_proj, L.up_proj, L.down_proj);
                h = h + mlp_out;
                P.dense_mlp_ms += P.sync_tock(h);
            }

            // Eval barrier (still needed even in profiled mode for KV correctness)
            if ((i + 1) % eval_cadence == 0 || i == (int)layers.size() - 1) {
                P.tick();
                std::vector<array> to_sync = {h};
                if (L.cache.has_data()) {
                    to_sync.push_back(*L.cache.keys);
                    to_sync.push_back(*L.cache.values);
                }
                eval(to_sync);
                P.eval_barrier_ms += P.tock_ms();
            }
        }

        P.tick();
        auto out = final_norm(h);
        P.final_norm_ms += P.sync_tock(out);
        P.report(S);

        return out;
    }
};

// ============================================================================
// Global state
// ============================================================================

static std::unique_ptr<QwenModel> g_model;
static std::unordered_map<std::string, array> g_weights;
static QwenConfig g_config;
static bool g_initialized = false, g_finalized = false;

// ============================================================================
// Weight helpers
// ============================================================================

static array extract_array(void* ptr) { return *static_cast<array*>(ptr); }

static array get_w(const std::string& key) {
    auto it = g_weights.find(key);
    if (it == g_weights.end()) throw std::runtime_error("Missing weight: " + key);
    return it->second;
}

static bool has_w(const std::string& key) { return g_weights.count(key) > 0; }

static QuantizedLinear make_qlinear(const std::string& prefix) {
    int gs = g_config.quant_group_size;
    if (has_w(prefix + ".scales")) {
        auto w = get_w(prefix + ".weight");
        auto s = get_w(prefix + ".scales");
        auto bi = get_w(prefix + ".biases");
        int w_last = w.shape(-1), s_last = s.shape(-1);
        int b = (s_last > 0) ? (w_last * 32) / (s_last * gs) : g_config.quant_bits;
        if (b == 0) b = g_config.quant_bits;
        std::optional<array> lb;
        if (has_w(prefix + ".bias")) lb = get_w(prefix + ".bias");
        return {w, s, bi, lb, gs, b, true};
    }
    std::optional<array> lb;
    if (has_w(prefix + ".bias")) lb = get_w(prefix + ".bias");
    return {get_w(prefix + ".weight"), array(0.0f), array(0.0f), lb, gs, g_config.quant_bits, false};
}

static int detect_bits(const std::string& wk, const std::string& sk) {
    if (!has_w(wk) || !has_w(sk)) return g_config.quant_bits;
    return (get_w(wk).shape(-1) * 32) / (get_w(sk).shape(-1) * g_config.quant_group_size);
}

static Norm make_norm(const std::string& prefix) {
    return {get_w(prefix + ".weight"), g_config.rms_norm_eps};
}

static Embedding make_embedding(const std::string& prefix) {
    Embedding emb;
    emb.weight = get_w(prefix + ".weight");
    eval({*emb.weight});
    if (has_w(prefix + ".scales")) {
        emb.scales = get_w(prefix + ".scales");
        emb.biases = get_w(prefix + ".biases");
        eval({*emb.scales, *emb.biases});
        emb.group_size = g_config.quant_group_size;
        emb.bits = detect_bits(prefix + ".weight", prefix + ".scales");
        emb.quantized = true;
    }
    return emb;
}

// ============================================================================
// Layer builders
// ============================================================================

static void build_dense_layer(QwenLayer& layer, int idx) {
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
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;

    layer.gate_proj = make_qlinear(p + ".mlp.gate_proj");
    layer.up_proj = make_qlinear(p + ".mlp.up_proj");
    layer.down_proj = make_qlinear(p + ".mlp.down_proj");
    layer.is_moe = false;

    // Force-eval layer weights to pin GPU buffers
    std::vector<array> lw = {
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
        lw.push_back(*layer.q_proj.linear_bias);
        lw.push_back(*layer.k_proj.linear_bias);
        lw.push_back(*layer.v_proj.linear_bias);
    }
    eval(lw);
}

static void build_moe_layer(QwenLayer& layer, int idx) {
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
    }

    layer.rope_theta = g_config.rope_theta;
    layer.rotary_dim = g_config.rotary_dim;
    layer.num_heads = g_config.num_heads;
    layer.num_kv_heads = g_config.num_kv_heads;
    layer.head_dim = g_config.head_dim;
    layer.group_size = g_config.quant_group_size;

    layer.is_moe = true;
    layer.moe.emplace();
    layer.moe->gate = make_qlinear(p + ".mlp.gate");
    layer.moe->gate_w = get_w(p + ".mlp.switch_mlp.gate_proj.weight");
    layer.moe->gate_s = get_w(p + ".mlp.switch_mlp.gate_proj.scales");
    layer.moe->gate_b = get_w(p + ".mlp.switch_mlp.gate_proj.biases");
    layer.moe->up_w = get_w(p + ".mlp.switch_mlp.up_proj.weight");
    layer.moe->up_s = get_w(p + ".mlp.switch_mlp.up_proj.scales");
    layer.moe->up_b = get_w(p + ".mlp.switch_mlp.up_proj.biases");
    layer.moe->down_w = get_w(p + ".mlp.switch_mlp.down_proj.weight");
    layer.moe->down_s = get_w(p + ".mlp.switch_mlp.down_proj.scales");
    layer.moe->down_b = get_w(p + ".mlp.switch_mlp.down_proj.biases");
    std::string moe_p = p + ".mlp.switch_mlp";
    layer.moe->gate_bits = detect_bits(moe_p + ".gate_proj.weight", moe_p + ".gate_proj.scales");
    layer.moe->up_bits = detect_bits(moe_p + ".up_proj.weight", moe_p + ".up_proj.scales");
    layer.moe->down_bits = detect_bits(moe_p + ".down_proj.weight", moe_p + ".down_proj.scales");
    layer.moe->num_experts_per_tok = g_config.num_experts_per_tok;

    // Fuse gate+up weights: concatenate along output dim (axis=1) for single gather_qmm.
    // Only valid when gate and up have identical quantization config.
    // NOTE: measured ~+1% at 1k but -3% at 4k — larger N hurts kernel efficiency.
    // Disabled by default. Set QWEN_FUSE_GATE_UP=1 to enable.
    const char* fuse_env = std::getenv("QWEN_FUSE_GATE_UP");
    if (fuse_env && std::string(fuse_env) == "1" && layer.moe->gate_bits == layer.moe->up_bits) {
        auto& gw = *layer.moe->gate_w;
        auto& uw = *layer.moe->up_w;
        auto& gs = *layer.moe->gate_s;
        auto& us = *layer.moe->up_s;
        auto& gb = *layer.moe->gate_b;
        auto& ub = *layer.moe->up_b;
        layer.moe->fused_gate_up_w = concatenate({gw, uw}, 1);
        layer.moe->fused_gate_up_s = concatenate({gs, us}, 1);
        layer.moe->fused_gate_up_b = concatenate({gb, ub}, 1);
        layer.moe->gate_up_split = gw.shape(1);  // split point in output dim
        eval({*layer.moe->fused_gate_up_w, *layer.moe->fused_gate_up_s,
              *layer.moe->fused_gate_up_b});
        layer.moe->use_fused_gate_up = true;
        if (idx == 0)
            fprintf(stderr, "[qwen] Fused gate+up: [%d,%d,%d] (split@%d)\n",
                layer.moe->fused_gate_up_w->shape(0),
                layer.moe->fused_gate_up_w->shape(1),
                layer.moe->fused_gate_up_w->shape(2),
                layer.moe->gate_up_split);
    }
}

// ============================================================================
// Model builder
// ============================================================================

static QwenModel build_model() {
    QwenModel m;
    m.config = g_config;

    // Tunable eval cadence from env
    const char* cadence_env = std::getenv("QWEN_EVAL_CADENCE");
    m.eval_cadence = cadence_env ? std::atoi(cadence_env) : 8;
    fprintf(stderr, "[qwen] eval_cadence=%d\n", m.eval_cadence);

    m.embed_tokens = make_embedding("model.embed_tokens");
    m.final_norm = make_norm("model.norm");

    int N = g_config.num_layers;
    m.layers.reserve(N);
    for (int i = 0; i < N; i++) {
        m.layers.emplace_back();
        if (g_config.is_moe()) {
            build_moe_layer(m.layers[i], i);
        } else {
            build_dense_layer(m.layers[i], i);
        }
    }

    // Log MoE weight shapes for optimization analysis
    if (g_config.is_moe() && N > 0 && m.layers[0].is_moe) {
        auto& moe = *m.layers[0].moe;
        auto shape_str = [](const array& a) {
            std::string s = "[";
            for (int d = 0; d < a.ndim(); d++) {
                if (d) s += ",";
                s += std::to_string(a.shape(d));
            }
            return s + "]";
        };
        fprintf(stderr, "[qwen] MoE shapes (layer 0): gate_w=%s up_w=%s down_w=%s\n",
            shape_str(*moe.gate_w).c_str(), shape_str(*moe.up_w).c_str(),
            shape_str(*moe.down_w).c_str());
        fprintf(stderr, "[qwen] MoE scales: gate_s=%s\n", shape_str(*moe.gate_s).c_str());
        fprintf(stderr, "[qwen] MoE bits: gate=%d up=%d down=%d, group_size=%d\n",
            moe.gate_bits, moe.up_bits, moe.down_bits, m.layers[0].group_size);
    }

    fprintf(stderr, "[qwen] Model built: %d layers, %s\n",
        N, g_config.is_moe() ? "MoE" : "dense");
    return m;
}

// ============================================================================
// JSON parsing (minimal)
// ============================================================================

static std::string json_str(const std::string& j, const std::string& k) {
    auto p = j.find("\"" + k + "\"");
    if (p == std::string::npos) return "";
    p = j.find(":", p); auto s = j.find("\"", p+1); auto e = j.find("\"", s+1);
    return j.substr(s+1, e-s-1);
}
static int json_int(const std::string& j, const std::string& k, int d=0) {
    auto p = j.find("\"" + k + "\"");
    if (p == std::string::npos) return d;
    p = j.find(":", p); auto s = j.find_first_of("-0123456789", p+1);
    auto e = j.find_first_not_of("-0123456789", s);
    return std::stoi(j.substr(s, e-s));
}
static float json_float(const std::string& j, const std::string& k, float d=0) {
    auto p = j.find("\"" + k + "\"");
    if (p == std::string::npos) return d;
    p = j.find(":", p); auto s = j.find_first_of("-0123456789.", p+1);
    auto e = j.find_first_not_of("-0123456789.eE+-", s);
    return std::stof(j.substr(s, e-s));
}
static bool json_bool(const std::string& j, const std::string& k, bool d=false) {
    auto p = j.find("\"" + k + "\"");
    if (p == std::string::npos) return d;
    p = j.find(":", p); auto s = j.find_first_not_of(" \t\n", p+1);
    return j.substr(s, 4) == "true";
}

// ============================================================================
// C ABI
// ============================================================================

extern "C" {

int qwen_init(const char* config_json) {
    try {
        std::string j(config_json);
        g_config.model_type = json_str(j, "model_type");
        g_config.hidden_size = json_int(j, "hidden_size", 2048);
        g_config.num_layers = json_int(j, "num_hidden_layers", 28);
        g_config.num_heads = json_int(j, "num_attention_heads", 16);
        g_config.num_kv_heads = json_int(j, "num_key_value_heads", 2);
        g_config.head_dim = json_int(j, "head_dim", 128);
        g_config.intermediate_size = json_int(j, "intermediate_size", 8960);
        g_config.vocab_size = json_int(j, "vocab_size", 151936);
        g_config.rms_norm_eps = json_float(j, "rms_norm_eps", 1e-6f);
        g_config.rope_theta = json_float(j, "rope_theta", 1000000.0f);
        g_config.rotary_dim = json_int(j, "rotary_dim", 0);
        g_config.tie_word_embeddings = json_bool(j, "tie_word_embeddings", false);
        g_config.use_qk_norm = json_bool(j, "use_qk_norm", false);
        g_config.num_local_experts = json_int(j, "num_local_experts", 0);
        g_config.num_experts_per_tok = json_int(j, "num_experts_per_tok", 0);

        g_weights.clear();
        g_model.reset();
        g_initialized = true;
        g_finalized = false;

        // Profiling: QWEN_PROFILE=1 enables per-stage timing
        const char* prof_env = std::getenv("QWEN_PROFILE");
        g_prof.enabled = prof_env && std::string(prof_env) == "1";
        if (g_prof.enabled)
            fprintf(stderr, "[qwen] PROFILING ENABLED — will eval+sync per stage (slower)\n");

        fprintf(stderr, "[qwen] Init: type=%s layers=%d hidden=%d heads=%d/%d hd=%d experts=%d\n",
            g_config.model_type.c_str(), g_config.num_layers,
            g_config.hidden_size, g_config.num_heads, g_config.num_kv_heads,
            g_config.head_dim, g_config.num_local_experts);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen] init error: %s\n", e.what());
        return -1;
    }
}

int qwen_set_weight(const char* key, void* arr_ptr) {
    if (!g_initialized || g_finalized) return -1;
    try {
        g_weights.insert_or_assign(std::string(key), extract_array(arr_ptr));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen] set_weight error '%s': %s\n", key, e.what());
        return -1;
    }
}

int qwen_finalize(void) {
    if (!g_initialized || g_finalized) return -1;
    try {
        fprintf(stderr, "[qwen] Finalizing with %zu weights\n", g_weights.size());
        g_model = std::make_unique<QwenModel>(build_model());
        g_finalized = true;
        g_weights.clear();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen] finalize error: %s\n", e.what());
        return -1;
    }
}

int qwen_run(void* token_array_ptr, double* out_elapsed_ms) {
    if (!g_model) return -1;
    try {
        auto& tokens = *static_cast<array*>(token_array_ptr);
        auto t0 = std::chrono::high_resolution_clock::now();

        for (auto& layer : g_model->layers) layer.cache.reset();

        auto output = g_prof.enabled
            ? g_model->forward_profiled(tokens)
            : g_model->forward(tokens);

        // Final eval of all KV caches
        std::vector<array> to_eval;
        for (auto& layer : g_model->layers) {
            if (layer.cache.has_data()) {
                to_eval.push_back(*layer.cache.keys);
                to_eval.push_back(*layer.cache.values);
            }
        }
        eval(to_eval);

        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_elapsed_ms)
            *out_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen] run error: %s\n", e.what());
        return -1;
    }
}

int qwen_num_cache_layers(void) {
    return g_model ? (int)g_model->layers.size() : 0;
}

void* qwen_get_k_ptr(int i) {
    if (!g_model || i < 0 || i >= (int)g_model->layers.size()) return nullptr;
    auto& c = g_model->layers[i].cache;
    return c.keys ? new array(*c.keys) : nullptr;
}

void* qwen_get_v_ptr(int i) {
    if (!g_model || i < 0 || i >= (int)g_model->layers.size()) return nullptr;
    auto& c = g_model->layers[i].cache;
    return c.values ? new array(*c.values) : nullptr;
}

int qwen_kv_shape(int i, int* kv_heads, int* seq_len, int* head_dim) {
    if (!g_model || i < 0 || i >= (int)g_model->layers.size()) return -1;
    auto& c = g_model->layers[i].cache;
    if (!c.has_data()) return -2;
    *kv_heads = c.keys->shape(1);
    *seq_len = c.keys->shape(2);
    *head_dim = c.keys->shape(3);
    return 0;
}

void qwen_cleanup(void) {
    g_model.reset();
    g_weights.clear();
    g_initialized = false;
    g_finalized = false;
}

} // extern "C"
