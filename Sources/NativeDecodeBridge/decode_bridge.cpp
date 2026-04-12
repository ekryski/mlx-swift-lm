// decode_bridge.cpp — Native C++ decode bridge for Gemma 4 E2B
//
// Runs the full decode loop in native C++, bypassing Swift MLXArray/ARC overhead.
// Same weight-sharing pattern as prefill_bridge_v2: Swift passes MLXArray.ctx
// pointers, we copy the mlx::core::array metadata (cheap, shares GPU buffer).
//
// Key difference from prefill bridge:
//   - Manages its own KV cache (appends each token's K/V)
//   - Runs lm_head to produce logits
//   - Can do argmax sampling in C++ (no Swift round-trip per token)
//   - Supports db_generate() for tight multi-token decode loops
//
// Build:
//   clang++ -std=c++20 -O3 -shared -fPIC \
//     -I/opt/homebrew/lib/python3.13/site-packages/mlx/include \
//     -I$(dirname $0) \
//     -L/opt/homebrew/lib/python3.13/site-packages/mlx/lib -lmlx \
//     -framework Metal -framework Foundation -framework Accelerate \
//     -Wl,-rpath,/opt/homebrew/lib/python3.13/site-packages/mlx/lib \
//     -o libdecode_bridge.dylib decode_bridge.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile_impl.h"
#include "decode_bridge.h"

using namespace mlx::core;

// ============================================================================
// Architecture constants
// ============================================================================
static int g_non_shared_layers = 0;  // layers with own KV cache (15)
static int g_all_layers       = 0;  // total layers including shared (35)
static int g_hidden_size      = 0;
static int g_num_heads        = 0;
static int g_num_kv_heads     = 0;
static int g_sliding_window   = 0;
static int g_sliding_pattern  = 0;
static int g_vocab_size       = 0;
static int g_total_layers     = 0;  // same as g_all_layers (for PLE indexing)
static int g_hidden_per_layer = 0;

static int g_head_dim_slide   = 256;
static int g_head_dim_full    = 512;

// Per-layer info: donor mapping and layer types
static std::vector<int> g_donor_map;        // donor_map[i] = cache index to use for layer i
static std::vector<bool> g_layer_is_shared; // true if layer reuses donor's KV
static std::vector<bool> g_layer_is_full;   // true if full attention (vs sliding)

static constexpr int BITS       = 4;
static constexpr int GROUP_SIZE = 64;
static constexpr float RMS_EPS  = 1e-6f;

static constexpr float ROPE_THETA_SLIDING = 10000.0f;
static constexpr float ROPE_THETA_FULL    = 1000000.0f;

// Stub flags for per-block profiling (set via db_set_stub)
static bool g_stub_mlp  = false;
static bool g_stub_attn = false;
static bool g_stub_ple  = false;

static bool is_full_attention(int layer_idx) {
    return g_sliding_pattern > 0 && ((layer_idx + 1) % g_sliding_pattern == 0);
}

// ============================================================================
// GELU approximate (must match Python/prefill bridge exactly)
// Precomputed bf16 constants to avoid heap allocations per call.
// ============================================================================
static array g_gelu_half = array(0.5f, bfloat16);
static array g_gelu_one = array(1.0f, bfloat16);
static array g_gelu_coeff = array(0.044715f, bfloat16);
static array g_gelu_sqrt2pi = array(0.7978845608028654f, bfloat16);
static array g_gelu_three = array(3, bfloat16);

static array gelu_approx(const array& x) {
    auto x3 = power(x, g_gelu_three);
    auto inner = g_gelu_sqrt2pi * (x + g_gelu_coeff * x3);
    return g_gelu_half * x * (g_gelu_one + tanh(inner));
}

// ============================================================================
// Layer components (same as prefill bridge)
// ============================================================================
struct QuantizedLinear {
    array weight, scales, biases;

    array operator()(const array& x) const {
        return quantized_matmul(
            x, weight, scales, biases,
            true, GROUP_SIZE, BITS, "affine");
    }
};

struct RMSNorm {
    array weight;
    array operator()(const array& x) const {
        return fast::rms_norm(x, weight, RMS_EPS);
    }
};

static array rms_norm_no_scale(const array& x) {
    return fast::rms_norm(x, std::nullopt, RMS_EPS);
}

struct QuantizedEmbedding {
    array weight, scales, biases;

    array operator()(const array& indices) const {
        auto w = take(weight, indices, 0);
        auto s = take(scales, indices, 0);
        auto b = take(biases, indices, 0);
        return dequantize(w, s, b, GROUP_SIZE, BITS, "affine");
    }
};

struct ScaledLinear {
    array weight;
    float scalar;

    array operator()(const array& x) const {
        auto out = matmul(x, transpose(weight));
        return out * array(scalar, bfloat16);
    }
};

// ============================================================================
// KV Cache — pre-allocated with slice assignment (matches Swift's KVCacheSimple)
//
// Avoids O(n²) concatenate-per-token. Pre-allocates in chunks of `step` and
// uses slice assignment for O(1) per-token append.
// ============================================================================
struct NativeKVCache {
    array keys = array(0.0f);    // [1, kv_heads, capacity, head_dim]
    array values = array(0.0f);  // [1, kv_heads, capacity, head_dim]
    int offset = 0;
    int capacity = 0;
    int kv_heads = 0;
    int head_dim = 0;
    bool has_data = false;
    static constexpr int step = 256;

    void init(int kv_h, int hd) {
        kv_heads = kv_h;
        head_dim = hd;
        offset = 0;
        capacity = 0;
        has_data = false;
    }

    // Import from external arrays (e.g., after prefill)
    void import_from(const array& k, const array& v) {
        keys = k;
        values = v;
        offset = k.shape(2);
        capacity = k.shape(2);
        kv_heads = k.shape(1);
        head_dim = k.shape(3);
        has_data = true;
    }

    // Ensure capacity for at least `needed` total tokens
    void ensure_capacity(int needed) {
        if (needed <= capacity) return;

        int new_cap = ((needed + step - 1) / step) * step;
        auto k_ext = zeros({1, kv_heads, new_cap - capacity, head_dim}, bfloat16);
        auto v_ext = zeros({1, kv_heads, new_cap - capacity, head_dim}, bfloat16);

        if (has_data) {
            // Only keep the valid portion [0..offset), then append zeros
            auto k_valid = slice(keys, {0, 0, 0, 0}, {1, kv_heads, offset, head_dim});
            auto v_valid = slice(values, {0, 0, 0, 0}, {1, kv_heads, offset, head_dim});
            keys = concatenate({k_valid, k_ext}, 2);
            values = concatenate({v_valid, v_ext}, 2);
        } else {
            keys = k_ext;
            values = v_ext;
        }
        capacity = new_cap;
        has_data = true;
    }

    // Append new K/V [1, kv_heads, 1, head_dim] and return valid history
    std::pair<array, array> update(const array& new_k, const array& new_v) {
        ensure_capacity(offset + 1);

        // In-place slice update — no new array allocation per token
        keys = slice_update(keys, new_k,
            {0, 0, offset, 0},
            {1, kv_heads, offset + 1, head_dim});
        values = slice_update(values, new_v,
            {0, 0, offset, 0},
            {1, kv_heads, offset + 1, head_dim});

        offset += 1;

        // Return valid portion [0..offset)
        auto ret_k = slice(keys, {0, 0, 0, 0}, {1, kv_heads, offset, head_dim});
        auto ret_v = slice(values, {0, 0, 0, 0}, {1, kv_heads, offset, head_dim});
        return {ret_k, ret_v};
    }
};

// ============================================================================
// Attention with KV cache for decode
// ============================================================================
struct Attention {
    QuantizedLinear q_proj, k_proj, v_proj, o_proj;
    RMSNorm q_norm, k_norm;
    int head_dim;
    int num_heads;
    int num_kv_heads;
    bool is_sliding;
    float rope_theta;
    array rope_freqs;     // raw freqs for fast::rope
    array inv_freqs;      // 1/freqs for fast::rms_norm_rope fusion

    // Normal path: own KV cache
    array operator()(const array& x, NativeKVCache& cache,
                     const array* input_norm_w = nullptr) const {
        return forward(x, &cache, nullptr, nullptr, 0, input_norm_w);
    }

    // Shared KV path: reuse donor's cached K/V
    array shared_kv(const array& x, const array& donor_k, const array& donor_v,
                    int donor_offset, const array* input_norm_w = nullptr) const {
        return forward(x, nullptr, &donor_k, &donor_v, donor_offset, input_norm_w);
    }

private:
    // Fused rms_norm + quantized GEMV (single kernel for T=1 decode)
    static array fused_norm_qgemv(const array& x, const array& norm_w,
                                   const QuantizedLinear& proj) {
        static int dbg = 0;
        if (dbg < 1) {
            fprintf(stderr, "[db] Using fused_norm_qgemv: x.shape=[%d,%d,%d] x.dtype=%s\n",
                (int)x.shape(0), x.ndim() > 1 ? (int)x.shape(1) : -1,
                x.ndim() > 2 ? (int)x.shape(2) : -1,
                x.dtype() == bfloat16 ? "bf16" : "other");
            dbg++;
        }
        return fast::rms_norm_qgemv(
            x, norm_w, proj.weight, proj.scales, proj.biases,
            RMS_EPS, GROUP_SIZE);
    }

    // Compiled QKV projections: norm + Q/K/V proj + reshape + transpose + V norm
    // Compiled per-layer (keyed by this+1 to distinguish from post-attn compile)
    struct QKVCompiledState {
        std::function<std::vector<array>(const std::vector<array>&)> fn;
        bool ready = false;
    };

    // Compiled Q-only projection for shared layers
    struct QCompiledState {
        std::function<std::vector<array>(const std::vector<array>&)> fn;
        bool ready = false;
    };

    array forward(const array& x, NativeKVCache* cache,
                  const array* shared_k, const array* shared_v,
                  int ext_offset, const array* input_norm_w) const {
        int B = x.shape(0);
        int S = x.shape(1);
        int rope_offset = cache ? cache->offset : ext_offset;

        array q = array(0.0f);
        array full_k = array(0.0f);
        array full_v = array(0.0f);
        int seq_len = 0;

        if (cache) {
            // Non-shared: compile Q/K/V proj + reshapes + V norm (x is already normed)
            static std::unordered_map<std::uintptr_t, QKVCompiledState> qkv_cache;
            auto& state = qkv_cache[(std::uintptr_t)this];
            if (!state.ready) {
                auto fn = [this](const std::vector<array>& ins) -> std::vector<array> {
                    auto qq = transpose(reshape(q_proj(ins[0]), {1, 1, num_heads, head_dim}), {0, 2, 1, 3});
                    auto kk = transpose(reshape(k_proj(ins[0]), {1, 1, num_kv_heads, head_dim}), {0, 2, 1, 3});
                    auto vv = rms_norm_no_scale(
                        transpose(reshape(v_proj(ins[0]), {1, 1, num_kv_heads, head_dim}), {0, 2, 1, 3}));
                    return {qq, kk, vv};
                };
                state.fn = mlx::core::detail::compile(
                    fn, (std::uintptr_t)this + 1, true);
                state.ready = true;
            }
            auto qkv = state.fn({x});  // x is already normed by TransformerLayer
            q = qkv[0];
            auto k = qkv[1];
            auto v = qkv[2];

            // Q/K norm + RoPE (uses offset — can't compile)
            if (inv_freqs.size() > 0) {
                q = fast::rms_norm_rope(q, q_norm.weight, inv_freqs,
                    RMS_EPS, rope_offset, num_heads, S);
                k = fast::rms_norm_rope(k, k_norm.weight, inv_freqs,
                    RMS_EPS, rope_offset, num_kv_heads, S);
            } else {
                q = q_norm(q);
                q = fast::rope(q, head_dim, false, rope_theta, 1.0f, rope_offset);
                k = k_norm(k);
                k = fast::rope(k, head_dim, false, rope_theta, 1.0f, rope_offset);
            }

            k = astype(k, bfloat16);
            v = astype(v, bfloat16);

            auto [ck, cv] = cache->update(k, v);
            full_k = ck;
            full_v = cv;
            seq_len = cache->offset;
        } else {
            // Shared layer: compile norm + Q proj + reshape + transpose
            static std::unordered_map<std::uintptr_t, QCompiledState> q_cache;
            auto& qstate = q_cache[(std::uintptr_t)this];
            if (!qstate.ready) {
                auto fn = [this](const std::vector<array>& ins) -> std::vector<array> {
                    auto qq = transpose(reshape(q_proj(ins[0]), {1, 1, num_heads, head_dim}), {0, 2, 1, 3});
                    return {qq};
                };
                qstate.fn = mlx::core::detail::compile(
                    fn, (std::uintptr_t)this + 2, true);
                qstate.ready = true;
            }
            q = q_cache[(std::uintptr_t)this].fn({x})[0];  // x already normed

            // Q norm + RoPE (uses offset)
            if (inv_freqs.size() > 0) {
                q = fast::rms_norm_rope(q, q_norm.weight, inv_freqs,
                    RMS_EPS, rope_offset, num_heads, S);
            } else {
                q = q_norm(q);
                q = fast::rope(q, head_dim, false, rope_theta, 1.0f, rope_offset);
            }

            full_k = *shared_k;
            full_v = *shared_v;
            seq_len = full_k.shape(2);
        }

        float scale = 1.0f;
        array attn_out = array(0.0f);

        if (is_sliding && seq_len > g_sliding_window) {
            int start = seq_len - g_sliding_window;
            auto windowed_k = slice(full_k, {0, 0, start, 0},
                                    {B, num_kv_heads, seq_len, head_dim});
            auto windowed_v = slice(full_v, {0, 0, start, 0},
                                    {B, num_kv_heads, seq_len, head_dim});
            attn_out = fast::scaled_dot_product_attention(
                q, windowed_k, windowed_v, scale, "");
        } else {
            attn_out = fast::scaled_dot_product_attention(
                q, full_k, full_v, scale, "");
        }

        attn_out = reshape(
            transpose(attn_out, {0, 2, 1, 3}),
            {B, S, num_heads * head_dim});

        return o_proj(attn_out);
    }
};

// Compiled GEGLU: matches Swift's compiledGeglu exactly.
// compile(shapeless: true) { gate, x in geluApproximate(gate) * x }
static std::function<std::vector<array>(const std::vector<array>&)> g_compiled_geglu;
static bool g_geglu_compiled = false;
static char g_geglu_id_anchor;  // stable address for compile ID

static array compiled_geglu(const array& gate, const array& up) {
    if (!g_geglu_compiled) {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            return {gelu_approx(inputs[0]) * inputs[1]};
        };
        g_compiled_geglu = mlx::core::detail::compile(
            fn, (std::uintptr_t)&g_geglu_id_anchor, /*shapeless=*/true);
        g_geglu_compiled = true;
    }
    return g_compiled_geglu({gate, up})[0];
}

// Compiled GeluMul for PLE: matches Swift's compiledGeluMul
static std::function<std::vector<array>(const std::vector<array>&)> g_compiled_gelu_mul;
static bool g_gelu_mul_compiled = false;
static char g_gelu_mul_id_anchor;

static array compiled_gelu_mul(const array& gate, const array& x) {
    if (!g_gelu_mul_compiled) {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            return {gelu_approx(inputs[0]) * inputs[1]};
        };
        g_compiled_gelu_mul = mlx::core::detail::compile(
            fn, (std::uintptr_t)&g_gelu_mul_id_anchor, /*shapeless=*/true);
        g_gelu_mul_compiled = true;
    }
    return g_compiled_gelu_mul({gate, x})[0];
}

struct MLP {
    QuantizedLinear gate_proj, up_proj, down_proj;

    array operator()(const array& x) const {
        // Matches Swift: downProj(compiledGeglu(gateProj(x), upProj(x)))
        return down_proj(compiled_geglu(gate_proj(x), up_proj(x)));
    }
};

struct TransformerLayer {
    RMSNorm input_layernorm;
    Attention self_attn;
    RMSNorm post_attention_layernorm;
    RMSNorm pre_feedforward_layernorm;
    MLP mlp;
    RMSNorm post_feedforward_layernorm;

    QuantizedLinear per_layer_input_gate;
    QuantizedLinear per_layer_projection;
    RMSNorm post_per_layer_input_norm;

    array layer_scalar;

    // Normal: own KV cache, fused input_layernorm + attention projections
    array operator()(const array& x, const array& per_layer_input,
                     NativeKVCache& cache) const {
        // Match Swift: TransformerBlock does input_layernorm, attention receives normed x
        auto normed = input_layernorm(x);
        auto attn_out = g_stub_attn ? x : self_attn(normed, cache, nullptr);
        return forward_impl(x, per_layer_input, attn_out);
    }

    array with_shared_kv(const array& x, const array& per_layer_input,
                         const array& donor_k, const array& donor_v, int donor_offset) const {
        auto normed = input_layernorm(x);
        auto attn_out = g_stub_attn ? x :
            self_attn.shared_kv(normed, donor_k, donor_v, donor_offset, nullptr);
        return forward_impl(x, per_layer_input, attn_out);
    }

private:
    // Compiled post-attention block — lazily initialized on first use
    // Uses static map keyed by layer address to avoid aggregate init issues
    struct CompiledState {
        std::function<std::vector<array>(const std::vector<array>&)> fn;
        bool ready = false;
    };

    array forward_impl(const array& x, const array& per_layer_input,
                       const array& attn_out_raw) const {
        // Use static map for per-layer compiled state (avoids aggregate init issues)
        static std::unordered_map<std::uintptr_t, CompiledState> compiled_cache;
        auto& state = compiled_cache[(std::uintptr_t)this];
        if (!state.ready) {
            auto fn = [this](const std::vector<array>& inputs) -> std::vector<array> {
                return {forward_impl_pure(inputs[0], inputs[1], inputs[2])};
            };
            state.fn = mlx::core::detail::compile(
                fn, (std::uintptr_t)this, /*shapeless=*/true);
            state.ready = true;
        }
        return state.fn({x, per_layer_input, attn_out_raw})[0];
    }

    array forward_impl_pure(const array& x, const array& per_layer_input,
                            const array& attn_out_raw) const {
        auto residual = x;
        auto h = attn_out_raw;
        h = post_attention_layernorm(h);
        h = residual + h;

        // Pre-norm MLP with residual
        residual = h;
        auto ff = pre_feedforward_layernorm(h);
        ff = mlp(ff);
        ff = post_feedforward_layernorm(ff);
        h = residual + ff;

        // Per-layer input gating
        residual = h;
        auto gate = per_layer_input_gate(h);
        gate = gelu_approx(gate) * per_layer_input;
        gate = per_layer_projection(gate);
        gate = post_per_layer_input_norm(gate);
        h = residual + gate;

        // Layer scalar
        h = h * layer_scalar;
        return h;
    }
};

struct DecodeModel {
    QuantizedEmbedding embed_tokens;
    QuantizedEmbedding embed_tokens_per_layer;
    ScaledLinear per_layer_model_projection;
    RMSNorm per_layer_projection_norm;
    std::vector<TransformerLayer> layers;
    RMSNorm final_norm;

    // lm_head: for tied embeddings, use embed_tokens dequantize + matmul
    // For separate lm_head, store it as QuantizedLinear
    bool tie_word_embeddings = true;
    QuantizedLinear lm_head;  // only used when !tie_word_embeddings

    // Logit softcapping
    float final_logit_softcapping = 0.0f;

    // KV caches (one per non-shared layer)
    std::vector<NativeKVCache> caches;

    // Persistent donor KV storage for shared layer access
    struct DonorKV {
        array k; array v; int offset;
        DonorKV() : k(0.0f), v(0.0f), offset(0) {}
    };
    std::vector<DonorKV> donor_kvs;

    // Precomputed scales and constants
    array embed_scale_arr = array(0.0f);
    array pl_embed_scale_arr = array(0.0f);
    array pl_input_scale_arr = array(0.0f);
    array softcap_arr = array(0.0f);
    array full_embed_dequant = array(0.0f);  // precomputed dequantized embedding table
    bool embed_dequant_ready = false;

    void precompute_scales() {
        embed_scale_arr = array(std::sqrt(static_cast<float>(g_hidden_size)), bfloat16);
        pl_embed_scale_arr = array(std::sqrt(static_cast<float>(g_hidden_per_layer)), bfloat16);
        pl_input_scale_arr = array(std::pow(2.0f, -0.5f), bfloat16);
        if (final_logit_softcapping > 0.0f) {
            softcap_arr = array(final_logit_softcapping, bfloat16);
        }
    }

    void precompute_embed_dequant() {
        if (!embed_dequant_ready && tie_word_embeddings) {
            auto raw = dequantize(
                embed_tokens.weight, embed_tokens.scales, embed_tokens.biases,
                GROUP_SIZE, BITS, "affine");
            eval(raw);
            fprintf(stderr, "[db] Embed raw shape: [%d,%d]\n",
                (int)raw.shape(0), (int)raw.shape(1));
            full_embed_dequant = transpose(raw);
            eval(full_embed_dequant);
            fprintf(stderr, "[db] Embed transposed shape: [%d,%d]\n",
                (int)full_embed_dequant.shape(0), (int)full_embed_dequant.shape(1));
            embed_dequant_ready = true;
        }
    }

    void init_caches() {
        caches.resize(g_non_shared_layers);
        for (int i = 0; i < g_non_shared_layers; i++) {
            int hd = g_layer_is_full[i] ? g_head_dim_full : g_head_dim_slide;
            caches[i].init(g_num_kv_heads, hd);
        }
        donor_kvs.resize(g_non_shared_layers);
    }

    // Single token decode step → logits [1, 1, vocab_size]
    // Accepts token as MLX array (avoids CPU extraction from previous step's argmax)
    array step_from_array(const array& token_arr) {
        // Input is [1, 1] from Swift's inputs array — use directly if shape matches
        if (token_arr.ndim() == 2 && token_arr.shape(0) == 1 && token_arr.shape(1) == 1) {
            return step_impl(token_arr.dtype() == int32 ? token_arr : astype(token_arr, int32));
        }
        // Fallback: reshape scalar or [1] to [1, 1]
        return step_impl(reshape(astype(token_arr, int32), {1, 1}));
    }

    array step(int32_t token_id) {
        auto tokens = array(&token_id, {1, 1}, int32);
        return step_impl(tokens);
    }

    array step_impl(const array& tokens) {

        auto h = embed_tokens(tokens) * embed_scale_arr;

        // Per-layer embeddings
        auto per_layer_inputs = embed_tokens_per_layer(tokens) * pl_embed_scale_arr;
        per_layer_inputs = reshape(per_layer_inputs,
            {1, 1, g_total_layers, g_hidden_per_layer});

        auto per_layer_proj = per_layer_model_projection(h);
        per_layer_proj = reshape(per_layer_proj,
            {1, 1, g_total_layers, g_hidden_per_layer});
        per_layer_proj = per_layer_projection_norm(per_layer_proj);

        auto combined_per_layer = (per_layer_proj + per_layer_inputs) * pl_input_scale_arr;

        // Layer loop: all 35 layers, shared layers use donor's KV
        // Reuse persistent donor KV storage (avoids 30 dummy array allocations per token)
        for (auto& dkv : donor_kvs) { dkv.offset = 0; }

        for (int i = 0; i < g_all_layers; i++) {
            auto pli = slice(combined_per_layer,
                             {0, 0, i, 0},
                             {1, 1, i + 1, g_hidden_per_layer});
            pli = reshape(pli, {1, 1, g_hidden_per_layer});

            if (!g_layer_is_shared[i]) {
                // Non-shared layer: has own cache
                int cache_idx = g_donor_map[i];  // maps to cache index (0-14)
                int pre_offset = caches[cache_idx].offset;
                h = layers[i](h, pli, caches[cache_idx]);
                // Store K/V for shared layers to reference
                auto ret_k = slice(caches[cache_idx].keys,
                    {0, 0, 0, 0}, {1, g_num_kv_heads, caches[cache_idx].offset,
                     caches[cache_idx].head_dim});
                auto ret_v = slice(caches[cache_idx].values,
                    {0, 0, 0, 0}, {1, g_num_kv_heads, caches[cache_idx].offset,
                     caches[cache_idx].head_dim});
                donor_kvs[cache_idx].k = ret_k;
                donor_kvs[cache_idx].v = ret_v;
                donor_kvs[cache_idx].offset = pre_offset;
            } else {
                // Shared layer: use donor's KV
                int donor_idx = g_donor_map[i];
                auto& dkv = donor_kvs[donor_idx];
                h = layers[i].with_shared_kv(h, pli, dkv.k, dkv.v, dkv.offset);
            }
        }

        h = final_norm(h);

        // lm_head → logits
        array logits = array(0.0f);
        if (tie_word_embeddings) {
            // Use quantized_matmul directly on packed embedding weights
            // Matches Swift's QuantizedEmbedding.asLinear() which calls quantizedMM
            logits = quantized_matmul(
                h, embed_tokens.weight, embed_tokens.scales, embed_tokens.biases,
                /*transpose=*/true, GROUP_SIZE, BITS, "affine");
        } else {
            logits = lm_head(h);
        }

        // Logit softcapping (precomputed constant)
        if (final_logit_softcapping > 0.0f) {
            logits = tanh(logits / softcap_arr) * softcap_arr;
        }


        return logits;
    }

    // Argmax sample from logits
    int32_t sample_argmax(const array& logits) {
        // logits: [1, 1, vocab_size] → take argmax over last dim
        auto token = argmax(reshape(logits, {-1}));
        eval(token);
        return token.item<int32_t>();
    }
};

// ============================================================================
// Global state
// ============================================================================
static std::unique_ptr<DecodeModel> g_model = nullptr;
static std::unordered_map<std::string, array> g_weight_store;
static bool g_initialized = false;
static bool g_finalized   = false;

// ============================================================================
// Weight extraction (same as prefill bridge)
// ============================================================================
static array extract_array(void* arr_ptr) {
    if (!arr_ptr) throw std::runtime_error("db: null arr_ptr");
    auto* cpp_array = static_cast<mlx::core::array*>(arr_ptr);
    return *cpp_array;
}

static array get_weight(const std::string& key) {
    auto it = g_weight_store.find(key);
    if (it != g_weight_store.end()) return it->second;
    fprintf(stderr, "[db] WARNING: weight not found: %s\n", key.c_str());
    throw std::runtime_error("Missing weight: " + key);
}

static QuantizedLinear make_quantized_linear(const std::string& prefix) {
    return {
        get_weight(prefix + ".weight"),
        get_weight(prefix + ".scales"),
        get_weight(prefix + ".biases"),
    };
}

static RMSNorm make_rms_norm(const std::string& prefix) {
    return {get_weight(prefix + ".weight")};
}

static QuantizedEmbedding make_quantized_embedding(const std::string& prefix) {
    return {
        get_weight(prefix + ".weight"),
        get_weight(prefix + ".scales"),
        get_weight(prefix + ".biases"),
    };
}

static Attention make_attention(const std::string& prefix, int layer_idx) {
    bool full = is_full_attention(layer_idx);
    int hd = full ? g_head_dim_full : g_head_dim_slide;
    float theta = full ? ROPE_THETA_FULL : ROPE_THETA_SLIDING;

    // Proportional RoPE freqs for full-attention layers
    array freqs({}, float32);
    if (full) {
        int rotated_dims = hd / 4;  // partial_rotary_factor=0.25
        auto exponents = arange(0, rotated_dims, 2, float32);
        exponents = exponents / static_cast<float>(hd);
        auto rotated_freqs = power(array(theta, float32), exponents);
        auto inf_freqs = mlx::core::full(
            {(hd - rotated_dims) / 2},
            std::numeric_limits<float>::infinity(), float32);
        freqs = concatenate({rotated_freqs, inf_freqs});
    }

    // Compute inv_freqs for fused rms_norm_rope
    // inv_freqs = 1.0 / freqs (inf → 0 for unrotated dims = identity)
    array inv = array({}, float32);
    if (freqs.size() > 0) {
        inv = array(1.0f) / freqs;
        eval(inv);
    } else if (!full) {
        // Sliding: compute standard inv_freqs
        auto exp = arange(0, hd, 2, float32) / static_cast<float>(hd);
        auto sfreqs = power(array(theta, float32), exp);
        inv = array(1.0f) / sfreqs;
        eval(inv);
    }

    return {
        make_quantized_linear(prefix + ".q_proj"),
        make_quantized_linear(prefix + ".k_proj"),
        make_quantized_linear(prefix + ".v_proj"),
        make_quantized_linear(prefix + ".o_proj"),
        make_rms_norm(prefix + ".q_norm"),
        make_rms_norm(prefix + ".k_norm"),
        hd, g_num_heads, g_num_kv_heads,
        !full, theta, freqs, inv,
    };
}

static MLP make_mlp(const std::string& prefix) {
    return {
        make_quantized_linear(prefix + ".gate_proj"),
        make_quantized_linear(prefix + ".up_proj"),
        make_quantized_linear(prefix + ".down_proj"),
    };
}

static TransformerLayer make_layer(int layer_idx) {
    std::string prefix = "layers." + std::to_string(layer_idx);
    return TransformerLayer {
        make_rms_norm(prefix + ".input_layernorm"),
        make_attention(prefix + ".self_attn", layer_idx),
        make_rms_norm(prefix + ".post_attention_layernorm"),
        make_rms_norm(prefix + ".pre_feedforward_layernorm"),
        make_mlp(prefix + ".mlp"),
        make_rms_norm(prefix + ".post_feedforward_layernorm"),
        make_quantized_linear(prefix + ".per_layer_input_gate"),
        make_quantized_linear(prefix + ".per_layer_projection"),
        make_rms_norm(prefix + ".post_per_layer_input_norm"),
        get_weight(prefix + ".layer_scalar"),
    };
}

static DecodeModel build_model() {
    auto embed_tokens = make_quantized_embedding("embed_tokens");
    auto embed_tokens_per_layer = make_quantized_embedding("embed_tokens_per_layer");

    float projection_scalar = 1.0f / std::sqrt(static_cast<float>(g_hidden_size));
    ScaledLinear per_layer_model_projection = {
        get_weight("per_layer_model_projection.weight"),
        projection_scalar,
    };
    auto per_layer_projection_norm = make_rms_norm("per_layer_projection_norm");

    std::vector<TransformerLayer> layers;
    layers.reserve(g_all_layers);
    for (int i = 0; i < g_all_layers; i++) {
        layers.push_back(make_layer(i));
    }

    auto final_norm = make_rms_norm("norm");

    // Check for separate lm_head
    bool tied = true;
    QuantizedLinear lm_head_linear = {array(0.0f), array(0.0f), array(0.0f)};
    try {
        lm_head_linear = make_quantized_linear("lm_head");
        tied = false;
    } catch (...) {
        // No lm_head weights → tied embeddings
        tied = true;
    }

    // Check for softcapping
    // TODO: make configurable — Gemma 4 E2B uses softcap=30.0
    float softcap = 30.0f;

    return {
        std::move(embed_tokens),
        std::move(embed_tokens_per_layer),
        std::move(per_layer_model_projection),
        std::move(per_layer_projection_norm),
        std::move(layers),
        std::move(final_norm),
        tied,
        std::move(lm_head_linear),
        softcap,
    };
}

// ============================================================================
// C ABI
// ============================================================================
extern "C" {

int db_init(int num_layers, int hidden_size, int num_heads, int num_kv_heads,
            int sliding_window, int sliding_window_pattern,
            int vocab_size, int total_layers, int hidden_per_layer) {
    try {
        g_non_shared_layers = num_layers;
        g_all_layers      = total_layers;
        g_hidden_size     = hidden_size;
        g_num_heads       = num_heads;
        g_num_kv_heads    = num_kv_heads;
        g_sliding_window  = sliding_window;
        g_sliding_pattern = sliding_window_pattern;
        g_vocab_size      = vocab_size;
        g_total_layers    = total_layers;
        g_hidden_per_layer = hidden_per_layer;

        g_head_dim_slide   = 256;
        g_head_dim_full    = 512;

        // Build per-layer info: types, donor mapping
        g_layer_is_full.resize(total_layers);
        g_layer_is_shared.resize(total_layers);
        g_donor_map.resize(total_layers);

        // Determine which layers are full vs sliding attention
        for (int i = 0; i < total_layers; i++) {
            g_layer_is_full[i] = is_full_attention(i);
        }

        // Build donor mapping (matches Swift's previousKVs logic):
        // Non-shared layers (0..M-1) map to themselves
        // Shared layers (M..N-1) map to last non-shared layer of same type
        int M = num_layers;  // non-shared count
        int last_sliding = -1, last_full = -1;
        for (int i = 0; i < M; i++) {
            g_donor_map[i] = i;
            g_layer_is_shared[i] = false;
            if (g_layer_is_full[i]) last_full = i;
            else last_sliding = i;
        }
        for (int i = M; i < total_layers; i++) {
            g_layer_is_shared[i] = true;
            g_donor_map[i] = g_layer_is_full[i] ? last_full : last_sliding;
        }

        g_weight_store.clear();
        g_model.reset();
        g_initialized = true;
        g_finalized   = false;

        fprintf(stderr, "[db] Initialized: %d non-shared + %d shared = %d layers, hidden=%d, heads=%d/%d\n",
                num_layers, total_layers - num_layers, total_layers,
                hidden_size, num_heads, num_kv_heads);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] init error: %s\n", e.what());
        return -1;
    }
}

int db_set_weight(const char* key, void* arr_ptr) {
    if (!g_initialized || g_finalized) return -1;
    try {
        g_weight_store.insert_or_assign(std::string(key), extract_array(arr_ptr));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] set_weight error for '%s': %s\n", key, e.what());
        return -1;
    }
}

int db_finalize(void) {
    if (!g_initialized || g_finalized) return -1;
    try {
        fprintf(stderr, "[db] Finalizing with %zu weight tensors\n", g_weight_store.size());
        g_model = std::make_unique<DecodeModel>(build_model());
        g_model->precompute_scales();
        g_model->precompute_embed_dequant();
        g_model->init_caches();
        g_finalized = true;
        g_weight_store.clear();
        fprintf(stderr, "[db] Model built, %d KV caches initialized\n", g_non_shared_layers);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] finalize error: %s\n", e.what());
        return -1;
    }
}

int db_import_kv(int layer_idx, void* k_ptr, void* v_ptr) {
    if (!g_model || layer_idx < 0 || layer_idx >= g_non_shared_layers) return -1;
    try {
        auto k = extract_array(k_ptr);
        auto v = extract_array(v_ptr);
        g_model->caches[layer_idx].import_from(k, v);
        if (layer_idx == 0) {
            eval(k);
            auto ks = astype(sum(k), float32);
            eval(ks);
            fprintf(stderr, "[db] Layer 0 KV imported: K shape=[%d,%d,%d,%d] sum=%.4f\n",
                (int)k.shape(0), (int)k.shape(1), (int)k.shape(2), (int)k.shape(3),
                ks.item<float>());
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] import_kv error layer %d: %s\n", layer_idx, e.what());
        return -1;
    }
}

int db_set_cache_offset(int offset) {
    if (!g_model) return -1;
    for (auto& cache : g_model->caches) {
        cache.offset = offset;
    }
    return 0;
}

int32_t db_step(int32_t token_id) {
    if (!g_model) return -1;
    try {
        auto logits = g_model->step(token_id);
        return g_model->sample_argmax(logits);
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] step error: %s\n", e.what());
        return -1;
    }
}

int db_step_logits(int32_t token_id, float* out_logits, int vocab_size) {
    if (!g_model) return -1;
    try {
        auto logits = g_model->step(token_id);
        // Flatten and convert to float32
        logits = astype(reshape(logits, {-1}), float32);
        eval(logits);

        int n = std::min(vocab_size, (int)logits.size());
        std::memcpy(out_logits, logits.data<float>(), n * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] step_logits error: %s\n", e.what());
        return -1;
    }
}

int db_generate(int32_t first_token_id, int max_tokens, int32_t* out_tokens,
                int32_t eos_token_id, double* out_elapsed_ms) {
    if (!g_model) return -1;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        int32_t current_token = first_token_id;
        int generated = 0;

        // Pipelined decode: build graph for token N+1 while GPU evaluates token N.
        // Step 0: build + eval synchronously (no previous work to overlap with)
        auto logits = g_model->step(current_token);
        auto token_arr = argmax(reshape(logits, {-1}));
        async_eval(token_arr);

        for (int i = 0; i < max_tokens; i++) {
            // Wait for current token (GPU work dispatched in previous iteration)
            int32_t next_token = token_arr.item<int32_t>();
            out_tokens[i] = next_token;
            generated++;

            if (next_token == eos_token_id || i == max_tokens - 1) break;

            // Build next step's graph while GPU finishes current step
            // (graph construction is 0.25ms, GPU eval is 4.8ms — plenty of overlap)
            logits = g_model->step(next_token);
            token_arr = argmax(reshape(logits, {-1}));
            async_eval(token_arr);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_elapsed_ms) {
            *out_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        return generated;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] generate error: %s\n", e.what());
        return -1;
    }
}

int db_get_cache_offset(void) {
    if (!g_model || g_model->caches.empty()) return -1;
    return g_model->caches[0].offset;
}

int db_export_kv(int layer_idx, void* out_k, void* out_v) {
    if (!g_model || layer_idx < 0 || layer_idx >= g_non_shared_layers) return -1;
    auto& cache = g_model->caches[layer_idx];
    if (cache.keys.size() == 0) return -2;
    eval({cache.keys, cache.values});
    std::memcpy(out_k, cache.keys.data<void>(), cache.keys.nbytes());
    std::memcpy(out_v, cache.values.data<void>(), cache.values.nbytes());
    return 0;
}

int db_kv_nbytes(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= g_non_shared_layers) return 0;
    auto& cache = g_model->caches[layer_idx];
    return cache.keys.size() == 0 ? 0 : (int)cache.keys.nbytes();
}

int db_kv_shape(int layer_idx, int* out_kv_heads, int* out_seq_len, int* out_head_dim) {
    if (!g_model || layer_idx < 0 || layer_idx >= g_non_shared_layers) return -1;
    auto& cache = g_model->caches[layer_idx];
    if (cache.keys.size() == 0) return -2;
    *out_kv_heads = cache.keys.shape(1);
    *out_seq_len = cache.keys.shape(2);
    *out_head_dim = cache.keys.shape(3);
    return 0;
}

void* db_step_logits_ptr(int32_t token_id) {
    if (!g_model) return nullptr;
    try {
        auto logits = g_model->step(token_id);

        // Async eval the logits — dispatches GPU work without blocking.
        // This starts GPU execution immediately, and Swift's .item() will
        // wait for completion. This enables overlap with CPU sampling work.
        async_eval(logits);

        // Heap-allocate a copy — Swift takes ownership via mlx_array { ctx }
        auto* result = new mlx::core::array(std::move(logits));
        return result;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] step_logits_ptr error: %s\n", e.what());
        return nullptr;
    }
}

void* db_step_logits_from_array(void* token_arr_ptr) {
    if (!g_model || !token_arr_ptr) return nullptr;
    try {
        // Pass token array directly to model — no CPU extraction needed.
        // The token stays on GPU as a lazy/evaluated MLX array.
        auto& token_arr = *static_cast<mlx::core::array*>(token_arr_ptr);
        // Async eval the input to dispatch previous step's work.
        // Dispatch previous step's work. Input is .newAxis wrapper (unscheduled)
        // around the asyncEval'd token from the previous iteration.
        async_eval(token_arr);
        auto logits = g_model->step_from_array(token_arr);
        // Return lazy — let Swift's asyncEval handle dispatch.
        auto* result = new mlx::core::array(std::move(logits));
        return result;
    } catch (const std::exception& e) {
        fprintf(stderr, "[db] step_logits_from_array error: %s\n", e.what());
        return nullptr;
    }
}

void db_set_stub(int stub_mlp, int stub_attn, int stub_ple) {
    g_stub_mlp  = stub_mlp != 0;
    g_stub_attn = stub_attn != 0;
    g_stub_ple  = stub_ple != 0;
    fprintf(stderr, "[db] Stubs: mlp=%d attn=%d ple=%d\n",
            g_stub_mlp, g_stub_attn, g_stub_ple);
}

void db_reset_caches(void) {
    if (g_model) {
        // Clear old cache arrays first (drops shared_ptr refs to Metal buffers)
        g_model->caches.clear();
        // Free unreferenced Metal buffers
        mlx::core::clear_cache();
        // Reinitialize fresh empty caches
        g_model->init_caches();
    }
}

void db_cleanup(void) {
    g_model.reset();
    g_weight_store.clear();
    g_initialized = false;
    g_finalized   = false;
    fprintf(stderr, "[db] Cleaned up\n");
}

} // extern "C"
