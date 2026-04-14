// prefill_bridge_v2.cpp — Weight-sharing native prefill bridge for Gemma 4 E2B
//
// Accepts EXTERNAL weight arrays from Swift instead of loading safetensors.
// The arr_ptr passed to pb2_set_weight() is MLXArray.ctx — a raw
// mlx::core::array* owned by Swift. We copy the mlx::core::array (cheap:
// ~100 bytes metadata, shared_ptr to underlying buffer) so the bridge holds
// a reference but does NOT own or free the Swift-side array.
//
// Forward pass is IDENTICAL to prefill_bridge.cpp:
//   - 15 non-shared layers (0–14)
//   - Sliding attention (head_dim=256) at layers 0-3, 5-8, 10-13
//   - Full attention (head_dim=512) at layers 4, 9, 14
//   - GeGLU MLP (gelu_approx), 4-bit quantized weights
//   - Per-layer input gating
//   - RMSNorm with direct weight (NOT 1+weight)
//   - Proportional RoPE for full layers, standard RoPE for sliding
//   - Causal SDPA
//
// Build:
//   clang++ -std=c++20 -O3 -shared -fPIC \
//     -I/opt/homebrew/lib/python3.13/site-packages/mlx/include \
//     -I/Users/tom/dev/mlx-swift/Source/Cmlx/mlx-c \
//     -I/tmp \
//     -L/opt/homebrew/lib/python3.13/site-packages/mlx/lib -lmlx \
//     -framework Metal -framework Foundation -framework Accelerate \
//     -Wl,-rpath,/opt/homebrew/lib/python3.13/site-packages/mlx/lib \
//     -o /tmp/libprefill_bridge_v2.dylib /tmp/prefill_bridge_v2.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <optional>
#include "mlx/mlx.h"
#include "prefill_bridge_v2.h"

// Private mlx-c header — gives us mlx_array_get_() to extract C++ array
// from the opaque C handle. We only use this to understand the struct layout;
// the actual arr_ptr from Swift is MLXArray.ctx which is already mlx::core::array*.
// Including it anyway for documentation and the inline helpers.
#include "mlx/c/private/array.h"

using namespace mlx::core;

// ============================================================================
// Architecture constants — set by pb2_init()
// ============================================================================
static int g_num_layers       = 0;
static int g_hidden_size      = 0;
static int g_num_heads        = 0;
static int g_num_kv_heads     = 0;
static int g_sliding_window   = 0;
static int g_sliding_pattern  = 0;  // every Nth layer is full attention

// Derived constants (Gemma 4 E2B defaults — recomputed in pb2_init)
static int g_head_dim_slide   = 256;
static int g_head_dim_full    = 512;
static int g_intermediate     = 0;
static int g_total_layers     = 0;
static int g_hidden_per_layer = 256;

// Quantization params (hardcoded for 4-bit Gemma 4 E2B)
static constexpr int BITS       = 4;
static constexpr int GROUP_SIZE = 64;
static constexpr float RMS_EPS  = 1e-6f;

// RoPE params
static constexpr float ROPE_THETA_SLIDING = 10000.0f;
static constexpr float ROPE_THETA_FULL    = 1000000.0f;

static bool is_full_attention(int layer_idx) {
    return g_sliding_pattern > 0 && ((layer_idx + 1) % g_sliding_pattern == 0);
}

// ============================================================================
// Helper: GELU approximate (tanh approximation)
// ============================================================================
static array gelu_approx(const array& x) {
    // Must match Python's nn.gelu_approx exactly:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
    //
    // Two critical details:
    //   1. x**3 = power(x, 3) — single op, one final round.
    //      x*x*x = two bf16 multiplies with intermediate rounding → different.
    //   2. Python float scalars are "universal" type that don't promote bf16.
    //      C++ array(float, bf16) pre-rounds; array(float) promotes to f32.
    //      Use bf16 constants (pre-rounded) + power() to get closest match.
    auto dt = x.dtype();
    static const float kSqrt2OverPi = 0.7978845608028654f;
    auto x3 = power(x, array(3, dt));
    auto inner = array(kSqrt2OverPi, dt) * (x + array(0.044715f, dt) * x3);
    return array(0.5f, dt) * x * (array(1.0f, dt) + tanh(inner));
}

// ============================================================================
// Layer components — same structs as v1 but weights stored as array copies
// ============================================================================
struct QuantizedLinear {
    array weight;  // uint32 packed
    array scales;  // bfloat16
    array biases;  // bfloat16

    array operator()(const array& x) const {
        return quantized_matmul(
            x, weight, scales, biases,
            /*transpose=*/true,
            /*group_size=*/GROUP_SIZE,
            /*bits=*/BITS,
            /*mode=*/"affine");
    }
};

struct RMSNorm {
    array weight;  // learned scale — used directly (NOT 1+weight)

    array operator()(const array& x) const {
        return fast::rms_norm(x, weight, RMS_EPS);
    }
};

static array rms_norm_no_scale(const array& x) {
    return fast::rms_norm(x, std::nullopt, RMS_EPS);
}

struct QuantizedEmbedding {
    array weight;  // [vocab, packed_cols] uint32
    array scales;  // [vocab, num_groups] bfloat16
    array biases;  // [vocab, num_groups] bfloat16

    array operator()(const array& indices) const {
        auto w = take(weight, indices, 0);
        auto s = take(scales, indices, 0);
        auto b = take(biases, indices, 0);
        return dequantize(w, s, b, GROUP_SIZE, BITS, "affine");
    }
};

struct ScaledLinear {
    array weight;  // [out, in] bfloat16
    float scalar;

    array operator()(const array& x) const {
        auto out = matmul(x, transpose(weight));
        // Python ScaledLinear: (x @ W.T) * scalar, where scalar is Python float
        // When W is bf16, x is bf16, matmul output is bf16
        // bf16 * Python float → bf16 (MLX keeps lower dtype)
        return out * array(scalar, bfloat16);
    }
};

struct Attention {
    QuantizedLinear q_proj, k_proj;
    std::shared_ptr<QuantizedLinear> v_proj;  // optional: null when k_eq_v
    QuantizedLinear o_proj;
    RMSNorm q_norm, k_norm;
    int head_dim;
    int num_heads;
    int num_kv_heads;
    bool is_sliding;
    float rope_theta;
    array rope_freqs;  // precomputed for full (proportional), empty for sliding

    mutable array last_k = array({}, bfloat16);
    mutable array last_v = array({}, bfloat16);
    mutable array debug_k_raw = array({}, bfloat16);  // K after proj, before norm/rope
    mutable array debug_attn_input = array({}, bfloat16);  // x input to attention (after layernorm)

    array operator()(const array& x) const {
        int B = x.shape(0);
        int S = x.shape(1);

        debug_attn_input = x;  // normed input to attention
        auto q = q_proj(x);
        auto k = k_proj(x);
        auto v = v_proj ? (*v_proj)(x) : k;  // k_eq_v: V = K when no v_proj

        debug_k_raw = k;  // K after proj, before norm/rope

        q = transpose(reshape(q, {B, S, num_heads, head_dim}), {0, 2, 1, 3});
        k = transpose(reshape(k, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});
        v = transpose(reshape(v, {B, S, num_kv_heads, head_dim}), {0, 2, 1, 3});

        q = q_norm(q);
        k = k_norm(k);
        v = rms_norm_no_scale(v);

        // RoPE: sliding = standard, full = proportional (precomputed freqs)
        if (is_sliding) {
            q = fast::rope(q, head_dim, false, rope_theta, 1.0f, 0);
            k = fast::rope(k, head_dim, false, rope_theta, 1.0f, 0);
        } else {
            q = fast::rope(q, head_dim, false, std::nullopt, 1.0f, 0, rope_freqs);
            k = fast::rope(k, head_dim, false, std::nullopt, 1.0f, 0, rope_freqs);
        }

        // Store K/V in bfloat16 to match Swift's cache dtype
        // Python's cache.update_and_fetch() stores bf16, then SDPA reads bf16 back.
        // We must use bf16 K/V for SDPA to match Python exactly.
        last_k = astype(k, bfloat16);
        last_v = astype(v, bfloat16);
        k = last_k;
        v = last_v;

        // SDPA (scale=1.0 for Gemma4)
        float scale = 1.0f;
        auto make_attn = [&]() -> array {
        if (is_sliding && S > g_sliding_window) {
            auto rinds = arange(S);
            auto linds = reshape(arange(S), {S, 1});
            auto mask_arr = (linds >= rinds) & (linds < (rinds + g_sliding_window));
            return fast::scaled_dot_product_attention(
                q, k, v, scale, /*mask_mode=*/"", /*mask_arr=*/mask_arr);
        } else if (is_sliding) {
            return fast::scaled_dot_product_attention(
                q, k, v, scale, "causal");
        } else {
            return fast::scaled_dot_product_attention(
                q, k, v, scale, "causal");
        }
        };
        auto attn_out = make_attn();

        attn_out = reshape(
            transpose(attn_out, {0, 2, 1, 3}),
            {B, S, num_heads * head_dim});

        return o_proj(attn_out);
    }
};

struct MLP {
    QuantizedLinear gate_proj, up_proj, down_proj;

    array operator()(const array& x) const {
        auto gate = gate_proj(x);
        auto up   = up_proj(x);
        return down_proj(gelu_approx(gate) * up);
    }
};

struct TransformerLayer {
    RMSNorm input_layernorm;
    Attention self_attn;
    RMSNorm post_attention_layernorm;
    RMSNorm pre_feedforward_layernorm;
    MLP mlp;
    RMSNorm post_feedforward_layernorm;

    // PLE fields — optional for models without PLE
    std::shared_ptr<QuantizedLinear> per_layer_input_gate;
    std::shared_ptr<QuantizedLinear> per_layer_projection;
    std::shared_ptr<RMSNorm> post_per_layer_input_norm;
    std::shared_ptr<array> layer_scalar;
    bool has_ple = false;

    // Debug: per-layer intermediate captures (only layer 0 is checked)
    mutable array debug_attn_out = array({}, bfloat16);
    mutable array debug_mlp_out = array({}, bfloat16);
    mutable array debug_h_out = array({}, bfloat16);

    array operator()(const array& x, const array& per_layer_input) const {
        // Pre-norm attention with residual
        auto residual = x;
        auto h = input_layernorm(x);
        h = self_attn(h);
        h = post_attention_layernorm(h);
        h = residual + h;
        debug_attn_out = h;

        // Pre-norm MLP with residual
        residual = h;
        auto ff = pre_feedforward_layernorm(h);
        ff = mlp(ff);
        ff = post_feedforward_layernorm(ff);
        h = residual + ff;
        debug_mlp_out = h;

        // Per-layer input gating (PLE) — only for models with PLE
        if (has_ple) {
            residual = h;
            auto gate = (*per_layer_input_gate)(h);
            gate = gelu_approx(gate);
            gate = gate * per_layer_input;
            gate = (*per_layer_projection)(gate);
            gate = (*post_per_layer_input_norm)(gate);
            h = residual + gate;

            h = h * (*layer_scalar);
        }
        debug_h_out = h;

        return h;
    }
};

struct Model {
    QuantizedEmbedding embed_tokens;
    std::shared_ptr<QuantizedEmbedding> embed_tokens_per_layer;
    std::shared_ptr<ScaledLinear> per_layer_model_projection;
    std::shared_ptr<RMSNorm> per_layer_projection_norm;
    bool has_ple = false;
    std::vector<TransformerLayer> layers;
    RMSNorm final_norm;

    // Precomputed scale constants (avoid per-call array creation)
    mutable array debug_h0 = array(0.0f);  // h before layer 0, for debugging
    array embed_scale_arr = array(0.0f);
    array pl_embed_scale_arr = array(0.0f);
    array pl_input_scale_arr = array(0.0f);

    void precompute_scales() {
        // Match Python: scalars stored as bf16 (Python float * bf16 → bf16)
        embed_scale_arr = array(std::sqrt(static_cast<float>(g_hidden_size)), bfloat16);
        pl_embed_scale_arr = array(std::sqrt(static_cast<float>(g_hidden_per_layer)), bfloat16);
        pl_input_scale_arr = array(std::pow(2.0f, -0.5f), bfloat16);
    }

    array forward(const array& token_ids) const {
        int B = token_ids.shape(0);
        int S = token_ids.shape(1);

        auto h = embed_tokens(token_ids) * embed_scale_arr;

        // PLE (Per-Layer Embeddings) — optional, only for models with hiddenSizePerLayerInput > 0
        array combined_per_layer = array(0.0f);
        if (has_ple) {
            auto per_layer_inputs = (*embed_tokens_per_layer)(token_ids) * pl_embed_scale_arr;
            per_layer_inputs = reshape(per_layer_inputs, {B, S, g_total_layers, g_hidden_per_layer});

            auto per_layer_proj = per_layer_model_projection->operator()(h);
            per_layer_proj = reshape(per_layer_proj, {B, S, g_total_layers, g_hidden_per_layer});
            per_layer_proj = (*per_layer_projection_norm)(per_layer_proj);

            combined_per_layer = (per_layer_proj + per_layer_inputs) * pl_input_scale_arr;
        }

        // Debug: save h before layer 0
        debug_h0 = h;

        // Run through non-shared layers
        for (int i = 0; i < g_num_layers; i++) {
            if (has_ple) {
                auto pli = slice(combined_per_layer,
                                 {0, 0, i, 0},
                                 {B, S, i + 1, g_hidden_per_layer});
                pli = reshape(pli, {B, S, g_hidden_per_layer});
                h = layers[i](h, pli);
            } else {
                // No PLE: pass zero PLE input
                auto pli = zeros({B, S, std::max(g_hidden_per_layer, 1)}, h.dtype());
                h = layers[i](h, pli);
            }
        }

        // Skip final norm during prefill — K/V eval doesn't need it
        return h;
    }
};

// ============================================================================
// Global state
// ============================================================================
static std::shared_ptr<Model> g_model = nullptr;
static std::unordered_map<std::string, array> g_weight_store;
static bool g_initialized = false;
static bool g_finalized   = false;

// ============================================================================
// Weight extraction from Swift's MLXArray.ctx
//
// Swift passes MLXArray.ctx which is the void* inside the mlx_array struct.
// That void* points to a heap-allocated mlx::core::array.
// We COPY the mlx::core::array (cheap — shares underlying data buffer via
// shared_ptr) so the bridge holds a reference without owning Swift's object.
// ============================================================================
static array extract_array(void* arr_ptr) {
    if (!arr_ptr) {
        throw std::runtime_error("pb2_set_weight: null arr_ptr");
    }
    // arr_ptr IS the mlx::core::array* (the ctx field from mlx_array struct)
    auto* cpp_array = static_cast<mlx::core::array*>(arr_ptr);
    // Copy — shares underlying data buffer, just bumps refcount (~100 bytes metadata)
    return *cpp_array;
}

// ============================================================================
// Build helpers — pull from g_weight_store
// ============================================================================
static array get_weight(const std::string& key) {
    auto it = g_weight_store.find(key);
    if (it != g_weight_store.end()) return it->second;
    fprintf(stderr, "[pb2] WARNING: weight not found: %s\n", key.c_str());
    throw std::runtime_error("Missing weight: " + key);
}

static bool has_weight(const std::string& key) {
    return g_weight_store.count(key) > 0;
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
    // partial_rotary_factor=0.25: only first 25% of dims are rotated,
    // remaining get inf (identity). Matches Python's ProportionalRoPE.
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

    // v_proj: present for sliding layers, absent for full layers with k_eq_v
    std::shared_ptr<QuantizedLinear> v;
    if (has_weight(prefix + ".v_proj.weight")) {
        auto _v = make_quantized_linear(prefix + ".v_proj");
        v = std::make_shared<QuantizedLinear>(std::move(_v));
    }

    return {
        make_quantized_linear(prefix + ".q_proj"),
        make_quantized_linear(prefix + ".k_proj"),
        std::move(v),
        make_quantized_linear(prefix + ".o_proj"),
        make_rms_norm(prefix + ".q_norm"),
        make_rms_norm(prefix + ".k_norm"),
        hd,
        g_num_heads,
        g_num_kv_heads,
        !full,
        theta,
        freqs,
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

    // Build core layer components
    auto attn = make_attention(prefix + ".self_attn", layer_idx);
    auto mlp_block = make_mlp(prefix + ".mlp");

    // PLE fields — only if weights exist
    std::shared_ptr<QuantizedLinear> ple_gate, ple_proj;
    std::shared_ptr<RMSNorm> ple_norm;
    std::shared_ptr<array> ple_scalar;
    bool layer_has_ple = false;
    if (has_weight(prefix + ".per_layer_input_gate.weight")) {
        auto _g = make_quantized_linear(prefix + ".per_layer_input_gate");
        ple_gate = std::make_shared<QuantizedLinear>(std::move(_g));
        auto _p = make_quantized_linear(prefix + ".per_layer_projection");
        ple_proj = std::make_shared<QuantizedLinear>(std::move(_p));
        auto _n = make_rms_norm(prefix + ".post_per_layer_input_norm");
        ple_norm = std::make_shared<RMSNorm>(std::move(_n));
        auto _s = get_weight(prefix + ".layer_scalar");
        ple_scalar = std::make_shared<array>(std::move(_s));
        layer_has_ple = true;
    }

    return TransformerLayer{
        make_rms_norm(prefix + ".input_layernorm"),
        std::move(attn),
        make_rms_norm(prefix + ".post_attention_layernorm"),
        make_rms_norm(prefix + ".pre_feedforward_layernorm"),
        std::move(mlp_block),
        make_rms_norm(prefix + ".post_feedforward_layernorm"),
        std::move(ple_gate), std::move(ple_proj),
        std::move(ple_norm), std::move(ple_scalar),
        layer_has_ple,
    };
}

static std::shared_ptr<Model> build_model() {
    auto embed_tokens = make_quantized_embedding("embed_tokens");

    // PLE: optional
    bool ple_active = has_weight("embed_tokens_per_layer.weight");
    std::shared_ptr<QuantizedEmbedding> embed_tokens_per_layer;
    std::shared_ptr<ScaledLinear> per_layer_model_projection;
    std::shared_ptr<RMSNorm> per_layer_projection_norm;
    if (ple_active) {
        auto _e = make_quantized_embedding("embed_tokens_per_layer");
        embed_tokens_per_layer = std::make_shared<QuantizedEmbedding>(std::move(_e));
        float projection_scalar = 1.0f / std::sqrt(static_cast<float>(g_hidden_size));
        auto _sl = ScaledLinear{get_weight("per_layer_model_projection.weight"), projection_scalar};
        per_layer_model_projection = std::make_shared<ScaledLinear>(std::move(_sl));
        auto _rn = make_rms_norm("per_layer_projection_norm");
        per_layer_projection_norm = std::make_shared<RMSNorm>(std::move(_rn));
        fprintf(stderr, "[pb2] PLE enabled\n");
    } else {
        fprintf(stderr, "[pb2] PLE disabled (no embed_tokens_per_layer weights)\n");
    }

    std::vector<TransformerLayer> layers;
    layers.reserve(g_num_layers);
    for (int i = 0; i < g_num_layers; i++) {
        fprintf(stderr, "[pb2] Building layer %d/%d (%s, head_dim=%d)\n",
                i, g_num_layers,
                is_full_attention(i) ? "full" : "sliding",
                is_full_attention(i) ? g_head_dim_full : g_head_dim_slide);
        layers.push_back(make_layer(i));
    }

    auto final_norm = make_rms_norm("norm");

    return std::shared_ptr<Model>(new Model{
        std::move(embed_tokens),
        std::move(embed_tokens_per_layer),
        std::move(per_layer_model_projection),
        std::move(per_layer_projection_norm),
        ple_active,
        std::move(layers),
        std::move(final_norm),
    });
}

// ============================================================================
// C ABI
// ============================================================================
extern "C" {

int pb2_init(int num_layers, int hidden_size, int num_heads, int num_kv_heads,
             int sliding_window, int sliding_window_pattern) {
    try {
        g_num_layers      = num_layers;
        g_hidden_size     = hidden_size;
        g_num_heads       = num_heads;
        g_num_kv_heads    = num_kv_heads;
        g_sliding_window  = sliding_window;
        g_sliding_pattern = sliding_window_pattern;

        // Derived constants for Gemma 4 E2B architecture
        // head_dim_slide = hidden_size * 2 / num_heads  (256 for 1536/8 * ... — actually from config)
        // For Gemma 4 E2B: slide=256, full=512, intermediate=6144
        // TODO: Make these configurable if needed for other models
        g_head_dim_slide   = 256;
        g_head_dim_full    = 512;
        g_intermediate     = hidden_size * 4;  // 6144 for hidden=1536
        g_total_layers     = 35;               // Gemma 4 E2B total
        g_hidden_per_layer = 256;              // hidden_size_per_layer_input

        g_weight_store.clear();
        g_model.reset();
        g_initialized = true;
        g_finalized   = false;

        fprintf(stderr, "[pb2] Initialized: %d layers, hidden=%d, heads=%d/%d, pattern=%d\n",
                num_layers, hidden_size, num_heads, num_kv_heads, sliding_window_pattern);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2] init error: %s\n", e.what());
        return -1;
    }
}

int pb2_set_weight(const char* key, void* arr_ptr) {
    if (!g_initialized || g_finalized) {
        fprintf(stderr, "[pb2] set_weight called in wrong state (init=%d, final=%d)\n",
                g_initialized, g_finalized);
        return -1;
    }

    try {
        array copied = extract_array(arr_ptr);
        g_weight_store.insert_or_assign(std::string(key), std::move(copied));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2] set_weight error for '%s': %s\n", key, e.what());
        return -1;
    }
}

int pb2_finalize(void) {
    if (!g_initialized || g_finalized) {
        fprintf(stderr, "[pb2] finalize called in wrong state\n");
        return -1;
    }

    try {
        fprintf(stderr, "[pb2] Finalizing with %zu weight tensors\n", g_weight_store.size());
        g_model = build_model();
        g_model->precompute_scales();
        g_finalized = true;

        // Clear the weight store — model now holds all the array copies
        g_weight_store.clear();

        fprintf(stderr, "[pb2] Model built successfully (zero-copy weight sharing with Swift)\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2] finalize error: %s\n", e.what());
        return -1;
    }
}

// Zero-copy variant: accepts raw mlx::core::array* (avoids GPU→CPU→GPU roundtrip for tokens)
extern "C"
int pb2_run_array(void* token_arr_ptr,
                  double* out_elapsed_ms, float* out_checksum) {
    if (!g_model || !token_arr_ptr) {
        fprintf(stderr, "[pb2] run_array called before finalize or with null ptr\n");
        return -1;
    }

    try {
        // Zero-copy: use the Swift MLXArray's underlying data directly
        auto& tokens_1d = *static_cast<mlx::core::array*>(token_arr_ptr);
        auto tokens = reshape(tokens_1d, {1, static_cast<int>(tokens_1d.size())});

        auto t0 = std::chrono::high_resolution_clock::now();
        auto output = g_model->forward(tokens);

        std::vector<array> cache_arrays;
        for (auto& layer : g_model->layers) {
            cache_arrays.push_back(layer.self_attn.last_k);
            cache_arrays.push_back(layer.self_attn.last_v);
        }
        eval(cache_arrays);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto& layer0_k = g_model->layers[0].self_attn.last_k;
        eval(layer0_k);
        auto checksum_arr = astype(sum(layer0_k), float32);
        eval(checksum_arr);
        float checksum = checksum_arr.item<float>();

        if (out_elapsed_ms) *out_elapsed_ms = elapsed_ms;
        if (out_checksum) *out_checksum = checksum;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2] run_array error: %s\n", e.what());
        return -1;
    }
}

int pb2_run(const int32_t* token_ids, int token_count,
            double* out_elapsed_ms, float* out_checksum) {
    if (!g_model) {
        fprintf(stderr, "[pb2] run called before finalize\n");
        return -1;
    }

    try {
        // Create token ID tensor [1, token_count]
        auto tokens = array(token_ids, {1, token_count}, int32);
        eval(tokens);

        // Timed forward pass
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output = g_model->forward(tokens);

        // Eval all layer K AND V tensors
        std::vector<array> cache_arrays;
        for (auto& layer : g_model->layers) {
            cache_arrays.push_back(layer.self_attn.last_k);
            cache_arrays.push_back(layer.self_attn.last_v);
        }
        eval(cache_arrays);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Checksum: sum of layer 0's K
        auto& layer0_k = g_model->layers[0].self_attn.last_k;
        eval(layer0_k);
        auto checksum_arr = astype(sum(layer0_k), float32);
        eval(checksum_arr);
        float checksum = checksum_arr.item<float>();

        if (out_elapsed_ms) *out_elapsed_ms = elapsed_ms;
        if (out_checksum) *out_checksum = checksum;

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2] run error: %s\n", e.what());
        return -1;
    }
}

void pb2_cleanup(void) {
    g_model.reset();
    g_weight_store.clear();
    g_initialized = false;
    g_finalized   = false;
    fprintf(stderr, "[pb2] Cleaned up\n");
}

} // extern "C"

// ============================================================================
// K/V export for Swift cache injection
// ============================================================================

// Get number of non-shared layers (= number of caches to populate)
extern "C" int pb2_num_layers() {
    return g_model ? (int)g_model->layers.size() : 0;
}

// Get K/V shape for a layer after forward: [1, kv_heads, seq_len, head_dim]
extern "C" int pb2_kv_shape(int layer_idx, int* out_kv_heads, int* out_seq_len, int* out_head_dim) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return -1;
    auto& k = g_model->layers[layer_idx].self_attn.last_k;
    if (k.size() == 0) return -2;
    // Shape: [B, kv_heads, seq_len, head_dim]
    *out_kv_heads = k.shape(1);
    *out_seq_len = k.shape(2);
    *out_head_dim = k.shape(3);
    return 0;
}

// Get byte size of K (or V) for a layer
extern "C" size_t pb2_kv_nbytes(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return 0;
    auto& k = g_model->layers[layer_idx].self_attn.last_k;
    return k.nbytes();
}

// Export K and V as CPU-side bfloat16 data.
// Caller allocates out_k and out_v buffers of pb2_kv_nbytes() each.
extern "C" int pb2_export_kv(int layer_idx, void* out_k, void* out_v) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return -1;
    auto& attn = g_model->layers[layer_idx].self_attn;
    if (attn.last_k.size() == 0) return -2;

    eval({attn.last_k, attn.last_v});

    std::memcpy(out_k, attn.last_k.data<void>(), attn.last_k.nbytes());
    std::memcpy(out_v, attn.last_v.data<void>(), attn.last_v.nbytes());
    return 0;
}

// Checksum of a layer's K: sum(K) as float
extern "C" float pb2_layer_k_checksum(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return 0;
    auto& k = g_model->layers[layer_idx].self_attn.last_k;
    if (k.size() == 0) return 0;
    eval(k);
    auto s = astype(sum(k), float32);
    eval(s);
    return s.item<float>();
}

// Self-test: validate one borrowed weight
extern "C" int pb2_test_weight(const char* key, void* arr_ptr) {
    fprintf(stderr, "[pb2-test] key=%s ptr=%p\n", key, arr_ptr);
    if (!arr_ptr) {
        fprintf(stderr, "[pb2-test] NULL pointer!\n");
        return -1;
    }
    try {
        auto* cpp_arr = static_cast<mlx::core::array*>(arr_ptr);
        fprintf(stderr, "[pb2-test] shape=(");
        for (int i = 0; i < cpp_arr->ndim(); i++) {
            if (i) fprintf(stderr, ",");
            fprintf(stderr, "%d", (int)cpp_arr->shape(i));
        }
        fprintf(stderr, ") size=%zu nbytes=%zu\n", cpp_arr->size(), cpp_arr->nbytes());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pb2-test] error: %s\n", e.what());
        return -1;
    }
}

// Zero-copy K/V access: return raw mlx::core::array* pointers
// Swift can pass these to mlx-c functions via the ctx field
extern "C" void* pb2_get_k_ptr(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return nullptr;
    return &(g_model->layers[layer_idx].self_attn.last_k);
}
extern "C" void* pb2_get_v_ptr(int layer_idx) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) return nullptr;
    return &(g_model->layers[layer_idx].self_attn.last_v);
}

// Create new heap-allocated mlx::core::array copies of K/V for a layer.
// Returns raw pointers via out params. Shares underlying GPU data buffer via shared_ptr.
// Caller takes ownership and must eventually delete (or pass to MLXArray which handles it).
extern "C" void pb2_get_kv_handles(int layer_idx, void** out_k_ptr, void** out_v_ptr) {
    if (!g_model || layer_idx < 0 || layer_idx >= (int)g_model->layers.size()) {
        if (out_k_ptr) *out_k_ptr = nullptr;
        if (out_v_ptr) *out_v_ptr = nullptr;
        return;
    }
    auto& attn = g_model->layers[layer_idx].self_attn;

    // new array(...) creates a lightweight copy that shares the underlying data buffer
    if (out_k_ptr) *out_k_ptr = new mlx::core::array(attn.last_k);
    if (out_v_ptr) *out_v_ptr = new mlx::core::array(attn.last_v);
}

// Debug: get checksum of h before layer 0
extern "C" int pb2_debug_h0_checksum(float* out_sum) {
    if (!g_model || g_model->debug_h0.size() == 0) return -1;
    eval(g_model->debug_h0);
    auto s = astype(sum(g_model->debug_h0), float32);
    eval(s);
    *out_sum = s.item<float>();
    return 0;
}

// Debug: get max/mean abs diff of h0 against externally provided reference
extern "C" int pb2_debug_h0_nbytes() {
    if (!g_model) return 0;
    return (int)g_model->debug_h0.nbytes();
}

extern "C" int pb2_debug_h0_export(void* out) {
    if (!g_model || g_model->debug_h0.size() == 0) return -1;
    eval(g_model->debug_h0);
    std::memcpy(out, g_model->debug_h0.data<void>(), g_model->debug_h0.nbytes());
    return 0;
}

extern "C" int pb2_debug_layer0_k_raw_checksum(float* out) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& kr = g_model->layers[0].self_attn.debug_k_raw;
    if (kr.size() == 0) return -2;
    eval(kr);
    auto s = astype(sum(kr), float32);
    eval(s);
    *out = s.item<float>();
    return 0;
}

extern "C" int pb2_debug_layer0_attn_input_checksum(float* out) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& ai = g_model->layers[0].self_attn.debug_attn_input;
    if (ai.size() == 0) return -2;
    eval(ai);
    auto s = astype(sum(ai), float32);
    eval(s);
    *out = s.item<float>();
    return 0;
}

extern "C" int pb2_debug_layer0_attn_input_export(void* out, int* out_nbytes) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& ai = g_model->layers[0].self_attn.debug_attn_input;
    if (ai.size() == 0) return -2;
    eval(ai);
    *out_nbytes = (int)ai.nbytes();
    std::memcpy(out, ai.data<void>(), ai.nbytes());
    return 0;
}

extern "C" int pb2_debug_layer0_attn_input_shape(int* B, int* S, int* D) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& ai = g_model->layers[0].self_attn.debug_attn_input;
    *B = ai.shape(0); *S = ai.shape(1); *D = ai.shape(2);
    return 0;
}

extern "C" int pb2_debug_layer0_k_raw_export(void* out, int* out_nbytes) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& kr = g_model->layers[0].self_attn.debug_k_raw;
    if (kr.size() == 0) return -2;
    eval(kr);
    *out_nbytes = (int)kr.nbytes();
    std::memcpy(out, kr.data<void>(), kr.nbytes());
    return 0;
}

extern "C" int pb2_debug_layer0_k_raw_shape(int* B, int* S, int* D) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& kr = g_model->layers[0].self_attn.debug_k_raw;
    *B = kr.shape(0); *S = kr.shape(1); *D = kr.shape(2);
    return 0;
}

// Layer 0 substep exports: attn_out, mlp_out, h_out
static int export_debug_tensor(const array& t, void* out, int* nbytes) {
    if (t.size() == 0) return -2;
    eval(t);
    *nbytes = (int)t.nbytes();
    std::memcpy(out, t.data<void>(), t.nbytes());
    return 0;
}
static size_t debug_tensor_nbytes(const array& t) {
    if (t.size() == 0) return 0;
    eval(t);
    return t.nbytes();
}

extern "C" size_t pb2_debug_layer0_attn_out_nbytes() {
    if (!g_model || g_model->layers.empty()) return 0;
    return debug_tensor_nbytes(g_model->layers[0].debug_attn_out);
}
extern "C" int pb2_debug_layer0_attn_out_export(void* out, int* nbytes) {
    if (!g_model || g_model->layers.empty()) return -1;
    return export_debug_tensor(g_model->layers[0].debug_attn_out, out, nbytes);
}

extern "C" size_t pb2_debug_layer0_mlp_out_nbytes() {
    if (!g_model || g_model->layers.empty()) return 0;
    return debug_tensor_nbytes(g_model->layers[0].debug_mlp_out);
}
extern "C" int pb2_debug_layer0_mlp_out_export(void* out, int* nbytes) {
    if (!g_model || g_model->layers.empty()) return -1;
    return export_debug_tensor(g_model->layers[0].debug_mlp_out, out, nbytes);
}

extern "C" size_t pb2_debug_layer0_h_out_nbytes() {
    if (!g_model || g_model->layers.empty()) return 0;
    return debug_tensor_nbytes(g_model->layers[0].debug_h_out);
}
extern "C" int pb2_debug_layer0_h_out_export(void* out, int* nbytes) {
    if (!g_model || g_model->layers.empty()) return -1;
    return export_debug_tensor(g_model->layers[0].debug_h_out, out, nbytes);
}

extern "C" int pb2_debug_h0_shape(int* S, int* D) {
    if (!g_model) return -1;
    auto& h0 = g_model->debug_h0;
    if (h0.ndim() == 2) {
        *S = h0.shape(0); *D = h0.shape(1);
    } else {
        *S = h0.shape(1); *D = h0.shape(2);
    }
    return 0;
}

extern "C" int pb2_debug_layer0_kproj_weight_checksum(double* w_sum, float* s_sum, float* b_sum) {
    if (!g_model || g_model->layers.empty()) return -1;
    auto& kp = g_model->layers[0].self_attn.k_proj;
    eval({kp.weight, kp.scales, kp.biases});
    // Use CPU stream for float64 (GPU doesn't support it)
    auto cpu = mlx::core::Device(mlx::core::Device::cpu);
    auto ws = astype(sum(astype(kp.weight, float32, cpu), cpu), float64, cpu);
    auto ss = sum(astype(kp.scales, float32, cpu), cpu);
    auto bs = sum(astype(kp.biases, float32, cpu), cpu);
    eval({ws, ss, bs});
    *w_sum = ws.item<double>();
    *s_sum = ss.item<float>();
    *b_sum = bs.item<float>();
    return 0;
}
