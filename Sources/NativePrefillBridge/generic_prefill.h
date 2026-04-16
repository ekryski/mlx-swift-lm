// generic_prefill.h — Model-agnostic native prefill bridge
//
// Supports multiple model architectures via runtime config.
// Same weight-sharing pattern as prefill_bridge_v2: Swift passes
// MLXArray.ctx pointers, bridge copies metadata (shares GPU buffer).
#ifndef GENERIC_PREFILL_H
#define GENERIC_PREFILL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize with model config JSON string
// Parses model_type, layer count, dimensions, etc.
int gp_init(const char* config_json);

// Set a weight tensor (pass Swift's MLXArray.ctx pointer)
int gp_set_weight(const char* key, void* arr_ptr);

// Mark initialization complete — builds model from weights
int gp_finalize(void);

// Run prefill on token array (pass MLXArray.ctx pointer)
// Returns 0 on success
int gp_run(void* token_array_ptr, double* out_elapsed_ms);

// Get number of KV cache layers
int gp_num_cache_layers(void);

// Get K/V array pointers for cache injection into Swift
// Returns heap-allocated mlx::core::array* (caller takes ownership)
void* gp_get_k_ptr(int layer_idx);
void* gp_get_v_ptr(int layer_idx);

// KV shape info
int gp_kv_shape(int layer_idx, int* kv_heads, int* seq_len, int* head_dim);

// Cleanup
void gp_cleanup(void);

// Allocator bug repro (standalone test, no model needed)
int gp_repro_allocator_bug(void);

#ifdef __cplusplus
}
#endif

#endif
