// prefill_bridge_qwen.h — Qwen-family native prefill bridge
//
// Dedicated prefill bridge for Qwen2, Qwen3, and Qwen3-MoE architectures.
// Separated from generic_prefill.cpp to allow Qwen-specific tuning
// (eval barriers, graph structure, MoE dispatch) without risking
// regressions on other model families.
//
// Integration: called from Qwen Swift model prepare() methods.
#ifndef PREFILL_BRIDGE_QWEN_H
#define PREFILL_BRIDGE_QWEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize with Qwen model config JSON
int qwen_init(const char* config_json);

// Set a weight tensor (pass Swift's MLXArray.ctx pointer)
int qwen_set_weight(const char* key, void* arr_ptr);

// Mark initialization complete — builds model from weights
int qwen_finalize(void);

// Run prefill on token array (pass MLXArray.ctx pointer)
int qwen_run(void* token_array_ptr, double* out_elapsed_ms);

// Get number of KV cache layers
int qwen_num_cache_layers(void);

// Get K/V array pointers for cache injection
void* qwen_get_k_ptr(int layer_idx);
void* qwen_get_v_ptr(int layer_idx);

// KV shape info
int qwen_kv_shape(int layer_idx, int* kv_heads, int* seq_len, int* head_dim);

// Cleanup
void qwen_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
