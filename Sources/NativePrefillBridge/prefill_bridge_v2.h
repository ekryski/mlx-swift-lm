// prefill_bridge_v2.h — Weight-sharing native prefill bridge
// Accepts external weight arrays instead of loading its own.
#ifndef PREFILL_BRIDGE_V2_H
#define PREFILL_BRIDGE_V2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize bridge with model architecture (no weight loading)
int pb2_init(int num_layers, int hidden_size, int num_heads, int num_kv_heads,
             int sliding_window, int sliding_window_pattern);

// Set a weight tensor for the bridge (pass Swift's mlx_array ctx pointer)
// key: weight name like "layers.0.self_attn.q_proj.weight"
// arr_ptr: the mlx_array opaque handle from Swift (MLXArray.ctx)
int pb2_set_weight(const char* key, void* arr_ptr);

// Mark initialization complete (all weights set)
int pb2_finalize(void);

// Run prefill
int pb2_run(const int32_t* token_ids, int token_count,
            double* out_elapsed_ms, float* out_checksum);

// Run prefill with MLXArray pointer (zero-copy)
int pb2_run_array(void* token_arr_ptr, double* out_elapsed_ms, float* out_checksum);

// Get K/V cache handles for a layer
void pb2_get_kv_handles(int layer_idx, void** out_k_ptr, void** out_v_ptr);

// Cleanup
void pb2_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
