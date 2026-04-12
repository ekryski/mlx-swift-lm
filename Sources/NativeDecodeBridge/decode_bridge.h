// decode_bridge.h — Native C++ decode bridge for Gemma 4 E2B
// Runs the full decode loop in C++, bypassing Swift MLXArray overhead.
// Shares weight-borrowing pattern with prefill_bridge_v2.
#ifndef DECODE_BRIDGE_H
#define DECODE_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize bridge with model architecture (same as pb2_init)
int db_init(int num_layers, int hidden_size, int num_heads, int num_kv_heads,
            int sliding_window, int sliding_window_pattern,
            int vocab_size, int total_layers, int hidden_per_layer);

// Set a weight tensor (pass Swift's MLXArray.ctx pointer)
int db_set_weight(const char* key, void* arr_ptr);

// Mark initialization complete
int db_finalize(void);

// Import KV cache from prefill (or Swift's cache)
// k_ptr/v_ptr are mlx::core::array* with shape [1, kv_heads, seq_len, head_dim]
int db_import_kv(int layer_idx, void* k_ptr, void* v_ptr);

// Set the current cache offset (seq position after prefill)
int db_set_cache_offset(int offset);

// Run one decode step: token_id in, token_id out (argmax sampling)
// Returns the sampled token ID, or -1 on error
int32_t db_step(int32_t token_id);

// Run one decode step returning logits for custom sampling
// out_logits: pointer to float array of size vocab_size (caller allocated)
// Returns 0 on success
int db_step_logits(int32_t token_id, float* out_logits, int vocab_size);

// Run N decode steps in a tight C++ loop (argmax only)
// out_tokens: caller-allocated buffer for N token IDs
// Returns number of tokens generated (stops early on EOS)
int db_generate(int32_t first_token_id, int max_tokens, int32_t* out_tokens,
                int32_t eos_token_id, double* out_elapsed_ms);

// Get current cache offset
int db_get_cache_offset(void);

// Export KV cache back to Swift (for cache management)
int db_export_kv(int layer_idx, void* out_k, void* out_v);
int db_kv_nbytes(int layer_idx);
int db_kv_shape(int layer_idx, int* out_kv_heads, int* out_seq_len, int* out_head_dim);

// Run one decode step, return pointer to the logits mlx::core::array (zero-copy)
// The returned pointer is valid until the next db_step call.
// Returns NULL on error.
void* db_step_logits_ptr(int32_t token_id);

// Same as db_step_logits_ptr but accepts an mlx_array* (avoids Swift-side .item() sync)
// token_arr_ptr: pointer to mlx::core::array containing the token ID
void* db_step_logits_from_array(void* token_arr_ptr);

// Reset KV caches (keep model, just clear cache state)
void db_reset_caches(void);

// Per-block stubbing for profiling (0=run, 1=stub/skip)
void db_set_stub(int stub_mlp, int stub_attn, int stub_ple);

// Cleanup (full teardown)
void db_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
