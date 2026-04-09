# Spec: Adaptive Memory-Aware Command Buffer Management

## Problem

MLX's command buffer commit heuristic uses `max_ops_per_buffer` (op count) and `max_mb_per_buffer` (input buffer size) to decide when to commit. This has two issues:

1. **Output allocations aren't tracked** — large outputs (attention score matrices, 256 MB during prefill) don't trigger early commits, causing unchecked memory growth.
2. **Fixed op count is a poor proxy** — 300 ops of decode (tiny tensors) uses ~2 MB total; 300 ops of prefill (large tensors) uses ~3+ GB. The same threshold can't serve both.

### Current behavior

```cpp
bool Device::command_buffer_needs_commit(int index) {
    auto& stream = get_stream_(index);
    return (stream.buffer_ops > max_ops_per_buffer_) ||
        ((stream.buffer_sizes >> 20) > max_mb_per_buffer_);
}
```

- `buffer_ops`: incremented per dispatch
- `buffer_sizes`: sum of **unique input** buffer data sizes (outputs not counted)
- Thresholds are **static** (set once at device init from chip type + env var)

## Proposed Design

### Core change: Track total referenced memory (inputs + outputs)

```cpp
// In CommandEncoder (device.h)
struct DeviceStream {
    size_t buffer_ops{0};
    size_t buffer_input_sizes{0};   // existing (rename from buffer_sizes)
    size_t buffer_output_sizes{0};  // NEW: track output allocations
    size_t buffer_total_sizes{0};   // NEW: inputs + outputs combined
    // ...
};
```

Add output tracking in `set_output_array`:

```cpp
void CommandEncoder::set_output_array(const array& a, int idx) {
    stream_.buffer_output_sizes += a.data_size();
    stream_.buffer_total_sizes = stream_.buffer_input_sizes + stream_.buffer_output_sizes;
    // ... existing code ...
}
```

### Adaptive threshold based on total memory pressure

Replace the static `max_mb_per_buffer_` with a memory-aware check:

```cpp
bool Device::command_buffer_needs_commit(int index) {
    auto& stream = get_stream_(index);

    // Op count limit (still useful as a ceiling)
    if (stream.buffer_ops > max_ops_per_buffer_) return true;

    // Memory-based limit: total referenced memory (inputs + outputs)
    // This naturally adapts to prefill (large tensors, frequent commits)
    // vs decode (small tensors, infrequent commits)
    size_t total_mb = stream.buffer_total_sizes >> 20;
    if (total_mb > max_mb_per_buffer_) return true;

    return false;
}
```

### Tuned defaults per chip class

| Chip | max_ops | max_mb (total) | Rationale |
|------|---------|----------------|-----------|
| Phone ('p') | 20 | 40 | Memory constrained |
| Base/Pro ('g') | 100 | 200 | Moderate memory |
| **Max ('s')** | **500** | **200** | High ops (fast decode), memory-limited commits during prefill |
| **Ultra ('d')** | **500** | **400** | More memory headroom |

The key insight: **set ops_per_buffer HIGH (500) for maximum decode speed, but let max_mb_per_buffer (200 MB) trigger early commits during prefill when large tensors accumulate.** This gives you the best of both worlds:

- **Decode (T=1)**: 200 small ops × ~10 KB each = ~2 MB total → never hits the 200 MB limit → all ops in one commit → fast
- **Prefill (T=2048)**: After ~3-4 attention layers, total memory reaches ~200 MB → commit fires → memory freed → next batch of layers

### Example: Prefill at T=2048 with max_mb=200

```
Layer 1: inputNorm(11 MB out) + QKV proj(3×23 MB out) + attention(256 MB out) + ...
         Total: ~340 MB → exceeds 200 MB at attention score allocation
         → COMMIT → GPU starts, memory reclaimable

Layer 2: starts fresh buffer, same pattern
         → COMMIT after attention

... 30 layers, each gets its own commit → peak memory ~400 MB (one layer's worth)
```

vs current behavior (ops_per_buffer=300, no output tracking):

```
Layers 1-10: all in one buffer, no commit triggered
         Total: ~3.4 GB of attention scores alive simultaneously
         → Peak memory: 3.4 GB + model weights + KV cache
```

### Environment variable override

Keep the existing env vars for user tuning:

```cpp
max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
```

Users can set `MLX_MAX_OPS_PER_BUFFER=300` and `MLX_MAX_MB_PER_BUFFER=100` to tune for their specific workload.

### Reset on commit

```cpp
void Device::commit_command_buffer(int index) {
    auto& stream = get_stream_(index);
    stream.buffer_ops = 0;
    stream.buffer_input_sizes = 0;
    stream.buffer_output_sizes = 0;
    stream.buffer_total_sizes = 0;
    // ... existing commit logic ...
}
```

## Implementation

### Files to modify

1. **`mlx/backend/metal/device.h`** — Add `buffer_output_sizes` and `buffer_total_sizes` to `DeviceStream`
2. **`mlx/backend/metal/device.cpp`** — Update `command_buffer_needs_commit()`, update defaults, reset on commit
3. **`mlx/backend/metal/eval.cpp`** — Ensure `set_output_array` calls track output sizes

### Migration path

1. Add output size tracking (non-breaking, just more accurate accounting)
2. Update defaults for Max/Ultra (higher ops, memory-limited commits)
3. Benchmark decode speed AND peak memory at multiple contexts
4. If results are positive, propose upstream to mlx

### Verification

| Test | Expected |
|------|----------|
| Decode 1K, E2B | ~93 tok/s (high ops, few commits) |
| Decode 16K, E2B | Comparable to current |
| Prefill 2048, E2B | Peak memory similar to ops_per_buffer=100 |
| Prefill 2048 + ops=500 | Peak memory should NOT blow up (mb limit triggers) |
| Long context 64K turbo4v2 | No OOM, memory stays bounded |

## Expected impact

- **Decode**: +11% (same benefit as ops_per_buffer=300, since decode never hits the MB limit)
- **Prefill memory**: bounded to ~1 layer's worth of intermediates (~400 MB) regardless of ops_per_buffer
- **No user-visible behavior change** for memory-constrained workloads (MB limit fires conservatively)
