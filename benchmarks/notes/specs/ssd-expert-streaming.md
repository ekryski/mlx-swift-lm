# Feature Spec: SSD Expert Streaming (P7)

**Status**: Research / Design
**Priority**: Medium — enables 122B+ MoE models on 64GB machines
**Applies to**: qwen3.5-35b-a3b, nemotron-30b-a3b (MoE architectures)

## Problem

Mixture-of-Experts (MoE) models like Qwen3.5-35B-A3B have 35B total parameters but only activate ~3B per token. The full weight set (~18GB at 4-bit) must reside in memory even though only ~10% is used per forward pass. On 64GB machines, this leaves insufficient room for KV cache at long contexts.

## Proposed Solution

Stream MoE expert weights directly from NVMe SSD to GPU command buffer on demand, keeping only the active experts in GPU memory. SwiftLM has proven this works on M5 Pro 64GB with Qwen3.5-122B-A3B.

## Reference Implementation: SwiftLM

SwiftLM's implementation (`LocalPackages/mlx-swift/Source/Cmlx/mlx/mlx/fast/moe_stream_op.h`):

### Architecture

1. **Pinned Metal memory allocation** — GPU-accessible buffers that don't page-fault
2. **Dispatch I/O for async weight streaming** — macOS dispatch_io for non-blocking NVMe reads
3. **Expert offsets as byte positions** — Tracked in safetensors file layout
4. **Rolling metrics** — 10-second aggregation windows tracking throughput, compression ratios

### Metal Kernel

`streamed_moe_gemm` in `moe_stream.metal`:
- Input X: Token embeddings [M, K] in bfloat16
- Weights W: 4-bit quantized expert weights [N, K/8] packed as uint32
- 4-bit unpacking: Each uint32 stores 8 4-bit weights, extracted via bit shifts
- Sign extension from 4-bit range (-8 to 7), scaled by 1/8.0
- Output: Result matrix [M, N] in float32

### Key Design Decisions

- **Zero-copy via mmap**: macOS page cache handles NVMe → GPU memory transfer. No explicit copy needed — the OS streams pages on demand
- **Only active expert pages loaded**: Router selects top-K experts, only those pages are touched
- **4-bit quantization enforced for MoE**: 2-bit breaks JSON grammar stability for tool calling
- **No backward pass**: VJP/JVP unsupported for streaming experts (inference-only)

## Implementation Plan for mlx-swift-lm

### Phase 1: Expert Weight Mapping

1. Parse safetensors file layout to identify expert weight byte ranges
2. Create an `ExpertStreamManager` that maps expert IDs → file offsets
3. Use `mmap()` on the safetensors file instead of loading all weights into MLXArray

### Phase 2: Streaming GEMM Kernel

1. Port `streamed_moe_gemm` Metal kernel to work with MLXFast.metalKernel
2. Kernel reads from mmap'd buffer, does 4-bit unpack + GEMM in one dispatch
3. Replace the standard MoE forward pass with streaming variant when `--stream-experts` flag is set

### Phase 3: Memory Management

1. Track which expert pages are resident in GPU memory
2. Implement eviction policy (LRU or frequency-based) when memory pressure detected
3. Add thermal awareness — pause generation at critical temperatures (SwiftLM does this)

### Phase 4: Integration

1. Add `--stream-experts` flag to generation parameters
2. Auto-detect when model doesn't fit in memory and suggest streaming
3. Wisdom-style auto-calibration: profile once per (model, hardware) pair

## Estimated Impact

| Metric | Without Streaming | With Streaming |
|--------|:-----------------:|:--------------:|
| Max model size (64GB) | ~27B (4-bit) | 122B+ (4-bit) |
| Expert load latency | 0ms (preloaded) | ~2-5ms per expert (NVMe) |
| Peak GPU memory | Full model | Active experts only (~10-20%) |
| Throughput | ~3-4 tok/s (122B, SwiftLM) | Same (SSD bandwidth limited) |

## Risks

1. **NVMe bandwidth bottleneck**: M1 Max SSD ~5-7 GB/s. Each expert ~100-500MB. Loading 2 experts per layer per token = significant I/O
2. **macOS page cache thrashing**: If experts don't fit in page cache, repeated reads slow down
3. **MLXFast.metalKernel limitations**: May not support mmap'd buffer inputs natively — may need C++ integration
4. **M1/M2 vs M5 performance**: SwiftLM tested on M5 Pro. Older chips have slower NVMe

## Dependencies

- macOS 14+ for optimal mmap + Metal interop
- NVMe SSD (not external storage)
- Safetensors format (not GGUF) for byte-level expert addressing
