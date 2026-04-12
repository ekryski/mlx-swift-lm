# GPU→CPU→GPU Roundtrips: Remaining Instances

**Date**: 2026-04-12
**Status**: Catalogued for future optimization

## Fixed This Session

| Location | Pattern | Fix |
|----------|---------|-----|
| `Gemma4.swift:148-172` | Bridge K/V injection: `memcpy` GPU→CPU→GPU per layer | Zero-copy `MLXArray.fromCppArray()` via `pb2_get_kv_handles` |
| `Gemma4.swift:1131-1132` | Bridge token IDs: `eval()` + `.asArray(Int32.self)` | Zero-copy `pb2_run_array()` — pass `mlx::core::array*` directly |

## Remaining: LLM Paths (Low Priority)

### N-gram Draft Tokens (Initialization Only)
**File**: `Libraries/MLXLMCommon/Evaluate.swift:961, 1009`
```swift
self.promptTokenIds = prompt.reshaped(-1).asArray(Int32.self)
```
- Only at `TokenIterator` init, not in hot decode loop
- Extracts prompt token IDs to CPU for n-gram hash table lookup
- Could pass MLXArray pointer if n-gram search were GPU-side

## Remaining: VLM Paths (Affects TTFT for Vision Models)

### 1. Token Index Searching (5+ VLM models)
Pattern: `.asArray(Int.self)` loop to find image/video token positions
- `Libraries/MLXVLM/Models/QwenVL.swift:31` — `inputIds.asArray(Int.self).enumerated()`
- `Libraries/MLXVLM/Models/LFM2VL.swift:953` — `inputIds.flattened().asArray(Int.self).enumerated()`
- `Libraries/MLXVLM/Models/FastVLM.swift:1115` — `inputIds.asArray(Int.self)`
- `Libraries/MLXVLM/Models/Mistral3.swift:682` — `inputIds[0].asArray(Int32.self)`
- `Libraries/MLXVLM/Models/Pixtral.swift:839` — `inputIds[0].asArray(Int32.self)`

**Fix**: Replace with GPU-side `MLX.argWhere(inputIds .== imageTokenId)` or similar.

### 2. Mask Filtering Loops (4 VLM models)
Pattern: `.asArray(Bool.self)` + enumerate/compactMap to find true indices
- `Libraries/MLXVLM/Models/Qwen3VL.swift:1201` — `mask.asType(.bool).asArray(Bool.self)`
- `Libraries/MLXVLM/Models/Qwen3VL.swift:1581` — `mask.asArray(Bool.self)`
- `Libraries/MLXVLM/Models/Qwen35.swift:1088` — `mask.asArray(Bool.self)`
- `Libraries/MLXVLM/Models/Gemma3.swift:858` — `imageMaskExpandedFlattened.asArray(Bool.self)`
- `Libraries/MLXVLM/Models/Gemma4.swift:1260` — `imageMaskExpandedFlattened.asArray(Bool.self)`

**Fix**: Use GPU-side `MLX.which()` / `argWhere()` / nonzero operations.

### 3. Sequence Length Accumulation (Qwen25VL)
**File**: `Libraries/MLXVLM/Models/Qwen25VL.swift:515, 527, 559`
```swift
let validIndices = indexFlattened.asArray(Int.self).enumerated().filter { ... }
cuWindowSeqlens.append(contentsOf: cuSeqlensTmp.asArray(Int.self))
let cuSeqlens = cuSeqlens.asArray(Int.self)  // then CPU loop to build mask
```
**Fix**: Build attention mask using GPU-side scatter/gather ops.

### 4. Video Timestamp Generation (3 instances)
**File**: `Libraries/MLXVLM/MediaProcessing.swift:305, 426, 479`
```swift
let sampledTimeValues = MLXArray.linspace(0, durationTimeValue, count: N).asArray(Int64.self)
```
**Fix**: Pure CPU arithmetic — no need for GPU linspace:
```swift
let sampledTimeValues = (0..<N).map { Int64(Double($0) * duration / Double(N)) }
```

### 5. Image Display Conversion (Unavoidable)
**File**: `Libraries/MLXLMCommon/UserInput.swift:129`
```swift
let arrayData = array.asData()  // GPU→CPU for CIImage construction
```
Necessary for Core Image interop. Can't avoid without custom Metal→display pipeline.

## General Anti-Pattern to Avoid

```swift
// BAD: GPU→CPU→GPU roundtrip
let cpuArray = gpuArray.asArray(Type.self)
for item in cpuArray { /* CPU processing */ }
let newGpuArray = MLXArray(processedList)

// GOOD: Stay on GPU
let result = gpuOperation(gpuArray)

// GOOD: Pass pointers for cross-boundary interop
let ptr = gpuArray.ctx.ctx  // raw mlx::core::array*
bridgeFunction(ptr)          // C++ reads directly
```
