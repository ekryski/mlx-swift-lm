# MLX Swift LM

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [Breaking Changes](#breaking-changes) for more information.

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience:

- Maximum freedom
  - Copy the protocol conformance code (~100 lines) from the [integration packages](#Tokenizer-and-Downloader-Integrations)
- Freedom and convenience
  - Use the [integration packages](#Tokenizer-and-Downloader-Integrations) for your preferred tokenizer and downloader packages
- Convenience
  - Use the macros for integration with Swift Transformers and Swift Hugging Face

### Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
```

Then add your preferred tokenizer and downloader integrations:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.1.0"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHuggingFace", package: "swift-hf-api-mlx"),
    ]),
```

### Tokenizer and Downloader Integrations

Tokenization and model downloading are handled by separate packages. Adapters make it easy to use your preferred tokenizer and downloader packages. For instructions on how to use them, see the readmes in the respective packages.

| Tokenizer package                                            | Adapter                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DePasqualeOrg/swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | [DePasqualeOrg/swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) |

| Downloader package                                           | Adapter                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface) | [DePasqualeOrg/swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) |
| [DePasqualeOrg/swift-hf-api](https://github.com/DePasqualeOrg/swift-hf-api) | [DePasqualeOrg/swift-hf-api-mlx](https://github.com/DePasqualeOrg/swift-hf-api-mlx) |


> **Note:** The adapters are offered for convenience and are not required. You can also use tokenizer and downloader packages directly by setting up the required protocol conformance for MLX Swift LM, just like the code in the integration packages. Alternatively, you can use the macros provided by this package to integrate with Swift Transformers and Swift Hugging Face.

### Quick Start

You can get started with a wide variety of open-weights LLMs and VLMs using this simplified API (for more details, see  [MLXLMCommon](Libraries/MLXLMCommon)):

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let model = try await loadModel(
    from: HubClient.default,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Loading from a local directory:

```swift
import MLXLLM
import MLXLMTokenizers

let modelDirectory = URL(filePath: "/path/to/model")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

Use a custom Hugging Face client:

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let hub = HubClient(token: "hf_...")
let container = try await loadModelContainer(
    from: hub,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
```

Use a custom downloader:

```swift
import MLXLLM
import MLXLMCommon
import MLXLMTokenizers

struct S3Downloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Download files and return a local directory URL.
        return URL(filePath: "/tmp/model")
    }
}

let container = try await loadModelContainer(
    from: S3Downloader(),
    using: TokenizersLoader(),
    id: "my-bucket/my-model"
)
```

Or use the underlying API to control every aspect of the evaluation.

## Migrating to Version 3

Version 3 of MLX Swift LM decouples the tokenizer and downloader implementations. See the [integrations](#Tokenizer-and-Downloader-Integrations) section for details.

### New dependencies

Add your preferred tokenizer and downloader adapters:

```swift
// Before (2.x) – single dependency
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x) – core + adapters
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "3.0.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.1.0"),
```

And add their products to your target:

```swift
.product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),

// If you use MLXEmbedders:
.product(name: "MLXEmbeddersTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXEmbeddersHFAPI", package: "swift-hf-api-mlx"),
```

### New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.x)
import MLXLLM
import MLXLMHFAPI      // Downloader adapter
import MLXLMTokenizers // Tokenizer adapter
```

If you use MLXEmbedders:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI      // Downloader adapter
import MLXEmbeddersTokenizers // Tokenizer adapter
```

### Loading API changes

The core APIs now include a `from:` parameter of type `URL` or `any Downloader` as well as a `using:` parameter for the tokenizer loader. Tokenizer integration packages may supply convenience methods with a default tokenizer loader, allowing you to omit the `using:` parameter.

The most visible call-site changes are:

- `hub:` → `from:`: Models are now loaded from a directory `URL` or  `Downloader`.
- `HubApi` → `HubClient`: A new implementation of the Hugging Face Hub client is used.

Example when downloading from Hugging Face:

```swift
// Before (2.x) – hub defaulted to HubApi()
let container = try await loadModelContainer(
    id: "mlx-community/Qwen3-4B-4bit"
)

// After (3.x) – Using Swift Hugging Face + Swift Tokenizers
let container = try await loadModelContainer(
    from: HubClient.default,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

At the lower-level core API, you can still pass any `Downloader` and any `TokenizerLoader` explicitly.

Loading from a local directory:

```swift
// Before (2.x)
let container = try await loadModelContainer(directory: modelDirectory)

// After (3.x)
let container = try await loadModelContainer(from: modelDirectory)
```

Loading with a model factory:

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    from: HubClient.default,
    configuration: modelConfiguration
)
```

Loading an embedder:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI
import MLXEmbeddersTokenizers

let container = try await loadModelContainer(
    from: HubClient.default,
    configuration: .configuration(id: "sentence-transformers/all-MiniLM-L6-v2")
)
```

### Renamed methods

`decode(tokens:)` is renamed to `decode(tokenIds:)` to align with the `transformers` library in Python:

```swift
// Before (2.x)
let text = tokenizer.decode(tokens: ids)

// After (3.0)
let text = tokenizer.decode(tokenIds: ids)
```

## Development Setup

Run once after cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

### Why `make` instead of `swift build`

Swift Package Manager (SPM) does not handle several parts of this repo's build pipeline:

- **Metal shaders** — SPM cannot compile `.metal` files. The MLX Metal kernels must be compiled separately into `mlx.metallib` and copied into the test bundle.
- **Submodule staleness** — When you modify C/C++ files deep in git submodules (`mlx-swift` → `mlx` → `mlx-c`), SPM's build cache may not detect the change. It tracks content signatures keyed by the dependency's git revision, so edits within a submodule can go stale.
- **Test bundle regeneration** — `swift build --build-tests` can regenerate the `.xctest` bundle, wiping previously-copied Metal shaders.
- **Stale repository cache** — SPM keeps a global bare-repo cache at `~/Library/Caches/org.swift.swiftpm/repositories/` that survives `swift package reset`. When a tracked branch advances (e.g. our `ekryski/mlx-swift` alpha picks up new submodule pins), that cache can serve a stale revision on the next resolve, producing build errors against missing symbols. `make clean-all` clears the cached entries for this project's three forked deps (`mlx-swift`, `mlx`, `mlx-c`) by inspecting each entry's origin URL — unrelated packages like `mlx-audio-swift` or `mlx-vlm` are left alone.

The project [Makefile](Makefile) wraps SPM and fills these gaps using file-timestamp dependency tracking. It only rebuilds what actually changed:

| What changed | What rebuilds | What stays cached |
|---|---|---|
| A `.metal` or kernel `.h` file | Metal shaders only | SPM targets |
| A `.cpp`/`.c`/`.h` in `mlx` or `mlx-c` | SPM's Cmlx target only | Swift targets, Metal |
| Swift sources | SPM incremental rebuild | Metal |
| Nothing | Artifact copy only (~instant) | Everything |

After every build, the metallib is copied to the release directory and test bundle automatically.

You do not need to use `make` directly for typical workflows — `setup-dev.sh` and `benchmark.sh` both call it internally. For manual builds or targeted rebuilds, see `make help`.

#### Manual builds

For targeted rebuilds when working on specific parts of the stack:

```bash
make                # Full incremental build (only rebuilds what changed)
make metal          # Recompile Metal shaders only
make spm            # Swift build only (with Cmlx cache invalidation)
make status         # Show what's built and what's stale
make doctor         # Verify resolved deps have required symbols + submodule pin consistency
make clean-cmlx     # Force SPM to recompile C/C++ on next build
make help           # Full reference
```

`make doctor` is a fast offline diagnostic that catches the two common ways the dep chain goes silently stale before you sit through a minute-long build that fails at link time:

1. **Stale SPM pin** — the resolved `mlx-swift` checkout is too old to have a symbol our Swift code calls (e.g. `MLXFast.turboBulkDequantRotated`). Manifests at compile time as `type 'MLXFast' has no member 'X'`.
2. **Submodule drift** — `mlx-swift`'s `Source/Cmlx/mlx` or `Source/Cmlx/mlx-c` checkouts have advanced past the SHAs the gitlink expects (e.g. someone manually `git pull`-ed inside a submodule).

For each it prints either OK or a one-line remediation hint. Add new `MLXFast` symbols our code requires to `DOCTOR_REQUIRED_SYMBOLS` in the Makefile so they are checked too.

## Testing

Tests require Metal and must be run via Xcode's build system so that the MLX Metal shaders (`default.metallib`) are built and available. Running `swift test` directly will fail with "Failed to load the default metallib" because SwiftPM does not build Metal shaders.

In Xcode: open the package and run tests (Cmd-U), or from the command line:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

Benchmarks are an exception — they're gated behind a Swift Testing filter (`swift test --filter benchmark`) and invoked via `./scripts/benchmark.sh`, which handles the Metal shader build via `make`. See the next section.

## Benchmarking

Inference benchmarks measure prefill throughput, decode tok/s, TTFT, perplexity, KL divergence, and GPU memory across models, quantization levels, and KV cache configurations. Benchmarks run in **release mode** and write markdown reports to `benchmarks/`.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete CLI reference, methodology details, and environment variable API.

This resolves Swift packages, compiles Metal shaders, does an initial release build, and copies all artifacts into the test bundle. After setup, all benchmark commands work immediately.

### Basic benchmark

Benchmark any registered model family or HuggingFace repo directly:

```bash
# Known model family (downloads automatically on first run)
./scripts/benchmark.sh --model qwen35-0.8b --context 128

# Any HuggingFace model by repo ID
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128

# With perplexity tracking
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128 --ppl

# Multi-model sweep — all rows land in one hardware-dated report file
./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2 --quick
```

Results are saved as hardware-dated markdown files in `benchmarks/`, one file per sweep per day (e.g. `benchmarks/m1-max-64gb-2026-04-16.md`). All runs for the same model across the sweep are grouped together in that file.

For more advanced benchmark combinations and options see [`benchmarks/README.md`](benchmarks/README.md).

### Choosing a deployment shape (Apple Silicon)

The 416-run batched sweep in [`benchmarks/batched-sweep-2026-04-29.md`](benchmarks/batched-sweep-2026-04-29.md) tells a clear story for inference workloads on Apple Silicon. Two patterns are worth promoting as defaults:

**For batched serving (B>1) — pick MoE.** MoE models with active-parameter sparsity (only a fraction of weights run per token) leave plenty of GeMM headroom for batching to amortize. Best observed speedups at B=2 ctx=1024:

| Model | B=1 tok/s | B=2 agg tok/s | Speedup |
|---|---:|---:|---:|
| `gemma4-26b-a4b` (active 4B) | 28.0 | 39.2 | **1.40×** |
| `qwen35-35b-a3b` (active 3B) | 64.6 | 85.8 | **1.32×** |
| `nemotron-30b-a3b` (active 3B) | 75.4 | 80.5 | 1.07× |

Dense models at the same parameter count peak around 1.05–1.20× and lose ground above ctx=4k. `gemma4-26b-a4b` is the standout — it holds **1.30× at ctx=8k**, the only model in the registry to do so.

**For single-stream (B=1) on memory-constrained hardware — `gemma4-e2b` / `qwen35-2b` / `qwen35-4b`.** Dense decode tok/s stays high through 16k context with peak GPU usage well under what a 16 GB Mac can wire (under 5 GB at ctx=16k for 9B 4-bit B=1). Pair with `--kv turbo4v2` if you want to fit a larger model class at the same memory budget.

**TurboQuant note.** At B=1 turbo4v2 closely matches no-quant on every model (within 5%). At B>1 long-context the gap widens significantly (0.60× on 9B at ctx=32k B=2) — this is a known regression filed in the follow-up issue list. Use turbo4v2 when memory matters; skip it when speed matters and you have RAM.

## Configuration

Knobs that change runtime behavior at the inference level — not the bench harness. Bench/profiling-only flags (`MLX_BENCH_*`, `MLX_METAL_PROFILE`) live in [`benchmarks/README.md`](benchmarks/README.md).

### `GenerateParameters` (programmatic API)

Set on the `GenerateParameters` struct passed to `generate(...)`. Defaults shown in parentheses.

| Field | Default | Notes |
|---|---|---|
| `temperature` | `0.6` | Sampling temperature. `0` selects greedy. |
| `topP` | `1.0` | Nucleus sampling threshold. |
| `topK` | `0` | Top-k cutoff (0 = disabled). |
| `minP` | `0.0` | Minimum probability filter. |
| `repetitionPenalty` | `nil` | DRY-style penalty for recent tokens. |
| `repetitionContextSize` | `20` | Window applied to `repetitionPenalty`. |
| `presencePenalty` / `frequencyPenalty` | `nil` | OpenAI-style penalties. |
| `maxTokens` | `nil` | Upper bound on generated tokens. |
| `maxKVSize` | `nil` | Hard cap on KV cache tokens; backs `StandardKVCache` in `.window` eviction mode (legacy alias `RotatingKVCache`). |
| `kvScheme` | `nil` | KV-cache compression scheme (`"turbo4"`, `"turbo4v2"`, `"affine4"`, etc.). When unset, models load `StandardKVCache` (legacy alias `KVCacheSimple`) or their architecture default. Parsed via `KVCache.CompressionAlgorithm.init?(_:)`. |
| `kvBits` / `kvGroupSize` | `nil` / `64` | Generic quantized KV cache (independent of `kvScheme`). |
| `quantizedKVStart` | `0` | First token at which generic-quant KV kicks in. |
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill — lower = lower peak GPU at the cost of prefill throughput. Falls back to the model's `defaultPrefillStepSize` (Qwen35 dense 1024 / Qwen35 MoE 4096 / Gemma 4 4096 / GPT-OSS 2048 / Nemotron 1024). M1 Max sweep on Qwen 2B / ctx=16k / `--kv none` (peak / prefill tok/s): 256 → 2.26 GB / 1106 · 512 → 2.27 GB / 1132 · 1024 → 2.38 GB / 1148 · 2048 → 2.51 GB / 1182. |
| `turboBoundarySkip` | `2` | TurboQuant codebook boundary skip; lowers raise PPL slightly but speed up encode. |
| `ngramSize` | `0` | Prompt-lookup n-gram speculative decoding (n-gram length). `0` disables. Net win only on repetitive output (code, templates). |
| `maxNgramDraftTokens` | `0` | Max draft tokens per speculation round. Pair with `ngramSize`. |
| `reasoningEffort` | `nil` | Hint passed to chat templates that support it (`"low"`, `"medium"`, `"high"`). |
| `thinkStartTokenId` / `thinkEndTokenId` | `nil` | Token IDs for thinking-phase boundaries; enables phase-separated logprob tracking when set. |
| `thinkingPhasePrefilled` | `false` | Set when the prompt already opens with `<think>`. |
| `harmonyChannelMarkerTokenId` / `harmonyThinking…` / `harmonyGeneration…` | `nil` / `[]` / `[]` | GPT-OSS harmony-format phase machine. |
| `collectPerTokenData` | `false` | Store per-token logprobs / IDs / phase labels for downstream KLD. |
| `trackPerplexity` | `false` | Accumulate logprobs for end-of-run PPL. |

### KV cache toggles

| Flag | Default | Notes |
|---|---|---|
| `TurboQuantizedKVCache.useCompressedAttention` | `true` | **B path is the default as of [`c5ca7a3`](https://github.com/ekryski/mlx-swift-lm/commit/c5ca7a3).** Decode runs the fused bulk-dequant Metal kernel + `MLXFast.scaledDotProductAttention` on the compressed cache — within ~3% of `--kv none` on Qwen 9B and faster than `--kv none` at short context. Set the constructor flag to `false` (or use the env override below) for the historic A path that keeps a raw fp16 cache. (Legacy alias `TurboQuantKVCache`.) |
| `BatchedKVCache.maxBatch` | constructor arg | Max simultaneous decode streams sharing one cache. Must match the request shape. |
| `StandardKVCache.reserve(_:)` | not called | **Opt-in workload-size hint** for windowed eviction. Pre-allocates the rotating buffer to a known size up front (typically `prompt_length + maxTokens`) instead of growing in `step`-sized chunks (default `step=256`). Idempotent: only takes effect before the first write. Clamped to `maxCacheSize`, floored at `step`. No-op when eviction is `.unbounded`. Most useful when `maxKVSize` is generous (e.g. a model's full context window) but the actual workload uses only a fraction — the hint sizes the buffer to the workload instead of growing incrementally or pre-allocating the full window. Behaviour is unchanged when never called. (Legacy alias path: `RotatingKVCache.reserve(_:)`.) <br/>`let cache = StandardKVCache(maxSize: 4096)` <br/>`cache.reserve(promptLen + maxTokens)`<br/><br/>**Programmatic construction:** `makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2), eviction: .unbounded)` — single source of truth for `kvScheme` string parsing. |

### Environment variable overrides

These env vars take precedence over the constructor / `GenerateParameters` defaults. They exist for **diagnostics, A/B testing, and tuning** — not as the primary user-facing API. Set them in the shell before launching an inference process; they are read once at first use and cached.

#### Cache / attention path

| Variable | Effect |
|---|---|
| `TURBO_COMPRESSED_ATTENTION=0` | Force the raw-fp16 working buffer "A" path globally — overrides the constructor's `useCompressedAttention=true` default. `TURBO_COMPRESSED_ATTENTION=1` forces the compressed "B" path. If unset honors the constructor value ("B" path compressed attention by default). From profiling the "A" path is faster but bloats memory because we have our Turbo compressed KV cache and a constant working fp16 buffer sent to the default SDPA metal kernel. With compressed attention it is more true to the Turbo compression algorithm because the compressed KV cache is accessed by either a custom fused Dequant + SDPA metal kernel (default due to speed) or the TurboFlash metal kernel (can be enabled by setting `TURBO_DEQUANT_SDPA=1`), both of which are currently slower than the default MLX SDPA. **The trade off is memory savings vs. speed.** |
| `TURBO_DEQUANT_SDPA=0` | Disable the fused-dequant + matrix-engine SDPA path; falls back to TurboFlash. Useful when sweeping over very long contexts where TurboFlash's per-token bit-unpack still wins (ie. ≥ 24k on Qwen 9B / Nemotron-class). |
| `TURBO_DEQUANT_JIT=1` | Force the JIT'd `MLXFast.metalKernel` bulk-dequant path instead of the precompiled `MLXFast.turboBulkDequantRotated`. Use for A/B comparison when iterating on the dequant kernel itself; the precompiled C code kernel path is the shipping default since it avoids the first-dispatch PSO compile in TTFT. |
| `TURBO_FLASH_BLOCK_SIZE=N` | Pin TurboFlash pass1's kernel block size (override the adaptive `tokenCount/32` heuristic). Powers of two only. A performance knob to tune for particular models and context sizes. |
| `TURBO_FLASH_NR0=N` | Set the number of query Rows handled per SIMD group in the first pass (Index/Pass `0`) of the TurboFlash decode kernel. Default `2` ; `1` falls back to single-row in first pass. The default `2` register cost (dim=128, NR0=2): ~24 extra floats per thread vs NR0=1 fits comfortably inside Apple's 96-register/thread budget. NR0=4/8 might pay off on bigger register files/future architecture but aren't instantiated in the metallib yet. Not really something to mess with unless profiling new hardware tuning command buffers. |
| `TURBO_SPARSE_V_THRESHOLD=N` | Skip-V threshold for the separated `mseWeightedSum` kernel. Default `1e-6`. `0.0` disables skip; `1e-4` is too aggressive and clips long-context attention. |
| `TURBO_DEBUG=1` | Verbose logging from `compressedAttention` (offsets, shapes, key-norm sanity). Only enable for short debugging because it will impact speed. |

#### Model-specific

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. May be removed in future. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. The architecture default is on for some configurations and off where the wrapper costs ~10 % decode. |
| `GDN_EVAL_INTERVAL=N` | GatedDelta (Qwen3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

#### Wired memory

`WiredMemoryUtils.resolveTicket(...)` honours these env vars when sizing a wired-memory ticket. Bench harness uses this directly; library callers can opt in via the same API.

| Variable | Effect |
|---|---|
| `MLX_MEMORY_LIMIT` | Explicit wired-memory limit. Accepts plain bytes or human-friendly units (`32g`, `32GB`, `512m`, `4k`, `1.5g`), case-insensitive. Bypasses the smart estimator and `MLX_SMART_MEMORY`. Clamped to `GPU.maxRecommendedWorkingSetBytes()` when available. |
| `MLX_SMART_MEMORY` | `0` disables the model-aware estimator (then ticket falls back to `GPU.maxRecommendedWorkingSetBytes()`). Anything else, including unset, leaves the smart estimator on (the default). The estimator computes `weights + kv(maxTokens × batchSize, kvScheme) + workspace` from the loaded model — accurate when callers pass `kvHeadsOverride`/`headDimOverride`, conservative heuristic otherwise. |

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Breaking Changes

### Loading API

The `hub` parameter (previously `HubApi`) has been replaced with `from` (any `Downloader` or `URL` for a local directory). Functions that previously defaulted to `defaultHubApi` no longer have a default – callers must either pass a `Downloader` explicitly or use the convenience methods in `MLXLMHuggingFace` / `MLXEmbeddersHuggingFace`, which default to `HubClient.default`.

For most users who were using the default Hub client, adding `import MLXLMHuggingFace` or `import MLXEmbeddersHuggingFace` and using the convenience overloads is sufficient.

Users who were passing a custom `HubApi` instance should create a `HubClient` instead and pass it as the `from` parameter. `HubClient` conforms to `Downloader` via `MLXLMHuggingFace`.

### `ModelConfiguration`

- `tokenizerId` and `overrideTokenizer` have been replaced by `tokenizerSource: TokenizerSource?`, which supports `.id(String)` for remote sources and `.directory(URL)` for local paths.
- `preparePrompt` has been removed. This shouldn't be used anyway, since support for chat templates is available.
- `modelDirectory(hub:)` has been removed. For local directories, pass the `URL` directly to the loading functions. For remote models, the `Downloader` protocol handles resolution.

### Tokenizer loading

`loadTokenizer(configuration:hub:)` has been removed. Tokenizer loading now uses `AutoTokenizer.from(directory:)` from Swift Tokenizers directly.

`replacementTokenizers` (the `TokenizerReplacementRegistry`) has been removed. Use `AutoTokenizer.register(_:for:)` from Swift Tokenizers instead.

### `defaultHubApi`

The `defaultHubApi` global has been removed. Hugging Face Hub access is now provided by `HubClient.default` from the `HuggingFace` module.

### Low-level APIs

- `downloadModel(hub:configuration:progressHandler:)` → `Downloader.download(id:revision:matching:useLatest:progressHandler:)`
- `loadTokenizerConfig(configuration:hub:)` → `AutoTokenizer.from(directory:)`
- `ModelFactory._load(hub:configuration:progressHandler:)` → `_load(configuration: ResolvedModelConfiguration)`
- `ModelFactory._loadContainer`: removed (base `loadContainer` now builds the container from `_load`)

