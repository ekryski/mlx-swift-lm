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
make clean-cmlx     # Force SPM to recompile C/C++ on next build
make help           # Full reference
```

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
| `maxKVSize` | `nil` | Hard cap on KV cache tokens; backs `RotatingKVCache`. |
| `kvScheme` | `nil` | KV-cache compression scheme (`"turbo4"`, `"turbo4v2"`, `"affine4"`, etc.). When unset, models load `KVCacheSimple` or their architecture default. |
| `kvBits` / `kvGroupSize` | `nil` / `64` | Generic quantized KV cache (independent of `kvScheme`). |
| `quantizedKVStart` | `0` | First token at which generic-quant KV kicks in. |
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill (defaults model-by-model). |
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
| `TurboQuantKVCache.useCompressedAttention` | `true` | **B path is the default as of [`c5ca7a3`](https://github.com/ekryski/mlx-swift-lm/commit/c5ca7a3).** Decode runs the fused bulk-dequant Metal kernel + `MLXFast.scaledDotProductAttention` on the compressed cache — within ~3% of `--kv none` on Qwen 9B and faster than `--kv none` at short context. Set the constructor flag to `false` (or use the env override below) for the historic A path that keeps a raw fp16 cache. |
| `BatchedKVCache.maxBatch` | constructor arg | Max simultaneous decode streams sharing one cache. Must match the request shape. |

### Environment variable overrides

These env vars take precedence over the constructor / `GenerateParameters` defaults. They exist for **diagnostics, A/B testing, and tuning** — not as the primary user-facing API. Set them in the shell before launching an inference process; they are read once at first use and cached.

#### Cache / attention path

| Variable | Effect |
|---|---|
| `TURBO_USE_ALPHA=1` | Force the raw-fp16 A path globally — overrides `useCompressedAttention=true`. Use when comparing decode tok/s before/after a codec change, or when bisecting a quality regression. |
| `TURBO_DEQUANT_SDPA=0` | Disable the fused-dequant + matrix-engine SDPA path; falls back to TurboFlash. Useful when sweeping over very long contexts where TurboFlash's per-token bit-unpack still wins (≥ 24k on Qwen 9B / Nemotron-class). |
| `TURBO_FLASH_BLOCK_SIZE=N` | Pin TurboFlash pass1's block size (override the adaptive `tokenCount/32` heuristic). Powers of two only. |
| `TURBO_FLASH_NR0=N` | TurboFlash multi-row queries (1 / 2). Default `2`; `1` falls back to single-row pass1. Higher values not instantiated in the metallib. |
| `TURBO_SPARSE_V_THRESHOLD=N` | Skip-V threshold for the separated `mseWeightedSum` kernel. Default `1e-6`. `0.0` disables skip; `1e-4` is too aggressive and clips long-context attention. |
| `TQ_DEBUG=1` | Verbose logging from `compressedAttention` (offsets, shapes, key-norm sanity). Hot-path; only enable for short repros. |

#### Model-specific

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. The architecture default is on for some configurations and off where the wrapper costs ~10 % decode. |
| `GDN_EVAL_INTERVAL=N` | GatedDelta (Qwen3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

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

