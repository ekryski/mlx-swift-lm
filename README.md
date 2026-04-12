# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large
language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key features include:

- Integration with the Hugging Face Hub to easily use thousands of LLMs with a single command.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM check out
the [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

# Using MLX Swift LM

The MLXLLM, MLXVLM, MLXLMCommon, and MLXEmbedders libraries are available
as Swift Packages.

Add the following dependency to your Package.swift:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
```

or use the latest release:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
```

Then add one or more libraries to the target as a dependency:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-lm/` to the
`Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in
Xcode.

# Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). You can get started with a wide
variety of open weights LLMs and VLMs using this simplified API:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Or use the underlying API to control every aspect of the evaluation.

# Why `make` instead of `swift build`

Swift Package Manager (SPM) does not handle several parts of the build pipeline:

- **Metal shaders** -- SPM cannot compile `.metal` files. The MLX Metal kernels must be compiled separately into `mlx.metallib` and copied into the test bundle.
- **Native C++ dylibs** -- The prefill bridge (`libprefill_bridge_v2.dylib`) is built via `clang++` and must be placed in the build output and test bundle manually.
- **Submodule staleness** -- When you modify C/C++ files deep in git submodules (`mlx-swift` -> `mlx` -> `mlx-c`), SPM’s build cache may not detect the change. It tracks content signatures keyed by the dependency’s git revision, so edits within a submodule can go stale.
- **Test bundle regeneration** -- `swift build --build-tests` can regenerate the `.xctest` bundle, wiping previously-copied Metal shaders and dylibs.

The project [Makefile](Makefile) wraps SPM and fills these gaps using file-timestamp dependency tracking. It only rebuilds what actually changed:

| What changed | What rebuilds | What stays cached |
|---|---|---|
| A `.metal` or kernel `.h` file | Metal shaders only | SPM targets, bridge dylib |
| A `.cpp`/`.c`/`.h` in `mlx` or `mlx-c` | SPM’s Cmlx target only | Swift targets, Metal, bridge |
| Swift sources | SPM incremental rebuild | Metal, bridge |
| `prefill_bridge_v2.cpp` | Bridge dylib only | SPM, Metal |
| Nothing | Artifact copy only (~instant) | Everything |

After every build, artifacts (metallib, dylibs) are copied to the release directory and test bundle automatically.

You do not need to use `make` directly for typical workflows -- `setup-dev.sh` and `benchmark.sh` both call it internally. For manual builds or targeted rebuilds, see `make help`.

# Testing

Tests require Metal and must be run via Xcode’s build system so that the MLX
Metal shaders (`default.metallib`) are built and available. Running `swift test`
will fail with “Failed to load the default metallib” because SwiftPM does not
build Metal shaders.

In Xcode: open the package and run tests (Ctrl-U), or from the command line:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination ‘platform=macOS’
```

# Benchmarking

Inference benchmarks measure prefill throughput, token generation speed, TTFT, **perplexity**, and GPU memory across models, quantization levels, and KV cache configurations. Benchmarks run in **release mode** and write markdown reports to `benchmarks/`.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete CLI reference, methodology details, and environment variable API.

## Setup

Run once after cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

This resolves Swift packages, compiles Metal shaders, builds the prefill bridge dylib, does an initial release build, and copies all artifacts into the test bundle. After setup, all benchmark commands work immediately.

## Basic Benchmark

Benchmark any registered model family or HuggingFace repo directly:

```bash
# Known model family (downloads automatically on first run)
./scripts/benchmark.sh --model qwen35-0.8b --context 128

# Any HuggingFace model by repo ID
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128

# With perplexity tracking
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128 --ppl
```

Results are saved as markdown tables in `benchmarks/<model-family>/`.

## Manual Builds

For targeted rebuilds when working on specific parts of the stack:

```bash
make                # Full incremental build (only rebuilds what changed)
make metal          # Recompile Metal shaders only
make bridge         # Recompile prefill bridge dylib only
make spm            # Swift build only (with Cmlx cache invalidation)
make status         # Show what’s built and what’s stale
make clean-cmlx     # Force SPM to recompile C/C++ on next build
make help           # Full reference
```

For more advanced benchmark combinations and options see [`benchmarks/README.md`](benchmarks/README.md).

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
