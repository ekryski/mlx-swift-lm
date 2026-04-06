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

# Testing

Tests require Metal and must be run via Xcode’s build system so that the MLX
Metal shaders (`default.metallib`) are built and available. Running `swift test`
will fail with “Failed to load the default metallib” because SwiftPM does not
build Metal shaders.

From the package root, run:

```bash
make test
```

or:

```bash
./scripts/test.sh
```

Alternatively, in Xcode: open the package and run tests (⌃U), or from the
command line:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

# Benchmarking

Inference benchmarks measure prefill throughput, token generation speed, TTFT, **perplexity**, and GPU memory across models, quantization levels, and KV cache configurations. Benchmarks run in **release mode** and write markdown reports to `benchmarks/`.

See [`Tests/Benchmarks/README.md`](Tests/Benchmarks/README.md) for the complete CLI reference, methodology details, and environment variable API.

## Setup

Metal shaders must be compiled before running benchmarks. Run once after cloning:

```bash
./scripts/setup-dev.sh
```

This resolves Swift packages, compiles the MLX Metal shaders, and does an initial release build. After setup, all benchmark commands work immediately.

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

## Comprehensive Benchmark

Sweep multiple context sizes, quantization levels, and KV cache strategies in one command:

```bash
# Full matrix: all quants × all KV configs, quick context ladder (128 / 1K / 4K / 32K)
./scripts/benchmark.sh --model qwen35-2b --quant all --kv all --quick

# Summarization across all 11 context sizes (128 → 128K) at one quant + KV config
./scripts/benchmark.sh --model qwen35-9b --method summarization --quant 4bit --kv turbo4

# KV compression quality sweep: measure KL divergence vs bf16 baseline
./scripts/benchmark.sh --model qwen35-2b --quant all --kv all --quick --kld

# WikiText-2 perplexity across context sizes (quality regression check)
./scripts/benchmark.sh --model qwen35-4b --method wikitext2 --quant all --context 512,2048,8192
```

### Model Families

| Shortname | Model |
|-----------|-------|
| `qwen35-0.8b` | Qwen3.5 0.8B |
| `qwen35-2b` | Qwen3.5 2B |
| `qwen35-4b` | Qwen3.5 4B |
| `qwen35-9b` | Qwen3.5 9B |
| `qwen35-27b` | Qwen3.5 27B |
| `qwen35-35b-a3b` | Qwen3.5 35B A3B (MoE) |
| `gpt-oss-20b` | GPT-OSS 20B |
| `nemotron-30b-a3b` | Nemotron Cascade 2 30B A3B |

Each family supports `bf16`, `8bit`, and `4bit` quantization via `--quant`. Use `--baseline` to auto-select the highest-fidelity variant that fits in GPU memory.

### KV Cache Configs

Available KV cache quantization via `--kv`: `none`, `affine4`, `turbo4`, `turbo3`.

Run `./scripts/benchmark.sh --help` for full usage, or see [`Tests/Benchmarks/README.md`](Tests/Benchmarks/README.md) for methodology details.

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
