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

# Benchmarks

Inference speed benchmarks measure prefill throughput, token generation speed, TTFT, **perplexity**, and GPU memory across models, quantization levels, and KV cache configurations. Benchmarks run in **release mode** for accurate results and write markdown reports to `benchmarks/`. For different context sizes they perform a summarization of a novel from [Project Gutenberg](https://www.gutenberg.org/).

Run the full matrix (all models, all KV configs, all context sizes):

```bash
./scripts/benchmark.sh
```

Common options:

```bash
./scripts/benchmark.sh --quick                              # Fast: 128 + 1024 + 4096 tokens only
./scripts/benchmark.sh --model qwen35-9b --kv turbo4        # Single model + KV config
./scripts/benchmark.sh --model qwen35-9b --quant bf16       # Use bf16 quantization
./scripts/benchmark.sh --context 1024                       # All models at one context size
./scripts/benchmark.sh --speed                              # Tool call + multi-turn tests only
./scripts/benchmark.sh --all                                # Context + speed + tool tests
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

Each family has multiple quantization variants: `bf16`, `8bit`, `4bit`, `nvfp4`, `mxfp4` (availability varies). Select with `--quant`.

### Quantization & Baseline

By default, benchmarks use **4-bit** quantized models. Use `--quant` to select a different quantization level:

```bash
./scripts/benchmark.sh --model qwen35-9b --quant 8bit
```

Use `--baseline` to automatically select the highest-fidelity variant (bf16 preferred) that fits in your system's GPU memory. If bf16 doesn't fit, it falls back to 8-bit, then 4-bit. If nothing fits, it reports an error with the model and memory sizes.

```bash
./scripts/benchmark.sh --baseline --model qwen35-9b --context 128
```

Baseline results are saved to `benchmarks/` for later comparison against quantized runs.

### Perplexity

Every benchmark run reports **perplexity** alongside speed metrics. Perplexity measures output quality — lower is better. Comparing perplexity across quantization levels (bf16 vs 4bit) and KV cache configurations (no-quant vs turbo4) shows the quality cost of compression.

### Custom Models

Benchmark any HuggingFace model by passing its repo ID:

```bash
./scripts/benchmark.sh --model hf:mlx-community/Qwen3.5-9B-4bit --context 128
```

### KV Cache Configs

Available KV cache quantization: `none`, `affine4`, `turbo4`, `turbo3`.

Context sizes range from 128 to 131,072 tokens. Results are printed with `[BENCH]` prefix for easy parsing and saved as markdown tables in `benchmarks/`.

Run `./scripts/benchmark.sh --help` for full usage.

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
