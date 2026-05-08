# Installation

`mlx-swift-lm` ships as a SwiftPM package with five products:

| Product | Use it for |
|---|---|
| `MLXLLM` | Text-only LLMs |
| `MLXVLM` | Vision-language models |
| `MLXEmbedders` | Encoder / embedding models |
| `MLXLMCommon` | Cross-cutting infra (KV cache, generate, wired memory) — pulled in transitively when you add LLM/VLM/Embedders |
| `MLXHuggingFace` | Optional macros for the official HuggingFace downloader / tokenizer stack |

## Xcode

1. **Project → Package Dependencies → `+`**
2. Enter `https://github.com/ml-explore/mlx-swift-lm` and pick a version /
   branch. (The current alpha branch tracks `3.32.x-alpha`.)
3. Add the products you need (`MLXLLM`, `MLXVLM`, …) to your target.

You also need a downloader and a tokenizer. Pick one of the three integration
paths — see [llm/using.md § Picking an integration](llm/using.md#picking-an-integration).

If you want the macros path:

- Add `https://github.com/huggingface/swift-huggingface`
- Add `https://github.com/huggingface/swift-transformers`
- Add `MLXHuggingFace`, `HuggingFace`, `Tokenizers` to your target's
  Frameworks/Libraries list.

Apple's [Adding package dependencies to your app](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app)
docs cover the UI flow.

## SwiftPM `Package.swift`

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.32.0-alpha")),
```

Then wire the products on your target. With the macros path:

```swift
.package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),

.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]
)
```

Or with an integration package (e.g. `swift-tokenizers-mlx` +
`swift-hf-api-mlx`):

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.2.0", traits: ["Swift"]),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.2.0"),

.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),
    ]
)
```

## Platform requirements

- **macOS 14.0+** / **iOS 18+** / **Mac Catalyst 18+** / **visionOS 2.4+**
- Apple Silicon (M-series). Inference uses Metal; CPU fallback is for
  small auxiliary tensors only.

## After install

- [Quick start](quickstart.md) — generate text in 5 lines.
- [Using an LLM](llm/using.md) — full integration choices and the lower-level API.
- [Using a VLM](vlm/using.md) — full integration choices and the lower-level API.
- [Models](models.md) — supported architectures and quantizations.
