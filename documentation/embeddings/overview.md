# Embeddings Overview

`MLXEmbedders` ships ports of popular encoder / embedding models. It uses the
same `Downloader` + `Tokenizer` integration as `MLXLLM` / `MLXVLM`, but the
inference shape is different — embedders run a single forward pass and return
a fixed-size vector per input, no autoregressive loop.

In tree:

- **BERT** — classic encoder.
- **NomicBERT** — Nomic Embed (v1, v1.5, multimodal).
- **ModernBERT** — modern encoder with rotary embeddings, alternating
  global / local attention.

Plus the pooling helpers (mean, CLS, weighted mean) needed to turn the
per-token outputs into a single embedding.

## Quick start

```swift
import MLXEmbedders
import MLXEmbeddersHuggingFace
import MLXLMTokenizers

let modelContainer = try await loadModelContainer(
    using: TokenizersLoader(),
    configuration: .nomic_text_v1_5
)

let inputs = [
    "search_query: Animals in tropical climates.",
    "search_document: Elephants",
    "search_document: Horses",
    "search_document: Polar bears",
]

let embeddings = await modelContainer.perform {
    (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
    // 1. Tokenize each input + add special tokens.
    let tokenIds = inputs.map { tokenizer.encode(text: $0, addSpecialTokens: true) }

    // 2. Pad to longest (model expects a rectangular [B, T] tensor).
    let maxLen = tokenIds.reduce(into: 16) { $0 = max($0, $1.count) }
    let padded = stacked(tokenIds.map { ids in
        MLXArray(ids + Array(repeating: tokenizer.eosTokenId ?? 0,
                             count: maxLen - ids.count))
    })

    // 3. Build the attention mask + token-type IDs.
    let mask        = (padded .!= tokenizer.eosTokenId ?? 0)
    let tokenTypes  = MLXArray.zeros(like: padded)

    // 4. Run forward + pool + normalise.
    let pooled = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
    )
    pooled.eval()
    return pooled.map { $0.asArray(Float.self) }
}
```

`pooled` is `[[Float]]` — one normalized vector per input, ready to compute
cosine similarity / nearest-neighbour search against.

## Loading from a local directory

```swift
let modelDirectory = URL(filePath: "/path/to/embedder")
let modelContainer = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

## Custom downloader

```swift
struct S3Downloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // download to a local directory and return its URL
        return URL(filePath: "/tmp/embedder")
    }
}

let modelContainer = try await loadModelContainer(
    from: S3Downloader(),
    using: TokenizersLoader(),
    configuration: .init(id: "my-bucket/my-embedder")
)
```

## Where the code lives

- `Libraries/MLXEmbedders/` — model ports and pooling helpers.
- `Libraries/MLXEmbedders/EmbeddingModel.swift` — the `EmbeddingModel` protocol
  the model classes conform to.
- `Libraries/MLXEmbedders/EmbeddingsModelFactory.swift` — the registry of
  curated checkpoint IDs.

## See also

- [LLM using](../llm/using.md#picking-an-integration) — same downloader /
  tokenizer integration choices.
- [Architecture](../architecture.md) — how `MLXEmbedders` fits alongside
  `MLXLLM` / `MLXVLM` / `MLXLMCommon`.

## Origin

Ported to Swift from [`taylorai/mlx_embedding_models`](https://github.com/taylorai/mlx_embedding_models)
(Apache 2.0).
