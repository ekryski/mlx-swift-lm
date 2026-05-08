# Using a Model

`mlx-swift-lm` has two layers — a high-level chat API for the common case, and a
lower-level `ModelFactory` + `generate(...)` flow when you need fine-grained
control over sampling, batching, or multi-modal input.

## The 5-line version

```swift
import MLXLLM
import MLXLMCommon

let model = try await loadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

That's a complete program. The first call resolves and downloads the checkpoint
on demand (cached under `~/Documents/huggingface/`), loads weights, applies
quantization, instantiates the tokenizer, and returns a container. `ChatSession`
maintains conversation state across turns.

> **Note** — the exact `loadModelContainer` line depends on which downloader /
> tokenizer integration you've chosen. See [Picking an integration](#picking-an-integration)
> below; the rest of the API is the same regardless.

## Customising the session

`ChatSession` exposes the things you usually want for chat: a system prompt,
sampling parameters, a token budget per response, and streaming.

```swift
let session = ChatSession(
    model,
    systemPrompt: "You are a concise, factual assistant.",
    parameters: GenerateParameters(
        temperature: 0.6,
        topP: 0.95,
        maxTokens: 256
    )
)

for try await chunk in session.streamResponse(to: "Summarise the Treaty of Westphalia in three sentences.") {
    print(chunk, terminator: "")
}
```

`GenerateParameters` is the same struct used by the lower-level API; the full
field reference (every sampling knob, KV-cache hint, thinking-mode option) lives
in the [generate-parameters.md](../generate-parameters.md).

## Picking an integration

`mlx-swift-lm` does **not** depend directly on a specific downloader or tokenizer
package. You pick one of three options, ordered roughly by convenience:

1. **MLXHuggingFace macros** (built-in adapter for the official
   `huggingface/swift-huggingface` + `huggingface/swift-transformers` stack).
   Drop in `import MLXHuggingFace`, then prefix with `#huggingFace…`. Good for
   parity with mlx-swift-lm 2.x.
2. **Integration packages** — small adapter libraries that wrap a concrete
   downloader / tokenizer behind the `Downloader` / `Tokenizer` /
   `TokenizerLoader` protocols. Pick one based on which downloader / tokenizer
   you already use:

    | Downloader implementation | Adapter |
    |---|---|
    | [DePasqualeOrg/swift-hf-api](https://github.com/DePasqualeOrg/swift-hf-api) | [DePasqualeOrg/swift-hf-api-mlx](https://github.com/DePasqualeOrg/swift-hf-api-mlx) |
    | [huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface) | [DePasqualeOrg/swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) |

    | Tokenizer implementation | Adapter |
    |---|---|
    | [DePasqualeOrg/swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) |
    | [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | [DePasqualeOrg/swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) |

3. **Implement the protocols yourself** — if you already have a custom
   downloader (e.g. a private mirror) or tokenizer, conform it to
   `Downloader` / `Tokenizer` / `TokenizerLoader`. The protocols are small;
   each has only a handful of methods. The adapter packages above are good
   reference implementations.

A minimal protocol-implementation example, using `HuggingFace.HubClient`:

```swift
import HuggingFace
import MLXLMCommon

struct HubDownloader: Downloader {
    let upstream: HubClient

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: { @MainActor progress in progressHandler(progress) }
        )
    }
}
```

Then plug it in:

```swift
let model = try await loadModel(
    from: HubDownloader(upstream: HubClient()),
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

## Lower-level: `ModelFactory` and `generate(...)`

When `ChatSession` doesn't fit — multi-modal input, custom samplers, batched
decode, speculative decoding, manual cache management — drop down to the
`ModelFactory` + `generate(...)` API.

### Loading

```swift
let modelFactory: ModelFactory = LLMModelFactory.shared       // or .vision, .embeddings
let modelConfiguration = LLMRegistry.llama3_8B_4bit            // or your own ModelConfiguration
let tokenizerLoader: any TokenizerLoader = ...

let container = try await modelFactory.loadContainer(
    using: tokenizerLoader,
    configuration: modelConfiguration
)
```

`container` is a Swift `actor` that owns a `ModelContext` (the bundled model +
processor + tokenizer). All inference must happen inside `container.perform { … }`.

You can also build a `ModelConfiguration` from a raw HuggingFace ID:

```swift
let modelConfiguration = ModelConfiguration(id: "mlx-community/llama3_8B_4bit")
```

### Evaluating

The lower-level evaluation flow is: `UserInput → LMInput → generate(...)`. You
prepare the input (which may include images for VLMs), then stream tokens.

```swift
let prompt = "Summarise the impact of the Erie Canal on US industrialisation."
var input = UserInput(prompt: prompt)
// (For VLMs: input = UserInput(prompt: prompt, images: image.map { .url($0) }) )

let parameters = GenerateParameters(temperature: 0.6, maxTokens: 256)

let result = try await container.perform { context in
    let lmInput = try context.processor.prepare(input: input)
    var detok = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

    return try MLXLMCommon.generate(
        input: lmInput, parameters: parameters, context: context
    ) { tokens in
        if let last = tokens.last { detok.append(token: last) }
        if let new = detok.next() {
            print(new, terminator: "")
            fflush(stdout)
        }
        return tokens.count >= 256 ? .stop : .more
    }
}
```

### Wired memory (optional)

`mlx-swift-lm` ships a policy-based wired-memory coordinator — useful when you
have several inference tasks competing for a single GPU memory budget, or when
you want long-lived weights to participate in admission control without
inflating the wired limit while idle.

```swift
let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes)

let stream = try MLXLMCommon.generate(
    input: lmInput,
    parameters: parameters,
    context: context,
    wiredMemoryTicket: ticket
)
```

Built-in policies: `WiredSumPolicy`, `WiredMaxPolicy`, `WiredFixedPolicy`. Use
`WiredMemoryTicket.withWiredLimit` for cancellation-safe pairing. Full reference
in [memory-management.md](../memory-management.md).

## Adding to your project

### Xcode

Project → Package Dependencies → `+` → enter `https://github.com/ml-explore/mlx-swift-lm`.

For the macros path also add:

- `https://github.com/huggingface/swift-huggingface`
- `https://github.com/huggingface/swift-transformers`

For an integration-package path add the adapter from the [Picking an integration](#picking-an-integration)
table above. Apple's [Adding package dependencies to your app](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app)
docs cover the UI flow.

### SwiftPM

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.32.0-alpha")),
```

Plus your chosen integration packages, and the dependency wiring on your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]),
```

## See also

- [Quick start](../quickstart.md) — the 5-line example in context.
- [Adding a model](adding-a-model.md) — porting a new architecture.
- [Evaluation](evaluation.md) — sampling, streaming, and tool-calling details.
- [Migrating from 2.x](../migrations/v2-to-v3.md) — if your code still uses
  the pre-3.0 `loadModelContainer(configuration:)` global helper.
- [Migrating from 3.x](../migrations/v3-to-v4.md) — KV-cache rewrite (spec 006):
  `kvScheme: "turbo4v2"` → `compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2)`,
  `maybeQuantizeKVCache` removed in favour of `makeAttentionCache(...)`.
