# Evaluation

The high-level `ChatSession` API loads a model and evaluates prompts in a few
lines. Most chat-style use cases — single-turn, multi-turn, streaming, and
single-image VLM input — work without dropping below it.

## One-shot and multi-turn

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

The second question refers back to context (the location) from the first —
that context lives inside the `ChatSession`. For one-shot prompts, just create
a session, evaluate, and discard. Multiple `ChatSession` instances let you keep
several independent conversations alive (at the cost of one KV cache each).

## Streaming output

`session.respond` returns the full response as a string. To see tokens as they
generate, use `streamResponse`:

```swift
let session = ChatSession(model)
for try await chunk in session.streamResponse(to: "Why is the sky blue?") {
    print(chunk, terminator: "")
}
print()
```

## VLMs (Vision Language Models)

The same API works for VLMs — pass an image (or video) into `respond`:

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
)
let session = ChatSession(model)

let answer1 = try await session.respond(
    to: "What kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
)
print(answer1)

// Follow-up referring back to the same image — context is maintained.
let answer2 = try await session.respond(to: "What is behind the dog?")
print(answer2)
```

For multi-image input, system-message construction, or processor customization
beyond what `ChatSession` exposes, see [VLM using](../vlm/using.md).

## Customising a session

`ChatSession` accepts these initializers:

- `instructions: String?` — system-style instructions for the chat
  (e.g. _"respond in rhyme"_, _"keep responses to one paragraph"_).
- `generateParameters: GenerateParameters` — sampling temperature, `topP`,
  `topK`, `maxTokens`, repetition penalties, KV-cache hints, etc. Full field
  reference in [generate-parameters.md](../generate-parameters.md).
- `processing: UserInputProcessing?` — image / video processing options
  (resize, frame sampling, etc.).

```swift
let session = ChatSession(
    model,
    instructions: "Respond like a 17th-century pirate.",
    generateParameters: GenerateParameters(temperature: 0.9, maxTokens: 200)
)
```

## When `ChatSession` isn't enough

Drop down to the lower-level `ModelFactory` + `MLXLMCommon.generate(...)` API
when you need:

- batched decode across multiple requests sharing one model
- speculative decoding (draft model or n-gram)
- custom logit processors / token sampling
- direct `KVCache` lifetime control (e.g. prefix-caching)
- per-token logprob collection / KL divergence tracking

See the [Lower-level: `ModelFactory` and `generate(...)`](using.md#lower-level-modelfactory-and-generate)
section in [using.md](using.md).
