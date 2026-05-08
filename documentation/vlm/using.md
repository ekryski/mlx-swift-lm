# Using a VLM

The high-level `ChatSession` API works the same for VLMs as for text-only LLMs
— pass an image (or video) into `respond(...)` and the rest is the same flow.
The lower-level `ModelFactory` + `generate(...)` API is also shared with
[`MLXLLM`](../llm/using.md); only the input preparation differs.

## The 5-line version

```swift
import MLXVLM
import MLXLMCommon

let model = try await loadModelContainer(
    configuration: VLMRegistry.qwen2_5VL3BInstruct4Bit
)

let session = ChatSession(model)
print(try await session.respond(
    to: "What kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
))
```

`image:` accepts a URL, a `CGImage`, or a `Data` containing a PNG / JPEG.
For multi-image input or video, see below.

## Follow-ups referencing the same image

`ChatSession` keeps the image context across turns:

```swift
let answer1 = try await session.respond(
    to: "What kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
)
print(answer1)  // "A golden retriever..."

let answer2 = try await session.respond(to: "What is behind the dog?")
print(answer2)  // "A wooden fence and..."
```

The second call references the dog from the first turn — the image plus the
prior response are in the KV cache, no need to re-attach the image.

## Streaming output

```swift
for try await chunk in session.streamResponse(
    to: "Describe what you see in detail.",
    image: .url(imageURL)
) {
    print(chunk, terminator: "")
}
```

## Multi-image / video

Use the lower-level API (`UserInput` + `MLXLMCommon.generate`) when you need
more than one image, video, or processor customization beyond what
`ChatSession` exposes.

```swift
import MLXVLM
import MLXLMCommon

let container = try await VLMModelFactory.shared.loadContainer(
    using: tokenizerLoader,
    configuration: VLMRegistry.qwen2_5VL3BInstruct4Bit
)

let prompt = "Compare these two images."
var input = UserInput(
    prompt: prompt,
    images: [.url(image1URL), .url(image2URL)]
)
input.processing.resize = .init(width: 768, height: 768)

let parameters = GenerateParameters(temperature: 0.6, maxTokens: 200)

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
        return tokens.count >= 200 ? .stop : .more
    }
}
```

For video input, pass the frames as an image array via the model-specific
processor (Qwen 2 VL, Qwen 3 VL, and SmolVLM 2 all support video — see the
specific model's `Processor.prepare(...)` for the video-frame path).

## Using a checkpoint not in `VLMRegistry`

```swift
let configuration = ModelConfiguration(
    id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    defaultPrompt: "Describe this image."
)
let container = try await VLMModelFactory.shared.loadContainer(
    using: tokenizerLoader,
    configuration: configuration
)
```

The factory dispatches based on `model_type` in the checkpoint's `config.json`.
If the architecture is in [models.md](../models.md), this works without
recompiling.

## Common pitfalls

1. **Token-loop on first turn (`"The!!!!!"`, `<pad>` flood, `"ThisThis"`)** —
   pre-existing bug class fixed across all consolidated VLMs in PRs
   [#172](https://github.com/ekryski/mlx-swift-lm/pull/172) – [#180](https://github.com/ekryski/mlx-swift-lm/pull/180).
   If you hit this on a custom port, add the prefill-sync barrier described
   in [vlm/overview.md](overview.md#pattern-prefill-sync-barrier-issue-169).
2. **Wrong image resize for the model** — Pixtral / Qwen-style VLMs are
   sensitive to image dims. Pass a sane `processing.resize` (typically 256–768
   square) or the processor's default.
3. **Image not attached to the right turn** — `ChatSession` only carries the
   image in the turn it was supplied; subsequent turns reference it via the KV
   cache. If you start a new conversation, re-attach.

## See also

- [VLM overview](overview.md) — what's in tree, where the code lives.
- [LLM using](../llm/using.md) — the lower-level `ModelFactory` flow is the
  same; the LLM doc is more detailed there.
- [Adding a VLM](adding-a-model.md) — porting a new architecture.
