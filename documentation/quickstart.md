# Quick Start

Generate text in 5 lines:

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

The first call resolves and downloads the checkpoint on demand (cached under
`~/Documents/huggingface/`), loads weights, applies quantization, instantiates
the tokenizer, and returns a container. `ChatSession` keeps conversation
context across turns — the second call references the location from the first
without you re-stating it.

## With an image

For VLMs, pass an image into `respond`:

```swift
import MLXVLM
import MLXLMCommon

let model = try await loadModelContainer(
    configuration: VLMRegistry.qwen2_5VL3BInstruct4Bit
)

let session = ChatSession(model)
print(try await session.respond(
    to: "What do you see in this image?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
))
```

## Streaming

```swift
for try await chunk in session.streamResponse(to: "Why is the sky blue?") {
    print(chunk, terminator: "")
}
```

## Next steps

| Want to … | Read |
|---|---|
| Add `mlx-swift-lm` to your project | [installation.md](installation.md) |
| Pick a downloader / tokenizer integration | [llm/using.md § Picking an integration](llm/using.md#picking-an-integration) |
| Customize sampling, system prompts, max tokens | [llm/evaluation.md § Customising a session](llm/evaluation.md#customising-a-session) |
| Use the lower-level `ModelFactory` + `generate(...)` API | [llm/using.md § Lower-level](llm/using.md#lower-level-modelfactory-and-generate) |
| Find which models are supported and what gaps exist | [models.md](models.md) |
| Understand the LLM ↔ VLM ↔ MLXLMCommon split | [architecture.md](architecture.md) |
| Port a new model | [llm/adding-a-model.md](llm/adding-a-model.md) / [vlm/adding-a-model.md](vlm/adding-a-model.md) |
