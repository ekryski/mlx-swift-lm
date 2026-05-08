# Adding a VLM

Adding a new vision-language model is the LLM porting flow plus a vision
encoder, a processor, and a chat template that knows where the image goes.
Use one of the existing models in
[`Libraries/MLXVLM/Models/`](../../Libraries/MLXVLM/Models/) as a template
(Qwen 2 VL, Qwen 2.5 VL, Gemma 3, Mistral 3 are all good references).

This guide assumes the upstream checkpoint ships:
- `config.json` with a `text_config` and a `vision_config`
- `tokenizer.json` / `tokenizer_config.json` / `chat_template.json` (or
  `chat_template` embedded in `tokenizer_config.json`)
- a `preprocessor_config.json` describing image preprocessing
- one or more `*.safetensors` shards containing both vision and text weights

The text-decoder side follows the LLM porting flow — start with
[adding an LLM](../llm/adding-a-model.md) and come back here for the vision
plumbing.

## 1. Configurations

VLMs typically have a wrapper config holding nested `text_config` and
`vision_config`:

```swift
public struct YourVLMConfiguration: Codable, Sendable {
    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let imageTokenIndex: Int
    public let videoTokenIndex: Int?
    public let modelType: String

    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        // ...
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let imageSize: Int
        public let patchSize: Int
        // ...
    }
}
```

Some checkpoints flatten these (Pixtral / Mistral 3 VL ships a flat
`preprocessor_config.json`). Add a custom `init(from:)` that tries the nested
path first then falls back to top-level keys — see
`Libraries/MLXVLM/Models/Mistral3.swift` for a worked example.

## 2. Vision encoder

A typical vision encoder is a patch-embed → ViT-style transformer → projector
to the text-decoder's hidden size:

```swift
fileprivate class YourVisionModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: YourPatchEmbed
    fileprivate let layers: [YourVisionBlock]
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: RMSNorm
    @ModuleInfo(key: "merger") var merger: YourPatchMerger

    public func callAsFunction(
        _ pixelValues: MLXArray, gridTHW: [THW]
    ) -> (MLXArray, MLXArray) {
        // Run vision blocks → projector → return hidden + (optional) deepstack outputs
    }
}
```

The encoder's output is the per-token visual feature tensor that gets
interleaved into the text embedding stream by the processor.

## 3. Processor

The processor is the bridge between user input and the model. It:

1. Resizes / normalizes images, computes pixel grids.
2. Builds a token sequence that has placeholder `<image>` tokens
   interleaved with text.
3. Runs the chat template if the model uses one.
4. Returns an `LMInput` containing tokens, pixel values, frame metadata,
   and any positional bookkeeping the model needs (M-RoPE positions,
   visual masks, etc.).

```swift
public class YourVLMProcessor: UserInputProcessor {
    public func prepare(input: UserInput) async throws -> LMInput {
        // 1. apply chat template to messages
        // 2. for each image, resize + normalize → pixelValues
        // 3. build token sequence with <image> placeholders
        // 4. compute frame metadata (THW per image)
        // 5. return LMInput(text:..., image: LMInput.ProcessedImage(pixels:..., frames:...))
    }
}
```

For multi-image / video, the processor concatenates pixel tensors and emits a
THW (time × height × width) descriptor per frame.

## 4. Outer model + multimodal interleave

The outer `*Model` class implements `VLMModel` and exposes `prepare(...)`,
which:

1. Embeds text tokens via the text decoder's embedding table.
2. Runs the vision encoder on the pixel values.
3. Calls `mergeInputIdsWithImageFeatures(...)` (or your own equivalent) to
   replace the image-token positions with the vision features.
4. Runs the prefill through the language model with the merged embeddings.
5. **Hard-syncs the cache and logits before returning** (see below).
6. Returns `.logits(output)`.

```swift
public func prepare(_ input: LMInput, cache: [any KVCache], windowSize _: Int?)
    throws -> PrepareResult
{
    let textEmbeds = languageModel.model.embedTokens(input.text.tokens)
    let visionFeatures = visionModel(input.image!.pixels, gridTHW: input.image!.frames)

    let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
        imageFeatures: visionFeatures,
        inputEmbeds: textEmbeds,
        inputIds: input.text.tokens,
        imageTokenIndex: config.imageTokenIndex,
        videoTokenIndex: config.videoTokenIndex
    )

    let output = languageModel(
        input.text.tokens,
        cache: castCache(cache),
        inputsEmbeds: mergedEmbeds,
        // ...
    )

    // Issue #169 prefill-sync barrier — see vlm/overview.md.
    var cacheArrays: [MLXArray] = []
    for c in cache {
        cacheArrays.append(contentsOf: c.innerState())
    }
    eval(cacheArrays + [output.logits])

    return .logits(output)
}
```

**Always include the barrier.** Without it, the iterator's first decode
forward can read stale K/V from the prefill writes — the bug class fixed
across all consolidated VLMs in PRs [#172–#180](https://github.com/ekryski/mlx-swift-lm/pulls?q=is%3Apr+is%3Aclosed+wsc).
For hybrid models (GDN + Attention), `SSMStateCache: ArraysCache` already
exposes its conv + recurrent state via `innerState()` — same iteration covers
the whole stack.

## 5. Sanitize weights

Most VLM checkpoints wrap weights under `language_model.` and `visual.` /
`vision_tower.` prefixes. Override `sanitize(weights:)` to remap them to your
`@ModuleInfo` keys. Drop tied LM-head weights when `tieWordEmbeddings` is true.

## 6. Register the model

In [`Libraries/MLXVLM/VLMModelFactory.swift`](../../Libraries/MLXVLM/VLMModelFactory.swift):

```swift
private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
    "yourVLM": create(YourVLMConfiguration.self, YourVLM.init),
    // ...
]

private var processorCreators: [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor] = [
    "yourVLM": create(YourVLMProcessorConfiguration.self, YourVLMProcessor.init),
    // ...
]
```

## 7. Add a registry entry

```swift
static public let yourVLM_4bit = ModelConfiguration(
    id: "mlx-community/YourVLM-4bit",
    defaultPrompt: "Describe what you see in this image."
)

static public func all() -> [ModelConfiguration] {
    [
        ...,
        yourVLM_4bit,
    ]
}
```

## 8. Smoke test

```bash
./scripts/benchmark.sh \
    --model mlx-community/YourVLM-4bit \
    --quant 4bit \
    --kv none \
    --method vision \
    --vision-prompt "Describe what you see in this image." \
    --vision-expect dog
```

Verify (a) coherent output, (b) the `--vision-expect` keyword matches, and
(c) no token-loop on the first turn (the prefill-sync barrier above).

## 9. Optional: bench-registry entry

Same flow as the [LLM bench-registry step](../llm/adding-a-model.md#7-optional-bench-registry-entry),
but use `--method vision` for the smoke command.
