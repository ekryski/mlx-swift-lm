# Adding an LLM

If the architecture follows the typical decoder-only LLM pattern, adding it
takes a few steps. Use one of the existing models in
[`Libraries/MLXLLM/Models/`](../../Libraries/MLXLLM/Models/) as a template
(Llama, Qwen 2, Mistral, GLM 4 are all good references).

This guide assumes the upstream checkpoint ships:
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- one or more `*.safetensors` shards

For a deeper dive into the porting flow — including dtype quirks, KV-cache
adapter shapes, and tracing weight tensors back to their Python originals —
see [the porting guide](../developing/porting.md).

## 1. Configuration struct

Match `config.json`. Use the `_field` / `var field` pattern for values that
need defaults so missing JSON keys don't fail decoding:

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let kvHeads: Int

    public let _layerNormEps: Float?
    public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

    enum CodingKeys: String, CodingKey {
        case hiddenSize    = "hidden_size"
        case hiddenLayers  = "num_hidden_layers"
        case kvHeads       = "num_key_value_heads"
        case _layerNormEps = "rms_norm_eps"
    }
}
```

If the model has a sibling VLM port with a nested `text_config`, give the
init a dual-decode path that looks for `text_config` first then falls back to
top-level keys (Gemma 3 / Mistral 3 / Qwen 3.5 are all examples).

## 2. Model class

The outer class conforms to `LLMModel` and ideally to `LoRAModel` /
`KVCacheDimensionProvider`:

```swift
public class YourModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let kvHeads: [Int]
    @ModuleInfo var model: YourModelInner

    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ args: YourModelConfiguration) {
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model   = YourModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }
}
```

The inner class (`YourModelInner`) is where you wire attention / MLP / decoder
layers. If your architecture overlaps with one of the consolidated families
(Gemma 3 / Gemma 4 / GLM 4 / LFM 2 / Mistral 3 / Qwen 2 / Qwen 3 / Qwen 3.5),
see [architecture.md](../architecture.md#llm--vlm-consolidation) — you may
already have a `MLXLMCommon.<Family>.MLP` (or similar) that you can typealias
into and avoid duplication.

## 3. Sanitize weights (optional)

If the upstream checkpoint's tensor keys don't match your `@ModuleInfo` keys
1:1, override `sanitize(weights:)` to remap them. Common patterns:

- Strip a prefix (`model.` → empty, `language_model.` → empty).
- Drop tied LM-head weights when `tieWordEmbeddings` is true.
- Fuse separate `gate_proj` / `up_proj` into `gate_up_proj` (Qwen 3 Next /
  Qwen 3.5 dense MLP).

## 4. Register the model

In [`Libraries/MLXLLM/LLMModelFactory.swift`](../../Libraries/MLXLLM/LLMModelFactory.swift):

```swift
public class ModelTypeRegistry: @unchecked Sendable {
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "yourModel": create(YourModelConfiguration.self, YourModel.init),
        ...
    ]
}
```

The string key is the `model_type` from the checkpoint's `config.json`.

## 5. Add a registry entry

```swift
public class LLMRegistry: @unchecked Sendable {
    static public let yourModel_4bit = ModelConfiguration(
        id: "mlx-community/YourModel-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?"
    )

    private static func all() -> [ModelConfiguration] {
        [
            ...,
            yourModel_4bit,
        ]
    }
}
```

## 6. Smoke test

Before opening a PR:

```bash
./scripts/benchmark.sh \
    --model mlx-community/YourModel-4bit \
    --quant 4bit \
    --kv none \
    --method simple
```

Verify coherent output and decode tok/s in the expected range. If you've
added it to the bench registry (`Tests/Benchmarks/Utils/ModelRegistry.swift`),
the short name `--model your-model` works directly.

## 7. Optional: bench-registry entry

For models you want to include in regression sweeps, add a `ModelFamily`
entry to [`Tests/Benchmarks/Utils/ModelRegistry.swift`](../../Tests/Benchmarks/Utils/ModelRegistry.swift).

```swift
static let yourModel_4B = ModelFamily(
    name: "YourModel 4B", shortName: "your-model-4b",
    variants: [
        .init(quantization: "bf16", repoId: "mlx-community/YourModel-4B-bf16"),
        .init(quantization: "8bit", repoId: "mlx-community/YourModel-4B-8bit"),
        .init(quantization: "4bit", repoId: "mlx-community/YourModel-4B-4bit"),
    ],
    temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
    presencePenalty: nil, repetitionPenalty: nil,
    extraEOSTokens: [],
    supportsThinking: false, reasoningEffort: nil
)
```

…and add `yourModel_4B` to `allFamilies`.
