# Supported Models

mlx-swift-lm dispatches a checkpoint to a Swift implementation by matching
the `model_type` field of `config.json` against an entry in
`LLMTypeRegistry` (text-only) or `VLMTypeRegistry` (vision-language). For
common checkpoints there is also a curated `LLMRegistry` / `VLMRegistry`
of `ModelConfiguration` values with sensible defaults.

This file is a snapshot — the registries are the source of truth. When in
doubt, grep for the registry constants in:

- `Libraries/MLXLLM/LLMModelFactory.swift` — `LLMTypeRegistry` (line 24+) and `LLMRegistry`
- `Libraries/MLXVLM/VLMModelFactory.swift` — `VLMTypeRegistry` (line 84+) and `VLMRegistry`

For LLM ↔ VLM consolidation status (which families share layer classes via
`MLXLMCommon.<Family>` namespaces) see
[`documentation/architecture.md`](../../../documentation/architecture.md#llm--vlm-consolidation).

## Quick reference

| Registry | Purpose | File |
|---|---|---|
| `LLMTypeRegistry` | `model_type` → LLM initializer | `MLXLLM/LLMModelFactory.swift` |
| `VLMTypeRegistry` | `model_type` → VLM initializer | `MLXVLM/VLMModelFactory.swift` |
| `LLMRegistry` | Curated `ModelConfiguration` values for LLM checkpoints | `MLXLLM/LLMModelFactory.swift` |
| `VLMRegistry` | Curated `ModelConfiguration` values for VLM checkpoints | `MLXVLM/VLMModelFactory.swift` |
| `EmbeddersRegistry` | Curated configurations for embedders | `MLXEmbedders/EmbeddingsModelFactory.swift` |

`MLXLMCommon.ModelRegistry` (the v2 typealias) is **deprecated** — use
`LLMRegistry` or `VLMRegistry` directly.

## LLM model types (LLMTypeRegistry)

Every entry below is registered in `LLMTypeRegistry.shared` and matches
`model_type` from `config.json`:

| `model_type` | Family | Notes |
|---|---|---|
| `llama`, `mistral` | Llama family | Llama 3, 3.1, 3.2, Mistral 7B, Mistral Nemo, CodeLlama |
| `phi`, `phi3`, `phimoe` | Phi family | Phi 2, Phi 3.5, Phi 3.5 MoE |
| `gemma`, `gemma2` | Gemma 1/2 | Standard RMSNorm with `1+weight` shift |
| `gemma3`, `gemma3_text` | Gemma 3 dense (LLM) | LLM-side text decoder; the text-only and VL forms are dispatched separately. Text-decoder layers are shared with `MLXVLM.Gemma3` via the `MLXLMCommon.Gemma3` namespace |
| `gemma3n` | Gemma 3n | Per-layer-input + altUp variant |
| `gemma4`, `gemma4_text` | Gemma 4 dense | Compiled QKV, fused norm+rope, conditional shared-MLP gate; LLM/VLM share `MLXLMCommon.Gemma4` (RMSNormNoScale / RMSNormZeroShift) |
| `qwen2` | Qwen 2 / 2.5 | LLM/VLM share `MLXLMCommon.Qwen2` (LayerArgs + Attention + MLP + DecoderLayer + ModelInner) |
| `qwen3` | Qwen 3 dense | LLM/VLM share `MLXLMCommon.Qwen3` (LayerArgs + MLP); LLM keeps `batchedForward` / `fullyBatchedForward` per-target |
| `qwen3_moe` | Qwen 3 MoE | |
| `qwen3_next` | Qwen 3 Next | Hybrid GDN+Attention (precursor to Qwen 3.5) |
| `qwen3_5`, `qwen3_5_text` | Qwen 3.5 dense (hybrid GDN+Attention) | LLM uses fused-`gate_up_proj` `Qwen3NextMLP`; VLM uses separate-projection MLP via `MLXLMCommon.Qwen35.MLP` |
| `qwen3_5_moe` | Qwen 3.5 MoE | Same hybrid pattern as the dense variant + sparse experts |
| `glm4` | GLM 4 dense | LLM/VLM share `MLXLMCommon.GLM4` (LayerArgs + fused-`gate_up_proj` MLP) |
| `glm4_moe` | GLM 4 MoE | |
| `glm4_moe_lite` | GLM 4 MoE Lite | |
| `mistral3` | Mistral 3 LLM | Shares `MLXLMCommon.Mistral3` with the VLM port |
| `lfm2` | LFM 2 dense | LLM/VLM share `MLXLMCommon.LFM2` (Configuration + Attention + ShortConv + MLP + DecoderLayer + ModelInner) |
| `lfm2_moe` | LFM 2 MoE | |
| `gpt_oss` | GPT-OSS | Harmony channel format; `--kv turbo*` supported end-to-end as of 2026-05-14 via DC-bias correction (`useBias: true` default-on full-attention layers) + hybrid sliding-FP16 policy + single-pass `turbo_flash_sdpa_v` sinks kernel. KV cache ~4× smaller than FP16 at 8K. Closes [#171](https://github.com/ekryski/mlx-swift-lm/issues/171) / [#130](https://github.com/ekryski/mlx-swift-lm/issues/130). |
| `nemotron_h` | Nemotron Cascade 2 | Hybrid Mamba + Attention; uses `SSMStateCache` for the linear-attention layers |
| `deepseek_v3`, `deepseek_v4` | DeepSeek R1 / V3 / V4 | V4 is phase-1 approximate math (header note in `Libraries/MLXLLM/Models/DeepseekV4.swift`) |
| `granite`, `granitemoehybrid` | IBM Granite | |
| `mimo`, `mimo_v2_flash` | MiMo | |
| `minimax`, `minimax_m2` | MiniMax | |
| `cohere` | Cohere | |
| `internlm2` | InternLM 2 | |
| `starcoder2` | StarCoder 2 | |
| `openelm` | OpenELM | |
| `bitnet` | BitNet b1.58 | |
| `smollm3` | SmolLM 3 | |
| `ernie4_5` | ERNIE 4.5 | |
| `exaone4` | EXAONE 4 | |
| `olmoe`, `olmo2`, `olmo3` | OLMo / OLMoE | |
| `bailing_moe` | Bailing MoE | |
| `nanochat` | NanoChat | |
| `acereason` | AceReason | Reuses `Qwen2Configuration` / `Qwen2Model` |
| `falcon_h1` | Falcon H1 | |
| `afmoe` | AfMoE | |
| `jamba_3b` | Jamba | |
| `apertus` | Apertus | |
| `baichuan_m1` | Baichuan M1 | |
| `minicpm` | MiniCPM | |

## VLM model types (VLMTypeRegistry)

| `model_type` | Family | Notes |
|---|---|---|
| `paligemma` | PaliGemma 3B | 8-bit only on `mlx-community` |
| `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` | Qwen VL line | Multi-image + video |
| `qwen3_5`, `qwen3_5_moe` | Qwen 3.5 VL | Hybrid GDN+Attention; ships with the VLM-side prefill-sync barrier (closes [#181](https://github.com/ekryski/mlx-swift-lm/issues/181)) |
| `idefics3` | Idefics 3 | |
| `gemma3` | Gemma 3 VL | 4B / 12B / 27B (`*-it-qat-4bit` are the canonical compressed checkpoints) |
| `gemma4` | Gemma 4 VL | E2B / E4B / 26B-A4B / 31B; text decoder shared with `MLXLLM.Gemma4TextModel` |
| `smolvlm` | SmolVLM 2 / Idefics3-style | See [#183](https://github.com/ekryski/mlx-swift-lm/issues/183) for `SmolVLM-Instruct-4bit` empty-output bug |
| `fastvlm`, `llava_qwen2` | FastVLM (Apple, Qwen 2 backbone) | bf16 only on `mlx-community` |
| `pixtral` | Pixtral | |
| `mistral3` | Mistral 3 VL | Pixtral-style processor with flat / nested config bridging |
| `lfm2_vl` | LFM 2 VL | |
| `glm_ocr` | GLM-OCR | Vision encoder + GLM 4 text decoder; uses M-RoPE |

## Curated `LLMRegistry` configurations

These are the `static public let` values in `LLMRegistry`. Pass them
directly to `loadModelContainer(configuration:)` / factory loaders.
Listing the major ones — see the source for the complete set (51 entries):

```swift
// Llama / Mistral
LLMRegistry.llama3_2_1B_4bit          // mlx-community/Llama-3.2-1B-Instruct-4bit
LLMRegistry.llama3_2_3B_4bit          // mlx-community/Llama-3.2-3B-Instruct-4bit
LLMRegistry.llama3_8B_4bit            // mlx-community/Meta-Llama-3-8B-Instruct-4bit
LLMRegistry.llama3_1_8B_4bit          // mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
LLMRegistry.mistral7B4bit             // mlx-community/Mistral-7B-Instruct-v0.3-4bit
LLMRegistry.mistralNeMo4bit           // mlx-community/Mistral-Nemo-Instruct-2407-4bit
LLMRegistry.codeLlama13b4bit          // mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX

// Qwen 2 / 2.5
LLMRegistry.qwen205b4bit              // mlx-community/Qwen1.5-0.5B-Chat-4bit
LLMRegistry.qwen2_5_1_5b              // mlx-community/Qwen2.5-1.5B-Instruct-4bit
LLMRegistry.qwen2_5_7b                // mlx-community/Qwen2.5-7B-Instruct-4bit

// Qwen 3
LLMRegistry.qwen3_0_6b_4bit           // mlx-community/Qwen3-0.6B-4bit
LLMRegistry.qwen3_1_7b_4bit           // mlx-community/Qwen3-1.7B-4bit
LLMRegistry.qwen3_4b_4bit             // mlx-community/Qwen3-4B-4bit
LLMRegistry.qwen3_8b_4bit             // mlx-community/Qwen3-8B-4bit
LLMRegistry.qwen3MoE_30b_a3b_4bit     // mlx-community/Qwen3-30B-A3B-4bit

// Gemma 1 / 2 / 3 / 3n / 4
LLMRegistry.gemma2bQuantized          // mlx-community/quantized-gemma-2b-it
LLMRegistry.gemma_2_2b_it_4bit        // mlx-community/gemma-2-2b-it-4bit
LLMRegistry.gemma_2_9b_it_4bit        // mlx-community/gemma-2-9b-it-4bit
LLMRegistry.gemma3_1B_qat_4bit        // mlx-community/gemma-3-1b-it-qat-4bit
LLMRegistry.gemma3n_E2B_it_lm_4bit    // mlx-community/gemma-3n-E2B-it-lm-4bit
LLMRegistry.gemma3n_E4B_it_lm_4bit    // mlx-community/gemma-3n-E4B-it-lm-4bit
LLMRegistry.gemma4_e2b_it_4bit        // mlx-community/gemma-4-e2b-it-4bit
LLMRegistry.gemma4_e4b_it_4bit        // mlx-community/gemma-4-e4b-it-4bit

// Phi
LLMRegistry.phi4bit                   // mlx-community/phi-2-hf-4bit-mlx
LLMRegistry.phi3_5_4bit               // mlx-community/Phi-3.5-mini-instruct-4bit
LLMRegistry.phi3_5MoE                 // mlx-community/Phi-3.5-MoE-instruct-4bit

// DeepSeek
LLMRegistry.deepSeekR1_7B_4bit        // mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit
LLMRegistry.deepseek_r1_4bit          // mlx-community/DeepSeek-R1-4bit

// GLM 4
LLMRegistry.glm4_9b_4bit              // mlx-community/GLM-4-9B-0414-4bit
                                       // (uses tool-call format `.glm4`)

// Granite / MiMo / Bitnet / SmolLM 3 / Baichuan
LLMRegistry.granite_3_3_2b_4bit       // mlx-community/granite-3.3-2b-instruct-4bit
LLMRegistry.miMo_7b_sft_4bit          // mlx-community/MiMo-7B-SFT-4bit
LLMRegistry.bitnet_b1_58_2b_4bit      // mlx-community/bitnet-b1.58-2B-4T-4bit
LLMRegistry.smolLM3_3b_4bit           // mlx-community/SmolLM3-3B-4bit
LLMRegistry.baichuan_m1_14B_ft_4bit   // mlx-community/Baichuan-M1-14B-Instruct-4bit-ft

// Aux
LLMRegistry.smolLM_135M_4bit          // mlx-community/SmolLM-135M-Instruct-4bit
LLMRegistry.openelm270M               // mlx-community/OpenELM-270M-Instruct
LLMRegistry.aceReason_7b_4bit         // mlx-community/AceReason-Nemotron-7B-4bit
LLMRegistry.ernie4_5_0_3b_pt_bf16_ft  // mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft
```

The bench-side registry (`Tests/Benchmarks/Utils/ModelRegistry.swift`) is a
*different* curated subset focused on the families we regression-sweep. See
[`documentation/models.md`](../../../documentation/models.md#bench-registry-families)
for the bench-side list.

## Curated `VLMRegistry` configurations

```swift
VLMRegistry.paligemma3bMix448_8bit              // mlx-community/paligemma-3b-mix-448-8bit
VLMRegistry.qwen2VL2BInstruct4Bit               // mlx-community/Qwen2-VL-2B-Instruct-4bit
VLMRegistry.qwen2_5VL3BInstruct4Bit             // mlx-community/Qwen2.5-VL-3B-Instruct-4bit
VLMRegistry.qwen3VL4BInstruct4Bit               // lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit
VLMRegistry.qwen3VL4BInstruct8Bit               // mlx-community/Qwen3-VL-4B-Instruct-8bit
VLMRegistry.smolvlminstruct4bit                 // mlx-community/SmolVLM-Instruct-4bit
VLMRegistry.lfm2_5_VL_1_6B_4bit                 // mlx-community/LFM2.5-VL-1.6B-4bit
VLMRegistry.lfm2_VL_1_6B_4bit                   // mlx-community/LFM2-VL-1.6B-4bit
VLMRegistry.ministral3_3B_4bit                  // mlx-community/Ministral-3-3B-Instruct-2512-4bit
VLMRegistry.gemma3_4B_qat_4bit                  // mlx-community/gemma-3-4b-it-qat-4bit
VLMRegistry.gemma3_12B_qat_4bit                 // mlx-community/gemma-3-12b-it-qat-4bit
VLMRegistry.gemma3_27B_qat_4bit                 // mlx-community/gemma-3-27b-it-qat-4bit
VLMRegistry.gemma4_E2B_it_4bit                  // mlx-community/gemma-4-e2b-it-4bit
VLMRegistry.gemma4_E4B_it_4bit                  // mlx-community/gemma-4-e4b-it-4bit
VLMRegistry.gemma4_31B_it_4bit                  // mlx-community/gemma-4-31b-it-4bit
VLMRegistry.gemma4_26BA4B_it_4bit               // mlx-community/gemma-4-26b-a4b-it-4bit
VLMRegistry.smolvlm                             // HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx
VLMRegistry.fastvlm                             // mlx-community/FastVLM-0.5B-bf16
VLMRegistry.qwen3_5_27B_4bit                    // mlx-community/Qwen3.5-27B-4bit
VLMRegistry.qwen3_5_35B_A3B_4bit                // mlx-community/Qwen3.5-35B-A3B-4bit
```

## Loading a model that's not in either registry

```swift
// Pass a raw HF id into ModelConfiguration; the factory dispatches based
// on `model_type` in the checkpoint's config.json. As long as the
// model_type is in LLMTypeRegistry / VLMTypeRegistry, this works without
// recompiling.
let config = ModelConfiguration(id: "mlx-community/SomeModel-4bit")

let container = try await LLMModelFactory.shared.loadContainer(
    from: hubDownloader,
    using: tokenizersLoader,
    configuration: config
)

// Pinned revision
let config = ModelConfiguration(id: "mlx-community/Model", revision: "v1.0")

// Local model
let config = ModelConfiguration(directory: URL(filePath: "/path/to/model"))
```

## Per-model overrides

`ModelConfiguration` exposes a few fields that aren't in the checkpoint
itself but tune behaviour:

```swift
// Extra stop tokens (some chat templates emit non-EOS terminators)
let config = ModelConfiguration(
    id: "...",
    extraEOSTokens: ["<end_of_turn>"]      // Gemma family
)

// Tool-call format override (most are auto-detected; override when
// auto-detection picks the wrong one or for new models pre-detection)
let config = ModelConfiguration(
    id: "mlx-community/GLM-4-9B-0414-4bit",
    toolCallFormat: .glm4
)

// Tokenizer class override
let config = ModelConfiguration(
    id: "...",
    overrideTokenizer: "PreTrainedTokenizer"
)

// Use tokenizer from a different model
let config = ModelConfiguration(
    id: "model-without-tokenizer",
    tokenizerId: "different-model-with-tokenizer"
)
```

The full list of `ToolCallFormat` cases (from
`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`):

```swift
public enum ToolCallFormat: String, Sendable, Codable, CaseIterable {
    case json                              // default; auto-detected for most chat templates
    case lfm2                              // LFM 2 / LFM 2 MoE
    case xmlFunction = "xml_function"      // Qwen 3 Coder-style XML
    case glm4                              // GLM 4 family
    case gemma                             // Gemma 3 / 3n / 4
    case kimiK2 = "kimi_k2"                // Kimi K2
    case minimaxM2 = "minimax_m2"          // MiniMax M2
    case mistral                           // Mistral / Mistral Nemo / Ministral 3
    case llama3                            // Llama 3 / 3.1 / 3.2
}
```

For the parser internals see [tool-calling.md](tool-calling.md).

## Adding a new model_type

```swift
// Custom model type registration
LLMTypeRegistry.shared.register(
    modelType: "custom_model",
    creator: { configData in
        let config = try JSONDecoder().decode(CustomConfig.self, from: configData)
        return CustomModel(config)
    }
)
```

For the full porting flow including `MLXLMCommon` shared-namespace usage
and the VLM prefill-sync barrier pattern see [model-porting.md](model-porting.md).

## Memory budgets (rough)

For 4-bit weight quantization, single-stream B=1, KV cache ≈ 5–15 % of
weights at typical context lengths:

| Model size | 4-bit weights | 8-bit weights | bf16 |
|---|---|---|---|
| 1B | ~0.5 GB | ~1 GB | ~2 GB |
| 3B | ~1.5 GB | ~3 GB | ~6 GB |
| 7B | ~3.5 GB | ~7 GB | ~14 GB |
| 13B | ~6.5 GB | ~13 GB | ~26 GB |
| 27B | ~14 GB | ~28 GB | ~54 GB |
| 30B (A3B) | ~15 GB | ~30 GB | (~60 GB) |
| 35B (A3B) | ~17 GB | ~35 GB | (~70 GB) |
| 70B | ~35 GB | ~70 GB | — |

Add `WiredMemoryUtils`-estimated KV + workspace; see
[wired-memory.md](wired-memory.md). For batched / long-context / KV-quant
combinations, see the deployment-shape table in
[`documentation/batched-decoding.md`](../../../documentation/batched-decoding.md).

## Notes

- The factory dispatches based on the **`model_type` field of `config.json`**, not on the HF repo id — so HF ids with vendor prefixes (`mlx-community/`, `lmstudio-community/`, etc.) work transparently.
- Mixed quantization is read from `quantization_config` in the model JSON (the per-layer-bits map).
- `mlx-community` MLX checkpoints are pre-quantized and pre-sanitized; locally-converted models must have a complete `config.json` and `*.safetensors` set.
