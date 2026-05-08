# Supported Models

`mlx-swift-lm` ships ~50 LLM architectures and ~20 VLM architectures, with
curated checkpoint lists in `LLMRegistry` and `VLMRegistry`. The bench
harness has its own subset in `Tests/Benchmarks/Utils/ModelRegistry.swift`
covering the architectures we sweep most often.

This page is the canonical landing for:

- which architectures are in tree
- what default sampling parameters / chat templates each family ships
- known gaps per model (broken quantizations, missing features, open issues)

For porting a new architecture, see [llm/adding-a-model.md](llm/adding-a-model.md)
or [vlm/adding-a-model.md](vlm/adding-a-model.md). For the LLM ↔ VLM consolidation
status (which families share layers via `MLXLMCommon`), see [architecture.md](architecture.md#llm--vlm-consolidation).

## Bench-registry families

Models in `Tests/Benchmarks/Utils/ModelRegistry.swift` are the curated set we
regression-sweep. Pass them to `./scripts/benchmark.sh --model <shortName>`
without typing the full HuggingFace path.

### Text-only LLMs

| Short name | Family | Sizes | Quantizations |
|---|---|---|---|
| `qwen35-0.8b` / `qwen35-2b` / `qwen35-4b` / `qwen35-9b` / `qwen35-27b` | Qwen 3.5 (hybrid GDN + Attention) | 0.8B – 27B | bf16 / 8bit / 4bit / nvfp4 / mxfp4 |
| `qwen36-27b` | Qwen 3.6 | 27B | 4bit |
| `qwen35-35b-a3b` | Qwen 3.5 MoE (A3B = 3B active) | 35B (35B/3B active) | 8bit / 4bit / nvfp4 / mxfp4 |
| `gpt-oss-20b` | GPT-OSS (Harmony channel format) | 20B | bf16 / 4bit / mxfp4 |
| `nemotron-30b-a3b` | Nemotron Cascade 2 (`nemotron_h` on MLX) | 30B (3B active) | 8bit / 4bit / nvfp4 / mxfp4 |
| `gemma4-e2b` / `gemma4-e4b` | Gemma 4 dense | 2B / 4B | bf16 / 8bit / 4bit / mxfp4 |
| `gemma4-31b` | Gemma 4 dense | 31B | bf16 / 8bit / 4bit / mxfp4 |
| `gemma4-26b-a4b` | Gemma 4 MoE (A4B = 4B active) | 26B (26B/4B active) | bf16 / 8bit / 4bit / mxfp4 |
| `lfm2-1.2b` | LFM 2 (LiquidAI) | 1.2B | bf16 / 8bit / 4bit |
| `ministral3-3b` | Ministral 3 (Mistral 3.x) | 3B | bf16 / 8bit / 4bit |
| `gemma3-1b` | Gemma 3 dense | 1B | bf16 / 8bit / 4bit-qat |
| `glm4-9b` / `glm4-32b` | GLM 4 (`0414` series) | 9B / 32B | bf16 / 8bit / 6bit / 4bit |

### Vision-language models

| Short name | Family | Sizes | Quantizations |
|---|---|---|---|
| `lfm2vl-1.6b` | LFM 2 VL | 1.6B | bf16 / 8bit / 4bit |
| `mistral3vl-24b` | Mistral Small 3.1 24B (Pixtral-style) | 24B | bf16 / 8bit / 4bit |
| `qwen2vl-2b` | Qwen 2 VL | 2B | bf16 / 8bit / 4bit |
| `qwen25vl-3b` | Qwen 2.5 VL | 3B | bf16 / 8bit / 4bit |
| `fastvlm-0.5b` | FastVLM (Apple, Qwen 2 backbone) | 0.5B | bf16 |
| `gemma3-4b` | Gemma 3 VL (text + vision) | 4B | bf16 / 8bit / 4bit-qat |
| `glm-ocr` | GLM-OCR (vision encoder + GLM 4 decoder) | (GLM 4 9B-class) | bf16 / 8bit / 6bit / 4bit |

The Gemma 4 entries above also expose vision when run with `--method vision`
(they're VLMs, just listed in the text-only block because they're often
benched as text models).

## Other registered checkpoints

`LLMRegistry` and `VLMRegistry` have additional curated checkpoints not in the
bench registry. See `Libraries/MLXLLM/LLMModelFactory.swift` and
`Libraries/MLXVLM/VLMModelFactory.swift` for the full list. Notable entries:

- **LLMs**: Llama 3 / 3.1 / 3.2, Qwen 1.5 / 2.5 / 3, Mistral 7B / Nemo, Phi 3.5 /
  Phi 3.5 MoE, Gemma 2 / 3n, DeepSeek R1 / Distill, OpenELM, CodeLlama, Granite,
  MiMo, AceReason, BitNet, Baichuan, SmolLM 1 / 2 / 3, ERNIE 4.5.
- **VLMs**: PaliGemma 3B, Qwen 3 VL 4B, SmolVLM, SmolVLM2, Gemma 3 12B / 27B
  (vision-enabled), the four Gemma 4 dense + MoE checkpoints (vision mode).

To use one not in either registry, pass the HuggingFace ID directly to
`ModelConfiguration(id:)` — the factory will resolve and download it. Example:
`./scripts/benchmark.sh --model mlx-community/SomeModel-7B-4bit`.

## Architecture port surface

These are the architectures with a Swift port in `Libraries/MLXLLM/Models/` or
`Libraries/MLXVLM/Models/`. Register a new HuggingFace checkpoint of any of
these families just by adding a `ModelConfiguration` to `LLMRegistry` /
`VLMRegistry`; you don't need to write a new model file.

**LLM**: BaichuanM1, BailingMoE, BitNet, Cohere, DeepseekV3 / V4, ERNIE,
Gemma 1 / 2 / 3 / 3n / 4, GLM 4, GLM 4 MoE Lite, GPT-OSS, Granite, GraniteMoeHybrid,
Llama, MiMo, MiMoV2 Flash, MiniCPM, MiniMax / MiniMaxM2, Mistral 3, Nanochat,
NemotronH, Olmo 2 / 3, OlmoE, OpenELM, Phi / Phi 3 / PhiMoE, Qwen 1.5 / 2 / 2.5
/ 3 / 3 MoE / 3 Next / 3.5 / 3.5 MoE, SmolLM 3, SSM, Starcoder 2.

**VLM**: FastVLM, Gemma 3 (VL), Gemma 4 (VL), GlmOcr, LFM 2 VL, Mistral 3,
PaliGemma, Qwen 2 VL, Qwen 2.5 VL, Qwen 3 VL, Qwen 3.5, SmolVLM, SmolVLM 2.

**Embedders** (`MLXEmbedders`): BERT, NomicBERT, ModernBERT.

## Known gaps

These are documented current behavioural gaps. File new ones as GitHub issues.

| Model / config | Issue | Status |
|---|---|---|
| `mlx-community/gemma-3-4b-it-qat-4bit` | [#170](https://github.com/ekryski/mlx-swift-lm/issues/170) — single-token loop on Gemma 3 4B QAT 4-bit | Open. Suspected: weight-sanitization fallback hardcodes `(64, 4, .affine)` when `quantization_config` missing in the JSON. Workaround: use the bf16 / 8bit variants. |
| `gpt-oss-20b` + `--kv turbo4v2` | [#171](https://github.com/ekryski/mlx-swift-lm/issues/171) — non-Harmony output on GPT-OSS-20B with TurboQuant value-bits=2 | Closed by default-guard. `supportsTurboQuantization: false` on `GPTOSS.swift` makes `--kv turbo4v2` silently fall back to no-quant. Path-A diagnosis (why the dequant working-buffer path fails on this model) is the open follow-up. |
| `gemma-4-*-VL` + `temp > 0` | [#169](https://github.com/ekryski/mlx-swift-lm/issues/169) — `<pad>` flood / `ThisThis` duplicate on Gemma 4 VLM | Closed. Hard `eval(cache + logits)` barrier in `Gemma4.prepare` flushes the prefill K/V writes before the iterator's first decode forward. |
| `mlx-community/Qwen3.5-27B-4bit` (vision) | [#181](https://github.com/ekryski/mlx-swift-lm/issues/181) — token-loop on vision input | Closed by [PR #180](https://github.com/ekryski/mlx-swift-lm/pull/180). Same prefill-sync race; the barrier covers both attention K/V and `SSMStateCache` state arrays via `innerState()`. |
| DeepSeek V4 | (no issue yet) | Phase-1 approximate math — see file header in `Libraries/MLXLLM/Models/DeepseekV4.swift`. Outputs are coherent; numerics tracked separately. |
| SmolVLM2 | (no issue yet) | Incomplete message generator + hardcoded processor config — see `Libraries/MLXVLM/Models/SmolVLM2.swift`. |
| FastVLM | (no issue yet) | Token-trim TODO in `Libraries/MLXVLM/Models/FastVLM.swift`. |

## Quantization notes

- **`4bit` / `8bit`** — the `mlx-community` standard; mixed-bit quant via
  `quantization_config` in the model JSON.
- **`mxfp4`** — Microscaling FP4 (E2M1 + per-block FP8 scales). Faster decode
  than 4-bit affine on some hardware, slightly higher PPL.
- **`nvfp4`** — NVIDIA-style FP4 with a different scale layout. Numerics
  similar to mxfp4; check the specific checkpoint for compatibility.
- **`qat-4bit`** — quantization-aware-trained 4-bit (Gemma 3 in particular).
  PPL is closer to 8-bit than to standard 4-bit.
- **`bf16`** — the unquantised reference. Use for max fidelity / baseline
  benchmarks.

For KV-cache quantization (different axis from weight quantization), see
[kv-cache.md](kv-cache.md).
