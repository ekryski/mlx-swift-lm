# VLM Overview

`MLXVLM` is the vision-language model family. It ports the `Blaizzy/mlx-vlm`
Python implementations to Swift. The text decoder is the same
infrastructure as `MLXLLM` (`MLXLMCommon` cache + `generate(...)` + chat
templates); the VLM-specific bits are the **vision encoder**, the
**processor** that prepares images / videos and interleaves vision tokens
into the text stream, and a chat template that knows where the image goes.

If you want a working VLM example in 5 lines, see [Quickstart](../quickstart.md).
The rest of this section drills into:

| Doc | What it covers |
|---|---|
| [Using a VLM](using.md) | `ChatSession` with images / videos, multi-image conversations, processor customization. |
| [Adding a VLM](adding-a-model.md) | How to port a new architecture into `MLXVLM` — vision encoder, processor, chat template, multimodal token interleave. |

## Where to find related docs

- **LLM (text-only)** — see [llm/overview.md](../llm/overview.md). The
  generate / sampling / KV-cache layer is shared.
- **Architecture map (LLM ↔ VLM ↔ MLXLMCommon)** — see [architecture.md](../architecture.md#llm--vlm-consolidation).
- **Per-model status (which VLMs are working, known gaps)** — see [models.md](../models.md).

## Where the code lives

- `Libraries/MLXVLM/Models/` — one Swift file per architecture (PaliGemma,
  Qwen 2 VL, Qwen 2.5 VL, Qwen 3 VL, Qwen 3.5 VL, FastVLM, LFM 2 VL, Mistral 3 VL,
  Gemma 3 VL, Gemma 4 VL, GLM-OCR, SmolVLM, SmolVLM 2). Each file defines the
  outer `*Model` class conforming to `VLMModel`, the vision encoder, the
  processor, and the chat template.
- `Libraries/MLXVLM/VLMModelFactory.swift` — the `VLMModelFactory.shared`
  singleton and the `VLMRegistry` of curated checkpoints + their default
  sampling parameters.
- `Libraries/MLXLMCommon/Models/<Family>.swift` — bit-identical text-decoder
  layers shared with the LLM port (Gemma 3 / Gemma 4 / LFM 2 / Mistral 3 /
  Qwen 2 / 3 / 3.5 / GLM 4 namespaces). See [architecture.md](../architecture.md#llm--vlm-consolidation).

## Currently in tree

| Family | Models | Notes |
|---|---|---|
| PaliGemma | 3B mix-448 | 8-bit only on `mlx-community` |
| Qwen 2 VL / 2.5 VL / 3 VL | 2B / 3B / 4B | Multi-image and video support |
| Qwen 3.5 VL | 27B / 35B-A3B | Hybrid GDN + Attention; needs prefill-sync barrier (fixed in [#181](https://github.com/ekryski/mlx-swift-lm/issues/181)) |
| FastVLM | 0.5B | Apple's Qwen 2 backbone; bf16 only |
| LFM 2 VL | 1.6B | LiquidAI; Pixtral-style processor |
| Mistral 3 VL | 24B (Mistral Small 3.1) | Pixtral-style; flat / nested config bridging |
| Gemma 3 VL | 4B / 12B / 27B (qat-4bit) | Text-decoder shared with `MLXLLM.Gemma3Model` |
| Gemma 4 VL | E2B / E4B / 26B-A4B / 31B | Text-decoder shared with `MLXLLM.Gemma4Model` |
| GLM-OCR | (GLM 4 9B-class) | Vision encoder + GLM 4 text decoder |
| SmolVLM / SmolVLM 2 | 500M (video) / 256M | Idefics3-style |

For the full VLMRegistry list and every checkpoint's quantization availability,
see [models.md](../models.md).

## Pattern: prefill-sync barrier (issue #169)

VLMs share a subtle bug class. Their `prepare(...)` runs the vision encoder
plus the text-decoder prefill in a single pass, then returns `.logits(...)`.
Without an explicit eval barrier the prefill K/V writes can stay pending in
the GPU command buffer, so the iterator's first decode forward reads stale
cache and produces a token-loop ("ThisThis", `<pad>` flood, `"The!!!!!"`).

The standard fix — applied to every consolidated VLM during the WS-C sprint:

```swift
var cacheArrays: [MLXArray] = []
for c in cache {
    cacheArrays.append(contentsOf: c.innerState())
}
eval(cacheArrays + [output.logits])

return .logits(output)
```

For hybrid GDN + Attention models (Qwen 3.5 VL), `SSMStateCache: ArraysCache`
already exposes its conv + recurrent tensors via `innerState()`, so the same
barrier covers the SSM state without infrastructure changes.

When porting a new VLM, apply this barrier at the end of `prepare(...)` —
it's cheap (one sync at the end of prefill, not per-token) and prevents the
class of bugs filed under #169 / #181.
