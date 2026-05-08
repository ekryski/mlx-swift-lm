# LLM Overview

`MLXLLM` is the text-decoder family of models. It ships ~50 reference architectures
(Llama 3.1 / 3.2, Qwen 2 / 2.5 / 3 / 3.5 / 3.6, Gemma 2 / 3 / 3n / 4, GLM 4, GPT-OSS,
DeepSeek V3 / V4, Mistral, Phi 3.5, Olmo 2 / 3, Nemotron Cascade 2, LFM 2, GraniteMoE,
SmolLM 2 / 3, BitNet, Qwen3-Next, …) and a small public surface for instantiating
them, sampling from them, and feeding them through the cross-cutting infrastructure
in [`MLXLMCommon`](../../Libraries/MLXLMCommon/README.md) (KV caches, wired memory,
chat templates, speculative decoding).

If you want a working example in 5 lines, see [Quick start](../quickstart.md).
The rest of this section drills into:

| Doc | What it covers |
|---|---|
| [Using a model](using.md) | The `ChatSession` high-level API and the lower-level `ModelFactory` / `generate(...)` flow when you need direct control. |
| [Adding a model](adding-a-model.md) | How to port a new architecture into `MLXLLM` — config decoding, layer wiring, weight sanitization. |
| [Evaluation](evaluation.md) | The `ChatSession` workflow plus the underlying `generate(...)` primitives. |

## Where to find related docs

- **VLM (text + vision)** — see the [VLM overview](../vlm/overview.md).
- **Embeddings (encoders)** — see the [embeddings overview](../embeddings/overview.md).
- **KV cache + compression** — see [kv-cache.md](../kv-cache.md).
- **Wired memory coordination** — see [wired-memory.md](../wired-memory.md).
- **Speculative decoding** — see [speculative-decoding.md](../speculative-decoding.md).
- **Architecture map (LLM ↔ VLM ↔ MLXLMCommon)** — see [architecture.md](../architecture.md).
- **Per-model status (known gaps + supported quantizations)** — see [models.md](../models.md).

## Where the code lives

- `Libraries/MLXLLM/Models/` — one Swift file per architecture; each defines a
  `*Model` outer class conforming to `LLMModel` plus its inner attention / MLP /
  decoder-layer types.
- `Libraries/MLXLLM/LLMModelFactory.swift` — the `LLMModelFactory.shared` singleton
  and the curated `LLMRegistry` of known checkpoint IDs + their default sampling
  parameters.
- `Libraries/MLXLMCommon/Models/` — bit-identical building blocks shared with the
  VLM stack (Gemma, Gemma 3, Gemma 4, GLM 4, LFM 2, Mistral 3, Qwen 2 / 3 / 3.5
  namespaces). See [architecture.md](../architecture.md) for why some classes live
  here vs per-target.
