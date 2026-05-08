# Architecture

`mlx-swift-lm` is split into four Swift modules. Each lives under `Libraries/`
and ships as a separate SwiftPM product.

```
Libraries/
├── MLXLMCommon/        ← cross-cutting infrastructure + shared layers
├── MLXLLM/             ← text-decoder models + LLM factory
├── MLXVLM/             ← vision-language models + VLM factory
├── MLXEmbedders/       ← encoder / embedding models
└── MLXHuggingFace/     ← optional HuggingFace integration macros
```

## What goes where

### `MLXLMCommon` — the foundation

Anything used by more than one of (LLM, VLM, Embedders) lives here. The big
buckets:

- **KV-cache implementations** (`KVCache.swift`, `KVCacheTypes.swift`) —
  `StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`,
  `SSMStateCache`, `BatchedKVCache`, plus the `makeAttentionCache(...)`
  factory and `KVCache.CompressionAlgorithm` parser. See [kv-cache.md](kv-cache.md).
- **Generation primitives** — `generate(...)`, `TokenIterator`,
  `BatchedTokenIterator`, `NaiveStreamingDetokenizer`, `GenerateParameters`,
  `LogitProcessor`.
- **Wired-memory coordination** — `WiredMemoryManager`, `WiredMemoryTicket`,
  policies (`WiredSumPolicy`, `WiredFixedPolicy`). See [wired-memory.md](wired-memory.md).
- **Speculative decoding** — n-gram prompt-lookup, draft-model coordination.
  See [speculative-decoding.md](speculative-decoding.md).
- **Model loading + downloader / tokenizer protocols** — `ModelFactory`,
  `Downloader`, `Tokenizer`, `TokenizerLoader`, `loadModel(...)`,
  `ChatSession`.
- **Shared layer building blocks** (`MLXLMCommon/Models/<Family>.swift`) —
  bit-identical layers used by both the LLM and VLM ports of a family. See
  [LLM ↔ VLM consolidation](#llm--vlm-consolidation) below.

### `MLXLLM` — text decoders

One Swift file per architecture under `Libraries/MLXLLM/Models/`, each
defining a `*Model` outer class conforming to `LLMModel` plus its inner
attention / MLP / decoder-layer types. Plus:

- `LLMModelFactory.swift` — the `LLMModelFactory.shared` singleton and the
  `LLMRegistry` of pre-baked checkpoint IDs and default sampling params.
- `Documentation` — see [llm/overview.md](llm/overview.md).

### `MLXVLM` — vision-language models

Same shape as `MLXLLM` but each model also defines a vision encoder, a
processor (image preprocessing + multimodal token interleaving), and a
chat template that knows how to lay out image / video tokens.

- `VLMModelFactory.swift` — analogous singleton + `VLMRegistry`.
- See [vlm/overview.md](vlm/overview.md).

### `MLXEmbedders` — encoders

Small set of encoder models (BERT-style, NomicBERT, ModernBERT) plus a
`HuggingFaceWrappedTokenizer`. See [embeddings/overview.md](embeddings/overview.md).

### `MLXHuggingFace` — integration macros

Optional. Provides `#huggingFaceLoadModelContainer`, `#hubDownloader`,
`#huggingFaceTokenizerLoader` macros that resolve to the official
`huggingface/swift-huggingface` + `huggingface/swift-transformers` stack.
See the [_Picking an integration_ section in llm/using.md](llm/using.md#picking-an-integration)
for alternatives.

## LLM ↔ VLM consolidation

Many model families have both LLM and VLM variants — e.g. Qwen 3 (text-only)
and Qwen 3 VL (text + vision). They share substantial code: the same MLP, the
same RMSNorm, often the same attention pattern. Where the layers are
**bit-identical** between targets, they live in `Libraries/MLXLMCommon/Models/<Family>.swift`
and both targets import them via `typealias`.

Per-family namespace files in tree (issue #168 sprint, PRs #172–#180):

| Family | Common namespace | What's shared | What stays per-target |
|---|---|---|---|
| Gemma | `MLXLMCommon.Gemma` | `Gemma.RMSNorm` (the `1+weight` shifted norm), shared sanitize helpers | Per-architecture decoder stacks |
| Gemma 3 | `MLXLMCommon.Gemma3` | `Configuration`, `Attention`, `MLP`, `TransformerBlock`, `Backbone` | Outer model + VLM vision encoder + chat template |
| Gemma 4 | `MLXLMCommon.Gemma4` | `RMSNormNoScale`, `RMSNormZeroShift` | Attention (alpha-branch fused norm+rope, compiled QKV, conditional MLP-gate compile), `SharedMLP` (compiled gate+down), `TransformerBlock`, `Backbone` |
| LFM 2 | `MLXLMCommon.LFM2` | `Configuration`, `Attention`, `ShortConv`, `MLP`, `DecoderLayer`, `ModelInner` | Outer model + VLM vision encoder + processor |
| Mistral 3 | `MLXLMCommon.Mistral3` | `LayerArgs`, `Attention` (with optional Llama-4 attn-scaling), `MLP`, `TransformerBlock`, `ModelInner` | VLM vision encoder + Pixtral processor (with flat-vs-nested config bridging) |
| Qwen 2 | `MLXLMCommon.Qwen2` | `LayerArgs`, `Attention`, `MLP`, `DecoderLayer`, `ModelInner` | LLM outer model + 3 VLM consumers (Qwen 2 VL / Qwen 2.5 VL / FastVLM) |
| Qwen 3 | `MLXLMCommon.Qwen3` | `LayerArgs`, `MLP` only — Attention left per-target | LLM `batchedForward` / `fullyBatchedForward` perf paths; VLM M-RoPE attention |
| GLM 4 | `MLXLMCommon.GLM4` | `LayerArgs`, `MLP` (fused `gate_up_proj`) | LLM standard rope vs VLM M-RoPE attention |
| Qwen 3.5 | `MLXLMCommon.Qwen35` | `MLP` (separate `gate_proj` / `up_proj` / `down_proj`) | LLM `Qwen3NextMLP` (fused), GDN fused-decode kernel, `fullyBatchedForward`, hybrid GDN+Attention dispatch |

A few things stay per-target on purpose:

1. **Alpha-branch perf work** — compiled QKV projections, `MLXFast.rmsNormRoPE`
   fused norm+rope kernels, `FusedGateUpSwitchGLU`, conditional
   `compile(shapeless:)` MLP wrappers tuned for specific MoE configs. Forcing
   these through a shared class either regresses the LLM or imposes an
   indirection on the hot path.
2. **VLM M-RoPE family** — Qwen3VL, GlmOcr, and Qwen3.5 VL all consume a
   pre-computed `(cos, sin)` tuple from a multimodal `RotaryEmbedding`. The LLM
   side uses the standard `applyRotaryPosition(rope, ...)` helper. A
   cross-family M-RoPE helper is tracked in [`specs/IMPLEMENTATION-PLAN.md`](../specs/IMPLEMENTATION-PLAN.md);
   when it lands, the VLM Attention classes shrink enough to consolidate.
3. **Hybrid SSM + Attention dispatch** (Qwen 3.5 / Nemotron-H / Jamba) —
   `SSMStateCache` instantiation lives inline in each target's `newCache(...)`.
   Spec 020 phase 2 (`SSMStateCache: TapeReplayCache` + Metal kernel) will
   reshape both factories at once, so consolidation happens then rather than
   pre-emptively here.

The umbrella tracking issue is [#168](https://github.com/ekryski/mlx-swift-lm/issues/168);
each WS-C PR documents what's deferred and why.

## Build pipeline

Swift Package Manager handles the Swift sources. Three things SPM does **not**
handle, which is why this repo wraps it in a Makefile:

- **Metal shaders** — `mlx.metallib` must be compiled separately from `.metal`
  sources and copied into the test bundle.
- **Submodule staleness** — when C/C++ files inside `mlx-swift` → `mlx` →
  `mlx-c` change, SPM's content-signature cache may not detect the edit.
  `make clean-cmlx` forces a rebuild.
- **Stale repo cache** — SPM's bare-repo cache at
  `~/Library/Caches/org.swift.swiftpm/repositories/` survives `swift package
  reset`. `make clean-all` clears the entries for our three forked deps
  (`mlx-swift`, `mlx`, `mlx-c`) without touching unrelated packages.

`make doctor` is a fast offline diagnostic for the dep chain. See
[developing.md](developing/developing.md) for the full local-development flow.

## Where to go next

- [Quickstart](quickstart.md) — minimal working example.
- [Models](models.md) — registry, supported architectures, known gaps per model.
- [Adding an LLM](llm/adding-a-model.md) / [Adding a VLM](vlm/adding-a-model.md) —
  porting a new architecture.
- [`specs/IMPLEMENTATION-PLAN.md`](../specs/IMPLEMENTATION-PLAN.md) — open work
  by tier (KV cache write-fusion, ANE LM head, chunkwise GDN, batched QKV
  fusion, RMSNorm + GEMV fusion, …).
