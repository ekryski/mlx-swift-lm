# Documentation

Table of contents for the `mlx-swift-lm` documentation. The top-level
[`README`](../README.md) is the curated landing page; this index lists every
page in the tree so you can jump straight to a topic.

## Getting started

- [Installation](installation.md) — SwiftPM / Xcode setup, picking integration
  packages.
- [Quick start](quickstart.md) — generate text in 5 lines (LLM and VLM).
- [Architecture](architecture.md) — module layout (`MLXLMCommon` /
  `MLXLLM` / `MLXVLM` / `MLXEmbedders` / `MLXHuggingFace`) and the LLM ↔ VLM
  consolidation map.
- [Models](models.md) — supported architectures, registries, per-model known
  gaps.

## LLM

- [Overview](llm/overview.md)
- [Using an LLM](llm/using.md) — `ChatSession` + the lower-level
  `ModelFactory` / `generate(...)` flow.
- [Evaluation](llm/evaluation.md) — sampling, streaming, multi-turn,
  customising a session.
- [Adding an LLM](llm/adding-a-model.md) — porting a new architecture.

## VLM

- [Overview](vlm/overview.md)
- [Using a VLM](vlm/using.md) — `ChatSession` with images / video,
  multi-image, processor customisation.
- [Adding a VLM](vlm/adding-a-model.md) — porting (vision encoder +
  processor + chat template + the issue-#169 prefill-sync barrier).

## Embeddings

- [Overview](embeddings/overview.md) — encoder / embedding models, pooling,
  batch usage.

## Cross-cutting topics

- [`GenerateParameters` reference](generate-parameters.md) — every sampling
  knob, prefill chunk-size, thinking-mode option, env-var override.
- [KV cache + compression](kv-cache.md) — algorithm matrix
  (Standard / Affine / TurboQuant / SSMStateCache / Batched), what's coming,
  constructor toggles, `TURBO_*` env vars.
- [Memory management](memory-management.md) — Apple Silicon unified memory,
  the smart-memory estimator, wired-memory tickets, policies, weight
  reservations, `MLX_MEMORY_LIMIT` / `MLX_SMART_MEMORY`.
- [Batched decoding](batched-decoding.md) — `generateBatched(...)`,
  multi-tenant serving, batch-size sizing, "what's coming — continuous
  batching".
- [Speculative decoding](speculative-decoding.md) — n-gram prompt-lookup +
  draft-model coordination.

## Migrations

- [v2 → v3](migrations/v2-to-v3.md) — decoupled tokenizer + downloader, new
  imports, loading API changes.
- [v3 → v4](migrations/v3-to-v4.md) — KV-cache rewrite under spec 006: class
  renames, typed `KVCache.CompressionAlgorithm`, `maybeQuantizeKVCache`
  removed in favour of `makeAttentionCache(...)`.

## Releases

- [Publishing a release](publishing-a-release.md) — manual-trigger pipeline,
  workflow inputs, semver guidance, hotfix branching, cross-repo
  coordination across `mlx-c → mlx → mlx-swift → mlx-swift-lm`.

## Local development

- [Developing in mlx-swift-lm](developing/developing.md) — what's where,
  how to develop locally, the `make` workflow.
- [Porting models from Python](developing/porting.md) — the long-form
  Python → Swift mapping, dtype quirks, KV-cache adapters.
- [Testing](developing/testing.md) — running tests, filter patterns, test-
  writing best practices.
- [Benchmarking](developing/benchmarking.md) — `./scripts/benchmark.sh`
  reference + the canonical methodology pointer.

## See also

- Top-level [`README`](../README.md) — the project landing page.
- [`specs/IMPLEMENTATION-PLAN.md`](../specs/IMPLEMENTATION-PLAN.md) — open
  work by tier (KV cache write fusion, ANE LM head, chunkwise GDN, batched
  QKV fusion, RMSNorm + GEMV fusion, etc.).
- [`benchmarks/README.md`](../benchmarks/README.md) — full bench-harness CLI
  reference, methodology, env-var API, report format.
