# MLX Swift LM

A Swift package for running large language models (LLMs), vision-language
models (VLMs), and embedding models on Apple Silicon, built on
[MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key capabilities:

- ~50 LLM and ~20 VLM reference architectures, plus ~3 embedder families. [See full list](documentation/models.md).
- High-level `ChatSession` API for chat-shaped use cases (text + images +
  video, multi-turn, streaming)
- Lower-level `ModelFactory` + `generate(...)` API for batched decode,
  speculative decoding, custom sampling, and direct cache management
- Multiple KV-cache compression algorithms (Affine, TurboQuant — symmetric
  and asymmetric K/V) plus a typed `compressionAlgorithm` parameter
- Wired-memory coordination so multiple inference tasks can share one GPU
  budget without stepping on each other
- Tool-call parsers for the major chat-template families (Qwen, Llama 3,
  Pythonic, Harmony, Hermes)

For example apps and tools that consume this package, see [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Everything lives under [`documentation/`](documentation/).

Start here:

- **[Installation](documentation/installation.md)** — SwiftPM / Xcode setup,
  picking integration packages
- **[Quick start](documentation/quickstart.md)** — generate text in 5 lines
  (LLM and VLM)
- **[Architecture](documentation/architecture.md)** — module layout and the
  LLM ↔ VLM consolidation map
- **[Models](documentation/models.md)** — supported architectures, registries,
  per-model known gaps

Then drill in:

| LLM | VLM | Embeddings |
|---|---|---|
| [Overview](documentation/llm/overview.md) | [Overview](documentation/vlm/overview.md) | [Overview](documentation/embeddings/overview.md) |
| [Using an LLM](documentation/llm/using.md) | [Using a VLM](documentation/vlm/using.md) | |
| [Evaluation](documentation/llm/evaluation.md) | | |
| [Adding an LLM](documentation/llm/adding-a-model.md) | [Adding a VLM](documentation/vlm/adding-a-model.md) | |

Cross-cutting topics:

- [`GenerateParameters` reference](documentation/generate-parameters.md)
- [KV cache + compression](documentation/kv-cache.md)
- [Memory management](documentation/memory-management.md)
- [Batched decoding](documentation/batched-decoding.md)
- [Speculative decoding](documentation/speculative-decoding.md)
- [Migrating to v3](documentation/migrations/v2-to-v3.md) /
  [Migrating to v4](documentation/migrations/v3-to-v4.md)
- [Publishing a release](documentation/publishing-a-release.md)

For local development:

- [Developing in mlx-swift-lm](documentation/developing/developing.md)
- [Porting models from Python](documentation/developing/porting.md)
- [Testing](documentation/developing/testing.md)
- [Benchmarking](documentation/developing/benchmarking.md)

## Quick start (5 lines)

```swift
import MLXLLM
import MLXLMCommon

let model = try await loadModelContainer(
    configuration: LLMRegistry.gemma4_e2b_it_4bit
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

For a VLM, replace `LLMRegistry.gemma4_e2b_it_4bit` with
`VLMRegistry.qwen2_5VL3BInstruct4Bit` and pass `image:` into `respond`.
Full walkthrough in [`documentation/quickstart.md`](documentation/quickstart.md).

## Installation

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.32.0-alpha")),
```

You also need a downloader and a tokenizer. Pick one of the three integration
paths — see [`documentation/llm/using.md § Picking an integration`](documentation/llm/using.md#picking-an-integration).

Full setup in [`documentation/installation.md`](documentation/installation.md).

## Upgrading

- **From 3.x → 4.x** — KV-cache architecture rewrite under spec 006: class
  renames, typed `KVCache.CompressionAlgorithm`, `maybeQuantizeKVCache`
  removed in favour of `makeAttentionCache(...)`. See
  [`documentation/migrations/v3-to-v4.md`](documentation/migrations/v3-to-v4.md).
- **From 2.x → 3.x** — decoupled tokenizer + downloader, new imports,
  loading API changes. See
  [`documentation/migrations/v2-to-v3.md`](documentation/migrations/v2-to-v3.md).

## Customizing Model Parameters

Inference behaviour is controlled by `GenerateParameters`. Pass it to
`ChatSession`, the lower-level `generate(...)` family, or the batched
decode entry points.

```swift
let parameters = GenerateParameters(
    temperature: 0.8,                                      // 0 = greedy
    topP: 0.95,
    maxTokens: 512,
    repetitionPenalty: 1.05,
    prefillStepSize: 2048,                                 // chunk long prompts
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2) // KV compression
)

let session = ChatSession(model, generateParameters: parameters)
```

Full field reference (every sampling knob, prefill chunk-size, thinking
mode, perplexity / KLD tracking, env-var overrides) in
[`documentation/generate-parameters.md`](documentation/generate-parameters.md).

## KV Cache Configuration

The KV cache stores per-layer K and V tensors so subsequent decode steps
don't re-compute attention over the entire prefix. `mlx-swift-lm` ships
several implementations — raw fp16 (default), affine-quantized (4 / 6 /
8 bit), and TurboQuant (asymmetric K/V bits) — selected via the typed
`compressionAlgorithm` knob.

```swift
// 4-bit keys, 2-bit values via TurboQuant ("turbo4v2"):
let parameters = GenerateParameters(
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2),
    maxKVSize: 4096   // optional: enable sliding-window eviction
)

let cache: [KVCache] = model.newCache(parameters: parameters)
let stream = try generate(
    input: lmInput,
    cache: cache,
    parameters: parameters,
    context: context
)
```

Full algorithm matrix, what's coming (`SSMStateCache: TapeReplayCache`,
KV write fusion, adaptive per-layer mixed precision), constructor
toggles, and the TurboQuant env-var set are in
[`documentation/kv-cache.md`](documentation/kv-cache.md).

## Memory Management

`mlx-swift-lm` runs on Apple Silicon's **unified memory**. By default
the OS reserves ~75 % of physical memory for the GPU's "wired" budget
(~48 GB on a 64 GB Mac). The library ships a **smart memory estimator**
that's on by default — it sizes a wired ticket from the loaded model
(`weights + kv(maxTokens × batchSize, compressionAlgorithm) + workspace`)
and hands it to Metal, so single-process inference works correctly out
of the box.

When you have multiple concurrent inference tasks (multi-tenant
serving, agent loops, parallel evaluation), explicit wired-memory
**tickets** let them coordinate one shared budget without stepping on
each other:

```swift
let policy = WiredSumPolicy(cap: 12 * 1024 * 1024 * 1024)  // 12 GB cap
let ticket = policy.ticket(size: estimatedBytes, kind: .active)

let stream = try generate(
    input: lmInput,
    parameters: parameters,
    context: context,
    wiredMemoryTicket: ticket
)
```

Active vs reservation tickets, the four built-in policies
(`WiredSumPolicy` / `WiredMaxPolicy` / `WiredFixedPolicy` /
`WiredBudgetPolicy`), measurement-driven budgeting, and the
`MLX_MEMORY_LIMIT` / `MLX_SMART_MEMORY` env-var overrides are in
[`documentation/memory-management.md`](documentation/memory-management.md).

## Batch Decoding

Batched decoding runs N concurrent decode streams through one model on
one set of weights, sharing GPU dispatch overhead across requests.
Useful for **multi-tenant serving** (several users sharing a model),
**speculative decoding** (draft + main verifier), and **N-best
sampling**. MoE models with active-parameter sparsity benefit most —
`gemma4-26b-a4b` holds a **1.30× speedup at ctx=8k**.

`generateBatched(...)` is the entry point. v1 requires equal-length
prompts:

```swift
let inputs = try prompts.map { prompt in
    try context.processor.prepare(input: UserInput(prompt: prompt))
}

let stream = try generateBatched(
    inputs: inputs,
    parameters: GenerateParameters(maxTokens: 32),
    context: context
)
```

**Continuous batching** (admitting new requests into an in-flight batch
as existing streams finish — the vLLM-style "iteration-level
scheduling") is the next direction; the cache slot-lifecycle primitives
already needed for hybrid GDN+Attention models in
`Libraries/MLXLMCommon/BatchedHybridCache.swift` will land first.

Full sweep numbers, picking a batch size against your wired budget, and
the v1 equal-length-prompt requirement details are in
[`documentation/batched-decoding.md`](documentation/batched-decoding.md).

## Building locally

After cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

For the full build pipeline reference (why `make` instead of `swift
build`, incremental rebuilds, `make doctor`, dep-chain diagnostics) see
[`documentation/architecture.md § Build pipeline`](documentation/architecture.md#build-pipeline)
and [`documentation/developing/developing.md`](documentation/developing/developing.md).
