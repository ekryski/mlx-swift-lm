# Batched Decoding

Batched decoding runs **N concurrent decode streams** through one model
on one set of weights, sharing GPU dispatch overhead across requests.
The classic single-stream loop pays one matmul per layer per token; with
B requests the same matmul runs against a `[B, T, D]` activation tensor
and amortizes that overhead across all B tokens-per-step.

## When it's useful

- **Multi-tenant serving.** When several users / agents are talking to
  the same model in parallel, batching lets them share inference
  capacity instead of serializing. On Apple Silicon the win is largest
  on **MoE models** (3B–4B active parameters out of 30B+ total) — the
  active matmul leaves plenty of GeMM headroom for batching to fill.
- **Speculative decoding.** Draft + main model evaluation; the main
  model's per-step verifier batches across draft tokens.
- **N-best sampling.** Generating diverse candidates from a single
  prompt for re-ranking / RAG / agent-loop exploration.

A 416-run batched sweep on M1 Max 64 GB, captured in
[`benchmarks/batched-sweep-2026-04-29.md`](../benchmarks/batched-sweep-2026-04-29.md):

| Model | B=1 tok/s | B=2 agg tok/s | Speedup |
|---|---:|---:|---:|
| `gemma4-26b-a4b` (active 4B) | 28.0 | 39.2 | **1.40×** |
| `qwen35-35b-a3b` (active 3B) | 64.6 | 85.8 | **1.32×** |
| `nemotron-30b-a3b` (active 3B) | 75.4 | 80.5 | 1.07× |

`gemma4-26b-a4b` is the standout — it holds **1.30× at ctx=8k**, the only
model in the registry to do so. Dense models at the same parameter count
peak around 1.05–1.20× and lose ground above ctx=4k, where the larger
activation tensor starts to compete with the working set.

## Quick example

`generateBatched(...)` is the entry point. v1 requires equal-length
prompts:

```swift
import MLXLLM
import MLXLMCommon

let container = try await loadModelContainer(
    configuration: LLMRegistry.gemma4_e2b_it_4bit
)

try await container.perform { context in
    // All inputs must be 1-D and the same length in v1.
    let prompts = [
        "Write a haiku about the ocean.",
        "Write a haiku about a city at night.",
        "Write a haiku about morning coffee.",
        "Write a haiku about a cat.",
    ]
    let inputs = try prompts.map { prompt in
        try context.processor.prepare(input: UserInput(prompt: prompt))
    }

    let stream = try generateBatched(
        inputs: inputs,
        parameters: GenerateParameters(maxTokens: 32),
        context: context
    )

    for await event in stream {
        switch event {
        case .chunks(let perStream):
            for (i, text) in perStream.enumerated() {
                if !text.isEmpty { print("[stream \(i)] \(text)") }
            }
        case .info(let info):
            print("done — agg tok/s = \(info.aggregateTokensPerSecond)")
        }
    }
}
```

For the full reference see
[`Libraries/MLXLMCommon/EvaluateBatched.swift`](../Libraries/MLXLMCommon/EvaluateBatched.swift)
and the `BatchedKVCache` / `BatchedHybridCache` implementations.

### Equal-length prompt requirement (v1)

The current `generateBatched(...)` runs one shared `BatchedKVCache` for
all streams; per-stream offset and per-stream early-EOS are tracked, but
**prompts must be 1-D and the same length** at prompt time. If your
prompts differ in length, pad them out to the longest with the
tokenizer's pad token before calling `generateBatched`.

## What's coming — continuous batching

The current shape is **synchronous**: you collect a batch of prompts up
front, run one prefill, then drive decode steps until every stream
finishes. New requests arriving mid-decode wait for the current batch.

Continuous batching — admitting new requests into an in-flight batch as
existing streams finish (vLLM-style "iteration-level scheduling") — is
the natural next step. It needs:

- **Variable-length prompt prefill** within one batch (paged or
  zero-padded with explicit per-stream lengths).
- **Per-stream slot lifecycle** — a `BatchedKVCache` that supports
  `addSlot` / `removeSlot` mid-decode without invalidating other
  streams. The `BatchedHybridLLM` protocol in
  `Libraries/MLXLMCommon/BatchedHybridCache.swift` already requires
  this for hybrid GDN+Attention models (Qwen 3.5 / Nemotron-H), so the
  cache surface is partly there.
- **Admission control** that hands the request scheduler a target batch
  size based on the wired-memory budget — see
  [memory-management.md](memory-management.md).

There's no single tracking spec yet — work will land alongside
[spec 020 phase 2](../specs/020-tape-replay-rollback-generalised.md)
(`SSMStateCache: TapeReplayCache`) and a new continuous-batching spec
that hasn't been written. The current `BatchedKVCache` lifecycle methods
(`copy`, `removeSlot`, `addSlot`) are already factored to support it.

## Picking a batch size

Memory scales linearly with B for the KV cache and roughly linearly for
working buffers. Use [`WiredMemoryUtils`](memory-management.md) with
`batchSize: B` to size your wired-memory ticket; the smart estimator's
KV term is `kv(maxTokens × batchSize, compressionAlgorithm)`, so you'll
see honest per-B memory predictions before you launch.

For B>1 you also need to pick a KV-cache compression algorithm carefully:

- **B=1**: `--kv turbo4v2` matches no-quant within ~5 % on every model.
- **B>1 long context**: TurboQuant decode regresses to ~0.60× on Qwen 9B
  at ctx=32k B=2. Use `--kv none` for B>1 long-context, or `--kv affine4`
  if memory matters more than throughput. The B>1 regression is filed in
  the follow-up issue list.

## See also

- [KV cache + compression](kv-cache.md) — `BatchedKVCache` and
  `BatchedHybridCache` reference.
- [Memory management](memory-management.md) — wired-memory tickets that
  size correctly for batched workloads.
- [Speculative decoding](speculative-decoding.md) — where the batched
  decode primitives also get used.
- [`benchmarks/batched-sweep-2026-04-29.md`](../benchmarks/batched-sweep-2026-04-29.md)
  — the full 416-run table this page summarises.
