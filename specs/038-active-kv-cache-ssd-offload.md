# 038 — Active KV cache SSD offload

**Status:** spec, design — depends on multiple unshipped prerequisites; do not start until #127/#128/#129 and spec 036 land.
**Branch:** TBD (new branch off `alpha` when prerequisites land)
**Depends on:** [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) (Metal paged-attention kernel) + [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) (`PagedKVCache` wired into model factories) + [#129](https://github.com/ekryski/mlx-swift-lm/issues/129) (TurboQuant + paged integration) + [spec 036](036-duoattention-retrieval-streaming-head-split.md) (DuoAttention page-temperature signal).

## The problem this solves

Spec 017's prefix KV cache (shipped 2026-05-12) covers **cross-request** caching: take a snapshot at request end, hydrate at the next request's start. That helps multi-turn chat TTFT by 2-11×.

This spec is a different idea: **inside a single in-flight generation**, swap older KV pages from GPU memory to SSD when they're not about to be read, then page them back in before the attention forward needs them. Use case: long-context decode where the full KV cache doesn't fit in unified memory.

Concretely: a 128K-context Qwen 3.5-9B at fp16 KV needs ~28 GB of cache. On a 32 GB Mac that leaves no room for weights + activations + dequant transients. With SSD offload you can run the same context on 16-24 GB hardware at the cost of some prefill / decode latency.

This is **not** the same as spec 017's L2 disk. L2 is read-once at request start to skip prefill; this spec is read-many during steady-state decode, with hard latency budgets on every step.

## Prior art

- **vLLM `swap_out` / `swap_in`** ([PR #1212](https://github.com/vllm-project/vllm/pull/1212), generalised in [#7831](https://github.com/vllm-project/vllm/pull/7831)) — CPU-pinned host memory pool, asynchronous H2D / D2H via CUDA streams. Used by the scheduler when there's GPU-memory pressure across concurrent requests; not used for single-request offload.
- **InfiniGen** ([OSDI '24](https://www.usenix.org/conference/osdi24/presentation/lee)) — dynamically prefetches KV pages predicted to be needed at the next layer's attention. Predictor is a small linear model trained per architecture. Cuts the I/O critical path by overlapping with previous-layer compute.
- **RaaS / FlexGen** — same family. RaaS overlaps SSD I/O with GPU compute; FlexGen is more aggressive (offloads to disk for memory-pressured inference on consumer GPUs).
- **CXL.mem prototypes** (academic) — DRAM-as-tier-3 via CXL. Not relevant on Apple Silicon — Apple's memory architecture doesn't expose CXL or fabric-attached memory.

The InfiniGen-style "predicted prefetch" pattern is the load-bearing technique. Naive swap-on-eviction (vLLM's pattern, designed for batched serving) doesn't help single-request decode because every step reads every layer's full cache. Without DuoAttention-style head classification or Quest-style top-k selection, **every page must be in GPU memory at every step**, so there's nothing safe to offload.

That's why spec 036 (DuoAttention) is a hard prerequisite: it tags each attention head as **retrieval** (full KV needed every step) or **streaming** (sink + recent window only). Streaming-head pages older than the window are never read again until they fall out of the window — they can be parked on SSD for as long as the request runs. Retrieval-head pages stay in GPU memory unless spec 034 (Quest) further reduces them to a top-k subset, in which case the non-selected pages also become offloadable.

## Design

### 1. Page lifetime classifier

A `KVPageTier` enum tags each cache page by its expected read pattern:

```swift
public enum KVPageTier: Sendable, Equatable {
    /// Retrieval-head full-cache page. Stays GPU-resident.
    case hot

    /// Streaming-head page outside the recent window, or Quest-rejected
    /// retrieval-head page. Safe to park on SSD; readers must page in
    /// before access.
    case warm

    /// Pre-fault candidate. The InfiniGen predictor flagged this page
    /// as "needed within the next k steps" — already paging in.
    case prefaulting
}
```

`PagedKVCache` (from #128) exposes per-page tier as a `[KVPageTier]` accessor. The model's attention forward reads this on each step to decide whether to issue a page-in request before invoking `MLXFast.scaledDotProductAttention`.

### 2. Offload thread

A dedicated `OffloadCoordinator` actor manages the SSD-resident page pool:

```swift
public actor OffloadCoordinator {
    /// Read N pages from the SSD-resident pool into a GPU buffer.
    /// Returns a Task; await the task before reading from the GPU buffer.
    public func pageIn(_ pageIDs: [PageID]) -> Task<MLXArray, Error>

    /// Schedule N pages to be written to SSD when GPU compute on them
    /// completes. Non-blocking; eviction is best-effort.
    public func pageOut(_ pageIDs: [PageID])

    /// Number of pages currently SSD-resident.
    public var residentCount: Int { get }
}
```

The implementation has two performance-critical paths:

- **Page-in**: read from SSD into a pinned page buffer, then issue a Metal blit into the cache's GPU buffer. M-series SSD reads at 5-7 GB/s sequential, and the Metal blit adds ~10 µs per page. For a typical 4 KB page that's <1 µs of read + 10 µs of blit — well under a single decode step (~30 ms on a 9B model).
- **Page-out**: GPU writes can stay in-place until the eviction policy decides to spill. Write to SSD happens via a background `DispatchQueue` (sync write, since the Mac unified memory model means we don't need explicit D2H copy — the GPU buffer is already in unified memory, so the write to disk is just a memcpy from unified memory into a `Data` blob).

### 3. Prefetch predictor (InfiniGen-style)

The naive policy is **fault on read**: page in only when the attention forward asks for a page that's currently SSD-resident. That's slow — every step stalls on SSD I/O.

The smart policy is to **predict the next k steps' page reads** and pre-page-in before the attention forward starts. InfiniGen trains a tiny linear model per architecture; we can copy the same pattern. The predictor input is the previous layer's attention output (or the current step's hidden state); the output is a sigmoid score per page. Pages with score > threshold get pre-paged at the start of the layer.

Predictor scope:
- **Phase 1**: heuristic — pre-fault all retrieval-head pages that DuoAttention's calibration flagged as load-bearing on > 80% of NIAH probes. No training; ship a per-model JSON sidecar.
- **Phase 2**: per-model trained predictor. Calibrate on the same NIAH set DuoAttention uses; predict from previous layer's `q @ k_compressed` (cheap signal that captures "which pages will be high-attention this step").

### 4. Latency budgets

For this spec to be a win, we need:

- **Decode step latency** must not increase by more than ~15% vs. all-in-memory baseline. On a 9B model at 50 tok/s that's 6 ms of slack per step. With M-series SSD at 5 GB/s and 4 KB pages, that buys ~7,500 pages per step — enough for an arbitrary cache size at typical retrieval-head selectivity.
- **Prefill latency** is allowed to grow more — long-context prefill is already expensive (multi-second on 128K). 2× prefill latency for 50% memory reduction is acceptable; users explicitly opting into offload accept this.
- **Page-in critical path** must be < 1 step's compute time. If we spend > 30 ms of SSD I/O serially per step, we've lost. The predictor must achieve ~99% pre-fault hit rate, with the < 1% fault path falling back to a sync read.

### 5. Eviction policy

Pages are evicted when:
1. They fall out of a streaming head's window (eviction is in fact mandatory — the streaming head will never read them again until they re-enter the window, which it can't).
2. Quest (spec 034) rejects them this step. Quest selects top-k per step; non-selected pages remain in GPU memory but are eviction candidates.
3. GPU memory pressure: when the wired-memory ticket is exceeded, the oldest non-hot pages are spilled.

Storage layout on SSD:
```
~/.cache/mlx-swift-lm/active-kv/<request-id>/
  pages/<page-id>.bin       — raw float16 page data
  manifest.json             — page-id → tier history (debug only)
```

Pages are not shared across requests (each request has its own directory), and the directory is deleted on request completion. This is **not** a persistent store — it's a runtime overflow buffer.

### 6. Configuration knobs

```swift
public struct ActiveKVOffloadConfig: Sendable {
    /// Maximum SSD bytes to use for this request's overflow buffer.
    /// Default 16 GiB.
    public var maxSSDBytes: Int

    /// Pages newer than this are never offloaded (recency window).
    /// Default 256 pages.
    public var recencyWindow: Int

    /// Per-step prefetch budget (number of pages to pre-fault).
    /// Default 32.
    public var prefetchBudget: Int

    /// Disable when the predictor's hit rate falls below this.
    /// Default 0.95.
    public var minPredictorHitRate: Float
}
```

Opt-in only — default off. Env override `MLX_ACTIVE_KV_OFFLOAD=1`.

## Phases

1. **Phase 1 — `OffloadCoordinator` actor + heuristic prefetch.** No predictor; pre-fault all DuoAttention-flagged retrieval pages at the start of each layer. Validates the I/O critical path can sustain decode-step latency on M2 Max NVMe. Bench: Qwen 3.5-9B @ 32K context, 32 GB Mac, measure decode tok/s vs. all-in-memory baseline. **Pass criterion**: ≥85% of baseline tok/s.

2. **Phase 2 — Trained predictor.** Per-model JSON sidecar predicting page reads from previous layer's hidden state. Same NIAH calibration set as DuoAttention. **Pass criterion**: prefetch hit rate ≥95% on NIAH and chat workloads.

3. **Phase 3 — Quest integration.** When spec 034 ships, Quest-rejected retrieval pages become eviction candidates. Composes multiplicatively with phase 1's streaming-head offload. **Pass criterion**: cache-resident fraction drops below 30% of full-cache size at 128K context without measurable accuracy loss on NIAH.

4. **Phase 4 — UI / observability.** `[ACTIVE-KV]` bench-log line per request: pages offloaded, pages faulted, predictor hit rate, mean page-in latency. Same shape as spec 017's `[PREFIX-CACHE]` line.

## Risks

1. **SSD wear**: every long-context request writes the working-set to SSD. A heavy user could write 100 GB/day; consumer NVMe is rated at ~600 TBW for the 1 TB tier, so this is ~16 years of daily heavy use. Document, but don't gate. Apple Silicon SSDs in laptops are non-replaceable; same caveat applies to running a swap-heavy workload generally.

2. **Predictor brittleness**: if the predictor drifts (e.g. new chat template, OOD prompt), pre-fault hit rate craters and every step stalls on sync I/O. `minPredictorHitRate` config auto-disables the offload below threshold and falls back to all-in-memory (potentially OOMing, but loudly).

3. **Power**: SSD I/O burns ~3-5 W on Apple Silicon laptops. On battery, a long offload-enabled session draws noticeably more than an all-in-memory session. Document.

4. **Apple's I/O model**: `URL.read(...)` goes through the filesystem cache, which is a process-shared resource. A second process doing heavy I/O can evict our pages from the FS cache, causing sync-read stalls even when we logically pre-paged. Mitigation: `mmap` the page files and `madvise(MADV_RANDOM)` to discourage FS read-ahead, or use `O_DIRECT` (not available on Apple's APIs in pure Swift; would need `Cmlx` bridging). Phase 1 ships without this and measures the failure mode.

5. **PagedKVCache prerequisite slippage**: #128 ("Wire PagedKVCache into Qwen 3 / Gemma 4 / etc model factories") is required and not started. If that work doesn't land, this spec is blocked at the same point. Track the dependency explicitly in the IMPLEMENTATION-PLAN.

## Files touched (projection)

**mlx-swift-lm**:

| File | What |
|---|---|
| `Libraries/MLXLMCommon/ActiveKVOffload.swift` (new) | `OffloadCoordinator` actor, `KVPageTier` enum, `ActiveKVOffloadConfig`, predictor sidecar loader. |
| `Libraries/MLXLMCommon/PagedKVCache.swift` (extend, after #128 wires it in) | Per-page tier tracking + offload hooks. |
| `Libraries/MLXLMCommon/AttentionUtils.swift` | Pre-fault hook before `MLXFast.scaledDotProductAttention` calls into a paged cache. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Read `ActiveKVOffloadConfig` from `GenerateParameters`, propagate to cache factory. |
| `scripts/active_kv_calibrate.py` (new) | InfiniGen-style predictor calibration on the DuoAttention NIAH set. |
| `Tests/MLXLMTests/ActiveKVOffloadTests.swift` (new) | Predictor hit-rate, page-in latency, eviction policy, OOM-fallback. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | New `--method long-context-offload` mode: pre-fill a 64K-128K prompt, decode 1024 tokens, measure tok/s + offload stats. |

## Out of scope

- **Persistent cross-request SSD cache** — that's spec 017 phase 4 (already shipped, opt-in). This spec is single-request overflow only.
- **Cross-process page sharing** — pages are per-request. Multiple processes running the same model don't share page files. (vLLM solves this with CPU swap pools shared via shm; not applicable here since we don't ship a multi-process server.)
- **Compression on SSD** — pages are written as raw float16. Compressing to int4 + scales doubles capacity but adds CPU work on the write path; phase 4 candidate if SSD bandwidth becomes the bottleneck.
- **CXL.mem / fabric memory** — Apple Silicon doesn't expose this. Spec out of scope.

## References

- vLLM swap_out PR series: [#1212](https://github.com/vllm-project/vllm/pull/1212), [#7831](https://github.com/vllm-project/vllm/pull/7831).
- [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management, OSDI '24](https://www.usenix.org/conference/osdi24/presentation/lee).
- [RaaS: Retrieval-augmented Speculation](https://arxiv.org/abs/2406.07522) (uses the same SSD-overflow pattern as a side primitive).
- [FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU, ICML '23](https://arxiv.org/abs/2303.06865) — the original SSD-tier inference paper. Targets batched serving, not single-request, but the I/O model carries over.
- [DuoAttention](https://arxiv.org/abs/2410.10819) — spec 036, retrieval/streaming head classification, the page-temperature signal this spec depends on.
- [Quest: Query-Aware Sparsity for Long-context Transformers](https://arxiv.org/abs/2406.10774) — spec 034, top-k page selection, additional eviction candidates.
- [Spec 017 — Cross-request prefix KV cache](017-prefix-kv-cache.md) — the cross-request side of "KV cache to SSD". Disjoint from this spec; phase 4 of 017 shipped 2026-05-12.
- [Issue #127 — Metal paged-attention kernel](https://github.com/ekryski/mlx-swift-lm/issues/127).
- [Issue #128 — Wire PagedKVCache into model factories](https://github.com/ekryski/mlx-swift-lm/issues/128).
- [Issue #129 — TurboQuant + paged integration](https://github.com/ekryski/mlx-swift-lm/issues/129).
