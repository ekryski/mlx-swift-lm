# 021 — Cross-compute-unit speculative decoding (ANE draft × GPU target)

**Status:** 🚧 Phase 1A scaffold landed ([PR #142](https://github.com/ekryski/mlx-swift-lm/pull/142) — protocol + registry + vocab gate). Phase 1B (real Core ML draft + integration) and Phase 2 (full iterator) **not started**; gated on [spec 025](025-ane-gpu-concurrency-primitives.md) (concurrency primitives + Phase 1 measurement harness).
**Branch:** Phase 1A merged via PR #142; later phases get fresh branches off alpha.
**Depends on:** spec 013 (n-gram path) only as fallback. **Independent** of specs 014/015/016/017/020.

**Reality update (post initial draft):** the [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM) project has already (a) ported Qwen 3.5 hybrid SSM+attention to Core ML, (b) shipped Swift implementations of PLD / MTP / EAGLE-3 / Lookahead / SuffixDecoding / cross-vocab speculative decoding / prefix cache, and (c) implemented `MirrorSpeculativeLoop` — running an EAGLE-3 draft on the GPU concurrently with target verify on the ANE. Apple themselves published [Mirror Speculative Decoding (arXiv:2510.13161, Jan 2026)](https://arxiv.org/abs/2510.13161) which formalises this pattern and reports **2.8–5.8× speedup** over baseline (30% average improvement over EAGLE-3). The "is this feasible" question has been answered upstream; this spec now becomes "how do we integrate it into mlx-swift-lm."

## The premise

Spec decode pays for itself when **draft cost ≪ target verify cost**. On Apple Silicon, single-stream decode is memory-bandwidth bound on the GPU (~30-40 ms/token on a 26B MoE), but the GPU is the same place we run the draft model. The draft model competes for the same memory bus, the same kernel-launch queue, and the same eval barriers.

The Apple Neural Engine (ANE) is a separate compute unit with its own memory pipe (still unified memory but a different access path), its own SRAM, and its own command queue. It's idle 100% of the time during normal LLM inference today.

This spec evaluates putting the **draft model** on the ANE while the **target model** stays on the GPU, running them concurrently per round. The hope: the draft pass overlaps GPU compute and (1) hides its own latency, (2) frees the GPU to focus on verification, (3) drops total power consumption by 50–80% on battery-bound deployments.

## What changed since the original draft of this spec

The first draft of this spec assumed:
- ANE LLM compilation was limited to dense-attention models (per ANEMLL's stated support).
- Qwen 3.5 hybrid SSM+attention couldn't run on ANE without research-scale porting.
- Cross-compute-unit drafting was unproven on consumer Apple Silicon.

All three turn out to be wrong:

1. **Qwen 3.5 hybrid SSM+attention runs on Core ML today** via [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM). The repo's v1.8.0 release ships Qwen 3.5 0.8B at ~48 tok/s and Qwen 3.5 2B at ~27 tok/s on iPhone 17 Pro A19 Pro, "99.9% ANE-resident". The conversion pipeline is open-source.
2. **Cross-compute-unit speculative decoding is published prior art**, not a research bet. Apple ML Research published [Mirror Speculative Decoding (arXiv:2510.13161)](https://arxiv.org/abs/2510.13161) in January 2026, formalising the pattern of running the draft on one accelerator while the target verifies on another. Headline numbers: **2.8–5.8× wall-time speedup**, 30% relative improvement over EAGLE-3 baseline.
3. **A reference Swift implementation exists**: `Sources/CoreMLLLM/MirrorSpeculativeLoop.swift` runs an EAGLE-3-style draft on the A19 Pro GPU's Neural Accelerators while target verify runs on the ANE. The repo also contains Swift implementations of PLD, MTP, Lookahead, SuffixDecoding, cross-vocab speculative decoding, and a prefix cache.

## The four interesting variants

With the upstream prereqs answered, this spec covers four variants — ordered roughly by how big the projected win is:

### Variant A — Mirror Speculative Decoding (the headline)

Apple's published pattern: **target on GPU + draft on ANE, run concurrently with cross-device synchronization at two rendezvous points per step.**

The Mirror-SD trick is the *early-exit channel*. The target doesn't wait for its full forward to complete before triggering the draft. After roughly N/2 layers, the target emits a top-κ token-and-logprob signal to the draft (small payload — `O(B·κ)` per step). The draft runs branch-complete rollouts on its accelerator while the target finishes its remaining layers on its accelerator. Final verification synchronises both back. The result is two pipelines running in parallel, each hiding the other's latency.

Concrete numbers from the paper:
- Qwen3-14B: 3.8× speedup
- Qwen3-32B: 3.78× speedup
- Peak: 5.84× on math reasoning
- Range: 2.8–5.8× across diverse tasks
- vs EAGLE-3 baseline: 30% average improvement

The published benchmarks were on **8× M2 Ultra GPUs paired with 8× NPUs** — server-scale. The *single-stream consumer Apple Silicon* version is what `MirrorSpeculativeLoop.swift` in CoreML-LLM is doing. They explicitly call out the v1 (serial-within-burst) and v2 (cross-burst pipelining) variants and project +15-20% from v1 over pure-ANE EAGLE-3 on iPhone 17 Pro.

For our integration: Variant A means running Mirror Speculative Decoding with target on GPU (via existing MLX) and draft on ANE (via Core ML, leveraging CoreML-LLM's existing Qwen 3.5 conversions). Expected speedup on Qwen 3.5 27B target: ~3-5× over baseline once the early-exit signal is wired up.

### Variant B — Same-family pairing (Qwen 3.5-0.8B → Qwen 3.5-27B)

Your specific pairing instinct, and now buildable today.

The original draft of this spec said GDN-on-ANE didn't exist. **It does** — `mlboydaisuke/qwen3.5-0.8B-CoreML` is publicly available on Hugging Face, achieving 48 tok/s on iPhone 17 Pro. Same tokenizer as Qwen 3.5-27B, distribution-aligned (small-of-the-family drafts for larger members typically hit 70-80% accept).

Variant B is essentially "Variant A without the early-exit signal" — simpler integration (no mid-target-forward IPC), but lower expected speedup. Sequential-within-burst:

```
wall = max(K · t_ANE_decode, t_GPU_verify_K+1) + sync
```

For Qwen 3.5-0.8B on ANE (~21 ms/token, conservative — the 48 tok/s headline likely reflects best-case batching) drafting K=4 for Qwen 3.5-27B on GPU (~30 ms decode, ~50 ms verify K+1=5):

```
ANE 4 drafts:    4 × 21 = 84 ms
GPU verify:      ~50 ms
wall:            max(84, 50) = 84 ms (ANE bottleneck)
emitted at 75%:  4 × 0.75 + 1 = 4 tokens
throughput:      4 / 0.084 = 48 tok/s     vs baseline 33.5 → 1.43×
```

K=2 is cleaner:

```
ANE 2 drafts:    2 × 21 = 42 ms
GPU verify K+1:  ~45 ms
wall:            max(42, 45) = 45 ms
emitted at 75%:  2 × 0.75 + 1 = 2.5 tokens
throughput:      2.5 / 0.045 = 56 tok/s   vs 33.5 → 1.67×
```

So Variant B as a sequential-within-burst loop gets ~1.5× over baseline. Less than Variant A's 3-5× ceiling, but no early-exit IPC required. Good fallback if Variant A's signal latency turns out worse than projected.

### Variant C — DFlash draft on ANE

DFlash's block-diffusion draft (16 tokens in one forward) running on ANE while DFlash's target runs on GPU. The pattern stacks on top of Variant A's parallelism:

- Variant A's win: cross-compute-unit overlap (~2× over single-unit baseline).
- Variant C's win: same overlap, but with K=16 emission per round and ~89% accept rate (DFlash's strength).

Expected combined speedup on Qwen 3.5-27B-4bit: 4-7× over baseline (vs Variant A's 3-5× and DFlash-on-GPU-only's 2.4×).

Buildable when:
1. The DFlash draft architecture (block-diffusion + cross-attention to target hidden states) is converted to Core ML. Non-trivial — the cross-attention input requires Core ML stateful-input plumbing, and block-diffusion's parallel-denoising loop must be expressible in Core ML's static graph. Probably 2-4 weeks of conversion work.
2. Spec 015 phases 1-3 have shipped (DFlash on GPU end-to-end), so we have a Swift-side reference for the draft architecture.

### Variant D — Generic-draft fallback (the original framing)

Use a small *generic* attention-based model (Qwen 3 0.6B, Llama 3.2 1B, Gemma 3 1B — all already in CoreML-LLM's supported set) on the ANE as drafter for any compatible target on the GPU. Modest accept rate (30-50%) due to cross-architecture distribution mismatch; sequential-within-burst loop.

Most attractive when the target is dense-attention and Variant B's same-family draft doesn't exist (e.g., Gemma 4 31B target — no Gemma 4 ~1B sibling exists, so use a Gemma 3 1B or Llama 3.2 1B draft).

## Quantitative feasibility

### Per-token cost numbers (sources cited at end)

**ANE on M4** (worst case: synchronous Core ML dispatch via ANEMLL):

| Model | tok/s on ANE | ms/token | Memory | Power |
|---|---|---|---|---|
| Llama 3.2 1B | 47–62 | 16–21 | ~600 MB | ~2 W |
| Qwen 3 0.6B | ~75 (est.) | ~13 | ~400 MB | ~2 W |
| DeepSeek R1 8B | 9.3 | 107 | ~5 GB | ~3 W |

ANE per-call dispatch overhead: **~0.095 ms** (XPC + IOKit).
ANE peak: **~19 TFLOPS FP16**, with a 32 MB SRAM cliff above ~4096-dim matmuls.

**GPU target decode latency** (ours, M1 Max, 4-bit):

| Target | tok/s | ms/token |
|---|---|---|
| Gemma 4 26B A4B (MoE) | 27 | 37 |
| Gemma 4 31B (dense) | 14 | 71 |
| Gemma 4 9B | ~30 | 33 |

### Pipelined cost model

```
wall_clock_per_round = max(K · t_ANE_decode, t_GPU_verify_K+1) + sync_overhead
expected_emitted     = accept_rate · K + 1
effective_tok_per_s  = expected_emitted / wall_clock_per_round
```

### Variant A — generic-draft cost model

Worked examples on Gemma 4 26B A4B target (37 ms decode, ~40 ms verify K+1=5), Qwen 0.6B draft on ANE (~13 ms/token), 60% accept:

| K | ANE draft time | GPU verify time | Bottleneck | Emitted | Throughput | Speedup |
|---|---|---|---|---|---|---|
| 2 | 26 ms | 38 ms | GPU | 2.2 | 58 tok/s | **2.15×** |
| 4 | 52 ms | 40 ms | ANE | 3.4 | 65 tok/s | **2.4×** |
| 6 | 78 ms | 42 ms | ANE | 4.6 | 59 tok/s | 2.2× |

Sweet spot near K=4: ANE just barely loses the race, but the higher emit-per-round ratio compensates.

The win evaporates on smaller GPU targets:

- Gemma 4 E2B (9 ms/token decode, K=4 verify ~14 ms): ANE draft 52 ms ≫ verify 14 ms → wall clock = 52 ms, emitted 3.4 → 65 tok/s vs baseline 110 tok/s. **0.6× — loss.**
- Gemma 4 9B (33 ms/token, K=4 verify ~38 ms): ANE 52 ms vs 38 ms → wall 52 ms, emit 3.4 → 65 tok/s vs 30 → **2.2×.** Win.

So Variant A's feasibility is target-size-bound: **only worth it for targets ≥9B**.

### Variant B — DFlash draft on ANE cost model

Different math because DFlash's draft is **block-diffusion** (one forward pass
producing 16 tokens) rather than autoregressive (16 sequential forwards). The
ANE-side draft is one big forward pass, not K small ones.

Estimating ANE-side block-diffusion forward time for a ~1B-parameter draft:

- ANE peak FP16 ≈ 19 TFLOPS, but realistic sustained ≈ 5-7 TFLOPS on
  block-shaped workloads (per Maderix's M4 ANE measurements).
- ~1B-parameter forward needs ~2-4 TFLOPs (depending on activation
  dimensions and block-diffusion denoising step count).
- Pessimistic estimate: 50-100 ms per draft forward (one forward producing
  16 candidates) including dispatch + IOSurface transfer overhead.

GPU-side verify: a 17-token forward on Qwen 3.5-27B-4bit, ~50-60 ms (verify
forward on the larger target with K+1=17 tokens).

Sequential (all-GPU DFlash today):
```
t_draft + t_verify ≈ 50 + 60 = 110 ms
emitted ≈ 0.89 × 16 + 1 ≈ 15.2 tokens
throughput ≈ 15.2 / 0.110 ≈ 138 tok/s         vs 33.5 baseline → 4.1×
```

Note this matches dflash-mlx's published 2.37× on Qwen3.5-27B-4bit reasonably
well after factoring in their measurement methodology (medians on a fixed
benchmark prompt). The headline number is in the right region.

ANE-offloaded (parallel overlap):
```
wall = max(t_ANE_draft, t_GPU_verify) ≈ max(50-100, 50-60)
     ≈ 60-100 ms (depending on which is the bottleneck)
emitted ≈ 15.2 tokens (unchanged)
throughput ≈ 15.2 / max(50, 60) to 15.2 / max(100, 60)
          ≈ 152 to 253 tok/s
```

Compared to the all-GPU DFlash baseline of 138 tok/s:

- **Pessimistic (ANE bottleneck at 100 ms)**: 152 tok/s → 1.10× DFlash, **4.5× absolute baseline**.
- **Optimistic (balanced ~60 ms)**: 253 tok/s → 1.83× DFlash, **7.5× absolute baseline**.

Even the pessimistic case is meaningful. The optimistic case would be the
fastest known speculative decode on Apple Silicon — though that depends on
the ANE draft fitting comfortably under 60 ms, which we'd need to measure.

**Variant B's per-target sensitivity differs from Variant A's.** With block-
diffusion, the K=16 emission rate is fixed regardless of target size; what
varies is whether the GPU verify on a smaller target is fast enough to make
the ANE draft *not* the bottleneck. On Qwen 3.5-9B (~30 ms decode, ~50 ms
verify K+1=17), ANE bottlenecks but the win is still 1.5× over all-GPU
DFlash. On Qwen 3.5-4B (~18 ms decode, ~30 ms verify), the GPU finishes
before the ANE — bottleneck is ANE — and the win is comparable.

The point: **Variant B doesn't have Variant A's "must be ≥9B" constraint.**
The K=16 amortisation works on any target size for which DFlash-on-GPU
already wins.

### Power efficiency

The headline number is power, not speed:

```
baseline (GPU only):            ~20 W steady-state
ANE-draft + GPU-verify:         ~2 W (ANE) + ~20 W (GPU verify) ≈ 22 W during compute
                               BUT verify duration drops 50% (verify only, no decode forward)
                               so average GPU power draw drops proportionally
                               net: ~12-14 W steady-state estimate
```

Combined with 2× throughput, **tokens-per-watt** improves by ~3–4×. For an unplugged MacBook or an iOS device, this is the dominant axis. Energy per token roughly halves at the same speed, or speed doubles at the same energy.

## Build vs depend vs bypass — three integration paths

A non-trivial design decision: do we *use* CoreML-LLM as a dependency, *fork* its relevant pieces into mlx-swift-lm, or *implement our own* against either Core ML or the private ANE API?

### Path 1 — Depend on CoreML-LLM as a Swift package

Pros:
- Fastest to working prototype: `import CoreMLLLM` and use their `MirrorSpeculativeLoop` directly, with our MLX-backed target as the verifier.
- All the model conversions are done. Their HuggingFace repos (`mlboydaisuke/qwen3.5-0.8B-CoreML` etc.) are downloadable.
- They've already absorbed the ANE-specific tricks (Conv2d-as-Linear, ANERMSNorm, pre-computed RoPE, explicit KV I/O, sliding-window cache).
- Their conversion pipeline (`conversion/`) lets us build new drafts as needed.
- Apache 2.0 license; fine to depend on.

Cons:
- New Swift Package dependency in mlx-swift-lm. Their dep graph (Core ML, Vision, AVFoundation) is heavier than what mlx-swift-lm currently pulls in.
- Their `ChunkedEngine` couples their decoder to their KV-cache layout. Using just their drafter (without their target engine) requires either a clean-extracted `Drafter` protocol from their code, or a fork.
- Versioning risk: they ship rapidly (v1.0 → v1.8 in a few months); breaking changes possible. Pin a tag, accept lag.

### Path 2 — Fork their drafter pieces into mlx-swift-lm

Pros:
- Zero external Swift Package dependency.
- Full control over the integration surface: extract just the parts we need (`MirrorSpeculativeLoop`, `PromptLookupLoop`, `MtpSpeculativeEngine`, `CrossVocabSpeculativeEngine`), drop the rest.
- We can adapt the chunked-engine assumptions to MLX-target-cache assumptions cleanly.

Cons:
- Lose upstream improvements. They've been shipping ANE optimisations rapidly; a fork drifts.
- Maintenance burden: every time they discover a new ANE trick (the v1.8.0 full-vocab repetition penalty fix; the v1.4.0 3-chunk decode), we have to re-port.
- License attribution / maintenance overhead.

### Path 3 — Reimplement against Core ML directly (no CoreML-LLM dep)

Pros:
- Cleanest dependency story.
- We pick our own protocol surface to integrate with `TokenIteratorProtocol`.

Cons:
- Re-deriving the ANE-specific tricks from scratch. The CoreML-LLM project's `docs/CONVERSION.md` and `docs/EXPERIMENTS.md` detail dozens of failed approaches before the working tricks. Re-walking that ground would cost months.
- We'd still need to convert the models — that's basically the same work as Path 1's conversion pipeline.

### Path 4 — Bypass Core ML entirely (private ANE API)

Pros:
- Zero Core ML dispatch overhead. Per the M4 ANE benchmarks, Core ML adds ~0.095 ms per call (XPC + IOKit). For our cycle (one ANE forward per round, several hundred rounds per generation) this is ~50 ms total — non-negligible at high tok/s.
- Theoretically faster wall-clock per round.

Cons:
- The private API is reverse-engineered (per Maderix's "Inside the M4 ANE" series). Apple has not published it.
- Apple can change it without notice between OS releases.
- **Almost certainly incompatible with App Store distribution** — apps using private APIs get rejected. Affects iOS deployment, which is one of this spec's primary motivations.
- Substantial implementation effort to match Core ML's compilation pipeline and ANE driver shim.

### Recommended path: 1 → 2

Start with **Path 1 (depend on CoreML-LLM)** for Phase 1A's measurement spike. If Mirror Speculative Decoding gives the expected speedup and the dependency works for our distribution model, ship it.

If the dependency is problematic (size, transitive deps, breaking changes), fork to **Path 2** at integration time. The cost of forking after measurements pass is bounded; the cost of re-deriving everything (Paths 3/4) is unbounded.

**Path 4 (private API) is interesting only as a measurement reference**: for Phase 1A we compare CoreML-LLM's Core ML path vs the private-API path on the same draft model and same workload. If the private-API path is materially faster (e.g., 20%+), and we can't ship it on iOS, we know the iOS deployment will leave performance on the table — useful information for the per-target routing predicate.

### Cross-Core-ML-vs-private-API benchmark

Phase 1A (below) explicitly includes a measurement that runs the same draft model through both:

- **Core ML / Mirror SD path** (CoreML-LLM dependency).
- **Private ANE API path** (custom shim — out of scope to ship, but in scope to measure).

Numbers we want:
- Per-call dispatch overhead (Core ML: ~0.095ms baseline).
- End-to-end draft throughput (tok/s for K-token autoregressive draft).
- Sustained ANE utilisation (% of theoretical peak).
- Sync barrier wall time per round.

Output: a single table that tells us how much of the ANE's potential we're actually using on the Core ML path. Informs whether to pursue the private-API path for Mac-only / sideloaded-app deployments where private APIs are tolerable.

## Design

### 1. Components

```
┌─────────────────────────────────┐
│  NGramSpeculativeTokenIterator  │   ← unchanged path: pure-CPU lookup
└─────────────────────────────────┘
┌─────────────────────────────────┐
│  ANESpeculativeTokenIterator    │   ← new
│   ┌──────────────────────────┐  │
│   │ ANEDraftBackend          │  │   Core ML model running on ANE
│   │  (per-target registered) │  │   produces K draft tokens
│   └──────────────────────────┘  │
│   ┌──────────────────────────┐  │
│   │ GPU target verify        │  │   MLX / mlx-swift-lm
│   │  (existing target model) │  │   verifies K+1 tokens in 1 forward
│   └──────────────────────────┘  │
└─────────────────────────────────┘
```

The iterator orchestrates two concurrent compute streams:
- ANE stream: autoregressive K-token draft via Core ML.
- GPU stream: K+1-token verify via MLX.

### 2. Cycle (per round)

```
T=0:   submit draft step 1 on ANE (asynchronous)
       (target idle)
T=t1:  draft 1 returns; submit draft step 2 on ANE
       submit verify forward on GPU using [y, draft_1, ...] (asynchronous)
       …
       (continue draft autoregressively as more tokens come back)
T=t2:  all K drafts ready
T=t3:  verify forward on GPU completes; sync verify argmax
       compare against drafts → accepted count
T=t4:  trim caches; emit accepted tokens; advance y
```

Critical design: **start the GPU verify as soon as enough drafts are available**, not after all K. This is similar to how llama.cpp's `speculative-simple.cpp` works but cross-compute-unit. With K=4, the GPU verify can start after draft tokens 1-2 return, while the ANE is still computing 3-4. By the time the verify forward needs positions 3-4 in its input, those tokens are ready.

This requires either:
- Submit verify with placeholder mask tokens initially, fill in later (Metal supports per-batch input update via shared buffers — possible on Apple Silicon).
- Wait for all K drafts before submitting verify (simpler, slightly less pipelined).

Phase 1 ships the latter; Phase 3 attempts the former.

### 3. Cross-framework plumbing

**Tokenizer alignment:** Draft and target must share a tokenizer up to a small Hamming distance (mirror llama.cpp's `SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128` check). Ship a one-shot vocabulary equivalence checker; refuse mismatched pairs at registry time.

**Token I/O:** Draft → target: just an `[Int]`, trivially passed through Swift. No tensor-level interop required.

**KV caches:** Draft cache lives inside the Core ML model (Core ML manages it via stateful inputs as of macOS 14). Target cache is `[KVCache]` in MLX. They are independent — no cross-framework sharing needed.

**Rollback on partial accept:** Both caches need to trim by `(K - accepted)`:
- GPU side: existing `trimPromptCache` in MLX.
- ANE side: Core ML stateful models support state rollback via `MLState.advance(by:)` or equivalent — depends on macOS version. Worst case, snapshot draft cache state at round entry and restore.

**Concurrent dispatch:** Core ML's `predictionAsync` and MLX's lazy graph + `asyncEval` are independent — they can run concurrently. Validate via Instruments' Metal System Trace + ANE Power Tools (M-series).

### 4. Draft model registry

Per-target mapping (mirrors `DRAFT_REGISTRY` in dflash-mlx but for Core ML drafts):

```swift
public struct ANEDraftRegistry {
    /// Maps target HF id (or family base name) to a Core ML mlpackage path
    /// containing the ANE-compiled draft model.
    static let drafts: [String: URL] = [
        "Gemma 4 9B":         resources.url(forResource: "Gemma-4-9B-draft.mlpackage"),
        "Gemma 4 26B A4B":    resources.url(forResource: "Gemma-4-26B-A4B-draft.mlpackage"),
        "GPT-OSS 20B":        resources.url(forResource: "GPT-OSS-20B-draft.mlpackage"),
        // …
    ]
}
```

The drafts themselves are compiled offline using ANEMLL's mlpackage pipeline. Initial set: a small Qwen 3 0.6B base trained on each target's output distribution (light distillation). Skipping this step (using a generic small draft) costs accept-rate but avoids per-target training.

### 5. Routing (`MLXLMCommon.generate`)

```
if dflash-eligible:                                     → DFlashSpeculativeTokenIterator
elif ane-draft-eligible:                                → ANESpeculativeTokenIterator
elif ngram-eligible (PLD):                              → NGramSpeculativeTokenIterator
else:                                                   → TokenIterator
```

`ane-draft-eligible` predicate:
- `parameters.aneDraftEnabled == true` (opt-in, like DFlash)
- target is registered in `ANEDraftRegistry`
- target's measured baseline tok/s on this hardware is below `MLX_ANE_DRAFT_GPU_THRESHOLD` (default 25 tok/s, i.e. won't activate on small fast targets where it loses)
- `temperature == 0` (greedy verify; non-greedy is phase 4)
- Core ML available + ANE present (`MLNeuralEngineDeviceSelectable`)

### 6. Phase 1A — measurement spike: Mirror SD via CoreML-LLM (1 week)

**Goal:** validate the cost model with the *easiest* path — depend on CoreML-LLM, use their pre-built Qwen 3.5 0.8B Core ML model and their `MirrorSpeculativeLoop`, target Qwen 3.5-27B on our MLX path.

This is now an **integration spike, not a research spike**. The architectural questions are answered upstream; we're validating that gluing our MLX target to their Core ML draft works at the expected speedup.

Steps:

1. Add `CoreML-LLM` Swift Package dependency (or vendored fork — decide later) to a feature branch.
2. Download `mlboydaisuke/qwen3.5-0.8B-CoreML` (their pre-built bundle).
3. Plumb their drafter to talk to our `Qwen35TextModel` MLX target via a thin adapter conforming to their `SpeculativeTarget` protocol. The adapter wraps our `[KVCache]` rollback semantics + verify forward.
4. Run Qwen 3.5-27B-4bit on MLX as the target.
5. Measure end-to-end tok/s vs our existing baseline.
6. **Add the Core-ML-vs-private-ANE-API benchmark side-by-side** (see "Build vs depend vs bypass" above): same draft model, two backends, compare per-call overhead, sustained throughput, and per-round wall time. This is the data we need for the path-4 decision.

Pass criteria (any two of):

- ≥3× throughput improvement vs MLX-only baseline on Qwen 3.5-27B-4bit (Mirror SD's lower bound).
- ≥4× tokens-per-watt improvement.
- Verified parallel ANE + GPU execution — both compute units showing ≥70% utilisation simultaneously during steady-state decode.
- Per-round wall time ≤ `max(t_ANE_draft, t_GPU_verify) × 1.2` (i.e. overhead < 20% of the parallel ideal).

Fail → fall back to Variant B (sequential-within-burst, no early-exit signal). Don't close the spec — Variant B's 1.5× is still a meaningful win on the same hardware.

### 7. Phase 1B — DFlash-on-ANE measurement spike (after spec 015 phases 1-3 ship + Phase 1A passes)

**Goal:** validate Variant C's cost model — DFlash draft on ANE, DFlash target on GPU.

DFlash's draft architecture (block-diffusion + cross-attention to target hidden states) does not exist as a Core ML model today. Either:

1. Convert one of `z-lab/Qwen3.5-*-DFlash` (the published DFlash draft weights) to Core ML using a pipeline derived from CoreML-LLM's `conversion/` tooling. Estimated 2-4 weeks of conversion work for the first model; subsequent ports are mechanical.
2. Train a new DFlash-style draft against a target we already support. Out of scope.

Once a Core ML DFlash draft is available, the spike harness:

1. Loads the converted DFlash draft on ANE via Core ML.
2. Runs the matching DFlash-eligible target (Qwen 3.5-9B / 27B / 35B-A3B) on GPU via MLX.
3. Wires the cross-attention input — target's hidden states from selected layers flow GPU → ANE through unified-memory `IOSurface` (or via Core ML's stateful-input API if it tolerates the access pattern).
4. Measures the cycle: ANE draft (1 forward producing 16 candidate tokens) + GPU verify (17-token forward), concurrently.

Pass criterion: ≥1.5× throughput improvement vs all-GPU DFlash on the same target (i.e. ≥3.5× over baseline on Qwen 3.5-27B), or ≥4× tokens-per-watt improvement.

Failure modes to plan for:
- Cross-attention input transfer bandwidth ANE↔GPU dominates the cycle. Mitigation: capture fewer target layers (DFlash supports a configurable layer list).
- Block-diffusion's parallel-denoising loop doesn't compile to Core ML's static graph cleanly. Mitigation: unroll to fixed step count.

### 8. Phase 2 — integrated iterator (3-4 weeks)

Once Phase 1A passes, implement `ANESpeculativeTokenIterator` as a first-class iterator:

- Adopt the dependency / fork decision from Phase 1A.
- `MirrorSpeculativeTokenIterator` wrapping the early-exit + concurrent-execution loop. Adapter layer translates between CoreML-LLM's `SpeculativeTarget` protocol (or our extracted equivalent) and our `[KVCache]` rollback semantics.
- Vocabulary-equivalence registry guard at iterator construction.
- Auto-routing in `MLXLMCommon.generate(...)`: when target is in the supported set AND a matching CoreML-LLM draft is available locally, prefer the Mirror SD path over plain PLD.
- `ANEDraftRegistry` mapping target HF id → CoreML-LLM draft bundle path.
- Bench harness method `--method mirror-sd` for repeatable measurement.

Acceptance:

- ≥3× throughput on Qwen 3.5-27B QA-requote / code-refactor / multi-turn-code prompts.
- Output byte-identical to MLX-only baseline at temperature 0 (strict-greedy guard equivalent applies).
- Iterator gracefully falls back to plain PLD when CoreML-LLM draft is missing or ANE is unavailable.

### 9. Phase 3 — Variant B fallback path

If Phase 1A's early-exit signal is impractical (cross-framework IPC latency dominates), ship Variant B (sequential-within-burst): same Mirror-SD draft+target pairing but without early-exit. ~1.5× speedup ceiling, simpler integration. This is a degraded version of the iterator that auto-selects when measured cycle time exceeds the Mirror-SD ideal by >50%.

### 10. Phase 4 — Variant C (DFlash-on-ANE) integration

After Phase 1B passes (DFlash draft → Core ML conversion landed). Mostly mechanical: same iterator skeleton, swap drafter implementation. Expected speedup: 4-7× over baseline on Qwen 3.5-27B-4bit (Mirror SD parallelism × DFlash's K=16 amortisation).

### 11. Phase 5 — Variant D (generic-draft fallback)

For targets without a same-family CoreML-LLM draft *and* without DFlash support: use a generic small ANE-supported model (Llama 3.2 1B, Gemma 3 1B). Lower expected speedup (~1.5-2×) due to cross-architecture acceptance tax. Last in priority because the targets it serves (Gemma 4 31B dense, etc.) already have other speedup paths (PLD with cross-request cache, spec 016).

## What we should not do

- Do not block DFlash work (spec 015) on this. DFlash is 3-4× on Qwen 3.5/3.6 with no cross-framework plumbing; that's a separate, higher-priority workstream.
- Do not pursue ANE-offload for **PLD** (n-gram lookup). The lookup is already CPU-µs cost; ANE adds nothing.
- Do not attempt to put the *target* on ANE. ANE's 32 MB SRAM cliff and dispatch model make it a bad fit for the high-bandwidth-per-token target forward. Keep target on GPU.
- Do not ship without the `MLX_ANE_DRAFT_GPU_THRESHOLD` guard. ANE-draft on a fast target is a net loss — silent regressions are unacceptable.

## Open questions

1. **GDN-on-ANE.** The most consequential prerequisite for Variant B and for
   the Qwen 3.5-0.8B → Qwen 3.5-27B pairing. Selective state-space layers
   (Mamba, GatedDeltaNet) involve input-dependent parameters, gated delta
   updates, and convolution state — none of which are mainstream Core ML
   ops. Two paths: (a) implement as composite of supported ops with possible
   utilisation cost, (b) ship a custom Core ML op via the Core ML Tools
   custom-layer API. Either is research-scale, weeks of effort. Track as a
   distinct sub-spec (021b?) before committing to Phase 1B.

2. **Core ML stateful model rollback API.** macOS 14 introduced stateful
   Core ML models with `MLState`. The exact rollback semantics on partial
   accept are documented loosely; need to verify they work for our case or
   fall back to per-round full snapshot/restore (slower).

3. **ANE peak utilisation under back-to-back forward passes.** ANEMLL
   reports 47-62 tok/s on Llama 3.2 1B with overhead patterns optimised for
   repeated single-shape inference. For Variant A's pattern (K=4
   single-token forwards in tight succession), or Variant B's pattern (one
   block-diff forward producing 16 tokens), measured throughput may differ
   from their published numbers. Phase 1A answers this.

4. **Hidden-state transfer bandwidth (Variant B only).** DFlash drafts
   consume target hidden states via cross-attention. With unified memory the
   transfer should be near-zero-copy, but the specific Core ML ↔ MLX
   interop API for hidden-state inputs is not documented for our use case.
   Possibly need to plumb through `IOSurface` directly, bypassing Core ML's
   higher-level prediction APIs. Medium-risk integration.

5. **Per-target draft training cost.** Variant A: a generic Qwen 3 0.6B
   draft probably gets 30-50% accept on cross-family targets (Gemma 4,
   GPT-OSS). Target-specific distillation (one day of training per target)
   lifts this to 60-70%. Variant B: DFlash drafts already exist for Qwen
   3.5/3.6 from z-lab; no additional training needed *if* GDN-on-ANE lands.
   Whose budget covers Variant A's distillation?

6. **Tokenizer mismatch.** Gemma 4 and Qwen 3 do *not* share tokenizers.
   For Variant A, need a Gemma-family draft and a Qwen-family draft
   separately — realistic registry covers ~3-4 distinct tokenizer families.
   For Variant B, the DFlash drafts ship with the matching tokenizer, so
   this is a non-issue.

7. **Qwen 3 → Qwen 3.5 cross-architecture pairing.** Qwen 3 (dense
   attention) and Qwen 3.5 (hybrid GDN) share the *same* tokenizer per the
   vLLM Qwen 3.5 recipes. Using Qwen 3-0.6B (which ANEMLL supports today)
   as the ANE-side draft for a Qwen 3.5 target on GPU is the closest
   currently-buildable approximation of "Qwen3.5-0.8B → Qwen3.5-27B". The
   accept-rate tax of cross-architecture drafting is unknown — possibly
   30-50% range; Phase 1A would measure.

8. **iOS deployment.** ANE-on-iOS is the most attractive deployment for
   this entire spec. iOS-specific testing is out of scope for the engine
   repo but should be flagged for the SwiftLM consumers.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/ANESpeculativeDecoding.swift` (new) | Iterator + cross-stream orchestration. ~600 lines. |
| `Libraries/MLXLMCommon/ANEDraftBackend.swift` (new) | Core ML wrapper, async prediction, state rollback. ~400 lines. |
| `Libraries/MLXLMCommon/ANEDraftRegistry.swift` (new) | Target → mlpackage map, vocab-equivalence checker. ~150 lines. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Routing predicate, eligibility check. |
| `Tests/MLXLMTests/ANEDraftSpikeTests.swift` (new, Phase 1) | Standalone measurement harness. |
| `Tests/MLXLMTests/ANESpeculativeTests.swift` (new, Phase 2) | Iterator integration tests. |
| `Resources/ane-drafts/*.mlpackage` (new, Phase 2) | Per-target compiled drafts. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | `--method ane-draft` plumbing. |

## Where this fits in the broader roadmap

This spec went from **research-grade and conditional** to **engineering-grade and high-priority** in the time it took to read one upstream repository. The pattern (cross-compute-unit speculative decoding) is published prior art (Apple's Mirror Speculative Decoding paper). The model-conversion work is shipped (CoreML-LLM has Qwen 3.5 hybrid SSM+attention). A reference Swift implementation exists. The remaining engineering is integration glue.

Updated ordering:

1. **Spec 015 phases 1-3 (DFlash on GPU)** — still first. Stand-alone 2-4× win on Qwen 3.5/3.6, no cross-framework risk. Gives us the Swift-side DFlash reference for Phase 1B of this spec.
2. **Spec 021 Phase 1A (Mirror SD measurement spike via CoreML-LLM)** — the foundational measurement question is "does CoreML-LLM's `MirrorSpeculativeLoop` give the projected 3× on Qwen 3.5-27B when paired with our MLX target?" One week of integration work.
3. **Spec 021 Phase 2 (integrated iterator)** — if Phase 1A passes, ship Mirror SD via either dependency or fork. This is now the **primary speedup path for Qwen 3.5/3.6 on Apple Silicon**, beating DFlash-on-GPU-alone.
4. **Spec 020 (tape-replay rollback)** — orthogonal; still required for PLD on Qwen 3.5/3.6 even with Mirror SD shipping (Mirror SD's rollback is on the draft side, not the target side).
5. **Spec 015 phases 4-6 + spec 021 Phase 4 (Variant C)** — DFlash-on-ANE composition. Multiplicative speedup (4-7× region). Requires DFlash → Core ML conversion.
6. Everything else (PLD optimisations, prefix cache, n-gram cache) — additive on top.

The TL;DR: your DFlash-on-ANE intuition was right *and* it composes with the broader Mirror-SD pattern that Apple published in January 2026. Your same-family pairing intuition was also right — Qwen 3.5-0.8B → Qwen 3.5-27B is exactly the working pairing in CoreML-LLM today. The original "GDN-on-ANE doesn't exist" objection in the first draft of this spec was wrong; it does exist and is shipping.

## Out of scope (and why)

- **MLX direct ANE backend.** Apple has not published a stable Metal-equivalent ANE programming API at this writing; everything goes through Core ML. Watch the Orion paper's findings — direct ANE programming would change the cost model significantly (the 0.095 ms dispatch overhead is the killer for small workloads). If/when MLX gains an ANE backend, this spec may need significant revision.
- **M5 Neural Accelerators in the GPU.** A *different* Apple Silicon feature: matmul-specialised hardware in the M5 GPU itself (akin to NVIDIA tensor cores), accessed via the existing MLX path. Faster than the ANE for compute-heavy operations and doesn't require Core ML. The "right" follow-up to spec decode on M5+ is exploiting these via MLX, not Core ML / ANE. This spec stays relevant for M1-M4 hardware where the ANE is the only co-processor option.

## References

### Upstream prior art (the foundation of this spec)

- [**`john-rocky/CoreML-LLM`**](https://github.com/john-rocky/CoreML-LLM) — the reference implementation. Has Qwen 3.5 0.8B / 2B on ANE (hybrid SSM+attn, 99.9% ANE-resident), `MirrorSpeculativeLoop` (Mirror SD), `MtpSpeculativeEngine`, `LookaheadEngine`, `SuffixSpeculativeEngine` + `SuffixTree`, `CrossVocabSpeculativeEngine`, `PromptLookupLoop`, `PrefixCache`, `PrefixKVCache`. Apache 2.0 licensed.
- [**Mirror Speculative Decoding (Apple ML Research, arXiv:2510.13161)**](https://arxiv.org/abs/2510.13161) — Apple's January 2026 paper formalising the cross-compute-unit speculative decoding pattern. Reports 2.8–5.8× speedup, 30% improvement over EAGLE-3. Server-scale benchmarks (M2 Ultra × 8) but the pattern is the same on consumer hardware.
- [**`bstnxbt/dflash-mlx` engine-v2**](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2) — DFlash on GPU, the reference for spec 015 and Variant C of this spec.
- [**Speculative Streaming (Apple ML Research, arXiv:2402.11131)**](https://arxiv.org/abs/2402.11131) — fuse the drafter into the target via multi-token-prediction heads. 1.8-3.1× speedups. Conceptually distinct from Mirror SD (single-model vs cross-device).
- [**Recurrent Drafter (Apple ML Research)**](https://machinelearning.apple.com/research/recurrent-drafter) — Apple's earlier work on small-recurrent-network drafters. Conceptual ancestor of the EAGLE-3 + draft-on-ANE pattern.

### Architecture details

- [Qwen3.5 0.8B Core ML weights (`mlboydaisuke/qwen3.5-0.8B-CoreML`)](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML) — pre-built bundle used by Phase 1A.
- [Qwen3.5 model lineup (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-9B) — confirms 0.8B / 2B / 4B / 9B / 27B / 35B-A3B / 122B-A10B / 397B-A17B series, all hybrid GatedDeltaNet.
- [GatedDeltaNet (Yang et al., NVIDIA / ICLR 2025)](https://github.com/NVlabs/GatedDeltaNet) — Qwen 3.5's hybrid linear-attention layer. CoreML-LLM ported this to Core ML for v1.0.0 (their first hybrid SSM+attention LLM on Core ML).
- [vLLM Qwen 3.5 / 3.6 Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) — confirms tokenizer compatibility across the Qwen 3 / 3.5 / 3.6 generations.

### ANE characterisation (for the path-4 measurement)

- [Inside the M4 Apple Neural Engine, Part 2: ANE Benchmarks (Maderix, 2025)](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — the dispatch-overhead (~0.095 ms), SRAM-cliff (32 MB), and TFLOPS (~19 FP16) numbers.
- [Orion: Characterizing and Programming Apple's Neural Engine (arXiv:2603.06728)](https://arxiv.org/html/2603.06728v1) — academic study, the closest thing to a private-API characterisation we have.
- [Apple Neural Engine for LLM Inference (InsiderLLM, 2025)](https://insiderllm.com/guides/apple-neural-engine-llm-inference/) — comparative ANE vs GPU benchmarks.
- [ANEMLL](https://github.com/Anemll/Anemll) — alternative Core ML LLM pipeline. Doesn't currently support GatedDeltaNet (CoreML-LLM does); supports Llama / Qwen 2.5 / Qwen 3 / Gemma 3 / DeepSeek R1 / DeepHermes.

### M5 Neural Accelerators (separate optimisation path, mentioned for context)

- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU (Apple ML Research)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) — M5's matmul-specialised hardware lives **in the GPU**, not in the ANE. Accessed via MLX. Different optimisation path than this spec; relevant for M5+ deployments where the GPU itself becomes faster.
