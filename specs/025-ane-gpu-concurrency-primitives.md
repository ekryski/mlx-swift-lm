# 025 — ANE+GPU concurrency primitives — race-free cross-device state for Mirror SD

**Status:** spec, ready to issue (precursor to spec 021)
**Branch:** new branch off alpha
**Depends on:** [issue #155](https://github.com/ekryski/mlx-swift-lm/issues/155) (`--dispatch-audit` Swift wrapper) for measurement; CoreML-LLM Swift Package availability for Phase 3
**Unblocks:** [spec 021](./021-ane-offloaded-draft-model.md) Phase 1A's concurrent-execution measurement gate, and Tier 4 row 13 (DFlash-on-ANE)
**Origin:** lessons from the retired AB/ICB track — see [`sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md`](/Users/eric/Development/personal/sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md)

## The insight

The AB/ICB track ran into one load-bearing problem: **per-step state crossed an asynchronous device boundary via mutate-in-place persistent storage, which forced a `Stream.synchronize` of 19–21 ms per step that wiped out every win.** The fix was identified (override-bound fresh allocation, the same pattern K/V `startArr` already uses race-free) but never implemented before the track was retired.

Spec 021 (ANE-offloaded draft model) is structurally the same problem with two devices instead of one. ANE produces draft tokens, GPU verifies them in parallel, CPU orchestrates. Per-step state flows in four directions: CPU → ANE (next draft input), ANE → CPU (draft logits), CPU → GPU (verify input), GPU → CPU (target logits). Each cross-device handoff is a potential race if storage is mutated in place from CPU while a device is mid-execution.

The plan's existing decision-point #2 calls this out:

> **021 Phase 1A's concurrent-execution check** — Apple Silicon truly running ANE + GPU in parallel without serialisation through XPC / IOSurface / shared queues. Failure here kills 021 entirely (everything in Tier 4 item 13 too).

That's the same question AB/ICB answered for single-device CPU-GPU concurrency: parallel execution is achievable iff per-step state is race-free at every async boundary. **This spec codifies the primitives we need before 021 phase 1A's measurement runs**, so the measurement evaluates concurrency itself, not integration-bug-prone integration code.

## What carries over from AB/ICB

Three architectural lessons map directly to ANE+GPU:

1. **Override-bound fresh-allocation is race-free; mutate-in-place persistent storage isn't.** K/V `startArr` (fresh `MLXArray` per step, lifecycle held by Metal command-buffer completion handler) ran in production without explicit synchronization. Persistent-AB shared-storage mutation forced a 19–21 ms `Stream.synchronize` per step. The same architectural rule applies across any async boundary: never mutate per-step state in place when a non-CPU device may still be reading.

2. **Pre-record + replay-with-overrides fits fixed-shape loops perfectly.** A draft-model decode loop has a fixed compute shape per model — exactly the workload ICB record/replay was designed for, just on a different compute device. Core ML's compiled prediction pipeline is already a "pre-recorded shape with per-call inputs" in this sense; the override mechanics translate directly.

3. **Dispatch instrumentation is the only way to verify parallelism actually happens.** AB's CPU-encoding microbench showed a 1.45× win that didn't translate to throughput because the wins were swallowed by serialization invisible to the microbench. Without per-device kernel/op counts and per-phase wall-clock, you can't tell whether ANE+GPU are actually overlapping or just round-tripping through XPC.

## What doesn't carry over

- **GPU-side CPU encoding overhead.** ANE doesn't have Metal command buffers. Core ML compiles the graph upfront; there's no per-step encoding cost to amortize. The +23% AB+PersistentAB win (which was specifically a CPU-encoding win) doesn't translate.
- **Argument Buffers themselves.** Core ML manages its own argument layout. The AB kernel work (`rms_norm_ab.metal` etc.) doesn't carry over verbatim.
- **`PersistentAb.swift`'s handle abstractions.** Those targeted Metal-specific buffer slots. ANE inputs go through `MLArray` / `MLMultiArray`, a different lifecycle model.

What carries forward is the **architectural rule**, not the code: every cross-device per-step buffer either (a) is a fresh allocation per step held alive by a completion handler, or (b) uses explicit double-buffering with completion-callback synchronization. **Never mutate persistent storage in place across an async device boundary.**

## Design

### 1. Cross-device state inventory for Mirror SD

Every per-step buffer crossing a device boundary in Mirror SD on ANE+GPU:

| Direction | What | Today (single-device GPU) | Cross-device pattern |
|---|---|---|---|
| CPU → ANE | next draft input tokens | `MLXArray` from `startArr` (race-free) | Fresh `MLArray` per `predict()` call; Core ML retains until completion |
| ANE → CPU | draft logits / sampled tokens | (does not exist yet) | Fresh output buffer per call; copy value out *inside* completion scope, never store the `MLMultiArray` ref |
| CPU → GPU | verify input (drafted tokens + cache slot) | already `MLXArray` per step | Unchanged — already follows the override-bound pattern |
| GPU → CPU | target logits | already `MLXArray` per step | Unchanged |

The two ANE-touching rows are the new race surface. Both follow the same pattern as K/V `startArr`: fresh allocation per step, lifecycle managed by the framework's completion path, never reused across calls.

### 2. The ANE input pattern

Today's K/V `startArr` pattern in `Evaluate.swift`:

```swift
let startArr = MLXArray([writeIdx])
overrides[.kvCacheStart] = startArr
// startArr is retained by Metal command-buffer completion; safe to drop ref
```

Adapt for ANE input — same shape, different framework:

```swift
let draftInput = MLArray.fromMLXArray(currentInputTokens)  // fresh per step
draftModel.predict(input: draftInput) { [weak self] result in
    self?.handleDraftResult(result)
}
// draftInput is retained by Core ML's completion path; safe to drop ref
```

**Critical invariant:** never reuse a `MLArray` across two `predict()` calls. Always allocate fresh. Memory pressure is fine — single-token-input arrays are tiny and the allocator amortizes.

### 3. The ANE output pattern

Core ML's prediction returns a fresh output buffer per call, retained until the completion handler runs:

```swift
draftModel.predict(input: draftInput) { result in
    let logits = result.featureValue(for: "logits")?.multiArrayValue
    // logits buffer is owned by Core ML output, retained for this scope only
    let draftToken = sampleAndCopy(logits)  // copy value out before scope ends
    onDraftReady(draftToken)
}
```

**Critical invariant:** **copy** the value out of the Core ML output before the completion handler scope ends. Never store a `MLMultiArray` reference for use by a subsequent GPU verify step — that crosses devices and re-introduces the AB shared-storage race in the worst possible form (silent corruption, no sync error).

### 4. Concurrency measurement harness

The concrete deliverable that gates spec 021 phase 1A's decision point.

Build a standalone test that runs:
- **(a)** GPU-only target forward, isolated
- **(b)** ANE-only draft forward, isolated
- **(c)** both concurrently, with the patterns from §2 and §3

Instrumentation:
- **`--dispatch-audit`** (issue #155) measures per-device kernel/op counts
- **`MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1`** signposts wall-clock per phase
- **`os_signpost` on the Core ML side** signposts ANE submission + completion
- **Instruments system trace** captures any XPC round-trips

Pass criteria:
- Concurrent wall-clock ≤ `max(GPU-only, ANE-only) + 10% overhead`
- No XPC round-trips visible per step in the system trace
- IOSurface-backed buffers used for any cross-device data (verifiable in trace)

Fail criteria identifies exactly where serialization happens (XPC, queue ordering, IOSurface contention, completion-handler thread hops).

### 5. The buffer lifecycle audit

Three things to verify *before* writing any Mirror SD integration code:

1. **`MLArray` retention semantics.** Does Core ML retain input arrays until prediction completes, even if the caller drops the ref immediately? Build a deliberate ARC stress test: weak-reference the input, drop the strong ref, verify the prediction still produces correct output.

2. **IOSurface lifetime under concurrent access.** Allocate an IOSurface, hand to Core ML as input, hand to MLX as a separate read, verify both can read concurrently without serialization. Measure in Instruments — if the trace shows queue serialization, IOSurface contention is real.

3. **Completion handler thread.** Where does Core ML's prediction completion run? Same thread as MLX's command-buffer completion? Different queue? Determines whether token handoff needs an explicit queue hop and how much latency that adds.

## Implementation phases

1. **Phase 1 — Concurrency measurement harness.** Build the standalone test from §4. Run on Apple Silicon. Reproducible numbers across runs. ~200 lines of Swift, no model integration. Output: pass/fail on concurrent execution, with diagnostic data if fail.

2. **Phase 2 — Buffer lifecycle audit.** Run the three verifications from §5. Document findings in `benchmarks/notes/ane-buffer-lifecycle-2026-MM-DD.md`. Determines whether the patterns in §2 and §3 are sufficient or need adjustment for Apple's actual ANE behavior.

3. **Phase 3 — Reference draft-loop primitive.** Build a minimal `ANEDraftLoop` Swift class that takes a Core ML model + an input `MLXArray`, runs draft forward, returns logits via the patterns from §2 and §3. Tested against a tiny placeholder Core ML model — no real Mirror SD draft yet. Validates the primitive class works before any spec 021 integration code is written. Depends on CoreML-LLM Swift Package dep landing (currently a 021 Phase 1A prerequisite).

4. **Phase 4 — Hand off to spec 021.** Deliver the primitive class + measurement harness to spec 021 Phase 1A. Phase 1A's measurement gate now evaluates a known-correct primitive instead of integration-bug-prone integration code.

## Expected impact

**This spec doesn't ship an end-user feature.** Its value is **de-risking spec 021 phase 1A's measurement gate** and providing reusable primitives for any future ANE-offloaded work.

If the harness shows ANE+GPU don't actually run in parallel, spec 021 fails fast with a clean diagnostic — we know the primitives, not the integration code, are the constraint. That cuts 1–2 weeks of debugging integration code looking for the bug that's actually in Apple's IOSurface or XPC layer. If the harness passes, spec 021 phase 1B starts on a known-good foundation.

Same value applies to Tier 4 row 13 (DFlash-on-ANE) — same primitives, different draft model.

## Risks

1. **Apple Silicon may not actually run ANE+GPU concurrently for this workload.** This is decision-point #2 from the implementation plan, just reframed. Phase 1's measurement harness is the test. If it fails, both spec 021 and Tier 4 row 13 die. Better to find out via a 200-line harness than through 021 phase 1B integration debugging.

2. **IOSurface contention.** Cross-device shared buffers may serialize through IOSurface internals on Apple Silicon, producing XPC round-trips invisible to single-process benchmarks. The §5.2 test explicitly checks for this; if it fails, the cross-device handoff pattern needs a different mechanism (probably copy-through-CPU, which costs latency).

3. **Core ML completion-handler thread surprises.** If Core ML's completion runs on a private queue with no MLX await primitive, integration needs a queue hop that adds latency. The §5.3 test surfaces this as a hidden cost line item for spec 021's perf model.

4. **The architectural lessons may not transfer perfectly.** The AB/ICB postmortem identified the rule "never mutate persistent storage in place across an async boundary" but the rule was derived from Metal-specific behavior. Core ML's `MLArray` lifecycle model may have its own quirks that change the rule. Phase 2 is the validation step.

## Files touched

| File | What |
|---|---|
| `Tests/Benchmarks/ANEConcurrencyHarness.swift` (new) | §4 measurement harness |
| `Tests/MLXLMTests/ANEBufferLifecycleTests.swift` (new) | §5 lifecycle audit |
| `Libraries/MLXLMCommon/ANEDraftLoop.swift` (new) | §3 primitive class |
| `benchmarks/notes/ane-concurrency-2026-MM-DD.md` (new) | results from phases 1–2 |
| `Package.swift` | CoreML-LLM Swift Package dep (shared with spec 021 phase 1A) |

## Why this is Tier 3 prework, not Tier 4

Spec 021 is Tier 3 with explicit decision-point gating on concurrent execution. Spec 025 *is* the concurrent-execution measurement, plus the primitives that 021's integration code uses. Same tier, executes immediately before 021 Phase 1A. Skipping it means measurement and integration debugging tangle together — exactly the failure mode that made the AB/ICB decode-loop ICB regression hard to diagnose (the 19–21 ms `Stream.synchronize` was the right diagnosis, but it took weeks of branch divergence to isolate).

The cost is small (~3–5 days for phases 1–2, another 2–3 for phase 3). The value is making spec 021 phase 1A a clean measurement of the *concurrency hypothesis*, not a measurement of "did we write the integration code correctly." That's exactly the architectural mistake that AB/ICB's decode-loop integration made.
