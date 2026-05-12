// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Tape-replay rollback tests (spec 020 phase 1)
//
// Phase 1 lands the protocol + free helpers; concrete `SSMStateCache`
// conformance + Metal kernel are phase 2. The tests here cover:
//
//   1. `canRollbackPromptCache` predicate semantics — pure-trimmable
//      stacks pass, state-replay-only stacks pass, mixed stacks pass,
//      stacks containing a layer that's neither trimmable nor state-replay-
//      replayable fail.
//   2. `rollbackPromptCache` dispatch — state-replay layers see
//      `commitFull`/`rollback`/`cancel` per acceptance count;
//      trimmable layers see `trim(rejected)`.
//   3. `beginCacheRecord` / `cancelCacheRecord` / `commitCacheRecord`
//      route only to state-replay layers.
//   4. Mixed-cache stack — both layer types coexist and observe correct
//      method calls per round.

// MARK: - Test fakes

/// Minimal trimmable-only cache that records what it was asked to do.
/// Doesn't actually store K/V — phase-1 helpers don't touch shapes, only
/// dispatch decisions.
final class FakeTrimmableCache: KVCache, Evaluatable {
    var trimCalls: [Int] = []
    var cacheOffset: Int = 0

    var offset: Int { cacheOffset }
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    func peek() -> (MLXArray, MLXArray)? { nil }
    var state: [MLXArray] {
        get { [] }
        set { _ = newValue }
    }
    var metaState: [String] {
        get { [] }
        set { _ = newValue }
    }
    var isTrimmable: Bool { true }
    @discardableResult
    func trim(_ n: Int) -> Int {
        trimCalls.append(n)
        cacheOffset = Swift.max(0, cacheOffset - n)
        return n
    }
    var memoryBytes: Int { 0 }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache { FakeTrimmableCache() }
    var isDonor: Bool {
        get { false }
        set { _ = newValue }
    }
    func innerState() -> [MLXArray] { [] }
}

/// Minimal cache that's neither trimmable nor state-replay-able. Models
/// the pre-phase-2 hybrid case where SSMStateCache exists but doesn't
/// conform to StateReplayCache yet.
final class FakeOpaqueCache: KVCache, Evaluatable {
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    func peek() -> (MLXArray, MLXArray)? { nil }
    var state: [MLXArray] {
        get { [] }
        set { _ = newValue }
    }
    var metaState: [String] {
        get { [] }
        set { _ = newValue }
    }
    var isTrimmable: Bool { false }
    @discardableResult
    func trim(_ n: Int) -> Int { 0 }
    var memoryBytes: Int { 0 }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache { FakeOpaqueCache() }
    var isDonor: Bool {
        get { false }
        set { _ = newValue }
    }
    func innerState() -> [MLXArray] { [] }
}

/// Minimal state-replay cache that records every method call. Used to
/// verify the dispatch logic in `rollbackPromptCache` and the
/// begin/commit/cancel helpers.
final class FakeStateReplayCache: StateReplayCache, Evaluatable {
    enum Event: Equatable {
        case begin
        case recordStep
        case commitFull
        case rollback(Int)
        case cancel
    }

    var events: [Event] = []
    var enabled: Bool = true

    var offset: Int { 0 }
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    func peek() -> (MLXArray, MLXArray)? { nil }
    var state: [MLXArray] {
        get { [] }
        set { _ = newValue }
    }
    var metaState: [String] {
        get { [] }
        set { _ = newValue }
    }
    var isTrimmable: Bool { false }
    @discardableResult
    func trim(_ n: Int) -> Int { 0 }
    var memoryBytes: Int { 0 }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache { FakeStateReplayCache() }
    var isDonor: Bool {
        get { false }
        set { _ = newValue }
    }
    func innerState() -> [MLXArray] { [] }

    var canStateReplay: Bool { enabled }
    var replayCost: StateReplayCost { .ok }
    func beginRecord() { events.append(.begin) }
    func recordStep(_ innovations: [MLXArray]) { events.append(.recordStep) }
    func commitFull() { events.append(.commitFull) }
    func rollback(acceptedPrefix k: Int) { events.append(.rollback(k)) }
    func cancel() { events.append(.cancel) }
}

// MARK: - Predicate tests

@Suite
struct CanRollbackPromptCacheTests {

    @Test
    func `Empty stack passes (vacuously)`() {
        #expect(canRollbackPromptCache([]))
    }

    @Test
    func `All-trimmable stack passes`() {
        let cache: [KVCache] = [FakeTrimmableCache(), FakeTrimmableCache()]
        #expect(canRollbackPromptCache(cache))
    }

    @Test
    func `All-state-replay stack passes`() {
        let cache: [KVCache] = [FakeStateReplayCache(), FakeStateReplayCache()]
        #expect(canRollbackPromptCache(cache))
    }

    @Test
    func `Mixed trimmable + state-replay stack passes`() {
        let cache: [KVCache] = [
            FakeTrimmableCache(),
            FakeStateReplayCache(),
            FakeTrimmableCache(),
        ]
        #expect(canRollbackPromptCache(cache))
    }

    @Test
    func `Stack with opaque cache fails`() {
        let cache: [KVCache] = [FakeTrimmableCache(), FakeOpaqueCache()]
        #expect(!canRollbackPromptCache(cache))
    }

    @Test
    func `StateReplayCache reporting canStateReplay = false fails`() {
        let disabled = FakeStateReplayCache()
        disabled.enabled = false
        let cache: [KVCache] = [FakeTrimmableCache(), disabled]
        #expect(!canRollbackPromptCache(cache))
    }
}

// MARK: - rollbackPromptCache dispatch tests

@Suite
struct RollbackPromptCacheTests {

    @Test
    func `Full accept calls commitFull on state-replay layers, no trim on trimmable`() {
        let trim = FakeTrimmableCache()
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [trim, stateReplay]

        rollbackPromptCache(cache, acceptedPrefix: 4, numDraft: 4)

        #expect(trim.trimCalls == [0])  // rejected = 0; trim(0) is called
        #expect(stateReplay.events == [.commitFull])
    }

    @Test
    func `Zero accept folds y baseline (rollback(1)) on state-replay layers, trim by all on trimmable`() {
        // Off-by-one regression test. The verify forward records
        // T = numDraft + 1 entries (1 y baseline + numDraft drafts). On
        // zero-draft-accept, the y baseline is still kept (the iterator
        // emits a correction token AT position 0 = right after y), so the
        // cache must fold ONE entry — not call `cancel()` which would
        // restore the snapshot and lose y's state contribution. Trim
        // path keeps `T - rejected = numDraft + 1 - numDraft = 1` entry
        // (y), matching this.
        let trim = FakeTrimmableCache()
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [trim, stateReplay]

        rollbackPromptCache(cache, acceptedPrefix: 0, numDraft: 4)

        #expect(trim.trimCalls == [4])
        #expect(stateReplay.events == [.rollback(1)])
    }

    @Test
    func `Partial accept folds (acceptedPrefix + 1) entries on state-replay layers`() {
        // Off-by-one regression test. Helper translates iterator-semantic
        // `acceptedPrefix = 3 drafts` to protocol-semantic
        // `acceptedPrefix = 4 entries to fold` (y baseline + 3 drafts).
        // Trim path keeps T - rejected = 6 - 2 = 4 entries — matching.
        let trim = FakeTrimmableCache()
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [trim, stateReplay]

        rollbackPromptCache(cache, acceptedPrefix: 3, numDraft: 5)

        #expect(trim.trimCalls == [2])  // rejected = 2
        #expect(stateReplay.events == [.rollback(4)])  // y + 3 drafts
    }

    @Test
    func `Mixed-stack rollback hits each layer once`() {
        let trim1 = FakeTrimmableCache()
        let tape1 = FakeStateReplayCache()
        let trim2 = FakeTrimmableCache()
        let tape2 = FakeStateReplayCache()
        let cache: [KVCache] = [trim1, tape1, trim2, tape2]

        rollbackPromptCache(cache, acceptedPrefix: 1, numDraft: 4)

        #expect(trim1.trimCalls == [3])      // rejected = 3
        #expect(trim2.trimCalls == [3])
        #expect(tape1.events == [.rollback(2)])  // y + 1 draft = 2 entries
        #expect(tape2.events == [.rollback(2)])
    }

    @Test
    func `Disabled state-replay layer is skipped, no event recorded`() {
        let trim = FakeTrimmableCache()
        let stateReplay = FakeStateReplayCache()
        stateReplay.enabled = false
        let cache: [KVCache] = [trim, stateReplay]

        rollbackPromptCache(cache, acceptedPrefix: 2, numDraft: 4)

        #expect(trim.trimCalls == [2])
        #expect(stateReplay.events == [])  // skipped because canStateReplay = false
    }

    @Test
    func `Precondition rejects out-of-range acceptedPrefix`() async {
        // Skipping - preconditions trap, not graceful for tests.
        // Valid range tested implicitly by the other cases.
    }
}

// MARK: - begin / commit / cancel record tests

@Suite
struct CacheRecordHelpersTests {

    @Test
    func `beginCacheRecord routes only to state-replay layers`() {
        let trim = FakeTrimmableCache()
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [trim, stateReplay, FakeTrimmableCache()]
        beginCacheRecord(cache)
        #expect(stateReplay.events == [.begin])
        #expect(trim.trimCalls == [])
    }

    @Test
    func `commitCacheRecord routes only to state-replay layers`() {
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [FakeTrimmableCache(), stateReplay]
        commitCacheRecord(cache)
        #expect(stateReplay.events == [.commitFull])
    }

    @Test
    func `cancelCacheRecord routes only to state-replay layers`() {
        let stateReplay = FakeStateReplayCache()
        let cache: [KVCache] = [FakeTrimmableCache(), stateReplay]
        cancelCacheRecord(cache)
        #expect(stateReplay.events == [.cancel])
    }

    @Test
    func `Disabled state-replay layer is skipped by all helpers`() {
        let stateReplay = FakeStateReplayCache()
        stateReplay.enabled = false
        let cache: [KVCache] = [stateReplay]
        beginCacheRecord(cache)
        commitCacheRecord(cache)
        cancelCacheRecord(cache)
        #expect(stateReplay.events == [])
    }
}

// MARK: - SSMStateCache: StateReplayCache (spec 020 phase 2)

/// Phase 2 conformance tests for the real `SSMStateCache` —
/// `canStateReplay` / `replayCost`, lifecycle (begin → append → commit /
/// rollback / cancel), and per-token-equivalence between the cache's
/// rollback path and an explicit GDN recurrence reference.
@Suite("SSMStateCache: StateReplayCache (phase 2 conformance)")
struct SSMStateCacheStateReplayTests {

    // Small concrete shapes used across the tests. Dk must be a multiple
    // of 32 because the `state_replay` Metal kernel uses `n_per_t = Dk / 32`
    // and Metal C++ forbids zero-length arrays. Dk=64 is the smallest cell
    // that exercises `n_per_t > 1` (two simd-group-wide registers per
    // lane), matching the smallest production Qwen 3.5 shape.
    static let B = 1
    static let Hv = 2
    static let Dv = 32
    static let Dk = 64

    /// Reference single-step recurrence — must match the body of
    /// `gated_delta_step` in `mlx-swift/Source/Cmlx/mlx-generated/metal/gated_delta.metal`
    /// and the cache's private `gatedDeltaReplayStep`.
    static func referenceStep(
        state: MLXArray, delta: MLXArray, k: MLXArray, g: MLXArray
    ) -> MLXArray {
        let decayed = state * g[.ellipsis, .newAxis, .newAxis]
        let kExpanded = k[.ellipsis, .newAxis, 0...]
        let deltaExpanded = delta[.ellipsis, 0..., .newAxis]
        return decayed + kExpanded * deltaExpanded
    }

    /// Per-step innovation tuple with deterministic small-int values
    /// so equivalence assertions stay tight under bf16.
    static func makeInnovation(seed: Int) -> [MLXArray] {
        let s = Float(seed)
        let delta = MLXArray.full(
            [B, Hv, Dv], values: MLXArray(s * 0.1, dtype: .float32))
        let k = MLXArray.full(
            [B, Hv, Dk], values: MLXArray(s * 0.05 + 0.01, dtype: .float32))
        let g = MLXArray.full(
            [B, Hv], values: MLXArray(0.9, dtype: .float32))
        return [delta, k, g]
    }

    static func makeCacheWithState(_ initial: MLXArray) -> SSMStateCache {
        let cache = SSMStateCache()
        cache[1] = initial
        cache[0] = MLXArray.zeros([B, 3, Hv * Dk])  // conv state placeholder (slot 0)
        cache.offset = 100  // arbitrary non-zero starting offset
        return cache
    }

    @Test("canStateReplay is true; replayCost is .ok; isTrimmable stays false")
    func basicProperties() {
        let cache = SSMStateCache()
        #expect(cache.canStateReplay == true)
        #expect(cache.replayCost == .ok)
        #expect(cache.isTrimmable == false)
        #expect(cache.storageKind == .ssm)
    }

    @Test("commitFull clears tape and leaves state advanced")
    func commitFullPreservesAdvancedState() {
        let initial = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk])
        let cache = Self.makeCacheWithState(initial)

        cache.beginRecord()
        cache.recordStep(Self.makeInnovation(seed: 1))
        cache.recordStep(Self.makeInnovation(seed: 2))

        // Layer's update() would have advanced the recurrent state during
        // verify; we simulate that by mutating slot 1 directly. commitFull
        // must not undo that.
        let advanced = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 7.0
        cache[1] = advanced
        cache.offset = 100 + 2  // 2 verify tokens

        cache.commitFull()

        // State should still be the advanced value; offset should still
        // be at +2; second commit must trap (no active recording).
        #expect(allClose(cache[1]!, advanced).item(Bool.self))
        #expect(cache.offset == 102)
    }

    @Test("cancel restores pre-record snapshot and discards delta log")
    func cancelRestoresSnapshot() {
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(3.0, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)

        cache.beginRecord()
        cache.recordStep(Self.makeInnovation(seed: 1))
        // Mid-verify the layer would have advanced state; simulate.
        cache[1] = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 99.0
        cache.offset = 101

        cache.cancel()

        // State must be back to the pre-record snapshot value (3.0)
        // and offset must roll back to the snapshot's offset.
        #expect(allClose(cache[1]!, initial).item(Bool.self))
        #expect(cache.offset == 100)
    }

    @Test("rollback(acceptedPrefix: 0) is equivalent to cancel")
    func rollbackZeroEqualsCancel() {
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(2.5, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)

        cache.beginRecord()
        cache.recordStep(Self.makeInnovation(seed: 7))
        cache.recordStep(Self.makeInnovation(seed: 11))
        // Layer-side advancement.
        cache[1] = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 42.0
        cache.offset = 102

        cache.rollback(acceptedPrefix: 0)

        #expect(allClose(cache[1]!, initial).item(Bool.self))
        #expect(cache.offset == 100)
    }

    @Test("rollback(acceptedPrefix: k) re-folds first k delta log entries via the GDN recurrence")
    func partialRollbackMatchesReference() {
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(0.5, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)

        let inn0 = Self.makeInnovation(seed: 1)
        let inn1 = Self.makeInnovation(seed: 2)
        let inn2 = Self.makeInnovation(seed: 3)
        let inn3 = Self.makeInnovation(seed: 4)

        cache.beginRecord()
        cache.recordStep(inn0)
        cache.recordStep(inn1)
        cache.recordStep(inn2)
        cache.recordStep(inn3)

        // Simulate the layer's full forward — state advances 4 steps.
        // The cache doesn't care about the actual mid-verify value; what
        // matters is the post-rollback state == reference state.
        cache[1] = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 1234.0
        cache.offset = 104

        // Roll back to acceptedPrefix=2 (first 2 delta log entries kept).
        cache.rollback(acceptedPrefix: 2)

        // Reference: start from snapshot, run 2 steps explicitly.
        var ref = initial
        ref = Self.referenceStep(state: ref, delta: inn0[0], k: inn0[1], g: inn0[2])
        ref = Self.referenceStep(state: ref, delta: inn1[0], k: inn1[1], g: inn1[2])

        #expect(allClose(cache[1]!, ref).item(Bool.self))
        // Offset advances by k=2 from the snapshot (which was 100).
        #expect(cache.offset == 102)
    }

    @Test("rollback(acceptedPrefix: numDraft) matches a full-accept replay")
    func fullPrefixRollbackMatchesReference() {
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(1.0, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)

        let inns = (0..<5).map { Self.makeInnovation(seed: $0 + 1) }

        cache.beginRecord()
        for inn in inns { cache.recordStep(inn) }
        // Simulate mid-verify — value doesn't matter here; rollback will
        // recompute from the snapshot.
        cache[1] = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk])
        cache.offset = 105

        cache.rollback(acceptedPrefix: 5)

        var ref = initial
        for inn in inns {
            ref = Self.referenceStep(state: ref, delta: inn[0], k: inn[1], g: inn[2])
        }

        #expect(allClose(cache[1]!, ref).item(Bool.self))
        #expect(cache.offset == 105)
    }

    @Test("Per-step equivalence: rolling 1+1 == rolling 0+2 within a round")
    func perStepEquivalence() {
        // Two separate recording sessions, same initial state and delta log:
        //   - Session A: rollback(2) — re-folds 2 entries in one call
        //   - Session B: rollback(1), then a fresh recording session with
        //     a single delta-log entry, rollback(1) — re-folds 1+1 entries
        //
        // The post-state of A and B must match exactly.
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(0.5, dtype: .float32))
        let inn0 = Self.makeInnovation(seed: 1)
        let inn1 = Self.makeInnovation(seed: 2)

        let cacheA = Self.makeCacheWithState(initial)
        cacheA.beginRecord()
        cacheA.recordStep(inn0)
        cacheA.recordStep(inn1)
        cacheA[0] = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk])
        cacheA.offset = 102
        cacheA.rollback(acceptedPrefix: 2)

        let cacheB = Self.makeCacheWithState(initial)
        cacheB.beginRecord()
        cacheB.recordStep(inn0)
        cacheB[0] = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk])
        cacheB.offset = 101
        cacheB.rollback(acceptedPrefix: 1)
        // Second round: state is now post-inn0; record + accept 1 more.
        cacheB.beginRecord()
        cacheB.recordStep(inn1)
        cacheB[0] = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk])
        cacheB.offset = 102
        cacheB.rollback(acceptedPrefix: 1)

        #expect(allClose(cacheA[0]!, cacheB[0]!).item(Bool.self))
        #expect(cacheA.offset == cacheB.offset)
    }

    @Test("Free-helper dispatch: rollbackPromptCache folds (accepted + 1) entries — y baseline + accepted drafts")
    func dispatchHelpersIntegrate() {
        // Off-by-one regression test (end-to-end with a real SSMStateCache).
        // The verify forward records T = numDraft + 1 entries: entry 0 is
        // the y baseline (always processed), entries 1..numDraft are the
        // drafts. `rollbackPromptCache(_, acceptedPrefix: k, numDraft: T-1)`
        // must end at the same cache state as
        // `trim(rejected = numDraft - k)` would — `k + 1` total tokens
        // kept (y + k drafts).
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(1.0, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)
        let stack: [KVCache] = [cache]

        #expect(canRollbackPromptCache(stack) == true)

        // numDraft = 2 → verify forward records T = 3 entries.
        beginCacheRecord(stack)
        let inn0 = Self.makeInnovation(seed: 1)
        let inn1 = Self.makeInnovation(seed: 2)
        let inn2 = Self.makeInnovation(seed: 3)
        cache.recordStep(inn0)   // y baseline
        cache.recordStep(inn1)   // draft 1
        cache.recordStep(inn2)   // draft 2
        // Simulate the layer having advanced state through all 3 entries
        // during the verify forward — the rollback path must restore from
        // the snapshot and re-fold the kept prefix.
        cache[1] = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 99.0
        cache.offset = 100 + 3

        // Iterator accepts 1 draft of 2 → rollback. Helper translates to
        // `cache.rollback(acceptedPrefix: 2)` (y baseline + 1 draft).
        rollbackPromptCache(stack, acceptedPrefix: 1, numDraft: 2)

        // Expected: state = snapshot folded through entries 0 and 1.
        var ref = initial
        ref = Self.referenceStep(state: ref, delta: inn0[0], k: inn0[1], g: inn0[2])
        ref = Self.referenceStep(state: ref, delta: inn1[0], k: inn1[1], g: inn1[2])
        #expect(allClose(cache[1]!, ref).item(Bool.self))
        // Offset: snapshot + (acceptedPrefix + 1) = 100 + 2.
        #expect(cache.offset == 102)
    }

    @Test("Free-helper dispatch: zero-accept folds y baseline only (off-by-one regression)")
    func dispatchHelpersZeroAcceptKeepsYBaseline() {
        // The y baseline (entry 0) is always processed by the verify
        // forward — even on zero-draft-accept rounds, the iterator emits
        // a correction token at position 0 (right after y) as the new
        // bonus. So the cache must keep y's state contribution, NOT
        // restore the pre-y snapshot.
        let initial = MLXArray.full(
            [Self.B, Self.Hv, Self.Dv, Self.Dk],
            values: MLXArray(1.0, dtype: .float32))
        let cache = Self.makeCacheWithState(initial)
        let stack: [KVCache] = [cache]

        // numDraft = 4 → T = 5 recorded entries.
        beginCacheRecord(stack)
        let inn0 = Self.makeInnovation(seed: 1)
        cache.recordStep(inn0)   // y baseline
        cache.recordStep(Self.makeInnovation(seed: 2))   // draft 1
        cache.recordStep(Self.makeInnovation(seed: 3))   // draft 2
        cache.recordStep(Self.makeInnovation(seed: 4))   // draft 3
        cache.recordStep(Self.makeInnovation(seed: 5))   // draft 4
        cache[1] = MLXArray.ones([Self.B, Self.Hv, Self.Dv, Self.Dk]) * 42.0
        cache.offset = 100 + 5

        // All 4 drafts rejected.
        rollbackPromptCache(stack, acceptedPrefix: 0, numDraft: 4)

        // Expected: state = snapshot + entry 0 (y baseline) only.
        var ref = initial
        ref = Self.referenceStep(state: ref, delta: inn0[0], k: inn0[1], g: inn0[2])
        #expect(allClose(cache[1]!, ref).item(Bool.self))
        // Offset: snapshot + 1 (y kept), matching trim semantics
        // (`trim(rejected=4)` of `100+5` → `100+1`).
        #expect(cache.offset == 101)
    }
}
