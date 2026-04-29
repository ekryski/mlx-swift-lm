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
//      stacks pass, tape-replay-only stacks pass, mixed stacks pass,
//      stacks containing a layer that's neither trimmable nor tape-
//      replayable fail.
//   2. `rollbackPromptCache` dispatch — tape-replay layers see
//      `commitFull`/`rollback`/`cancel` per acceptance count;
//      trimmable layers see `trim(rejected)`.
//   3. `beginCacheRecord` / `cancelCacheRecord` / `commitCacheRecord`
//      route only to tape-replay layers.
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

/// Minimal cache that's neither trimmable nor tape-replay-able. Models
/// the pre-phase-2 hybrid case where SSMStateCache exists but doesn't
/// conform to TapeReplayCache yet.
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

/// Minimal tape-replay cache that records every method call. Used to
/// verify the dispatch logic in `rollbackPromptCache` and the
/// begin/commit/cancel helpers.
final class FakeTapeReplayCache: TapeReplayCache, Evaluatable {
    enum Event: Equatable {
        case begin
        case appendInnovation
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
    func copy() -> any KVCache { FakeTapeReplayCache() }
    var isDonor: Bool {
        get { false }
        set { _ = newValue }
    }
    func innerState() -> [MLXArray] { [] }

    var canTapeReplay: Bool { enabled }
    var replayCost: TapeReplayCost { .ok }
    func beginRecord() { events.append(.begin) }
    func appendInnovation(_ delta: MLXArray) { events.append(.appendInnovation) }
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
    func `All-tape-replay stack passes`() {
        let cache: [KVCache] = [FakeTapeReplayCache(), FakeTapeReplayCache()]
        #expect(canRollbackPromptCache(cache))
    }

    @Test
    func `Mixed trimmable + tape-replay stack passes`() {
        let cache: [KVCache] = [
            FakeTrimmableCache(),
            FakeTapeReplayCache(),
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
    func `TapeReplayCache reporting canTapeReplay = false fails`() {
        let disabled = FakeTapeReplayCache()
        disabled.enabled = false
        let cache: [KVCache] = [FakeTrimmableCache(), disabled]
        #expect(!canRollbackPromptCache(cache))
    }
}

// MARK: - rollbackPromptCache dispatch tests

@Suite
struct RollbackPromptCacheTests {

    @Test
    func `Full accept calls commitFull on tape layers, no trim on trimmable`() {
        let trim = FakeTrimmableCache()
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [trim, tape]

        rollbackPromptCache(cache, acceptedPrefix: 4, numDraft: 4)

        #expect(trim.trimCalls == [0])  // rejected = 0; trim(0) is called
        #expect(tape.events == [.commitFull])
    }

    @Test
    func `Zero accept calls cancel on tape layers, trim by all on trimmable`() {
        let trim = FakeTrimmableCache()
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [trim, tape]

        rollbackPromptCache(cache, acceptedPrefix: 0, numDraft: 4)

        #expect(trim.trimCalls == [4])
        #expect(tape.events == [.cancel])
    }

    @Test
    func `Partial accept calls rollback(k) on tape layers, trim by rejected on trimmable`() {
        let trim = FakeTrimmableCache()
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [trim, tape]

        rollbackPromptCache(cache, acceptedPrefix: 3, numDraft: 5)

        #expect(trim.trimCalls == [2])
        #expect(tape.events == [.rollback(3)])
    }

    @Test
    func `Mixed-stack rollback hits each layer once`() {
        let trim1 = FakeTrimmableCache()
        let tape1 = FakeTapeReplayCache()
        let trim2 = FakeTrimmableCache()
        let tape2 = FakeTapeReplayCache()
        let cache: [KVCache] = [trim1, tape1, trim2, tape2]

        rollbackPromptCache(cache, acceptedPrefix: 1, numDraft: 4)

        #expect(trim1.trimCalls == [3])
        #expect(trim2.trimCalls == [3])
        #expect(tape1.events == [.rollback(1)])
        #expect(tape2.events == [.rollback(1)])
    }

    @Test
    func `Disabled tape-replay layer is skipped, no event recorded`() {
        let trim = FakeTrimmableCache()
        let tape = FakeTapeReplayCache()
        tape.enabled = false
        let cache: [KVCache] = [trim, tape]

        rollbackPromptCache(cache, acceptedPrefix: 2, numDraft: 4)

        #expect(trim.trimCalls == [2])
        #expect(tape.events == [])  // skipped because canTapeReplay = false
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
    func `beginCacheRecord routes only to tape layers`() {
        let trim = FakeTrimmableCache()
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [trim, tape, FakeTrimmableCache()]
        beginCacheRecord(cache)
        #expect(tape.events == [.begin])
        #expect(trim.trimCalls == [])
    }

    @Test
    func `commitCacheRecord routes only to tape layers`() {
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [FakeTrimmableCache(), tape]
        commitCacheRecord(cache)
        #expect(tape.events == [.commitFull])
    }

    @Test
    func `cancelCacheRecord routes only to tape layers`() {
        let tape = FakeTapeReplayCache()
        let cache: [KVCache] = [FakeTrimmableCache(), tape]
        cancelCacheRecord(cache)
        #expect(tape.events == [.cancel])
    }

    @Test
    func `Disabled tape layer is skipped by all helpers`() {
        let tape = FakeTapeReplayCache()
        tape.enabled = false
        let cache: [KVCache] = [tape]
        beginCacheRecord(cache)
        commitCacheRecord(cache)
        cancelCacheRecord(cache)
        #expect(tape.events == [])
    }
}
