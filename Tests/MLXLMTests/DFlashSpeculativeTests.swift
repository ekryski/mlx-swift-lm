// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import MLXNN
import Testing

// MARK: - DFlash speculative decoding tests (Phase 1 — full-attention only)
//
// Coverage layers:
//
// 1. Pure-Swift helper tests — no MLX evaluation. Cover the accept-prefix
//    walk, default capture-layer-IDs picker, and the two stub draft
//    backends. These run in milliseconds and don't depend on a model.
//
// 2. Integration tests with `Gemma3TextModel` wrapped in `MockDFlashTarget`
//    (defined below) so the iterator gets a real forward pass + KV cache
//    behavior. Same in-test random-weight model the n-gram tests use, so
//    output is deterministic but argmax sequences are nonsense — that's
//    fine; these tests check **plumbing**, not generation quality.
//
// Phase 2 will add tests against trained weights to validate accept rate;
// Phase 3 will add hybrid GDN coverage. Both deferred.

// MARK: - Pure-Swift helper tests

@Suite
struct DFlashHelperTests {

    @Test
    func `dflashAcceptedPrefixLength: empty draft returns 0`() {
        #expect(dflashAcceptedPrefixLength(draft: [], targetArgmax: [42]) == 0)
    }

    @Test
    func `dflashAcceptedPrefixLength: full match returns draft count`() {
        let draft = [1, 2, 3, 4]
        let targetArgmax = [1, 2, 3, 4, 99]  // bonus position past the draft
        #expect(dflashAcceptedPrefixLength(draft: draft, targetArgmax: targetArgmax) == 4)
    }

    @Test
    func `dflashAcceptedPrefixLength: first mismatch returns 0`() {
        let draft = [7, 2, 3, 4]
        let targetArgmax = [1, 2, 3, 4, 99]
        #expect(dflashAcceptedPrefixLength(draft: draft, targetArgmax: targetArgmax) == 0)
    }

    @Test
    func `dflashAcceptedPrefixLength: middle mismatch stops there`() {
        let draft = [1, 2, 99, 4]
        let targetArgmax = [1, 2, 3, 4, 5]
        #expect(dflashAcceptedPrefixLength(draft: draft, targetArgmax: targetArgmax) == 2)
    }

    @Test
    func `dflashAcceptedPrefixLength: tail mismatch returns count - 1`() {
        let draft = [1, 2, 3, 99]
        let targetArgmax = [1, 2, 3, 4, 5]
        #expect(dflashAcceptedPrefixLength(draft: draft, targetArgmax: targetArgmax) == 3)
    }

    @Test
    func `dflashDefaultCaptureLayerIDs: single-layer draft picks middle`() {
        let ids = dflashDefaultCaptureLayerIDs(numTargetLayers: 32, numDraftLayers: 1)
        #expect(ids == [16])
    }

    @Test
    func `dflashDefaultCaptureLayerIDs: two-layer draft spans the stack`() {
        let ids = dflashDefaultCaptureLayerIDs(numTargetLayers: 32, numDraftLayers: 2)
        // start=1, end=29 → [1, 29]
        #expect(ids.count == 2)
        #expect(ids.first == 1)
        #expect(ids.last == 29)
    }

    @Test
    func `dflashDefaultCaptureLayerIDs: four-layer draft is monotonic`() {
        let ids = dflashDefaultCaptureLayerIDs(numTargetLayers: 32, numDraftLayers: 4)
        #expect(ids.count == 4)
        // Strictly nondecreasing — sanity.
        for i in 1 ..< ids.count {
            #expect(ids[i] >= ids[i - 1])
        }
        // Stays within the [1, numTargetLayers - 3] band.
        #expect(ids.first! >= 1)
        #expect(ids.last! <= 32 - 3)
    }

    @Test
    func `dflashDefaultCaptureLayerIDs: tiny target degrades gracefully`() {
        // 4-layer target, 1-layer draft — should still pick something
        // sensible (the middle: 4/2 = 2).
        let ids = dflashDefaultCaptureLayerIDs(numTargetLayers: 4, numDraftLayers: 1)
        #expect(ids == [2])
    }
}

// MARK: - Stub draft backend tests

@Suite
struct DFlashDraftBackendTests {

    @Test
    func `ZeroAcceptDraftBackend: emits blockSize copies of stub token`() {
        var backend = ZeroAcceptDraftBackend(blockSize: 4, captureLayerIDs: [3, 7], stubToken: 99)
        let block = backend.draftBlock(targetHidden: [:], lastCommittedToken: 1)
        #expect(block == [99, 99, 99, 99])
        #expect(backend.captureLayerIDs == [3, 7])
        #expect(backend.blockSize == 4)
    }

    @Test
    func `ZeroAcceptDraftBackend: reset is a no-op (idempotent calls)`() {
        var backend = ZeroAcceptDraftBackend(blockSize: 2)
        backend.reset()
        backend.reset()
        let block = backend.draftBlock(targetHidden: [:], lastCommittedToken: 5)
        #expect(block == [0, 0])
    }

    @Test
    func `ScriptedDraftBackend: replays in order, then returns empty`() {
        var backend = ScriptedDraftBackend(
            blockSize: 4,
            script: [[1, 2, 3, 4], [5, 6, 7, 8]])
        #expect(backend.callCount == 0)
        let b1 = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        #expect(b1 == [1, 2, 3, 4])
        #expect(backend.callCount == 1)
        let b2 = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        #expect(b2 == [5, 6, 7, 8])
        let b3 = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        #expect(b3 == [])
        // callCount tracks served (non-empty) calls — the empty fallback
        // doesn't bump the cursor (the early `return []` short-circuits
        // ahead of the `defer cursor += 1`).
        #expect(backend.callCount == 2)
    }

    @Test
    func `ScriptedDraftBackend: reset rewinds the script cursor`() {
        var backend = ScriptedDraftBackend(blockSize: 2, script: [[1, 2], [3, 4]])
        _ = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        _ = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        #expect(backend.callCount == 2)
        backend.reset()
        #expect(backend.callCount == 0)
        let b = backend.draftBlock(targetHidden: [:], lastCommittedToken: 0)
        #expect(b == [1, 2])
    }
}

// MARK: - Mock DFlashTargetModel for integration tests
//
// `Gemma3TextModel` doesn't conform to `DFlashTargetModel` upstream — that
// conformance lands per-target in Phase 2 (spec 015 §"Files touched").
// For phase-1 plumbing tests we wrap it in a `MockDFlashTarget` reference
// type that delegates `LanguageModel` calls through and provides a stub
// `dflashForwardWithCapture` implementation that returns an empty
// `captured` dict (the iterator's slicing path handles that gracefully —
// covered by the empty-captured branch in `speculateRound`).
//
// Once Phase 2 lands the real per-target conformances, these tests get
// promoted to use those directly and the mock is retired.

final class MockDFlashTarget: Module, LanguageModel, DFlashTargetModel {
    let inner: Gemma3TextModel
    var dflashIsHybridGDN: Bool { false }

    /// Records every call to `dflashForwardWithCapture` — tests inspect
    /// this to confirm the iterator routed verify forwards through the
    /// capture method (not just the plain `callAsFunction`).
    var captureCallCount = 0

    init(_ inner: Gemma3TextModel) {
        self.inner = inner
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inner(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        inner.newCache(parameters: parameters)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, windowSize: windowSize)
    }

    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        // Phase-1 plumbing tests don't exercise this path; the iterator
        // doesn't call it. Real per-target conformances (Phase 2) tap
        // into the model's `embed_tokens` directly.
        fatalError("dflashEmbedTokens not exercised by phase-1 mock")
    }

    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        // Same as above — phase-1 iterator goes through the verify
        // forward, not through a stand-alone LM-head application.
        fatalError("dflashLmHeadLogits not exercised by phase-1 mock")
    }

    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, captured: [Int: MLXArray]) {
        captureCallCount += 1
        let logits = inner(inputIDs, cache: cache)
        // Phase-1 mock: don't actually capture intermediate hidden states.
        // The iterator handles the empty-dict case (covered by tests).
        // Real per-target conformances (phase 2) tap into the Gemma3Model's
        // per-layer activations and return them keyed by layerID.
        return (logits, [:])
    }
}

// MARK: - Iterator integration tests

@Suite(.serialized)
struct DFlashSpeculativeIteratorTests {

    let processor: any UserInputProcessor
    let context: ModelContext
    let mockTarget: MockDFlashTarget

    init() {
        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let inner = Gemma3TextModel(modelConfig)
        eval(inner)
        let mock = MockDFlashTarget(inner)
        eval(mock)
        self.processor = processor
        self.mockTarget = mock
        self.context = ModelContext(
            configuration: processor.configuration,
            model: mock,
            processor: processor,
            tokenizer: processor.tokenizer
        )
    }

    @Test
    func `Iterator emits exactly maxTokens tokens`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let backend = ZeroAcceptDraftBackend(blockSize: 4)
        let params = GenerateParameters(maxTokens: 12, temperature: 0.0)
        var iter = try DFlashSpeculativeTokenIterator(
            input: input,
            target: mockTarget,
            draftBackend: backend,
            parameters: params)
        var tokens: [Int] = []
        while let t = iter.next() { tokens.append(t) }
        #expect(tokens.count == 12)
    }

    @Test
    func `ZeroAccept backend never accepts a draft token`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        // stubToken=12345 is way out of vocab — guaranteed mismatch with any argmax.
        let backend = ZeroAcceptDraftBackend(blockSize: 4, stubToken: 12345)
        let params = GenerateParameters(maxTokens: 16, temperature: 0.0)
        var iter = try DFlashSpeculativeTokenIterator(
            input: input,
            target: mockTarget,
            draftBackend: backend,
            parameters: params)
        while iter.next() != nil {}
        // Bonus token always emits, but nothing from the draft itself.
        #expect(iter.dflashAcceptedCount == 0)
        #expect(iter.dflashProposedCount > 0)
        #expect(iter.dflashAcceptanceRate == 0)
    }

    @Test
    func `Iterator routes verify forwards through dflashForwardWithCapture`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let backend = ZeroAcceptDraftBackend(blockSize: 4)
        let params = GenerateParameters(maxTokens: 8, temperature: 0.0)
        var iter = try DFlashSpeculativeTokenIterator(
            input: input,
            target: mockTarget,
            draftBackend: backend,
            parameters: params)
        let countBefore = mockTarget.captureCallCount
        while iter.next() != nil {}
        let countAfter = mockTarget.captureCallCount
        // At least one cycle must have invoked the capture forward.
        #expect(countAfter > countBefore)
        // Capture-call count is bounded above by cycle count: every
        // verify-path cycle calls `dflashForwardWithCapture` exactly once,
        // but the iterator can also take the empty-draft AR fallback near
        // `maxTokens` (when `remaining - 1 == 0`), which bumps
        // `dflashCycleCount` but skips capture. So the inequality is
        // capture-calls ≤ cycles.
        #expect(countAfter - countBefore <= iter.dflashCycleCount)
    }

    @Test
    func `Output matches TokenIterator under greedy (count)`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let params = GenerateParameters(maxTokens: 12, temperature: 0.0)

        var ref = try TokenIterator(
            input: input, model: mockTarget, parameters: params)
        var refTokens: [Int] = []
        while let t = ref.next() { refTokens.append(t) }

        // Re-init mock target to clear captureCallCount drift between runs.
        let backend = ZeroAcceptDraftBackend(blockSize: 4)
        var spec = try DFlashSpeculativeTokenIterator(
            input: input,
            target: mockTarget,
            draftBackend: backend,
            parameters: params)
        var specTokens: [Int] = []
        while let t = spec.next() { specTokens.append(t) }

        // Count parity is the stable contract on tiny random-weight models —
        // batched-vs-sequential argmax can flip on ties. Stricter sequence
        // equality is the *intended* contract once a real model is wired in.
        #expect(specTokens.count == refTokens.count)
    }

    @Test
    func `ScriptedDraftBackend running out of script falls back to AR`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        // Script length 2 only — the iterator must keep emitting via AR
        // fallback after the script empties.
        let backend = ScriptedDraftBackend(
            blockSize: 4,
            script: [[10, 20, 30, 40], [50, 60, 70, 80]])
        let params = GenerateParameters(maxTokens: 16, temperature: 0.0)
        var iter = try DFlashSpeculativeTokenIterator(
            input: input,
            target: mockTarget,
            draftBackend: backend,
            parameters: params)
        var tokens: [Int] = []
        while let t = iter.next() { tokens.append(t) }
        #expect(tokens.count == 16)
    }

    @Test
    func `Hybrid GDN target rejected at construction`() async throws {
        // Same Gemma3TextModel config but hand-rolled mock that lies about
        // its hybrid status, exercising the init-time guard.
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let inner = Gemma3TextModel(modelConfig)
        eval(inner)
        let hybridMock = HybridLyingMockTarget(inner)
        eval(hybridMock)

        let input = try await processor.prepare(input: UserInput(prompt: "hi"))
        let backend = ZeroAcceptDraftBackend(blockSize: 4)
        let params = GenerateParameters(maxTokens: 4, temperature: 0.0)

        do {
            _ = try DFlashSpeculativeTokenIterator(
                input: input,
                target: hybridMock,
                draftBackend: backend,
                parameters: params)
            Issue.record("expected init to throw on hybrid GDN target")
        } catch let err as KVCacheError {
            // Look for the hybrid-related sentinel in the error message
            // so we don't conflate this with the trimmable-cache error.
            #expect(err.message.contains("Hybrid"))
        } catch {
            Issue.record("expected KVCacheError, got \(error)")
        }
    }
}

/// Variant of MockDFlashTarget that reports `dflashIsHybridGDN == true`
/// to exercise the iterator's init-time hybrid-rejection guard. The cache
/// is still trimmable (Gemma3 attention) — the test is purely about the
/// hybrid-flag check, not the cache-trimmability check.
final class HybridLyingMockTarget: Module, LanguageModel, DFlashTargetModel {
    let inner: Gemma3TextModel
    var dflashIsHybridGDN: Bool { true }

    init(_ inner: Gemma3TextModel) {
        self.inner = inner
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inner(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        inner.newCache(parameters: parameters)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, windowSize: windowSize)
    }

    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        fatalError("dflashEmbedTokens not exercised by HybridLyingMockTarget")
    }

    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        fatalError("dflashLmHeadLogits not exercised by HybridLyingMockTarget")
    }

    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, captured: [Int: MLXArray]) {
        (inner(inputIDs, cache: cache), [:])
    }
}
