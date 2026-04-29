// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import MLXNN
import Testing

// MARK: - Mirror Speculative Decoding tests (spec 021 phase 1A scaffold)
//
// Three suites:
//   1. `ANEDraftRegistryTests` — pure-Swift register/lookup behaviour.
//   2. `ANEVocabEquivalenceTests` — size + disagreement gate at the
//      llama.cpp-default boundaries.
//   3. `ANEDraftBackendStubTests` — `ScriptedANEDraftBackend` and
//      `ZeroAcceptANEDraftBackend` semantics.
//   4. `MirrorSpeculativeIteratorTests` — integration with
//      `Gemma3TextModel`. Exercises emit count, cycle bookkeeping,
//      vocab-gate rejection, AR fallback, and count-parity vs.
//      `TokenIterator`.

// MARK: - Registry tests

@Suite
struct ANEDraftRegistryTests {

    @Test
    func `Registry default is empty`() {
        let reg = ANEDraftRegistry()
        #expect(reg.count == 0)
        #expect(reg.isRegistered(targetID: "anything") == false)
        #expect(reg.entry(for: "anything") == nil)
    }

    @Test
    func `Register, lookup, remove round-trips`() {
        let reg = ANEDraftRegistry()
        let entry = ANEDraftRegistryEntry(
            targetID: "Qwen3.5-27B-Instruct-4bit",
            draftBundleURL: URL(fileURLWithPath: "/tmp/draft.mlpackage"),
            draftLength: 4,
            expectedTokenizerID: "Qwen/Qwen3.5-27B-Instruct")
        reg.register(entry)
        #expect(reg.count == 1)
        #expect(reg.isRegistered(targetID: "Qwen3.5-27B-Instruct-4bit"))
        #expect(reg.entry(for: "Qwen3.5-27B-Instruct-4bit") == entry)

        reg.remove(targetID: "Qwen3.5-27B-Instruct-4bit")
        #expect(reg.count == 0)
        #expect(reg.entry(for: "Qwen3.5-27B-Instruct-4bit") == nil)
    }

    @Test
    func `Re-register replaces the previous entry`() {
        let reg = ANEDraftRegistry()
        let v1 = ANEDraftRegistryEntry(
            targetID: "X", draftBundleURL: URL(fileURLWithPath: "/v1"),
            draftLength: 4, expectedTokenizerID: "tok")
        let v2 = ANEDraftRegistryEntry(
            targetID: "X", draftBundleURL: URL(fileURLWithPath: "/v2"),
            draftLength: 8, expectedTokenizerID: "tok")
        reg.register(v1)
        reg.register(v2)
        #expect(reg.count == 1)
        #expect(reg.entry(for: "X")?.draftBundleURL.path == "/v2")
        #expect(reg.entry(for: "X")?.draftLength == 8)
    }

    @Test
    func `registeredTargetIDs returns all registered IDs`() {
        let reg = ANEDraftRegistry()
        for id in ["A", "B", "C"] {
            reg.register(ANEDraftRegistryEntry(
                targetID: id,
                draftBundleURL: URL(fileURLWithPath: "/tmp/\(id)"),
                draftLength: 4,
                expectedTokenizerID: "tok"))
        }
        let ids = Set(reg.registeredTargetIDs())
        #expect(ids == Set(["A", "B", "C"]))
    }
}

// MARK: - Vocabulary equivalence tests

@Suite
struct ANEVocabEquivalenceTests {

    /// Build a synthetic vocab `[0..count)` → "tok_<id>".
    private static func mkVocab(_ count: Int, mutate: (inout [Int: String]) -> Void = { _ in }) -> [Int: String] {
        var v: [Int: String] = [:]
        for i in 0 ..< count { v[i] = "tok_\(i)" }
        mutate(&v)
        return v
    }

    @Test
    func `Identical vocabularies pass`() {
        let v = Self.mkVocab(1000)
        let r = aneCheckVocabEquivalence(draftVocab: v, targetVocab: v)
        #expect(r.isCompatible)
        #expect(r.disagreementCount == 0)
        #expect(r.sizeDifference == 0)
    }

    @Test
    func `Disagreement above zero threshold rejects (default policy)`() {
        var draft = Self.mkVocab(1000)
        let target = Self.mkVocab(1000)
        draft[100] = "different_string"
        let r = aneCheckVocabEquivalence(draftVocab: draft, targetVocab: target)
        #expect(!r.isCompatible)
        #expect(r.disagreementCount == 1)
    }

    @Test
    func `Disagreement under custom threshold passes`() {
        var draft = Self.mkVocab(1000)
        let target = Self.mkVocab(1000)
        draft[100] = "different_string"
        let r = aneCheckVocabEquivalence(
            draftVocab: draft, targetVocab: target, maxDisagreements: 1)
        #expect(r.isCompatible)
        #expect(r.disagreementCount == 1)
    }

    @Test
    func `Tokens below startTokenID are skipped`() {
        var draft = Self.mkVocab(1000)
        let target = Self.mkVocab(1000)
        // Disagree on tokens 0..<5 (BOS/EOS/etc.) — should not count.
        for i in 0 ..< 5 { draft[i] = "alt_\(i)" }
        let r = aneCheckVocabEquivalence(draftVocab: draft, targetVocab: target)
        #expect(r.isCompatible)
        #expect(r.disagreementCount == 0)
    }

    @Test
    func `Size difference above gate rejects without walking tokens`() {
        let draft = Self.mkVocab(1000)
        let target = Self.mkVocab(2000)
        let r = aneCheckVocabEquivalence(
            draftVocab: draft, targetVocab: target,
            maxSizeDifference: 128)
        #expect(!r.isCompatible)
        #expect(r.sizeDifference == 1000)
        #expect(r.comparedTokenCount == 0)  // bailed out early
    }

    @Test
    func `Size difference exactly at gate passes`() {
        let draft = Self.mkVocab(1000)
        let target = Self.mkVocab(1128)
        let r = aneCheckVocabEquivalence(
            draftVocab: draft, targetVocab: target,
            maxSizeDifference: 128)
        #expect(r.isCompatible)
        #expect(r.sizeDifference == 128)
    }

    @Test
    func `One-side-only token counts as disagreement`() {
        var draft = Self.mkVocab(1000)
        var target = Self.mkVocab(1000)
        draft[500] = nil  // hole on draft side, target still has it
        target[500] = "present"
        let r = aneCheckVocabEquivalence(draftVocab: draft, targetVocab: target)
        #expect(!r.isCompatible)
        #expect(r.disagreementCount >= 1)
    }

    @Test
    func `Sparse-on-both-sides hole is not a disagreement`() {
        var draft = Self.mkVocab(1000)
        var target = Self.mkVocab(1000)
        draft[500] = nil
        target[500] = nil
        let r = aneCheckVocabEquivalence(draftVocab: draft, targetVocab: target)
        #expect(r.isCompatible)
        #expect(r.disagreementCount == 0)
    }

    @Test
    func `disagreementRate is computed correctly`() {
        var draft = Self.mkVocab(100)
        let target = Self.mkVocab(100)
        // Disagree on tokens 50, 51, 52 (above startTokenID).
        for i in 50 ..< 53 { draft[i] = "diff_\(i)" }
        let r = aneCheckVocabEquivalence(
            draftVocab: draft, targetVocab: target, maxDisagreements: 100)
        #expect(r.disagreementCount == 3)
        // comparedTokenCount = (100 - 5) = 95
        #expect(r.comparedTokenCount == 95)
        #expect(abs(r.disagreementRate - 3.0/95.0) < 1e-9)
    }
}

// MARK: - Stub backend tests

@Suite
struct ANEDraftBackendStubTests {

    @Test
    func `ScriptedANEDraftBackend replays in order then returns empty`() {
        var b = ScriptedANEDraftBackend(draftLength: 3, script: [[1, 2, 3], [4, 5, 6]])
        #expect(b.callCount == 0)
        #expect(b.draftBlock(committedTokens: [], lastCommittedToken: 0) == [1, 2, 3])
        #expect(b.draftBlock(committedTokens: [], lastCommittedToken: 0) == [4, 5, 6])
        #expect(b.draftBlock(committedTokens: [], lastCommittedToken: 0) == [])
        // Exhausted call doesn't bump cursor (early-return short-circuits
        // before defer increment) — same convention as the DFlash side.
        #expect(b.callCount == 2)
    }

    @Test
    func `ScriptedANEDraftBackend reset rewinds cursor`() {
        var b = ScriptedANEDraftBackend(draftLength: 2, script: [[1, 2], [3, 4]])
        _ = b.draftBlock(committedTokens: [], lastCommittedToken: 0)
        _ = b.draftBlock(committedTokens: [], lastCommittedToken: 0)
        #expect(b.callCount == 2)
        b.reset()
        #expect(b.callCount == 0)
        #expect(b.draftBlock(committedTokens: [], lastCommittedToken: 0) == [1, 2])
    }

    @Test
    func `ZeroAcceptANEDraftBackend emits draftLength stub tokens`() {
        var b = ZeroAcceptANEDraftBackend(draftLength: 5, stubToken: 999)
        let block = b.draftBlock(committedTokens: [1, 2, 3], lastCommittedToken: 3)
        #expect(block == [999, 999, 999, 999, 999])
    }

    @Test
    func `Default draftCacheMemoryBytes is nil`() {
        let b = ZeroAcceptANEDraftBackend(draftLength: 4)
        #expect(b.draftCacheMemoryBytes == nil)
    }
}

// MARK: - Iterator integration tests

@Suite(.serialized)
struct MirrorSpeculativeIteratorTests {

    let processor: any UserInputProcessor
    let model: Gemma3TextModel

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
        let model = Gemma3TextModel(modelConfig)
        eval(model)
        self.processor = processor
        self.model = model
    }

    @Test
    func `Iterator emits exactly maxTokens tokens`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let backend = ZeroAcceptANEDraftBackend(draftLength: 4)
        let params = GenerateParameters(maxTokens: 10, temperature: 0.0)
        var iter = try MirrorSpeculativeTokenIterator(
            input: input, target: model,
            draftBackend: backend, parameters: params)
        var tokens: [Int] = []
        while let t = iter.next() { tokens.append(t) }
        #expect(tokens.count == 10)
    }

    @Test
    func `ZeroAccept backend never accepts`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let backend = ZeroAcceptANEDraftBackend(draftLength: 4, stubToken: 12345)
        let params = GenerateParameters(maxTokens: 12, temperature: 0.0)
        var iter = try MirrorSpeculativeTokenIterator(
            input: input, target: model,
            draftBackend: backend, parameters: params)
        while iter.next() != nil {}
        #expect(iter.mirrorAcceptedCount == 0)
        #expect(iter.mirrorProposedCount > 0)
        #expect(iter.mirrorAcceptanceRate == 0)
    }

    @Test
    func `Output count parity with TokenIterator`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hello world"))
        let params = GenerateParameters(maxTokens: 10, temperature: 0.0)

        var ref = try TokenIterator(input: input, model: model, parameters: params)
        var refTokens: [Int] = []
        while let t = ref.next() { refTokens.append(t) }

        let backend = ZeroAcceptANEDraftBackend(draftLength: 4)
        var spec = try MirrorSpeculativeTokenIterator(
            input: input, target: model,
            draftBackend: backend, parameters: params)
        var specTokens: [Int] = []
        while let t = spec.next() { specTokens.append(t) }

        // Tiny random-weight model: argmax ties can flip on
        // batched-vs-sequential paths; assert count parity.
        #expect(specTokens.count == refTokens.count)
    }

    @Test
    func `Vocab equivalence gate rejects incompatible report`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hi"))
        let backend = ZeroAcceptANEDraftBackend(draftLength: 4)
        let params = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let badReport = ANEVocabEquivalenceReport(
            comparedTokenCount: 1000, disagreementCount: 50,
            sizeDifference: 0, isCompatible: false)
        do {
            _ = try MirrorSpeculativeTokenIterator(
                input: input, target: model,
                draftBackend: backend, parameters: params,
                vocabReport: badReport)
            Issue.record("expected init to throw on incompatible vocab")
        } catch let err as KVCacheError {
            #expect(err.message.contains("vocabulary"))
        } catch {
            Issue.record("expected KVCacheError, got \(error)")
        }
    }

    @Test
    func `Compatible vocab report passes through`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hi"))
        let backend = ZeroAcceptANEDraftBackend(draftLength: 4)
        let params = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let goodReport = ANEVocabEquivalenceReport(
            comparedTokenCount: 1000, disagreementCount: 0,
            sizeDifference: 0, isCompatible: true)
        // Should construct without error.
        _ = try MirrorSpeculativeTokenIterator(
            input: input, target: model,
            draftBackend: backend, parameters: params,
            vocabReport: goodReport)
    }

    @Test
    func `Empty-script ScriptedBackend still progresses via AR`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "hi"))
        // Empty script — backend always returns []; iterator must keep
        // emitting via AR fallback.
        let backend = ScriptedANEDraftBackend(draftLength: 4, script: [])
        let params = GenerateParameters(maxTokens: 6, temperature: 0.0)
        var iter = try MirrorSpeculativeTokenIterator(
            input: input, target: model,
            draftBackend: backend, parameters: params)
        var tokens: [Int] = []
        while let t = iter.next() { tokens.append(t) }
        #expect(tokens.count == 6)
        // Never proposed anything because every call returned empty.
        #expect(iter.mirrorProposedCount == 0)
    }
}
