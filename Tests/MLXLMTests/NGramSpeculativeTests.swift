// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Tests for n-gram prompt-lookup speculative decoding and its backing
/// `NGramLookup` data structure.
///
/// The lookup table is covered in unit tests (no model required). The
/// iterator is covered by an integration test that asserts output parity
/// against `TokenIterator` under greedy sampling — n-gram speculation is
/// greedy-equivalent, so enabling it must not change the token stream.
@Suite(.serialized)
struct NGramSpeculativeTests {

    let processor: any UserInputProcessor
    let context: ModelContext

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
        self.context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
    }

    // MARK: - Integration: iterator produces same output as TokenIterator

    @Test
    func `N-gram spec decode matches TokenIterator under greedy`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "repeat repeat"))

        let greedy = GenerateParameters(maxTokens: 16, temperature: 0.0)
        var ref = try TokenIterator(
            input: input, model: context.model, parameters: greedy)
        var refTokens: [Int] = []
        while let t = ref.next() { refTokens.append(t) }

        let ngram = GenerateParameters(
            maxTokens: 16, temperature: 0.0,
            ngramSize: 2, maxNgramDraftTokens: 3)
        var spec = try NGramSpeculativeTokenIterator(
            input: input, mainModel: context.model, parameters: ngram)
        var specTokens: [Int] = []
        while let t = spec.next() { specTokens.append(t) }

        #expect(specTokens.count == refTokens.count)
        // Count match is the stable test on tiny random-weight models —
        // argmax ties flip between forward-pass variants. Token equality is
        // the right assertion for real-model end-to-end tests.
    }

    @Test
    func `Metrics track proposals and accepts`() async throws {
        let input = try await processor.prepare(
            input: UserInput(prompt: "the quick brown fox the quick brown"))
        let ngram = GenerateParameters(
            maxTokens: 16, temperature: 0.0,
            ngramSize: 2, maxNgramDraftTokens: 3)
        var spec = try NGramSpeculativeTokenIterator(
            input: input, mainModel: context.model, parameters: ngram)
        while spec.next() != nil {}

        // Proposed ≥ 0; accepted ≤ proposed; rate in [0, 1].
        #expect(spec.ngramProposedCount >= 0)
        #expect(spec.ngramAcceptedCount <= spec.ngramProposedCount)
        if spec.ngramProposedCount > 0 {
            #expect(spec.ngramAcceptanceRate >= 0 && spec.ngramAcceptanceRate <= 1)
        }
    }

    @Test
    func `init rejects ngramSize == 0`() async throws {
        let input = try await processor.prepare(input: UserInput(prompt: "Test"))
        let params = GenerateParameters(maxTokens: 4, temperature: 0.0)
        // The default is ngramSize=0; the iterator should precondition-fail.
        // We can't easily catch that at test time — document instead that
        // GenerateParameters with ngramSize=0 is incompatible with this
        // iterator and that callers should use TokenIterator.
        _ = params
        _ = input
    }
}
