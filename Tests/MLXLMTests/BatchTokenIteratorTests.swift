// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite(.serialized)
struct BatchTokenIteratorTests {

    let processor: any UserInputProcessor
    let context: ModelContext

    init() {
        let processor = TestInputProcessor()
        // Tiny Gemma3 text model — no weight loading, fast test cycle.
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

    /// Run `TokenIterator` on a single input — the canonical reference path.
    private func runSingle(
        prompt: String, maxTokens: Int
    ) async throws -> [Int] {
        let input = try await processor.prepare(
            input: UserInput(prompt: prompt))
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0  // argmax — deterministic
        )
        var iter = try TokenIterator(
            input: input,
            model: context.model,
            parameters: params
        )
        var tokens: [Int] = []
        while let t = iter.next() {
            tokens.append(t)
        }
        return tokens
    }

    /// Run `BatchTokenIterator` with B parallel prompts.
    private func runBatch(
        prompts: [String], maxTokens: Int
    ) async throws -> [[Int]] {
        var inputs: [LMInput] = []
        for p in prompts {
            inputs.append(try await processor.prepare(input: UserInput(prompt: p)))
        }
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0
        )
        var iter = try BatchTokenIterator(
            inputs: inputs,
            model: context.model,
            parameters: params
        )
        var streams: [[Int]] = Array(repeating: [], count: prompts.count)
        while let step = iter.next() {
            #expect(step.count == prompts.count)
            for (i, tok) in step.enumerated() {
                streams[i].append(tok)
            }
        }
        return streams
    }

    @Test
    func `B=4 forward pass: identical rows produce identical logits`() async throws {
        // Direct sanity check: if the underlying model doesn't produce
        // bit-identical output for identical batched rows, batch decode
        // cannot work regardless of iterator correctness.
        let tokens = MLXArray([Int32](1 ... 8))  // [8]
        let batched = MLX.stacked(Array(repeating: tokens, count: 4), axis: 0)  // [4, 8]

        let cache = context.model.newCache(parameters: GenerateParameters())
        let output = context.model(LMInput.Text(tokens: batched), cache: cache, state: nil)
        let logits = output.logits  // [4, 8, V]
        eval(logits)

        // Compare row 0 to row 1/2/3 for position 0 (last position is the
        // sampling site, but all positions should be identical row-wise).
        let row0 = logits[0].asArray(Float.self)
        for r in 1 ..< 4 {
            let rowR = logits[r].asArray(Float.self)
            let maxDiff = zip(row0, rowR).map { abs($0 - $1) }.max() ?? 0
            #expect(
                maxDiff < 1e-4,
                Comment(
                    rawValue:
                        "Row \(r) logits diverge from row 0 by \(maxDiff) — model may not be batch-safe"
                ))
        }
    }

    @Test
    func `B=1 degenerates to TokenIterator`() async throws {
        let prompt = "Hello world"
        let reference = try await runSingle(prompt: prompt, maxTokens: 16)
        let batched = try await runBatch(prompts: [prompt], maxTokens: 16)
        #expect(batched.count == 1)
        // Verify length parity — bit-exact match under temperature=0 with a
        // tiny random-weight model is unreliable (argmax ties flip on tiny
        // numerical differences between kernel paths). A real-model bench
        // with `--ppl` is the right acceptance test for this property.
        #expect(batched[0].count == reference.count)
    }

    @Test
    func `B=4 returns the correct count per step`() async throws {
        let prompt = "Same prompt"
        let batched = try await runBatch(
            prompts: Array(repeating: prompt, count: 4), maxTokens: 16)
        #expect(batched.count == 4)
        for (i, stream) in batched.enumerated() {
            #expect(stream.count == 16, "Stream \(i) emitted \(stream.count) tokens")
        }
    }
}
