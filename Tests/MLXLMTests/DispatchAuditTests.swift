// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Uses `GPU.totalDispatches()` to enumerate the actual per-token Metal
/// kernel-launch count for a typical decoder forward pass. The output is the
/// input to the Metal Indirect Command Buffer (ICB) feasibility decision:
///
/// - Low dispatch count per token (say ≤ 20) → ICB payoff is bounded.
///   `max_cmd_count` on MTLIndirectCommandBuffer is 16,384; the overhead of
///   maintaining the ICB is fixed, so the win scales with how many dispatches
///   it skips re-encoding. At very low counts the CPU-side encoder isn't the
///   bottleneck.
/// - High dispatch count per token (say ≥ 50) → ICB could save meaningful
///   per-token CPU time, especially if most dispatches have stable bindings
///   (weight buffers) and only a handful need per-token rewrites (input
///   token buffer, KV offset, position).
///
/// This suite runs on a tiny random-weight Gemma3 model because loading a
/// real model weights is a separate session and the per-layer dispatch
/// pattern is determined by the architecture, not the weights.
@Suite(.serialized)
struct DispatchAuditTests {

    let processor: any UserInputProcessor
    let context: ModelContext
    let numLayers: Int

    init() {
        let processor = TestInputProcessor()
        let numLayers = 8
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: numLayers, intermediateSize: 64,
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
        self.numLayers = numLayers
        self.context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
    }

    /// Count Metal dispatches issued during a forward-and-sync of the given
    /// closure. Any lazy MLXArrays the closure builds are eval'd before the
    /// counter is read, then `Stream.synchronize()` waits for async kernels
    /// (TokenIterator uses asyncEval internally, so without the sync the
    /// dispatches may not have been issued to the encoder yet).
    private func countDispatches(_ body: () -> [MLXArray]) -> UInt64 {
        GPU.resetDispatchCounter()
        let outputs = body()
        if !outputs.isEmpty {
            eval(outputs)
        }
        Stream.gpu.synchronize()
        return GPU.totalDispatches()
    }

    @Test
    func `Multi-step decode dispatch average`() async throws {
        let input = try await processor.prepare(
            input: UserInput(prompt: "Audit across multiple decode steps."))
        let params = GenerateParameters(maxTokens: 6, temperature: 0.0)

        var iter = try TokenIterator(
            input: input, model: context.model, parameters: params)
        _ = iter.next()  // prefill + step 1

        let steps = 5
        var nextIter = iter
        let totalDispatches = countDispatches {
            var outputs: [MLXArray] = []
            for _ in 0 ..< steps {
                guard nextIter.next() != nil else { break }
            }
            return outputs
        }
        let avg = Double(totalDispatches) / Double(steps)

        print(
            """
            [DISPATCH-AUDIT] Gemma3 tiny (\(numLayers) layers), \(steps) decode steps:
              total dispatches  = \(totalDispatches)
              avg per step      = \(avg)
              avg per layer     = \(avg / Double(numLayers))
            """)

        #expect(totalDispatches > 0)
    }

    @Test
    func `Prefill dispatch count for short context`() async throws {
        let input = try await processor.prepare(
            input: UserInput(prompt: "Prefill dispatch audit for a short prompt."))

        // Prefill shape expected by the model: [1, L].
        let tokens = input.text.tokens
        let tokenCount = tokens.dim(0)
        let batchedTokens = tokens.reshaped([1, tokenCount])

        let dispatchesPrefill = countDispatches {
            let cache = context.model.newCache(parameters: GenerateParameters())
            let result = context.model(
                LMInput.Text(tokens: batchedTokens), cache: cache, state: nil)
            return [result.logits]
        }

        print(
            """
            [DISPATCH-AUDIT] Gemma3 tiny prefill (\(tokenCount) tokens, \(numLayers) layers):
              total dispatches  = \(dispatchesPrefill)
              per token (approx)= \(Double(dispatchesPrefill) / Double(tokenCount))
              per layer (approx)= \(Double(dispatchesPrefill) / Double(numLayers))
            """)

        #expect(dispatchesPrefill > 0)
    }
}
