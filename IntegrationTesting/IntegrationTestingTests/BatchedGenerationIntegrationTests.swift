// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// Verify that the batched forward path on a real Qwen3.5 model produces
/// the same token sequence per batch member as the single-stream path
/// when fed identical prompts under greedy decoding.
///
/// This is the central acceptance criterion for #136: the B-dim forward must
/// be numerically equivalent to running B copies of the single-stream path.
@Suite(.serialized)
struct BatchedGenerationIntegrationTests {

    @Test func qwen35BatchedMatchesSingleStream() async throws {
        let container = try await models.qwen35Container()
        let prompt = "List five common European capital cities, separated by commas."
        let maxTokens = 32
        let batchSize = 4

        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0  // greedy → deterministic
        )

        // ── Reference: single-stream generate over the same prompt ───────
        let referenceTokens: [Int] = try await {
            let input = try await container.prepare(input: UserInput(prompt: prompt))
            let stream = try await container.generateTokens(
                input: input,
                parameters: params
            )
            var tokens: [Int] = []
            for await event in stream {
                if let token = event.token {
                    tokens.append(token)
                }
            }
            return tokens
        }()

        #expect(referenceTokens.count > 0, "Reference run produced no tokens")

        // ── Batched: B identical inputs through the new generateBatched ──
        var inputs: [LMInput] = []
        for _ in 0 ..< batchSize {
            inputs.append(try await container.prepare(input: UserInput(prompt: prompt)))
        }

        var batchedStreams: [[Int]] = Array(repeating: [], count: batchSize)
        let stream = try await container.generateBatched(
            inputs: inputs,
            parameters: params
        )
        for await generation in stream {
            switch generation {
            case .step(let tokens):
                #expect(tokens.count == batchSize)
                for (i, token) in tokens.enumerated() {
                    batchedStreams[i].append(token)
                }
            case .info:
                break
            }
        }

        // ── Compare every batch member against the reference ────────────
        let prefixLen = min(referenceTokens.count, batchedStreams[0].count)
        #expect(prefixLen >= 16, "Run produced too few tokens to verify")

        for (i, stream) in batchedStreams.enumerated() {
            let prefix = Array(stream.prefix(prefixLen))
            let refPrefix = Array(referenceTokens.prefix(prefixLen))
            #expect(
                prefix == refPrefix,
                Comment(
                    rawValue:
                        "Batch member \(i) diverges from single-stream reference. "
                        + "Reference: \(refPrefix). Got: \(prefix)."
                )
            )
        }
    }
}
