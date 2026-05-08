// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Google Gemma 4 MTP drafter benchmark on M5 Max.
//
// Google released MTP drafters (small "assistant" models) for the Gemma 4
// family on 2026-05-05, claimed up to 3× decode speedup via speculative
// decoding with no quality loss. Drafters share the target's tokenizer
// and (per Google's blog) "seamlessly utilize the target model's
// activations and share its KV cache."
//
// This benchmark uses the standard mlx-swift-lm SpeculativeTokenIterator
// path (port of mlx-lm's speculative_generate_step) — separate KV caches
// for main and draft. So this measures the floor of what MTP gives us
// before any KV-share plumbing.
//
// Pairs:
//   target: mlx-community/gemma-4-e4b-it-4bit
//   draft:  mlx-community/gemma-4-E4B-it-assistant-bf16
//
// Captures per cell: prefill_s, decode_s, decode_tps, accepted_total,
// proposed_total, acceptance_rate.
//
// Gated:
//   RUN_MTP_SPEC=1 swift test --filter "MTPSpecDecode"

import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let mtpDownloader: any Downloader = #hubDownloader()
private let mtpTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("Gemma 4 MTP drafter speedup on M5", .serialized)
struct MTPSpecDecode {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_MTP_SPEC"] == "1"
    }

    // 31B dense — matches the community Python MLX benchmark Tom shared
    // (1.99× speedup at ~20 tps baseline on bf16). Going with 4bit target
    // to keep download manageable (~17GB vs 62GB bf16). 31B drafter is
    // also `use_ordered_embeddings=false` per its config — no masked
    // embedder needed.
    private static let targetId = "mlx-community/gemma-4-31b-it-4bit"
    private static let draftId  = "mlx-community/gemma-4-31B-it-assistant-bf16"

    // numDraftTokens values to sweep. Includes k=1 and k=3 to find the
    // real sweet spot after drafter quantization shifted the optimum.
    // Defaults to short-ctx full sweep; at long ctx (MTP_PROMPT_TOK set)
    // we trim to k=2 (proven optimum) to keep the run fast.
    private static var draftBudgets: [Int] {
        let isLong = ProcessInfo.processInfo.environment["MTP_PROMPT_TOK"]
            .flatMap(Int.init).map { $0 >= 16_000 } ?? false
        return isLong ? [2] : [1, 2, 3, 4, 6, 8]
    }

    private static let shortPrompt = """
        Write a detailed explanation of how speculative decoding works in
        modern language model inference. Cover the role of the draft \
        model, why acceptance rate matters, the relationship between draft \
        length and end-to-end speedup, and the typical failure modes when \
        the draft and target diverge in distribution.
        """

    /// Generate a long filler prompt targeting roughly `tokenTarget` tokens.
    /// Filler tokenizes at ~1 token / 7 chars on Gemma's tokenizer.
    private static func longPrompt(tokenTarget: Int) -> String {
        let charTarget = max(64, tokenTarget * 7)
        let filler = "the routine quarterly procedure required " +
            "careful review of the documented operational records " +
            "during the assessment cycle. "
        var s = ""
        while s.count < charTarget { s += filler }
        return s + "\n\nSummarize the above in one paragraph."
    }

    /// Resolve the prompt: if MTP_PROMPT_TOK env is set, build a long
    /// filler prompt to that token target. Otherwise use the short one.
    private static var prompt: String {
        if let s = ProcessInfo.processInfo.environment["MTP_PROMPT_TOK"],
           let n = Int(s), n > 0
        {
            return longPrompt(tokenTarget: n)
        }
        return shortPrompt
    }

    private static let maxTokens = 256

    @Test("Gemma 4 E4B + E4B-assistant MTP speedup")
    func mtpRamp() async throws {
        guard Self.enabled else {
            print("[mtp] skipped: set RUN_MTP_SPEC=1")
            return
        }

        unsetenv("VLLM_TRIATT_ENABLED")
        unsetenv("LONGCTX_ENDPOINT")

        print("[mtp] loading target \(Self.targetId)...")
        let tStart = Date()
        let mainContext = try await loadModel(
            from: mtpDownloader, using: mtpTokenizerLoader,
            id: Self.targetId, progressHandler: { _ in }
        )
        print("[mtp] target loaded in " +
              "\(String(format: "%.1f", Date().timeIntervalSince(tStart)))s")

        print("[mtp] loading drafter \(Self.draftId)...")
        let dStart = Date()
        let draftContext = try await loadModel(
            from: mtpDownloader, using: mtpTokenizerLoader,
            id: Self.draftId, progressHandler: { _ in }
        )
        print("[mtp] drafter loaded in " +
              "\(String(format: "%.1f", Date().timeIntervalSince(dStart)))s")

        // Optional: quantize drafter to 4-bit at load. Saves ~75% of
        // drafter weight bandwidth on every forward step. Profile shows
        // drafter ≈ 2.5ms/step at bf16; expected ~0.7ms at 4-bit.
        if ProcessInfo.processInfo.environment["MTP_DRAFTER_4BIT"] == "1",
           let drafter = draftContext.model as? Gemma4AssistantModel
        {
            let qStart = Date()
            MLXNN.quantize(
                model: drafter,
                filter: { _, module in
                    // Quantize Linear and Embedding layers, group=64 bits=4.
                    if module is Linear || module is Embedding {
                        return (groupSize: 64, bits: 4, mode: .affine)
                    }
                    return nil
                }
            )
            eval(drafter)
            print("[mtp] drafter quantized to 4-bit in " +
                  "\(String(format: "%.1f", Date().timeIntervalSince(qStart)))s")
        }

        let resolvedPrompt = Self.prompt
        let messages: [[String: String]] = [
            ["role": "user", "content": resolvedPrompt],
        ]
        let userInput = UserInput(prompt: .messages(messages))
        let input = try await mainContext.processor.prepare(input: userInput)
        let promptTok = input.text.tokens.size
        print("[mtp] prompt tokens: \(promptTok)")

        let params = GenerateParameters(
            maxTokens: Self.maxTokens, temperature: 0.0
        )

        print("\n==== GEMMA 4 MTP DRAFTER BENCH (M5 Max) ====")
        print("target: \(Self.targetId)")
        print("draft:  \(Self.draftId)")
        print("prompt_tok=\(promptTok), max_new=\(Self.maxTokens), temp=0.0\n")
        print("config         prefill   decode   tps     tokens  speedup")

        // Baseline (no draft model)
        var baselineTps: Double = 0
        do {
            let r = try await Self.runOne(
                input: input, params: params,
                mainContext: mainContext, draftModel: nil, numDraftTokens: 0
            )
            baselineTps = r.decodeTps
            Self.printRow(
                tag: "baseline       ",
                r: r, speedup: 1.0
            )
        }

        // Spec-decode sweep
        for k in Self.draftBudgets {
            do {
                let r = try await Self.runOne(
                    input: input, params: params,
                    mainContext: mainContext,
                    draftModel: draftContext.model, numDraftTokens: k
                )
                let speedup = baselineTps > 0
                    ? r.decodeTps / baselineTps : 0.0
                Self.printRow(
                    tag: "spec k=\(k)         ",
                    r: r, speedup: speedup
                )
            } catch {
                print("spec k=\(k)   ERROR: \(error)")
            }
        }
    }

    private static func printRow(tag: String, r: CellResult, speedup: Double) {
        let prefillStr = String(format: "%.2f", r.prefillSec)
        let decodeStr = String(format: "%.2f", r.decodeSec)
        let tpsStr = String(format: "%.1f", r.decodeTps)
        let speedupStr = String(format: "%.2f", speedup)
        print("\(tag)\(prefillStr)s   \(decodeStr)s   \(tpsStr)   "
              + "\(r.tokensGen)     \(speedupStr)×")
    }

    private struct CellResult {
        let prefillSec: Double
        let decodeSec: Double
        let decodeTps: Double
        let tokensGen: Int
    }

    private static func runOne(
        input: LMInput,
        params: GenerateParameters,
        mainContext: ModelContext,
        draftModel: (any LanguageModel)?,
        numDraftTokens: Int
    ) async throws -> CellResult {
        // MTP path: drafter is a Gemma4AssistantModel, target a
        // Gemma4TextModel — use the dedicated MTP iterator.
        if let drafter = draftModel as? Gemma4AssistantModel,
           let mainGemma = mainContext.model as? Gemma4TextModel
        {
            let blockSize = max(2, numDraftTokens + 1)
            var tokensGen = 0
            let info = try runMTPSpeculative(
                input: input, mainModel: mainGemma, drafter: drafter,
                parameters: params, blockSize: blockSize,
                onToken: { _ in tokensGen += 1 })
            let prefillSec = info.prefillSec
            let decodeSec = info.decodeSec
            let tps = decodeSec > 0
                ? Double(max(1, tokensGen - 1)) / decodeSec : 0.0
            print(String(format:
                "  [mtp k=%d] proposed=%d accepted=%d (rate=%.2f%%)",
                numDraftTokens, info.proposedTotal, info.acceptedTotal,
                info.proposedTotal > 0
                    ? 100.0 * Double(info.acceptedTotal) /
                      Double(info.proposedTotal)
                    : 0.0))
            return CellResult(
                prefillSec: prefillSec, decodeSec: decodeSec,
                decodeTps: tps, tokensGen: tokensGen)
        }

        // Baseline (no drafter) — standard generate.
        let stream = try MLXLMCommon.generate(
            input: input, parameters: params, context: mainContext
        )

        let tStart = Date()
        var firstTokenTime: TimeInterval? = nil
        var tokensGen = 0
        for await gen in stream {
            switch gen {
            case .chunk:
                if firstTokenTime == nil {
                    firstTokenTime = Date().timeIntervalSince(tStart)
                }
                tokensGen += 1
            case .info, .toolCall:
                break
            }
        }
        let totalSec = Date().timeIntervalSince(tStart)
        let prefillSec = firstTokenTime ?? totalSec
        let decodeSec = max(0.001, totalSec - prefillSec)
        let decodeTps = Double(max(1, tokensGen - 1)) / decodeSec
        return CellResult(
            prefillSec: prefillSec,
            decodeSec: decodeSec,
            decodeTps: decodeTps,
            tokensGen: tokensGen
        )
    }
}
