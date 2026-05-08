// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MTPSpec — Multi-Token Prediction speculative decoding driver for
// Gemma 4 + Gemma4Assistant pairs. Distinct from
// SpeculativeTokenIterator (which assumes the drafter is autoregressive
// on token IDs); this iterator threads parent's last hidden state +
// shared K/V into the drafter every round.
//
// Loop:
//   1. Prefill parent on prompt → cache populated, last token + last
//      hidden state captured.
//   2. While more tokens needed:
//      a. Extract parent's last full-attn + last sliding-attn K/V.
//      b. drafter.setSharedKV(...) with current parent cache offset.
//      c. drafter.draftBlock(lastBonus, lastHidden, blockSize) →
//         (blockSize-1) candidate tokens.
//      d. Verify input = [lastBonus, c1, c2, ..., c_{K-1}] of length K.
//      e. parent forward → K logits + K hidden states.
//      f. Greedy verify: argmax(logits[i]) must match c_{i+1} for
//         accept. Stop at first mismatch.
//      g. If all matched, accept argmax(logits[K-1]) as bonus.
//      h. Trim parent cache by (K - acceptedCount) cells.
//      i. Yield accepted tokens. lastBonus = last accepted, lastHidden
//         = hidden at that position.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct MTPGeneration: Sendable {
    public enum Kind: Sendable {
        case token(Int)
        case info(MTPInfo)
    }
    public let kind: Kind
}

public struct MTPInfo: Sendable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let prefillSec: Double
    public let decodeSec: Double
    public let proposedTotal: Int
    public let acceptedTotal: Int
    public var acceptanceRate: Double {
        proposedTotal > 0
            ? Double(acceptedTotal) / Double(proposedTotal) : 0
    }
}

/// Run MTP speculative generation synchronously. Caller is responsible
/// for tokenizing the prompt (use mainContext.processor.prepare on a
/// UserInput). Token IDs are reported via the `onToken` callback as
/// they're accepted; the function returns once generation is complete.
public func runMTPSpeculative(
    input: LMInput,
    mainModel: Gemma4TextModel,
    drafter: Gemma4AssistantModel,
    parameters: GenerateParameters,
    blockSize: Int = 4,
    onToken: (Int) -> Void
) throws -> MTPInfo {
    return try runMTPSpeculativeBody(
        input: input,
        mainModel: mainModel,
        drafter: drafter,
        parameters: parameters,
        blockSize: blockSize,
        onToken: onToken
    )
}

private func runMTPSpeculativeBody(
    input: LMInput,
    mainModel: Gemma4TextModel,
    drafter: Gemma4AssistantModel,
    parameters: GenerateParameters,
    blockSize: Int,
    onToken: (Int) -> Void
) throws -> MTPInfo {
    precondition(blockSize >= 2,
                 "MTP requires blockSize >= 2 (drafter generates blockSize-1)")

    // Bind drafter to parent's input embedding.
    let embedScale = sqrt(Float(mainModel.config.hiddenSize))
    drafter.bind(
        parentEmbedding: mainModel.model.embedTokens,
        embedScale: embedScale)

    // Build parent cache.
    let cache = mainModel.newCache(parameters: parameters)

    // Prefill via parent.prepare — chunks through all but last token.
    let promptTokens = input.text.tokens.size
    let tStart = Date()
    let prepareResult = try mainModel.prepare(
        input, cache: cache, windowSize: nil)
    var remaining: LMInput.Text
    switch prepareResult {
    case .tokens(let y):
        remaining = y
    default:
        // logits-output prepare not handled here
        throw NSError(
            domain: "MTPSpec", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "prepare() returned non-token result"])
    }

    // Process the remaining (typically 1) token to get initial
    // last_hidden + bootstrap the first generated token.
    let lastChunk = remaining.tokens[.newAxis, 0...]
    let initHidden = mainModel.model(lastChunk, cache: cache)
    let initLogits: MLXArray
    if mainModel.config.tieWordEmbeddings {
        initLogits = mainModel.model.embedTokens.asLinear(initHidden)
    } else {
        initLogits = mainModel.lmHead!(initHidden)
    }
    // Greedy first token from the last position's logits. The argMax
    // .item() call forces eval of the dependency graph here — no need
    // for an explicit eval() barrier.
    let lastT = initLogits.dim(1) - 1
    var lastBonus = argMax(initLogits[0, lastT], axis: -1).item(Int.self)
    // Hidden at the last position: shape [1, 1, hidden]. Eval forces
    // it onto a fresh root so subsequent rounds don't keep the entire
    // initial graph alive.
    var lastHidden = initHidden[0..<1, lastT..<lastT + 1, 0...]
    eval(lastHidden)

    let prefillSec = Date().timeIntervalSince(tStart)
    onToken(lastBonus)

    var generatedCount = 1
    var proposedTotal = 0
    var acceptedTotal = 0
    let decodeStart = Date()

    let layerTypes = mainModel.config.layerTypes
    let previousKVs = mainModel.model.previousKVs

    // Optional per-phase profiling. Set MTP_PROFILE=1 to enable.
    // Costs ~0.5ms per round in extra evals — only use for diagnosis.
    let profileEnabled =
        ProcessInfo.processInfo.environment["MTP_PROFILE"] == "1"
    var tDraft: Double = 0
    var tVerify: Double = 0
    var tSync: Double = 0
    var tCache: Double = 0
    var roundsCounted = 0

    let maxTokens = parameters.maxTokens ?? 256
    while generatedCount < maxTokens {
        // 1. Extract shared K/V.
        guard let shared = extractDrafterSharedKV(
            layerTypes: layerTypes,
            previousKVs: previousKVs,
            caches: cache)
        else {
            // Fall back to single-token decode if extraction failed.
            let inp = MLXArray([Int32(lastBonus)]).reshaped([1, 1])
            let h = mainModel.model(inp, cache: cache)
            let l: MLXArray
            if mainModel.config.tieWordEmbeddings {
                l = mainModel.model.embedTokens.asLinear(h)
            } else {
                l = mainModel.lmHead!(h)
            }
            let nextTok = argMax(l[0, 0], axis: -1).item(Int.self)
            lastBonus = nextTok
            lastHidden = h[0..<1, 0..<1, 0...]
            eval(lastHidden)
            onToken(nextTok)
            generatedCount += 1
            continue
        }

        // 2. Drafter setup. Position = parent cache's full-attn offset
        //    (where the next token would land).
        let parentOffset = cache.first { c in
            // first non-shared, non-zero cache reflects current pos
            c.offset > 0
        }?.offset ?? cache[0].offset
        drafter.setSharedKV(
            fullAttn: shared.full,
            slidingAttn: shared.sliding,
            kvOffset: parentOffset,
            position: parentOffset)

        // 3. Draft K-1 candidates — stays as MLXArray, no sync yet.
        let tDraftStart = profileEnabled ? Date() : Date.distantPast
        let candidatesArr = drafter.draftBlock(
            lastBonus: lastBonus,
            hidden: lastHidden,
            blockSize: blockSize)
        if profileEnabled {
            eval(candidatesArr)  // force eval to measure drafter alone
            tDraft += Date().timeIntervalSince(tDraftStart)
        }
        let numCandidates = blockSize - 1
        proposedTotal += numCandidates

        // 4. Build verification input = [[bonus, c1, c2, ..., c_{K-1}]].
        // Concat lazily so the entire drafter chain + parent verify
        // forward fuses into one MLX graph evaluation.
        let bonusArr = MLXArray([Int32(lastBonus)]).reshaped([1, 1])
        let verifyInput = concatenated([bonusArr, candidatesArr], axis: 1)
        let K = blockSize  // verifyInput length

        // 5. Parent forward.
        let tVerifyStart = profileEnabled ? Date() : Date.distantPast
        let vHidden = mainModel.model(verifyInput, cache: cache)
        let vLogits: MLXArray
        if mainModel.config.tieWordEmbeddings {
            vLogits = mainModel.model.embedTokens.asLinear(vHidden)
        } else {
            vLogits = mainModel.lmHead!(vHidden)
        }
        if profileEnabled {
            eval(vLogits)
            tVerify += Date().timeIntervalSince(tVerifyStart)
        }

        // 6. Single sync per round: pull both candidates (drafter
        // outputs) and target predictions in one shot. Concatenate
        // [candidates_flat, preds] into one tensor and read it back
        // with a single asArray call — collapses 3 sync points
        // (K-1 in drafter + 1 here) into 1.
        let tSyncStart = profileEnabled ? Date() : Date.distantPast
        let predsTensor = argMax(vLogits[0], axis: -1).asType(.int32)
        let combined = concatenated(
            [candidatesArr.flattened(), predsTensor], axis: 0)
        let combinedCPU: [Int32] = combined.asArray(Int32.self)
        if profileEnabled {
            tSync += Date().timeIntervalSince(tSyncStart)
        }
        let candidatesCPU = Array(combinedCPU[0..<numCandidates])
        let predsCPU = Array(combinedCPU[numCandidates...])

        var accepted: [Int] = []
        var lastAcceptedIdx = -1
        for i in 0..<numCandidates {
            let pred = Int(predsCPU[i])
            if pred == Int(candidatesCPU[i]) {
                accepted.append(pred)
                lastAcceptedIdx = i
            } else {
                accepted.append(pred)
                lastAcceptedIdx = i
                break
            }
        }
        // 7. Bonus token if all candidates matched.
        var allMatched = false
        if accepted.count == numCandidates {
            let bonus = Int(predsCPU[K - 1])
            accepted.append(bonus)
            lastAcceptedIdx = K - 1
            allMatched = true
        }
        acceptedTotal += accepted.count

        // 8. Trim parent cache. Parent wrote K cells; we accepted
        //    accepted.count tokens. If accepted.count < K, trim
        //    K - accepted.count cells.
        let tCacheStart = profileEnabled ? Date() : Date.distantPast
        let toTrim = K - accepted.count
        if toTrim > 0 {
            for c in cache {
                if c.isTrimmable {
                    _ = c.trim(toTrim)
                }
            }
        }
        if profileEnabled {
            tCache += Date().timeIntervalSince(tCacheStart)
            roundsCounted += 1
        }

        // 9. Yield accepted tokens (capped to maxTokens).
        for tok in accepted {
            if generatedCount >= maxTokens { break }
            onToken(tok)
            generatedCount += 1
        }

        // 10. Update lastBonus + lastHidden for next round.
        lastBonus = accepted.last!
        // Hidden index in vHidden: position lastAcceptedIdx (if matched
        // all, use K-1; if partial match at i, use i).
        let hIdx = allMatched ? (K - 1) : lastAcceptedIdx
        lastHidden = vHidden[0..<1, hIdx..<hIdx + 1, 0...]
        eval(lastHidden)

        // EOS check (greedy).
        if lastBonus == 0 || lastBonus == 1 || lastBonus == 2 {
            // Generic EOS guards (Gemma uses 1 = EOS, 2 = BOS)
            break
        }
    }

    let decodeSec = Date().timeIntervalSince(decodeStart)
    if profileEnabled, roundsCounted > 0 {
        let n = Double(roundsCounted)
        print(String(format:
            "[mtp-profile] rounds=%d  draft=%.2fms  verify=%.2fms  " +
            "sync=%.2fms  cache_trim=%.2fms  per_round_total=%.2fms",
            roundsCounted,
            tDraft / n * 1000.0,
            tVerify / n * 1000.0,
            tSync / n * 1000.0,
            tCache / n * 1000.0,
            decodeSec / n * 1000.0))
    }
    return MTPInfo(
        promptTokens: promptTokens,
        generatedTokens: generatedCount,
        prefillSec: prefillSec,
        decodeSec: decodeSec,
        proposedTotal: proposedTotal,
        acceptedTotal: acceptedTotal)
}
