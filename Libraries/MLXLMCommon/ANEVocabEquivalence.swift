// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Vocabulary equivalence gate (Mirror Speculative Decoding)
//
// Mirror SD pairs a draft and a target that train independently — they
// can be totally different architectures (one trained on ANE-friendly
// shapes, one trained for GPU). The cross-framework token I/O works only
// if their tokenizers agree token-for-token: a draft saying "token 12345"
// has to mean the *same string* on the target side, otherwise the verify
// step is comparing apples to oranges and the iterator silently produces
// nonsense.
//
// llama.cpp solves this with `SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128` plus
// a token-by-token sample check. We mirror that policy here so the gate
// is identical in spirit — refuse a pairing whose tokenizers disagree on
// more than a handful of token IDs.
//
// The gate is **pure-Swift** — no dependency on any specific tokenizer
// implementation. Callers pass two `[Int: String]` maps (token ID → token
// string); we walk the intersection of their keysets and tally
// disagreements. This keeps the gate testable without standing up real
// tokenizers and without coupling to the `Tokenizers` package.

/// Result of comparing two tokenizer vocabularies for Mirror SD pairing.
public struct ANEVocabEquivalenceReport: Equatable, Sendable {
    /// Number of token IDs present in both vocabularies (the comparable
    /// set — IDs only present on one side are not counted as a
    /// disagreement, but they do reduce the comparable set so a vastly-
    /// different-size vocabulary will hit the size-difference gate first).
    public let comparedTokenCount: Int

    /// Number of compared token IDs whose strings differ.
    public let disagreementCount: Int

    /// Absolute size difference (`||draft| - |target||`).
    public let sizeDifference: Int

    /// True when both gates pass: sizes within `maxSizeDifference` AND
    /// disagreement count within `maxDisagreements`.
    public let isCompatible: Bool

    public var disagreementRate: Double {
        guard comparedTokenCount > 0 else { return 0 }
        return Double(disagreementCount) / Double(comparedTokenCount)
    }
}

/// llama.cpp's default — see `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`. Refuses
/// pairings whose vocabularies differ in size by more than this many
/// tokens.
public let aneVocabDefaultMaxSizeDifference: Int = 128

/// llama.cpp's default — `SPEC_VOCAB_CHECK_START_TOKEN_ID = 5`. Tokens
/// 0..<5 are usually special (BOS/EOS/PAD/UNK + one reserved) and
/// occasionally drift between draft and target without affecting
/// generation. Skipping them avoids false negatives.
public let aneVocabDefaultStartTokenID: Int = 5

/// Maximum tolerated token-string disagreements. llama.cpp uses zero
/// (any disagreement above the start ID rejects); we expose it as a
/// parameter so tests can probe the boundary cleanly.
public let aneVocabDefaultMaxDisagreements: Int = 0

/// Compare two tokenizer vocabularies for Mirror SD eligibility.
///
/// Both maps are token-ID → token-string. We walk every ID in
/// `[startTokenID, min(draft.size, target.size))` and count
/// disagreements. The size check is independent — vocabularies whose
/// sizes differ by more than `maxSizeDifference` fail outright, even
/// when their shared prefix matches.
///
/// - Parameter draftVocab: token ID → token string for the draft model.
/// - Parameter targetVocab: token ID → token string for the target.
/// - Parameter startTokenID: skip tokens below this ID. Defaults to
///   `aneVocabDefaultStartTokenID = 5`.
/// - Parameter maxSizeDifference: size-difference gate. Default 128.
/// - Parameter maxDisagreements: disagreement gate. Default 0.
public func aneCheckVocabEquivalence(
    draftVocab: [Int: String],
    targetVocab: [Int: String],
    startTokenID: Int = aneVocabDefaultStartTokenID,
    maxSizeDifference: Int = aneVocabDefaultMaxSizeDifference,
    maxDisagreements: Int = aneVocabDefaultMaxDisagreements
) -> ANEVocabEquivalenceReport {
    let draftSize = draftVocab.count
    let targetSize = targetVocab.count
    let sizeDiff = abs(draftSize - targetSize)

    // Size gate first — bail without walking tokens when this fails.
    if sizeDiff > maxSizeDifference {
        return ANEVocabEquivalenceReport(
            comparedTokenCount: 0,
            disagreementCount: 0,
            sizeDifference: sizeDiff,
            isCompatible: false)
    }

    // Walk the shared ID range and tally disagreements.
    let upperBound = Swift.min(draftSize, targetSize)
    var compared = 0
    var disagreements = 0
    if startTokenID < upperBound {
        for id in startTokenID ..< upperBound {
            // A token only present on one side is a disagreement: the
            // draft and target have different ideas about what ID
            // means. Skipping such IDs would mask real mismatches.
            let draftStr = draftVocab[id]
            let targetStr = targetVocab[id]
            switch (draftStr, targetStr) {
            case (nil, nil):
                // Neither side has the token — sparse vocabularies. Don't
                // count this slot at all; nothing to compare.
                continue
            case (let d?, let t?):
                compared += 1
                if d != t { disagreements += 1 }
            default:
                // One side missing → disagreement.
                compared += 1
                disagreements += 1
            }
        }
    }

    let pass = sizeDiff <= maxSizeDifference && disagreements <= maxDisagreements
    return ANEVocabEquivalenceReport(
        comparedTokenCount: compared,
        disagreementCount: disagreements,
        sizeDifference: sizeDiff,
        isCompatible: pass)
}
