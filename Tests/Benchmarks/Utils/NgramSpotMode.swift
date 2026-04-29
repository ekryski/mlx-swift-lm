// Copyright © 2026 Apple Inc.
//
// Pure-Swift helpers for the `ngram-spot` and `ngram-sweep-summary` bench
// methods (spec 018). Everything in this file is GPU-free and unit-tested
// in `NgramSpotModeTests.swift`.
//
// The integration glue (loading a model, running generations, recording
// per-cell tok/s) lives in `InferenceBenchmark.swift`. This file contains
// only the data shapes + transformations.

import Foundation

// MARK: - Cell specification

/// A single spot-mode candidate configuration. Compared against a baseline
/// (`ngramSize == 0`) and, optionally, other cells in a one-prompt sweep.
public struct NgramSpotCell: Equatable, Hashable, Sendable {
    public let ngramSize: Int
    public let maxDraftTokens: Int
    public let minHits: Int
    public let useAdaptive: Bool
    public let useStrictGreedy: Bool
    public let useDominance: Bool
    public let useMultiCandidate: Bool

    public init(
        ngramSize: Int,
        maxDraftTokens: Int,
        minHits: Int = 1,
        useAdaptive: Bool = false,
        useStrictGreedy: Bool = false,
        useDominance: Bool = false,
        useMultiCandidate: Bool = false
    ) {
        self.ngramSize = ngramSize
        self.maxDraftTokens = maxDraftTokens
        self.minHits = minHits
        self.useAdaptive = useAdaptive
        self.useStrictGreedy = useStrictGreedy
        self.useDominance = useDominance
        self.useMultiCandidate = useMultiCandidate
    }

    /// Short human-readable label for the summary table — `"n=3 D=4"` plus
    /// a trailing flag list when any opt-in flags are set.
    public var label: String {
        var s = "n=\(ngramSize) D=\(maxDraftTokens)"
        if minHits != 1 { s += " H=\(minHits)" }
        var flags: [String] = []
        if useAdaptive { flags.append("adaptive") }
        if useStrictGreedy { flags.append("strict") }
        if useDominance { flags.append("dominance") }
        if useMultiCandidate { flags.append("multi") }
        if !flags.isEmpty { s += " + " + flags.joined(separator: " + ") }
        return s
    }

    /// The "baseline" sentinel — `ngramSize == 0` disables speculative decoding
    /// entirely. Always the first cell of a spot-mode run; everything else
    /// is measured relative to it.
    public static let baseline = NgramSpotCell(ngramSize: 0, maxDraftTokens: 0)

    public var isBaseline: Bool { ngramSize == 0 }
}

// MARK: - Cell-spec parsing

/// Parse a comma-separated cell spec string into a list of cells.
///
/// Format: `n:D[:H][:flag,flag,...],...` — semicolons would have collided
/// with shell quoting, so flags are slash-delimited within a cell.
///
/// Examples:
///   - `"3:2"` → one cell, `n=3 D=2 H=1`, no flags.
///   - `"3:4:2"` → `n=3 D=4 H=2`, no flags.
///   - `"3:12:1:adaptive/strict"` → `n=3 D=12 H=1`, adaptive + strict-greedy on.
///   - `"3:2,3:4,3:8,3:12:1:adaptive/strict"` → four cells (default-ish).
///
/// Unknown flags are silently ignored (forward-compatible). A baseline
/// cell is **never** parsed from this string — callers always prepend the
/// baseline as cell 0.
public func parseSpotCells(_ raw: String) -> [NgramSpotCell] {
    raw.split(separator: ",", omittingEmptySubsequences: true)
        .compactMap { spec -> NgramSpotCell? in
            let parts = spec.split(separator: ":")
            guard parts.count >= 2,
                  let n = Int(parts[0]), n >= 1,
                  let d = Int(parts[1]), d >= 1
            else { return nil }
            let h = parts.count >= 3 ? Int(parts[2]) ?? 1 : 1
            let flagToken = parts.count >= 4 ? String(parts[3]) : ""
            let flagSet = Set(flagToken.split(separator: "/").map(String.init))
            return NgramSpotCell(
                ngramSize: n,
                maxDraftTokens: d,
                minHits: h,
                useAdaptive: flagSet.contains("adaptive"),
                useStrictGreedy: flagSet.contains("strict"),
                useDominance: flagSet.contains("dominance"),
                useMultiCandidate: flagSet.contains("multi"))
        }
}

/// The default candidate-cell list for `--method ngram-spot`. Covers the
/// regions where wins live in the recipe-bulk + qa-requote sweeps:
///   - low-overhead (`n=3 D=2`)
///   - mid-range (`n=3 D=4`)
///   - long-amortising (`n=3 D=8`)
///   - mixed-content default with adaptive + strict (`n=3 D=12`).
///
/// All five cells run at temperature 0; spec decode greedy-equivalence is
/// the safety net.
public let defaultSpotCells: [NgramSpotCell] = [
    NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
    NgramSpotCell(ngramSize: 3, maxDraftTokens: 4),
    NgramSpotCell(ngramSize: 3, maxDraftTokens: 8),
    NgramSpotCell(
        ngramSize: 3, maxDraftTokens: 12,
        useAdaptive: true, useStrictGreedy: true),
]

// MARK: - Per-cell measurement

/// One row of a spot-mode run: a measured cell and its computed deltas vs.
/// the baseline cell.
public struct NgramSpotResult: Equatable, Sendable {
    public let cell: NgramSpotCell

    /// Decode tok/s reported by `runGenerationBenchmark`. The user-visible
    /// average (post-prefill, includes warmup tokens 1..10).
    public let decodeTokPerSec: Double

    /// Steady decode tok/s — tokens 11..end average. `nil` when the cell
    /// generated ≤ 10 tokens.
    public let steadyTokPerSec: Double?

    /// Speculative-decode acceptance count and proposal count. `(0, 0)` for
    /// the baseline cell.
    public let accepted: Int
    public let proposed: Int

    /// Output text emitted by the cell (post-detokenization). Truncated to
    /// the first `outputSampleLimit` characters before storage to keep
    /// summary memory bounded on long generations.
    public let outputSample: String

    /// Token-stream prefix-equality match against the baseline cell's
    /// output. `nil` for the baseline cell itself.
    public let matchesBaseline: Bool?

    /// Acceptance rate in `[0, 1]`. Zero when no drafts were proposed.
    public var acceptanceRate: Double {
        proposed == 0 ? 0 : Double(accepted) / Double(proposed)
    }
}

/// Cap stored output samples at this many characters per cell. Generated
/// output past this length is dropped from the summary structure but
/// still flows through the bench's normal `[BENCH] Output:` line.
public let outputSampleLimit = 256

// MARK: - Output-match check

/// Check whether `cellOutput` is a token-prefix match of `baselineOutput`.
/// Spec-decode greedy-equivalence guarantees they should match exactly up
/// to `min(len(baseline), len(cell))` characters. We compare on character
/// prefix (not token prefix) for harness simplicity — the integration
/// callsite has already stripped detokenization artefacts equally on both
/// sides.
public func outputMatchesBaseline(baseline: String, cell: String) -> Bool {
    let limit = min(baseline.count, cell.count)
    guard limit > 0 else { return baseline.count == cell.count }
    return baseline.prefix(limit) == cell.prefix(limit)
}

// MARK: - Summary table renderer

/// Render the post-run markdown table the user sees at the end of a
/// `ngram-spot` invocation. Pure; stable to feed into snapshot tests.
public func formatSpotSummary(
    prompt: String,
    modelLabel: String,
    quantization: String,
    results: [NgramSpotResult]
) -> String {
    guard let baseline = results.first(where: { $0.cell.isBaseline }) else {
        return "[NGRAM-SPOT] no baseline cell — cannot compute speedup table"
    }
    let baselineRate = baseline.decodeTokPerSec
    let promptLine = prompt.split(separator: "\n").first.map(String.init) ?? prompt
    let promptTrimmed = promptLine.count > 64
        ? String(promptLine.prefix(64)) + "…"
        : promptLine

    var out = ""
    out += "[NGRAM-SPOT] \(promptTrimmed) @ \(modelLabel) \(quantization)\n"
    out += "| Cell                              | tok/s | Speedup | Accept | Match |\n"
    out += "|-----------------------------------|------:|--------:|-------:|:-----:|\n"
    for r in results {
        let label = r.cell.label.padding(toLength: 33, withPad: " ", startingAt: 0)
        let tps = String(format: "%6.1f", r.decodeTokPerSec)
        let speedup: String
        if r.cell.isBaseline {
            speedup = " 1.00×"
        } else if baselineRate > 0 {
            speedup = String(format: "%5.2f×", r.decodeTokPerSec / baselineRate)
        } else {
            speedup = "    —"
        }
        let accept = r.cell.isBaseline
            ? "    — "
            : String(format: "%5.1f%%", r.acceptanceRate * 100)
        let match: String
        if r.cell.isBaseline {
            match = " ref "
        } else {
            match = (r.matchesBaseline ?? false) ? "  ✓  " : "  ✗  "
        }
        out += "| \(label) | \(tps) | \(speedup) | \(accept) | \(match) |\n"
    }
    if let best = pickBestSpotCell(results: results), !best.cell.isBaseline {
        let speedup = baselineRate > 0
            ? (best.decodeTokPerSec / baselineRate - 1.0) * 100
            : 0
        out += String(format: "[NGRAM-SPOT] best: %@ (%+.1f%%)\n", best.cell.label, speedup)
    }
    return out
}

// MARK: - Best-cell picker

/// Pick the highest-throughput cell that still passes the output-match
/// check. The match check is a hard requirement — a faster-but-divergent
/// cell wouldn't be lossless and isn't a valid recommendation.
///
/// Returns `nil` when no cells are present. Returns the baseline when it
/// is the only matching cell (no spec-decode gain available on this
/// workload).
public func pickBestSpotCell(results: [NgramSpotResult]) -> NgramSpotResult? {
    let valid = results.filter { r in
        // Baseline always passes the match check (it's the reference) and
        // every cell that token-matches it is eligible.
        r.cell.isBaseline || r.matchesBaseline == true
    }
    return valid.max { a, b in a.decodeTokPerSec < b.decodeTokPerSec }
}

// MARK: - Sweep-summary roll-up (for `ngram-sweep-summary`)

/// One row of the sweep-summary mode: the best cell pick per (category, prompt).
public struct SweepBestPick: Equatable, Sendable {
    public let category: String
    public let promptName: String
    public let bestCell: NgramSpotCell
    public let baselineTokPerSec: Double
    public let bestTokPerSec: Double
    public var speedup: Double {
        baselineTokPerSec > 0 ? bestTokPerSec / baselineTokPerSec : 0
    }
}

/// One per-category aggregate row. The "default config" recommendation
/// for that category is the cell that wins the most prompts in the category;
/// ties broken by mean speedup.
public struct SweepCategorySummary: Equatable, Sendable {
    public let category: String
    public let promptCount: Int
    public let recommendedCell: NgramSpotCell
    public let meanSpeedup: Double
    public let medianSpeedup: Double
    public let minSpeedup: Double
    public let maxSpeedup: Double
}

/// Build the per-category summary from a flat list of best-pick results.
/// "Recommended cell" = the cell that wins the most prompts in that
/// category; ties broken by higher mean speedup.
public func summarizeSweepByCategory(_ picks: [SweepBestPick]) -> [SweepCategorySummary] {
    let byCategory = Dictionary(grouping: picks, by: \.category)
    var out: [SweepCategorySummary] = []
    for (category, rows) in byCategory.sorted(by: { $0.key < $1.key }) {
        let cellCounts = Dictionary(grouping: rows, by: \.bestCell)
            .mapValues { rows -> (count: Int, meanSpeedup: Double) in
                let mean = rows.map(\.speedup).reduce(0, +) / Double(rows.count)
                return (rows.count, mean)
            }
        let recommended = cellCounts.max { a, b in
            if a.value.count != b.value.count { return a.value.count < b.value.count }
            return a.value.meanSpeedup < b.value.meanSpeedup
        }?.key ?? rows.first!.bestCell

        let speedups = rows.map(\.speedup).sorted()
        let mean = speedups.reduce(0, +) / Double(speedups.count)
        let median = speedups[speedups.count / 2]
        out.append(SweepCategorySummary(
            category: category,
            promptCount: rows.count,
            recommendedCell: recommended,
            meanSpeedup: mean,
            medianSpeedup: median,
            minSpeedup: speedups.first ?? 0,
            maxSpeedup: speedups.last ?? 0))
    }
    return out
}

/// Render the per-category summary table at the end of a
/// `ngram-sweep-summary` run.
public func formatSweepSummary(_ summaries: [SweepCategorySummary]) -> String {
    var out = "[NGRAM-SWEEP-SUMMARY] per-category best-cell roll-up\n"
    out += "| Category             | Prompts | Best cell                      | Mean ×  | Median × | Range            |\n"
    out += "|----------------------|--------:|--------------------------------|--------:|---------:|------------------|\n"
    for s in summaries {
        let cat = s.category.padding(toLength: 20, withPad: " ", startingAt: 0)
        let cell = s.recommendedCell.label.padding(toLength: 30, withPad: " ", startingAt: 0)
        out += String(
            format: "| %@ | %7d | %@ | %6.2f× | %7.2f× | %.2f×–%.2f× |\n",
            cat, s.promptCount, cell, s.meanSpeedup, s.medianSpeedup,
            s.minSpeedup, s.maxSpeedup)
    }
    return out
}
