// Copyright © 2026 Apple Inc.
//
// Pure-Swift unit tests for `NgramSpotMode.swift`. No GPU, no model load —
// these run as part of the regular `swift test` invocation (not under the
// `--filter benchmark` filter that drives the heavy bench cells).

import Foundation
import Testing

@Suite("NgramSpotMode — cell parsing")
struct NgramSpotModeCellParsingTests {

    @Test
    func `parseSpotCells handles n:D form`() {
        let cells = parseSpotCells("3:2")
        #expect(cells.count == 1)
        #expect(cells.first == NgramSpotCell(ngramSize: 3, maxDraftTokens: 2))
    }

    @Test
    func `parseSpotCells handles n:D:H form`() {
        let cells = parseSpotCells("4:8:2")
        #expect(cells.count == 1)
        #expect(cells.first == NgramSpotCell(
            ngramSize: 4, maxDraftTokens: 8, minHits: 2))
    }

    @Test
    func `parseSpotCells handles n:D:H:flags form`() {
        let cells = parseSpotCells("3:12:1:adaptive/strict")
        #expect(cells.count == 1)
        let c = cells.first!
        #expect(c.ngramSize == 3)
        #expect(c.maxDraftTokens == 12)
        #expect(c.minHits == 1)
        #expect(c.useAdaptive)
        #expect(c.useStrictGreedy)
        #expect(!c.useDominance)
        #expect(!c.useMultiCandidate)
    }

    @Test
    func `parseSpotCells handles all four flags`() {
        let cells = parseSpotCells("3:4:1:adaptive/strict/dominance/multi")
        let c = cells.first!
        #expect(c.useAdaptive)
        #expect(c.useStrictGreedy)
        #expect(c.useDominance)
        #expect(c.useMultiCandidate)
    }

    @Test
    func `parseSpotCells silently ignores unknown flags`() {
        // Forward-compat — adding new flags shouldn't break older parsers.
        let cells = parseSpotCells("3:4:1:adaptive/futuristic-flag")
        #expect(cells.first?.useAdaptive == true)
        #expect(cells.first?.useStrictGreedy == false)
    }

    @Test
    func `parseSpotCells handles comma-separated multi-cell list`() {
        let cells = parseSpotCells("3:2,3:4,3:8")
        #expect(cells.count == 3)
        #expect(cells.map(\.maxDraftTokens) == [2, 4, 8])
    }

    @Test
    func `parseSpotCells matches the default cell list shape`() {
        // Round-trip: encoded form of the default cells parses back into
        // the same NgramSpotCell values. Locks the default list against
        // accidental drift.
        let encoded = "3:2,3:4,3:8,3:12:1:adaptive/strict"
        let cells = parseSpotCells(encoded)
        #expect(cells == defaultSpotCells)
    }

    @Test
    func `parseSpotCells rejects malformed entries silently`() {
        // Single-token spec — no `:` — should be dropped.
        #expect(parseSpotCells("3").isEmpty)
        // Non-integer fields dropped.
        #expect(parseSpotCells("foo:bar").isEmpty)
        // Zero / negative ngramSize dropped.
        #expect(parseSpotCells("0:4").isEmpty)
        // One bad entry doesn't poison the list — others survive.
        let mixed = parseSpotCells("3:4,bogus,4:2")
        #expect(mixed.count == 2)
        #expect(mixed.map(\.ngramSize) == [3, 4])
    }

    @Test
    func `baseline cell never parses out of a spec`() {
        // We always prepend the baseline ourselves; users shouldn't be able
        // to inject one via the cell spec (would silently produce two
        // baselines and break the "baseline = first cell" invariant).
        let cells = parseSpotCells("0:0")
        #expect(cells.isEmpty)
    }
}

@Suite("NgramSpotMode — output match check")
struct NgramSpotModeMatchTests {

    @Test
    func `outputMatchesBaseline accepts identical strings`() {
        #expect(outputMatchesBaseline(
            baseline: "the cat sat on the mat",
            cell: "the cat sat on the mat"))
    }

    @Test
    func `outputMatchesBaseline accepts cell as prefix of baseline`() {
        // Cell may have stopped at EOS earlier. Spec-decode greedy guarantees
        // its emitted tokens match the baseline up to its length.
        #expect(outputMatchesBaseline(
            baseline: "the cat sat on the mat",
            cell: "the cat sat"))
    }

    @Test
    func `outputMatchesBaseline accepts baseline as prefix of cell`() {
        // Inverse direction — if the baseline EOS'd earlier, the cell is
        // still valid as long as it agrees on the shared prefix.
        #expect(outputMatchesBaseline(
            baseline: "the cat sat",
            cell: "the cat sat on the mat"))
    }

    @Test
    func `outputMatchesBaseline rejects divergent strings`() {
        #expect(!outputMatchesBaseline(
            baseline: "the cat sat on the mat",
            cell: "the dog sat on the mat"))
    }

    @Test
    func `outputMatchesBaseline handles empty inputs`() {
        // Both empty — vacuously true.
        #expect(outputMatchesBaseline(baseline: "", cell: ""))
        // One empty, one not — different lengths but same (empty) prefix.
        // The current contract returns `false` in that case because the
        // baseline produced output and the cell produced none — that's a
        // real divergence.
        #expect(!outputMatchesBaseline(baseline: "abc", cell: ""))
        #expect(!outputMatchesBaseline(baseline: "", cell: "abc"))
    }
}

@Suite("NgramSpotMode — best-cell picker")
struct NgramSpotModeBestCellTests {

    /// Helper to build a fixture list. Default `matchesBaseline` is `true`
    /// for non-baseline cells unless overridden — most fixtures want
    /// matching cells.
    private func makeResult(
        cell: NgramSpotCell,
        decodeTokPerSec: Double,
        accepted: Int = 0,
        proposed: Int = 0,
        matchesBaseline: Bool? = nil
    ) -> NgramSpotResult {
        NgramSpotResult(
            cell: cell,
            decodeTokPerSec: decodeTokPerSec,
            steadyTokPerSec: decodeTokPerSec,
            accepted: accepted,
            proposed: proposed,
            outputSample: "",
            matchesBaseline: cell.isBaseline ? nil : (matchesBaseline ?? true))
    }

    @Test
    func `picks fastest matching cell over baseline when both pass match`() {
        let results = [
            makeResult(cell: .baseline, decodeTokPerSec: 27.0),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
                decodeTokPerSec: 34.0),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 4),
                decodeTokPerSec: 31.0),
        ]
        let best = pickBestSpotCell(results: results)
        #expect(best?.cell == NgramSpotCell(ngramSize: 3, maxDraftTokens: 2))
        #expect(best?.decodeTokPerSec == 34.0)
    }

    @Test
    func `excludes mismatching cells from contention`() {
        // Faster but divergent cell must NOT be picked. Spec-decode is
        // supposed to be lossless — a divergent cell is broken by definition.
        let results = [
            makeResult(cell: .baseline, decodeTokPerSec: 27.0),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
                decodeTokPerSec: 34.0,
                matchesBaseline: false),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 4),
                decodeTokPerSec: 31.0,
                matchesBaseline: true),
        ]
        let best = pickBestSpotCell(results: results)
        // The cheating cell (D=2 at 34 tok/s but mismatch) is filtered out.
        #expect(best?.cell == NgramSpotCell(ngramSize: 3, maxDraftTokens: 4))
    }

    @Test
    func `picks baseline when no spec cell beats it`() {
        let results = [
            makeResult(cell: .baseline, decodeTokPerSec: 27.0),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
                decodeTokPerSec: 25.0),
            makeResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 4),
                decodeTokPerSec: 24.0),
        ]
        let best = pickBestSpotCell(results: results)
        #expect(best?.cell.isBaseline == true)
    }

    @Test
    func `returns nil on empty input`() {
        #expect(pickBestSpotCell(results: []) == nil)
    }
}

@Suite("NgramSpotMode — summary table renderer")
struct NgramSpotModeSummaryTests {

    private func sampleResults() -> [NgramSpotResult] {
        [
            NgramSpotResult(
                cell: .baseline,
                decodeTokPerSec: 27.3, steadyTokPerSec: 27.5,
                accepted: 0, proposed: 0,
                outputSample: "Hello world",
                matchesBaseline: nil),
            NgramSpotResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
                decodeTokPerSec: 34.1, steadyTokPerSec: 34.2,
                accepted: 6, proposed: 10,
                outputSample: "Hello world",
                matchesBaseline: true),
            NgramSpotResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 4),
                decodeTokPerSec: 31.0, steadyTokPerSec: 31.1,
                accepted: 5, proposed: 12,
                outputSample: "Hello world",
                matchesBaseline: true),
        ]
    }

    @Test
    func `summary contains baseline reference row`() {
        let out = formatSpotSummary(
            prompt: "Test prompt",
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: sampleResults())
        #expect(out.contains("n=0 D=0"))
        #expect(out.contains(" 1.00×"))
        #expect(out.contains(" ref "))
    }

    @Test
    func `summary computes speedup vs baseline`() {
        let out = formatSpotSummary(
            prompt: "Test prompt",
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: sampleResults())
        // n=3 D=2 at 34.1 vs baseline 27.3 → 1.249× ≈ "1.25×"
        #expect(out.contains("1.25×"))
        // n=3 D=4 at 31.0 vs baseline 27.3 → 1.135× ≈ "1.14×"
        #expect(out.contains("1.14×"))
    }

    @Test
    func `summary computes acceptance rate`() {
        let out = formatSpotSummary(
            prompt: "Test prompt",
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: sampleResults())
        // 6/10 = 60.0%
        #expect(out.contains(" 60.0%"))
        // 5/12 = 41.667%
        #expect(out.contains(" 41.7%"))
    }

    @Test
    func `summary picks best non-baseline cell at the bottom`() {
        let out = formatSpotSummary(
            prompt: "Test prompt",
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: sampleResults())
        #expect(out.contains("[NGRAM-SPOT] best: n=3 D=2"))
    }

    @Test
    func `summary trims long prompts to 64 chars`() {
        let longPrompt = String(repeating: "long prompt content ", count: 20)
        let out = formatSpotSummary(
            prompt: longPrompt,
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: sampleResults())
        // Find the prompt line — the first line of the summary.
        let firstLine = out.split(separator: "\n").first.map(String.init) ?? ""
        // Should have the ellipsis marker at the prompt boundary.
        #expect(firstLine.contains("…"))
    }

    @Test
    func `summary emits short message when no baseline present`() {
        let resultsWithoutBaseline = [
            NgramSpotResult(
                cell: NgramSpotCell(ngramSize: 3, maxDraftTokens: 2),
                decodeTokPerSec: 34.0, steadyTokPerSec: 34.0,
                accepted: 6, proposed: 10,
                outputSample: "",
                matchesBaseline: true),
        ]
        let out = formatSpotSummary(
            prompt: "Test prompt",
            modelLabel: "Gemma 4 26B A4B",
            quantization: "4bit",
            results: resultsWithoutBaseline)
        #expect(out.contains("no baseline cell"))
    }
}

@Suite("NgramSpotMode — sweep-summary roll-up")
struct NgramSpotModeSweepSummaryTests {

    private let n3D2 = NgramSpotCell(ngramSize: 3, maxDraftTokens: 2)
    private let n3D4 = NgramSpotCell(ngramSize: 3, maxDraftTokens: 4)
    private let n4D2 = NgramSpotCell(ngramSize: 4, maxDraftTokens: 2)

    @Test
    func `categorizes picks by category`() {
        let picks = [
            SweepBestPick(category: "qa-requote", promptName: "01-bug-report",
                          bestCell: n3D2, baselineTokPerSec: 27.0, bestTokPerSec: 34.0),
            SweepBestPick(category: "qa-requote", promptName: "02-recipe",
                          bestCell: n3D2, baselineTokPerSec: 26.0, bestTokPerSec: 33.0),
            SweepBestPick(category: "code-refactor", promptName: "01-python",
                          bestCell: n4D2, baselineTokPerSec: 28.0, bestTokPerSec: 32.0),
        ]
        let summary = summarizeSweepByCategory(picks)
        #expect(summary.count == 2)
        #expect(summary[0].category == "code-refactor")
        #expect(summary[1].category == "qa-requote")
    }

    @Test
    func `recommends most-frequent-winning cell per category`() {
        let picks = [
            // qa-requote: n3D2 wins twice, n3D4 wins once.
            SweepBestPick(category: "qa-requote", promptName: "p1",
                          bestCell: n3D2, baselineTokPerSec: 27.0, bestTokPerSec: 34.0),
            SweepBestPick(category: "qa-requote", promptName: "p2",
                          bestCell: n3D2, baselineTokPerSec: 27.0, bestTokPerSec: 33.0),
            SweepBestPick(category: "qa-requote", promptName: "p3",
                          bestCell: n3D4, baselineTokPerSec: 27.0, bestTokPerSec: 30.0),
        ]
        let summary = summarizeSweepByCategory(picks)
        #expect(summary[0].recommendedCell == n3D2)
    }

    @Test
    func `tie-breaks by mean speedup`() {
        // Each cell wins exactly one prompt — tie on count.
        // n3D2: 1.5× speedup. n3D4: 1.1× speedup. n3D2 should win the tiebreak.
        let picks = [
            SweepBestPick(category: "test", promptName: "p1",
                          bestCell: n3D2, baselineTokPerSec: 20.0, bestTokPerSec: 30.0),
            SweepBestPick(category: "test", promptName: "p2",
                          bestCell: n3D4, baselineTokPerSec: 20.0, bestTokPerSec: 22.0),
        ]
        let summary = summarizeSweepByCategory(picks)
        #expect(summary[0].recommendedCell == n3D2)
    }

    @Test
    func `category summary computes range correctly`() {
        let picks = [
            SweepBestPick(category: "x", promptName: "p1",
                          bestCell: n3D2, baselineTokPerSec: 10.0, bestTokPerSec: 15.0),
            SweepBestPick(category: "x", promptName: "p2",
                          bestCell: n3D2, baselineTokPerSec: 10.0, bestTokPerSec: 20.0),
            SweepBestPick(category: "x", promptName: "p3",
                          bestCell: n3D2, baselineTokPerSec: 10.0, bestTokPerSec: 18.0),
        ]
        let summary = summarizeSweepByCategory(picks)
        let s = summary[0]
        #expect(s.minSpeedup == 1.5)
        #expect(s.maxSpeedup == 2.0)
        // Mean = (1.5 + 2.0 + 1.8) / 3 ≈ 1.7667
        #expect(abs(s.meanSpeedup - 1.7666666) < 0.001)
    }

    @Test
    func `formatSweepSummary produces well-formed table`() {
        let picks = [
            SweepBestPick(category: "qa-requote", promptName: "p1",
                          bestCell: n3D2, baselineTokPerSec: 27.0, bestTokPerSec: 34.0),
        ]
        let summary = summarizeSweepByCategory(picks)
        let rendered = formatSweepSummary(summary)
        #expect(rendered.contains("[NGRAM-SWEEP-SUMMARY]"))
        #expect(rendered.contains("qa-requote"))
        #expect(rendered.contains("n=3 D=2"))
    }
}
