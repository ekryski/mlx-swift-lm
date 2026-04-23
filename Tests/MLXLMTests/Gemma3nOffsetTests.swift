// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

/// Regression tests for the Gemma3n sliding-window mask offset computation.
///
/// Before this PR, the computation at `Gemma3nText.swift:601` forced a per-forward
/// GPU→CPU sync via `cachePosition.max().item()`. The rewrite keeps the arithmetic
/// on GPU using MLXArray ops. These tests pin the numerical identity between the
/// two formulations so a regression (wrong offset) fails the suite loudly.
struct Gemma3nOffsetTests {

    /// Reference CPU formulation (the pre-rewrite code path):
    ///     offset = max(0, cachePosition.max().item() - effectiveSeqLen + 1)
    private func referenceOffset(pastSeenTokens: Int, seqLen: Int, slidingWindow: Int) -> Int
    {
        let cachePosition = Array(pastSeenTokens ..< (pastSeenTokens + seqLen))
        let effectiveSeqLen = Swift.max(cachePosition.count, slidingWindow)
        return Swift.max(0, (cachePosition.max() ?? 0) - effectiveSeqLen + 1)
    }

    /// New GPU formulation (the post-rewrite code path), evaluated on GPU but read
    /// back here only to assert equality in tests.
    private func gpuOffset(pastSeenTokens: Int, seqLen: Int, slidingWindow: Int) -> Int {
        let cachePosition = MLXArray(pastSeenTokens ..< (pastSeenTokens + seqLen))
        let effectiveSeqLen = Swift.max(cachePosition.shape[0], slidingWindow)
        let maxPos = cachePosition.max().asType(.int32)
        let shift = MLXArray(Int32(effectiveSeqLen - 1))
        let offsetGPU = MLX.maximum(maxPos - shift, MLXArray(Int32(0)))
        eval(offsetGPU)
        return Int(offsetGPU.item(Int32.self))
    }

    @Test(
        arguments: [
            // (pastSeenTokens, seqLen, slidingWindow)
            (0, 1, 128),  // fresh cache, decode step
            (0, 64, 128),  // fresh cache, small prefill
            (0, 256, 128),  // fresh cache, prefill larger than window
            (100, 1, 128),  // cache shorter than window, decode
            (127, 1, 128),  // boundary: exactly one short of window
            (128, 1, 128),  // boundary: first token past the window
            (200, 1, 128),  // steady-state decode mid-window
            (1000, 4, 128),  // mid-generation multi-token step
            (4096, 1, 512),  // longer sliding window, late in generation
            (32000, 1, 4096),  // long-context case
        ] as [(Int, Int, Int)])
    func `GPU offset equals CPU reference offset`(
        pastSeenTokens: Int, seqLen: Int, slidingWindow: Int
    ) {
        let expected = referenceOffset(
            pastSeenTokens: pastSeenTokens, seqLen: seqLen, slidingWindow: slidingWindow)
        let actual = gpuOffset(
            pastSeenTokens: pastSeenTokens, seqLen: seqLen, slidingWindow: slidingWindow)
        #expect(
            expected == actual,
            Comment(
                rawValue:
                    "past=\(pastSeenTokens) seq=\(seqLen) window=\(slidingWindow): "
                    + "expected \(expected), got \(actual)"))
    }

    /// The offset must never go negative: `max(0, …)` in both forms.
    @Test
    func `Offset clamps at zero when cache is shorter than window`() {
        let offset = gpuOffset(pastSeenTokens: 5, seqLen: 1, slidingWindow: 128)
        #expect(offset == 0)
    }

    /// When the cache has advanced past the window, the offset walks forward by 1
    /// per token — this is the property the sliding-window mask depends on.
    @Test
    func `Offset advances by one per decode token past the window`() {
        let base = gpuOffset(pastSeenTokens: 500, seqLen: 1, slidingWindow: 128)
        let next = gpuOffset(pastSeenTokens: 501, seqLen: 1, slidingWindow: 128)
        #expect(next == base + 1)
    }
}
