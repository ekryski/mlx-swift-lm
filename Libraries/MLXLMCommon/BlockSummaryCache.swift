// Copyright © 2026 Eric Kryski.
//
// Per-block K summaries for spec 034's K-side top-k decode selection.
// See specs/034-decode-side-kv-selection.md for the broader design.

import Foundation
import MLX

/// Per-block K summaries computed once at prefill end and queried at decode
/// time by a top-k selector. Both fields cover the first
/// `blockSize * blockCount` stored tokens; tokens beyond that range are
/// scored densely without a summary.
public struct BlockKSummaries {
    /// Per-block, per-dim mean of K. Shape `[B, kvHeads, blockCount, headDim]`.
    public let means: MLXArray

    /// Per-block, per-dim max of `|K|`. Same shape as `means`. Used by
    /// selectors that need a tighter upper bound than the mean.
    public let maxAbs: MLXArray

    /// Stored tokens per summary block.
    public let blockSize: Int

    public var blockCount: Int { means.dim(2) }
}

/// Caches that can expose per-block K summaries. Conformers compute the
/// summaries on demand (typically once at the end of prefill) and invalidate
/// them whenever the underlying K is mutated in a way that breaks coverage.
///
/// Selectors consume `blockSummaries` to estimate per-block relevance (e.g.,
/// `Q · means` for a block-mean LSH) before issuing exact SDPA against the
/// top-k blocks. The summary computation is cache-agnostic — only the K
/// extraction differs per backend (raw FP16 for `StandardKVCache`, packed
/// + dequantized for `TurboQuantizedKVCache` until the kernel-direct path
/// lands in spec 034 phase 7).
public protocol BlockSummaryCache: KVCache {
    /// Currently cached summaries, or `nil` if none have been computed (or
    /// they have been invalidated). Reads are O(1).
    var blockSummaries: BlockKSummaries? { get }

    /// Compute (or recompute) summaries over the currently stored K. Idempotent.
    /// Typically called once at prefill end.
    ///
    /// - Parameter blockSize: Tokens per summary block. 64 is the canonical
    ///   default — fine-grained enough for ~1.5% selection resolution at
    ///   typical context lengths, coarse enough to keep the summary tensor
    ///   at <2% of the K cache footprint.
    func computeBlockSummaries(blockSize: Int)

    /// Drop any cached summaries. Called automatically when the cache is
    /// mutated in a way that invalidates summary coverage (e.g., `trim`).
    func invalidateBlockSummaries()
}

extension BlockSummaryCache {
    /// Convenience overload using the canonical block size of 64.
    public func computeBlockSummaries() {
        computeBlockSummaries(blockSize: 64)
    }
}

/// Compute (mean, maxAbs) per contiguous block of `blockSize` tokens along
/// the sequence axis of a K tensor.
///
/// - Parameters:
///   - keys: K tensor with shape `[B, kvHeads, T, headDim]`. Only the first
///     `blockSize * (T / blockSize)` tokens contribute; any trailing partial
///     block is dropped (callers typically retain those as the dense decode
///     tail).
///   - blockSize: Tokens per summary block. Must be ≥ 1.
/// - Returns: Summaries with shape `[B, kvHeads, T / blockSize, headDim]` for
///   both `means` and `maxAbs`, or `nil` if `T < blockSize`.
public func computeKBlockSummaries(
    keys: MLXArray, blockSize: Int
) -> BlockKSummaries? {
    precondition(blockSize >= 1, "blockSize must be at least 1")
    precondition(keys.ndim == 4, "keys must be rank-4 [B, kvHeads, T, headDim]")

    let T = keys.dim(2)
    let blockCount = T / blockSize
    guard blockCount > 0 else { return nil }
    let usable = blockCount * blockSize

    let B = keys.dim(0)
    let kvHeads = keys.dim(1)
    let headDim = keys.dim(3)

    let truncated = (usable == T) ? keys : keys[.ellipsis, ..<usable, 0...]
    let reshaped = truncated.reshaped([B, kvHeads, blockCount, blockSize, headDim])

    let means = reshaped.mean(axis: 3)
    let maxAbs = abs(reshaped).max(axis: 3)

    return BlockKSummaries(means: means, maxAbs: maxAbs, blockSize: blockSize)
}
