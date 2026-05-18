// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Configuration + Johnson-Lindenstrauss projection helper for the batched
// retrieval-attention path (selector index + sparse-aware KV cache).
//
// This file contains the minimal subset of the broader retrieval-attention
// infrastructure required by the BATCHED decode path. Per-request (B=1)
// retrieval-attention machinery (cache, kernel zoo, scheduler) is not
// included — those types are not used by the batched stack here.

import Foundation
import MLX

// MARK: - Config

/// Tunables for block-sparse retrieval attention. Defaults are the same
/// ship configuration validated in the broader retrieval-attention work
/// (static prefix 128, sliding window 2048, fine-block size 64, fine top-K
/// 32 with adaptive scaling, coarse rescue 2x1024 enabled).
///
/// The batched sparse decode path is BW-bound by K/V loads at long context;
/// the selector picks per-(B, KV-head) top-K blocks and dispatches a
/// fused fp16 additive mask + MLXFast SDPA so the SDPA tiled kernel walks
/// the masked region as a no-op.
public struct RetrievalAttentionConfig: Sendable {

    // ----- Static / sliding window

    /// Number of initial tokens always attended.
    public var staticInit: Int = 128

    /// Number of trailing tokens always attended.
    public var slidingWindow: Int = 2048

    // ----- Fine block retrieval

    /// Tokens per fine block. 64 is the published sweet spot for
    /// block-sparse attention (NSA/SeerAttention) — small enough for
    /// fine selectivity, large enough for coalesced gather.
    public var fineBlockSize: Int = 64

    /// Floor for fine top-K. When `adaptiveTopK` is true, the effective
    /// top-K at gather time is `max(fineTopK, ceil(seqLen / divisor))`.
    public var fineTopK: Int = 32

    /// Scale fineTopK with sequence length at gather time.
    public var adaptiveTopK: Bool = true

    /// Divisor for adaptive top-K scaling. `topK = ceil(seqLen / divisor)`.
    /// 256 = sweet spot on long-context Qwen at decode.
    public var adaptiveTopKDivisor: Int = 256

    /// Compute the effective fine top-K to use at gather time given the
    /// current cache length.
    public func effectiveFineTopK(seqLen: Int) -> Int {
        guard adaptiveTopK else { return fineTopK }
        let scaled = (seqLen + adaptiveTopKDivisor - 1) / adaptiveTopKDivisor
        return max(fineTopK, scaled)
    }

    // ----- Coarse rescue

    /// Tokens per coarse rescue block.
    public var coarseBlockSize: Int = 1024

    /// Number of coarse blocks to retrieve per query.
    public var coarseTopK: Int = 2

    /// Whether the coarse-rescue branch is enabled.
    public var coarseRescueEnabled: Bool = true

    // ----- Selector dim

    /// JL content-projection output dim.
    public var contentDim: Int = 16

    /// Trig position-features dim (sin/cos of relative positions). Unused
    /// in the batched path (selector is pure content / λ=0). Kept on the
    /// config struct for symmetry with the non-batched surface.
    public var trigDim: Int = 16

    /// Total selector dim — content + trig.
    public var selectorDim: Int { contentDim + trigDim }

    /// Content-vs-position blend. 0 = pure content (the ship default —
    /// no trig features computed, indexB asserts λ == 0).
    public var lambdaPos: Float = 0.0

    /// True if the selector path needs to compute / store trig features.
    /// Always false at λ=0 (the batched path's only supported mode).
    public var usesTrigFeatures: Bool { lambdaPos > 0.0 }

    /// Effective stored selector width — equals `contentDim` at λ=0.
    public var effectiveSelectorDim: Int {
        usesTrigFeatures ? selectorDim : contentDim
    }

    // ----- Per-layer hybrid

    /// Number of initial dense layers (run dense, skip sparse selector).
    public var denseFirstN: Int = 4

    /// Number of trailing dense layers.
    public var denseLastN: Int = 4

    /// True if this layer index should run sparse attention. Layers at
    /// the front (first-N) and back (last-N) of the stack stay dense —
    /// they're the high-recall layers in practice.
    public func isSparseLayer(layerIdx: Int, totalLayers: Int) -> Bool {
        guard layerIdx >= denseFirstN else { return false }
        guard layerIdx < totalLayers - denseLastN else { return false }
        return true
    }

    /// Minimum cache size before sparse kicks in. Below this the selector
    /// + mask construction overhead outweighs the BW saving from masking
    /// out unselected positions.
    public var sparseMinContext: Int = 16384

    // ----- Sparse prefill (chunked attention with L > 1)

    /// F-83 — enable sparse prefill (chunked attention with prior-chunks
    /// gather + within-chunk dense). When false, prefill falls through
    /// to dense SDPA. Default false to keep existing behavior; opt-in
    /// per cache (consumer can flip via `VSM_SPARSE_PREFILL=1` env or by
    /// passing a flipped config when constructing the cache).
    public var sparsePrefillEnabled: Bool = false

    /// F-83 — minimum prior cache length (in tokens) before sparse
    /// prefill engages. Below this, the chunk runs dense. Mirrors
    /// `sparseMinContext` for the decode path. Default 16384 = ~2.6x
    /// dedupe pre-budget at default config.
    public var sparsePrefillMinContext: Int = 16384

    public init() {}

    /// Deterministic seed for the JL random projection. Same matrix
    /// across prefill / decode / chunk boundaries — critical for the
    /// dot-product preservation argument in JL.
    public static var jlSeed: UInt32 { 0x4F50_4E53 }  // "OPNS"
}

// MARK: - JL projection

/// Build the Johnson-Lindenstrauss random-projection matrix used to
/// compress d_head down to `contentDim`. Entries are i.i.d.
/// `N(0, 1/contentDim)` — preserves `E[||Wx||^2] = ||x||^2`. Same matrix
/// applied to both Q and K so `(Wq) . (Wk) = q . (W^T W) . k ~ q . k`
/// in expectation.
///
/// - Parameters:
///   - dHead: head dim of the underlying model.
///   - config: backend config (for `contentDim` and JL seed).
/// - Returns: `[contentDim, dHead]` fp32 array.
public func retrievalAttentionJLProjection(
    dHead: Int,
    config: RetrievalAttentionConfig = RetrievalAttentionConfig()
) -> MLXArray {
    // Use a scoped RandomState seeded with the JL seed so we don't perturb
    // the caller's global PRNG. Sampling with `key:` scopes randomness.
    let key = MLXRandom.key(UInt64(RetrievalAttentionConfig.jlSeed))
    let scale = Float(1.0 / Float(config.contentDim).squareRoot())
    let W = MLXRandom.normal([config.contentDim, dHead], key: key) * scale
    return W.asType(.float32)
}
