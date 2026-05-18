// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Conformance hook for hybrid LLMs (interleaved attention + GDN / Mamba)
// that can route their attention sublayers through the batched sparse
// decode path. The bridge `as?`-casts a loaded model to this protocol
// and dispatches `fullyBatchedSparseDecode` when its session pool has
// a `BatchedHybridCache` whose attention slots use `.sparseAttention`.

import Foundation
import MLX

/// Hybrid models (e.g. Qwen 3.5 / 3.6, Nemotron H) that can interleave a
/// batched GDN / Mamba path with a batched sparse attention path. The
/// caller builds a `BatchedHybridCache` whose layers mix
/// `.sparseAttention(...)` for attention layers and `.gdn(...)` for
/// linear layers.
///
/// Inputs:
///   - `[B, 1]` token IDs.
///
/// Returns:
///   - `[B, 1, vocab]` logits.
public protocol BatchedHybridSparseLLM: BatchedHybridLLM {

    /// Single-step batched decode with sparse attention on attention layers.
    /// Per-layer dispatch is the model's responsibility — `.gdn` cases route
    /// to the existing GDN batched path, `.sparseAttention` cases route to
    /// the batched sparse decode site.
    func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        caches: BatchedHybridCache
    ) -> MLXArray

    /// Build a fresh `BatchedHybridCache` whose attention layers use
    /// `.sparseAttention(BatchedRetrievalAttentionKVCache)` wrappers built
    /// from the supplied `raConfig`. Linear (GDN / Mamba) layers stay on
    /// the standard `.gdn(BatchedMambaCache)` path.
    func newBatchedHybridSparseCache(
        maxBatch: Int,
        parameters: GenerateParameters?,
        raConfig: RetrievalAttentionConfig
    ) -> BatchedHybridCache
}
