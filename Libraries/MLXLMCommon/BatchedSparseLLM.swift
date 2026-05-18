// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Conformance hook for pure-attention LLMs that can route decode through
// the batched sparse attention path. The bridge / engine `as?`-casts a
// loaded model to this protocol and dispatches `fullyBatchedSparseDecode`
// when a `BatchedRetrievalAttentionKVCache` list exists for the active
// session pool.

import Foundation
import MLX

/// Models that can decode one step in a single batched call with sparse
/// attention should conform to this protocol. The decode path expects a
/// pre-built per-layer `BatchedRetrievalAttentionKVCache` list — the
/// caller is responsible for cache lifetime (slot add / remove / reset).
///
/// Inputs:
///   - `[B, 1]` token IDs.
///
/// Returns:
///   - `[B, 1, vocab]` logits.
public protocol BatchedSparseLLM: AnyObject {
    func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray
}
