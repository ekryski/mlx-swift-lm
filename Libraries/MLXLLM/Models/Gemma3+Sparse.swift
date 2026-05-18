// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedSparseLLM conformance for Gemma3TextModel. Pairs with the shared
// `Gemma3.{Attention, TransformerBlock, Backbone}.fullyBatchedSparseForward`
// hooks in MLXLMCommon/Models/Gemma3+Sparse.swift.

import Foundation
import MLX
import MLXLMCommon

extension Gemma3TextModel: BatchedSparseLLM {

    /// Single-step batched sparse decode. Per-layer
    /// `BatchedRetrievalAttentionKVCache` list must match
    /// `model.layers.count`. Sliding vs global layer mix is reflected in
    /// each layer's own RoPE — the dense fallback path uses the inner
    /// cache's `getCachedWithMask` which respects per-layer offsets.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        return lmHead(out)
    }
}
