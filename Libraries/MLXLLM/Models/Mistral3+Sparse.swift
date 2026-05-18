// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedSparseLLM conformance for Mistral3TextModel. Pairs with the
// shared `Mistral3.{Attention, TransformerBlock, ModelInner}.fullyBatchedSparseForward`
// hooks in MLXLMCommon/Models/Mistral3+Sparse.swift.

import Foundation
import MLX
import MLXLMCommon

extension Mistral3TextModel: BatchedSparseLLM {

    /// Single-step batched sparse decode. Per-layer
    /// `BatchedRetrievalAttentionKVCache` list must match `model.layers.count`.
    /// Llama-4 scaling and sliding-vs-full layer dispatch are handled inside
    /// the shared `Mistral3.ModelInner` extension.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }
}
