// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedSparseLLM conformance for Qwen2Model. Pairs with the
// `Qwen2.{Attention, DecoderLayer, ModelInner}.fullyBatchedSparseForward`
// hooks in the shared layer stack (MLXLMCommon/Models/Qwen2+Sparse.swift).

import Foundation
import MLX
import MLXLMCommon

extension Qwen2Model: BatchedSparseLLM {

    /// Single-step batched sparse decode. The caller is responsible for
    /// providing a per-layer `BatchedRetrievalAttentionKVCache` list whose
    /// inner caches have B slots reserved and per-slot offsets that reflect
    /// completed prefill.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }
}
