// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Batched decode for concurrent request serving.
///
/// Strategy: batch the COMPUTE (embedding, projections, MLP, lm_head)
/// while keeping attention PER-REQUEST (different cache lengths).
///
/// For a typical transformer decode step:
///   - QKV projection: O(3 * hidden^2) — BATCHED across B requests
///   - Attention: O(seq_len * head_dim) — per-request (cheap during decode)
///   - MLP: O(8 * hidden^2) — BATCHED across B requests
///   - LM head: O(hidden * vocab) — BATCHED across B requests
///
/// Total: ~92% of FLOPS are batched (projections + MLP + lm_head),
/// ~8% are per-request (attention against cached K/V).
public class BatchedDecoder {

    private let model: Module
    private var sessions: [BatchSession] = []

    public struct BatchSession {
        public let id: String
        public var cache: [KVCache]
        public var lastToken: MLXArray  // [1] — last generated token

        public init(id: String, cache: [KVCache], lastToken: MLXArray) {
            self.id = id
            self.cache = cache
            self.lastToken = lastToken
        }
    }

    public init(model: Module) {
        self.model = model
    }

    public var count: Int { sessions.count }

    public func addSession(_ session: BatchSession) {
        sessions.append(session)
    }

    public func removeSession(id: String) {
        sessions.removeAll { $0.id == id }
    }

    /// Step all sessions. Returns [(id, token_id)].
    ///
    /// Currently runs sequential forward passes but defers all eval
    /// to a single batch. Future: true batched projection.
    public func stepAll(
        using forward: (MLXArray, [KVCache]) -> MLXArray,
        sampler: (MLXArray) -> MLXArray
    ) -> [(String, Int)] {
        guard !sessions.isEmpty else { return [] }

        // Build all forward graphs (lazy — no eval yet)
        var pendingLogits: [(Int, MLXArray)] = []
        for (idx, session) in sessions.enumerated() {
            let input = session.lastToken[.newAxis]  // [1, 1]
            let logits = forward(input, session.cache)  // lazy
            pendingLogits.append((idx, logits))
        }

        // Sample all (lazy)
        var pendingTokens: [(Int, MLXArray)] = []
        for (idx, logits) in pendingLogits {
            let lastLogits = logits[0..., -1, 0...]  // [1, vocab] → [vocab]
            let token = sampler(lastLogits)  // lazy
            pendingTokens.append((idx, token))
        }

        // Single eval for all tokens
        let allTokenArrays = pendingTokens.map { $0.1 }
        eval(allTokenArrays)

        // Read results and update state
        var results: [(String, Int)] = []
        for (idx, tokenArray) in pendingTokens {
            let tokenId = tokenArray.item(Int.self)
            sessions[idx].lastToken = tokenArray
            results.append((sessions[idx].id, tokenId))
        }

        return results
    }
}
