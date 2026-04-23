// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Batch decode iterator for concurrent request serving.
///
/// Steps multiple `TokenIterator` sessions together using the two-phase
/// `stepAsync()`/`readToken()` API. All forward graphs are built before
/// any GPU sync, letting MLX batch the Metal command buffers.
///
/// ```swift
/// var batch = BatchTokenIterator()
/// batch.addRequest(id: "r1", input: input1, model: model, parameters: params)
/// batch.addRequest(id: "r2", input: input2, model: model, parameters: params)
///
/// while batch.hasActive {
///     let results = batch.stepAll()
///     for (id, token) in results { /* process */ }
/// }
/// ```
public struct BatchTokenIterator {

    struct Session {
        var iterator: TokenIterator
        var finished: Bool = false
    }

    private var sessions: [String: Session] = [:]

    public init() {}

    public var activeCount: Int {
        sessions.values.count { !$0.finished }
    }

    public var hasActive: Bool {
        sessions.values.contains { !$0.finished }
    }

    public var requestIds: [String] {
        Array(sessions.keys)
    }

    /// Add a new request. Prefill runs synchronously inside TokenIterator.init.
    @discardableResult
    public mutating func addRequest(
        id: String,
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> Int {
        var iterator = try TokenIterator(
            input: input,
            model: model,
            parameters: parameters
        )
        guard let firstToken = iterator.next() else { return -1 }
        sessions[id] = Session(iterator: iterator)
        return firstToken
    }

    public mutating func removeRequest(id: String) {
        sessions.removeValue(forKey: id)
    }

    /// Step all active sessions with batched GPU execution.
    ///
    /// Phase 1: call `stepAsync()` on every active session — builds
    /// lazy forward graphs and queues asyncEval. No GPU sync yet.
    ///
    /// Phase 2: call `readToken()` on every session — forces GPU sync.
    /// By this point, MLX has had a chance to batch all N forward passes
    /// into fewer Metal command buffer submissions.
    public mutating func stepAll() -> [(String, Int?)] {
        let activeIds = sessions.keys.filter { !sessions[$0]!.finished }

        // Phase 1: build all graphs (no sync)
        var stepped: [String] = []
        for id in activeIds {
            guard var session = sessions[id] else { continue }
            if session.iterator.stepAsync() {
                stepped.append(id)
            } else {
                session.finished = true
            }
            sessions[id] = session
        }

        // Phase 2: read all results (sync happens here)
        var results: [(String, Int?)] = []
        for id in stepped {
            guard var session = sessions[id] else { continue }
            let token = session.iterator.readToken()
            sessions[id] = session
            results.append((id, token))
        }

        // Report finished sessions
        for id in activeIds where !stepped.contains(id) {
            results.append((id, nil))
        }

        return results
    }
}
