// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Tier 2 HTTP client — POSTs evicted spans to longctx-svc /evict/write
// when V3's eviction round fires. Mirrors the Python callback at
// vllm-turboquant/vllm/v1/attention/triattention/backend_helpers.py:
// _evict_to_longctx_callback.
//
// Wire shape (matches longctx_svc/app.py:138-200):
//
//   POST /evict/write
//   {
//     "session_id": "v3-single-session",
//     "chunks": [
//       {"text": "...", "token_range": [start, end], "layer": -1, "score": 0.0}
//     ]
//   }
//
//   POST /evict/retrieve
//   {"session_id": "...", "query": "...", "top_k": 8, "score_floor": 0.0}
//   -> {"chunks": [...], "session_total": N}
//
// Tokenizer access lives outside this module (Bridge.swift owns it
// because it has the Swift Tokenizers handle). Caller decodes evicted
// positions to text spans before handing them to `writeEvicted`.
import Foundation

/// One evicted span ready to ship to longctx-svc.
public struct EvictionSpan: Sendable {
    public var text: String
    public var tokenStart: Int
    public var tokenEnd: Int
    public var layer: Int
    public var score: Float

    public init(
        text: String, tokenStart: Int, tokenEnd: Int,
        layer: Int = -1, score: Float = 0.0
    ) {
        self.text = text
        self.tokenStart = tokenStart
        self.tokenEnd = tokenEnd
        self.layer = layer
        self.score = score
    }
}

/// Retrieved chunk from /evict/retrieve.
public struct RetrievedSpan: Sendable, Decodable {
    public var text: String
    public var tokenRange: [Int]
    public var layer: Int
    public var score: Float

    enum CodingKeys: String, CodingKey {
        case text
        case tokenRange = "token_range"
        case layer
        case score
    }
}

/// HTTP client for the longctx-svc /evict/* routes. Reads the base URL
/// from `LONGCTX_ENDPOINT` lazily; returns nil endpoints make every
/// method a no-op so the engine can leave the callback installed without
/// caring whether longctx is wired this run.
public final class TriAttentionLongctxClient: @unchecked Sendable {

    public static let shared = TriAttentionLongctxClient()

    public var sessionID: String = "v3-single-session"
    public var timeoutSeconds: TimeInterval = 10.0
    private let session: URLSession

    private init() {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 10.0
        cfg.timeoutIntervalForResource = 10.0
        self.session = URLSession(configuration: cfg)
    }

    private var baseURL: URL? {
        guard let s = ProcessInfo.processInfo.environment["LONGCTX_ENDPOINT"],
              !s.isEmpty,
              let u = URL(string: s)
        else { return nil }
        return u
    }

    /// POST /evict/write. Synchronous (callback is already off the hot
    /// decode path — fires only when V3's policy finalizes a round).
    /// Errors are logged + swallowed so a misbehaving rescue path can't
    /// crash decoding. Returns true on 2xx.
    @discardableResult
    public func writeEvicted(_ spans: [EvictionSpan]) -> Bool {
        guard let base = baseURL, !spans.isEmpty else { return false }
        let url = base.appendingPathComponent("evict/write")
        let body: [String: Any] = [
            "session_id": sessionID,
            "chunks": spans.map { s -> [String: Any] in
                [
                    "text": s.text,
                    "token_range": [s.tokenStart, s.tokenEnd],
                    "layer": s.layer,
                    "score": s.score,
                ]
            },
        ]
        return postJSON(url: url, body: body, expectChunks: false) != nil
    }

    /// POST /evict/retrieve. Returns the decoded chunks list, or empty
    /// on any error (network / non-2xx / decode).
    public func retrieveEvicted(
        query: String, topK: Int = 8, scoreFloor: Float = 0.0
    ) -> [RetrievedSpan] {
        guard let base = baseURL else { return [] }
        let url = base.appendingPathComponent("evict/retrieve")
        let body: [String: Any] = [
            "session_id": sessionID,
            "query": query,
            "top_k": topK,
            "score_floor": scoreFloor,
        ]
        guard let data = postJSON(url: url, body: body, expectChunks: true)
        else { return [] }
        struct RetrieveResp: Decodable {
            let chunks: [RetrievedSpan]
            let sessionTotal: Int
            enum CodingKeys: String, CodingKey {
                case chunks
                case sessionTotal = "session_total"
            }
        }
        do {
            return try JSONDecoder().decode(
                RetrieveResp.self, from: data).chunks
        } catch {
            return []
        }
    }

    /// Synchronous POST, returns response body on 2xx, nil otherwise.
    /// Uses a semaphore to block — this is fine because callers are
    /// already off the hot decode path (V3 finalize_evict_round, or
    /// the Tier 3 prefill hook which runs before the model forward).
    private func postJSON(
        url: URL, body: [String: Any], expectChunks: Bool
    ) -> Data? {
        guard let payload = try? JSONSerialization.data(withJSONObject: body)
        else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = payload
        req.timeoutInterval = timeoutSeconds

        var responseData: Data? = nil
        var responseStatus = 0
        let sem = DispatchSemaphore(value: 0)
        let task = session.dataTask(with: req) { data, resp, err in
            defer { sem.signal() }
            if err != nil { return }
            if let http = resp as? HTTPURLResponse {
                responseStatus = http.statusCode
                if (200..<300).contains(http.statusCode) {
                    responseData = data
                }
            }
        }
        task.resume()
        _ = sem.wait(timeout: .now() + timeoutSeconds)
        if responseStatus != 0 && !(200..<300).contains(responseStatus) {
            // Non-2xx — caller will treat as failure (returns nil).
        }
        return responseData
    }
}
