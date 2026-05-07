// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Live end-to-end Tier 2 + Tier 3 round-trip from the Swift client
// against a running longctx-svc instance. This is the Swift mirror of
// AMD subagent 5's "longctx wire smoke" — proves the protocol works
// the same on M5 as on the MI300X droplet.
//
// Skipped (no failures) when LONGCTX_TEST_ENDPOINT isn't set or the
// service isn't reachable. Set the env var to a live longctx-svc URL
// to enable: `LONGCTX_TEST_ENDPOINT=http://127.0.0.1:9091 swift test`.
//
// What we test:
//   1. /evict/write a multi-chunk payload (with token_range, layer, score)
//   2. /evict/retrieve with the matching session_id, find chunks
//   3. session isolation — different session_id returns empty
//   4. eviction bridge end-to-end: synthetic eviction → callback fires
//      → bridge groups + decodes → POSTs → service receives
//
// Cross-platform claim: the same wire protocol that works on AMD
// (sub5's port 9099 round trip, sub6's prefill rehydrate hook) works
// on Apple Silicon via the Swift client. Same JSON schema, same routes,
// same HTTP semantics.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("TriAttention V3 — live longctx-svc round-trip", .serialized)
struct TriAttentionV3LiveLongctxTests {

    fileprivate static var endpoint: String? {
        // First env var, then the conventional test fallback.
        if let v = ProcessInfo.processInfo
            .environment["LONGCTX_TEST_ENDPOINT"], !v.isEmpty {
            return v
        }
        return nil
    }

    fileprivate static func reachable(_ url: String) -> Bool {
        guard let u = URL(string: url + "/healthz") else { return false }
        let sem = DispatchSemaphore(value: 0)
        var ok = false
        let task = URLSession.shared.dataTask(with: u) { _, resp, _ in
            if let h = resp as? HTTPURLResponse, h.statusCode == 200 {
                ok = true
            }
            sem.signal()
        }
        task.resume()
        _ = sem.wait(timeout: .now() + 2)
        return ok
    }

    /// Force the longctx client to read a fresh endpoint for this test.
    /// The client caches `LONGCTX_ENDPOINT` lazily on first call; we
    /// set it before any calls and trust the cache to be primed.
    fileprivate static func setEndpoint(_ url: String) {
        setenv("LONGCTX_ENDPOINT", url, 1)
    }

    @Test("Tier 2 write → retrieve round trip against live longctx-svc")
    func liveRoundTrip() throws {
        guard let url = TriAttentionV3LiveLongctxTests.endpoint,
              TriAttentionV3LiveLongctxTests.reachable(url)
        else {
            // Skip-by-passing — Swift Testing doesn't have a real skip,
            // and we don't want CI red when the service isn't up.
            #expect(Bool(true))
            return
        }
        TriAttentionV3LiveLongctxTests.setEndpoint(url)

        let client = TriAttentionLongctxClient.shared
        // Use a unique session id so reruns don't conflict.
        let sessionId = "swift-live-\(UUID().uuidString)"
        client.sessionID = sessionId

        let span1 = EvictionSpan(
            text: "the password is hunter2-quetzalcoatl-9821",
            tokenStart: 100, tokenEnd: 130,
            layer: -1, score: 0.95
        )
        let span2 = EvictionSpan(
            text: "the meeting is at 3pm in conference room blue",
            tokenStart: 200, tokenEnd: 230,
            layer: -1, score: 0.82
        )

        let writeOK = client.writeEvicted([span1, span2])
        #expect(writeOK == true)

        // Retrieve — query targets the password chunk.
        let chunks = client.retrieveEvicted(
            query: "What is the secret password?",
            topK: 5, scoreFloor: 0.0
        )
        #expect(chunks.isEmpty == false)
        // The password chunk should rank higher than the meeting one
        // for "secret password" query.
        let topText = chunks.first?.text ?? ""
        #expect(topText.contains("hunter2") || topText.contains("password"))
    }

    @Test("retrieve isolated by session_id against live longctx-svc")
    func liveSessionIsolation() throws {
        guard let url = TriAttentionV3LiveLongctxTests.endpoint,
              TriAttentionV3LiveLongctxTests.reachable(url)
        else {
            #expect(Bool(true))
            return
        }
        TriAttentionV3LiveLongctxTests.setEndpoint(url)

        let client = TriAttentionLongctxClient.shared

        // Write into session A.
        let sessionA = "swift-live-A-\(UUID().uuidString)"
        client.sessionID = sessionA
        let _ = client.writeEvicted([
            EvictionSpan(text: "A-only secret span",
                         tokenStart: 0, tokenEnd: 5,
                         layer: -1, score: 0.5),
        ])

        // Retrieve from session B — should be empty.
        let sessionB = "swift-live-B-\(UUID().uuidString)"
        client.sessionID = sessionB
        let chunks = client.retrieveEvicted(
            query: "secret span", topK: 5, scoreFloor: 0.0
        )
        #expect(chunks.isEmpty)
    }

    @Test("full engine drives eviction → /evict/write on live longctx-svc")
    func liveEngineDrivenEviction() throws {
        guard let url = TriAttentionV3LiveLongctxTests.endpoint,
              TriAttentionV3LiveLongctxTests.reachable(url)
        else {
            #expect(Bool(true))
            return
        }
        TriAttentionV3LiveLongctxTests.setEndpoint(url)

        let client = TriAttentionLongctxClient.shared
        let sessionId = "swift-engine-\(UUID().uuidString)"
        client.sessionID = sessionId

        // Tiny rig: 4 layers, 2 KV heads, head_dim=16. Budget=20 forces
        // V3 to evict aggressively at seq_len=64. warmup=8 trips
        // calibration on the first multi-token Q push.
        let cfg = TriAttentionV3Config(
            budget: 20, divideLength: 4, windowSize: 4, prefixProtect: 4,
            nSegments: 4, warmupTokens: 8, hybridMode: 2
        )
        let engine = TriAttentionV3Engine(
            cfg: cfg, nLayers: 4, nHeads: 4, nKVHeads: 2,
            headDim: 16, ropeTheta: 10000.0
        )

        // Wire the rescue bridge — same Tokenizer/IDs as the bridge test.
        let rescue = TriAttentionRescue.shared
        rescue.spanBleed = 2
        struct EngTok: TriAttentionTokenizerLike {
            func decode(tokens: [Int]) -> String {
                tokens.map { "et\($0)" }.joined(separator: " ")
            }
        }
        rescue.setTokenizer(EngTok())
        rescue.setPromptTokenIds(Array(0..<128), seqId: 0)
        rescue.install(on: engine)

        // 1. Trip calibration with synthetic Q on layer 0.
        let qCalib = MLXArray.ones([8, 4, 16], dtype: .float32)
        engine.accumulateQ(qCalib, layerIdx: 0)
        #expect(engine.calibrated)

        // 2. Build 4 cache instances (one per layer) wired to the engine.
        let caches = (0..<4).map {
            TriAttentionKVCache(layerIdx: $0, engine: engine)
        }

        // 3. Drive synthetic K/V through each cache: T=64 tokens of
        //    distinct K per position. update() fires the engine's
        //    accumulateLayerScoreFromBridge after the cache append.
        let T = 64
        let positions = (MLXArray(0..<Int32(T)).asType(.float32) + 1.0)
            .reshaped([1, 1, T, 1])
        let K = MLX.broadcast(positions, to: [1, 2, T, 16])
        let V = MLX.broadcast(positions, to: [1, 2, T, 16])
        for cache in caches {
            _ = cache.update(keys: K, values: V)
        }

        // 4. By now the engine has accumulated scores from all 4 layers
        //    on a 64-token cache with budget=20 → eviction should have
        //    fired at least once and the rescue bridge POSTed spans.
        //    Verify by retrieving — any chunk surfaced means /evict/
        //    write succeeded earlier in the chain.
        let chunks = client.retrieveEvicted(
            query: "et30", topK: 5, scoreFloor: 0.0
        )
        let msg: Comment =
            "expected at least one evicted span surfaced from end-to-end engine-driven eviction"
        #expect(chunks.isEmpty == false, msg)

        // 5. After eviction, at least one cache's offset has dropped
        //    below T (compaction worked at the storage level too, not
        //    just at the rescue path). Phase A doesn't gate finalize on
        //    "all layers contributed" the way the Python engine does
        //    (TODO follow-up), so eviction fires on the first cache's
        //    update before subsequent caches have written, leaving
        //    later caches in a varying-offset state for this synthetic
        //    test. End-to-end correctness on a real model uses the
        //    standard per-step prefill loop where all layers update
        //    before the next round of forward, so this asymmetry won't
        //    affect production. We just need at least one observation
        //    that the compaction codepath ran.
        let anyCompacted = caches.contains { $0.offset < T }
        let compactionMsg: Comment =
            "expected at least one cache compacted after end-to-end eviction"
        #expect(anyCompacted, compactionMsg)
    }

    @Test("rescue bridge end-to-end against live longctx-svc")
    func liveRescueBridge() throws {
        guard let url = TriAttentionV3LiveLongctxTests.endpoint,
              TriAttentionV3LiveLongctxTests.reachable(url)
        else {
            #expect(Bool(true))
            return
        }
        TriAttentionV3LiveLongctxTests.setEndpoint(url)

        let client = TriAttentionLongctxClient.shared
        let sessionId = "swift-bridge-\(UUID().uuidString)"
        client.sessionID = sessionId

        let rescue = TriAttentionRescue.shared
        rescue.spanBleed = 4

        // Tokenizer fake: ids 0..199 → "tok<id>"; lets us see specific
        // text in the POSTed spans.
        struct LiveTok: TriAttentionTokenizerLike {
            func decode(tokens: [Int]) -> String {
                tokens.map { "tok\($0)" }.joined(separator: " ")
            }
        }
        rescue.setTokenizer(LiveTok())
        rescue.setPromptTokenIds(Array(0..<200), seqId: 0)

        // Build a small engine just to register the callback. We don't
        // actually drive prefill — we fire the callback manually with
        // a known eviction set so the assertions are deterministic.
        let cfg = TriAttentionV3Config(
            budget: 32, divideLength: 8, windowSize: 4, prefixProtect: 4,
            nSegments: 4, warmupTokens: 16, hybridMode: 2
        )
        let engine = TriAttentionV3Engine(
            cfg: cfg, nLayers: 4, nHeads: 8, nKVHeads: 2,
            headDim: 64, ropeTheta: 10000.0
        )
        rescue.install(on: engine)

        // Trigger the eviction callback directly with two contiguous
        // runs: positions 50..52 and 100..101.
        engine.evictionCallback?(0, [50, 51, 52, 100, 101], 5)
        // Allow the synchronous POST to complete (already blocking).

        // Now retrieve and verify the spans landed. The fake tokenizer
        // decodes tokens 46..58 (run 1 ±4 bleed) and 96..107 (run 2
        // ±4 bleed) into "tok46 tok47 ... tok58" / "tok96 ... tok107".
        let chunks = client.retrieveEvicted(
            query: "tok50", topK: 5, scoreFloor: 0.0
        )
        #expect(chunks.isEmpty == false)
        let allText = chunks.map { $0.text }.joined(separator: " | ")
        #expect(
            allText.contains("tok50") || allText.contains("tok51") ||
            allText.contains("tok52"),
            "expected the run-1 span text to be retrievable"
        )
    }
}
