// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Cross-request prefix KV cache tests (spec 017)
//
// All pure-Swift / pure-array — no model forward, no tokenizer, no chat
// template (for the L1 + policy tests). Covers:
//
//   1. `StablePrefixPolicy`: Identity + FixedTrim + LastAssistantOpener.
//   2. `PrefixKey`: equality + hash invariance + formatVersion = 2.
//   3. `PrefixKVCache`:
//      - lookup miss / hit / longest-prefix selection
//      - cross-key isolation, fingerprintRejects accounting
//      - LRU bump on hit, lookup(..., record: false) doesn't bump
//      - byte-budget eviction + byteBudgetEvictions counter
//      - entry-count cap eviction
//      - skippedTooLong on oversized inserts
//      - exactHits vs partialHits classification
//      - prefillTokensSaved accumulation
//      - close() semantics (idempotent + everything throws after)
//      - formatVersion mismatch rejection at insert
//      - clear / resetStats
//
// Phase 1B serialise/hydrate tests live in `PrefixKVCacheSerialisationTests.swift`.

// MARK: - Helpers

private func makeKey(model: String = "test", layers: Int = 4) -> PrefixKey {
    PrefixKey(modelID: model, layerCount: layers, kvHeadDim: 64, kvBits: nil)
}

private func makeSnapshot(
    model: String = "test",
    tokens: [Int],
    layers: Int = 4,
    bytesPerLayer: Int = 1024
) -> PrefixSnapshot {
    let key = makeKey(model: model, layers: layers)
    // Use fp16 (Float16 = 2 bytes) — `bytesPerLayer / 2` Float16 values
    // gives an array of exactly `bytesPerLayer` bytes.
    let elementCount = bytesPerLayer / 2
    let arr: () -> MLXArray = { MLXArray.zeros([elementCount], dtype: .float16) }
    let layerStates = (0 ..< layers).map { _ in
        LayerCacheState(
            kind: .standardUnbounded, tokenCount: tokens.count,
            arrays: [arr()])
    }
    return PrefixSnapshot(key: key, tokens: tokens, layerStates: layerStates)
}

// MARK: - Stable-prefix policy

@Suite
struct StablePrefixPolicyTests {

    @Test
    func `IdentityPolicy returns full token count`() {
        let p = IdentityPolicy()
        #expect(p.stablePrefixLen([1, 2, 3, 4, 5]) == 5)
        #expect(p.stablePrefixLen([]) == 0)
    }

    @Test
    func `FixedTrimPolicy trims fixed suffix`() {
        let p = FixedTrimPolicy(trimSuffix: 3)
        #expect(p.stablePrefixLen([1, 2, 3, 4, 5, 6, 7]) == 4)
    }

    @Test
    func `FixedTrimPolicy floors at zero on short prompts`() {
        let p = FixedTrimPolicy(trimSuffix: 10)
        #expect(p.stablePrefixLen([1, 2, 3]) == 0)
    }

    @Test
    func `FixedTrimPolicy with zero trim is identity-equivalent`() {
        let p = FixedTrimPolicy(trimSuffix: 0)
        #expect(p.stablePrefixLen([1, 2, 3]) == 3)
    }

    @Test
    func `LastAssistantOpenerPolicy finds opener at tail`() {
        // prompt: [system tokens..., user message, opener]
        let opener = [100, 101, 102]
        let p = LastAssistantOpenerPolicy(opener: opener)
        let prompt = [1, 2, 3, 4, 5] + opener
        #expect(p.stablePrefixLen(prompt) == 5)
    }

    @Test
    func `LastAssistantOpenerPolicy returns rightmost match`() {
        // Two openers in the prompt; we should match the rightmost so the
        // most recent user turn becomes the new boundary.
        let opener = [100, 101]
        let p = LastAssistantOpenerPolicy(opener: opener)
        let prompt = [1, 100, 101, 2, 3, 100, 101]
        // rightmost opener starts at index 5.
        #expect(p.stablePrefixLen(prompt) == 5)
    }

    @Test
    func `LastAssistantOpenerPolicy no-match falls back per policy`() {
        let opener = [99, 99, 99]
        let identityFallback = LastAssistantOpenerPolicy(opener: opener, fallback: .identity)
        #expect(identityFallback.stablePrefixLen([1, 2, 3]) == 3)

        let refuseFallback = LastAssistantOpenerPolicy(opener: opener, fallback: .refuse)
        #expect(refuseFallback.stablePrefixLen([1, 2, 3]) == 0)

        let trimFallback = LastAssistantOpenerPolicy(
            opener: opener, fallback: .fixedTrim(suffix: 2))
        #expect(trimFallback.stablePrefixLen([1, 2, 3, 4, 5]) == 3)
    }

    @Test
    func `LastAssistantOpenerPolicy handles opener longer than prompt`() {
        let p = LastAssistantOpenerPolicy(opener: [1, 2, 3, 4, 5])
        #expect(p.stablePrefixLen([1, 2]) == 2)  // identity fallback
    }

    @Test
    func `LastAssistantOpenerPolicy handles empty prompt`() {
        let p = LastAssistantOpenerPolicy(opener: [1])
        #expect(p.stablePrefixLen([]) == 0)
    }

    @Test
    func `AssistantOpener.rawString matches expected sentinels`() {
        #expect(AssistantOpener.qwenChatML.rawString == "<|im_start|>assistant\n")
        #expect(AssistantOpener.gemma4.rawString == "<start_of_turn>model\n")
        #expect(AssistantOpener.gemma4Turn.rawString == "<|turn>model\n")
        #expect(
            AssistantOpener.gemma4WithThought.rawString
                == "<|turn>model\n<|channel>thought\n<channel|>")
        #expect(AssistantOpener.gptOSSHarmony.rawString == "<|start|>assistant<|channel|>")
        #expect(AssistantOpener.custom("foo").rawString == "foo")
    }

    @Test
    func `AssistantOpener.detect maps Qwen family to ChatML`() {
        #expect(AssistantOpener.detect(forModelID: "Qwen/Qwen3.5-9B-Instruct") == .qwenChatML)
        #expect(AssistantOpener.detect(forModelID: "mlx-community/Qwen3.5-0.8B-4bit") == .qwenChatML)
        #expect(AssistantOpener.detect(forModelID: "qwen3-next-32b") == .qwenChatML)
        // QwQ also uses ChatML.
        #expect(AssistantOpener.detect(forModelID: "Qwen/QwQ-32B-Preview") == .qwenChatML)
    }

    @Test
    func `AssistantOpener.detect maps Gemma 1-3 to legacy gemma4 opener`() {
        // Legacy `<start_of_turn>model\n` form for Gemma 1 / 2 / 3.
        #expect(AssistantOpener.detect(forModelID: "google/gemma-2-2b-it") == .gemma4)
        #expect(AssistantOpener.detect(forModelID: "google/gemma-3-4b") == .gemma4)
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma-3-27b-it-qat-4bit")
                == .gemma4)
    }

    @Test
    func `AssistantOpener.detect maps Gemma 4 small to gemma4Turn opener (issue #196)`() {
        // Gemma 4 small (E2B / E4B) uses the new `<|turn>` special token.
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma-4-E2B-it-4bit")
                == .gemma4Turn)
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma-4-e4b-it-4bit")
                == .gemma4Turn)
    }

    @Test
    func `AssistantOpener.detect maps Gemma 4 large to gemma4WithThought opener (issue #196)`() {
        // Large Gemma 4 (26B-A4B, 31B) ships add_generation_prompt with
        // channel-thought scaffolding — needs the longer opener.
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma-4-26b-a4b-it-4bit")
                == .gemma4WithThought)
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma-4-31b-it-4bit")
                == .gemma4WithThought)
        #expect(AssistantOpener.detect(forModelID: "GEMMA-4-31B") == .gemma4WithThought)
        // Some bench / model-id schemes use `gemma4-` without the dash
        // between the family and version digits.
        #expect(
            AssistantOpener.detect(forModelID: "mlx-community/gemma4-26b-a4b-it")
                == .gemma4WithThought)
    }

    @Test
    func `AssistantOpener.detect maps GPT-OSS to harmony`() {
        #expect(AssistantOpener.detect(forModelID: "loan-star/gpt-oss-20b-mlx-4Bit") == .gptOSSHarmony)
        #expect(AssistantOpener.detect(forModelID: "gpt_oss_20b") == .gptOSSHarmony)
    }

    @Test
    func `AssistantOpener.detect returns nil for unknown families`() {
        #expect(AssistantOpener.detect(forModelID: "meta-llama/Llama-3.2-3B") == nil)
        #expect(AssistantOpener.detect(forModelID: "microsoft/Phi-4") == nil)
        #expect(AssistantOpener.detect(forModelID: "mistralai/Mistral-7B-Instruct-v0.3") == nil)
        #expect(AssistantOpener.detect(forModelID: "") == nil)
        #expect(AssistantOpener.detect(forModelID: "random-string-with-no-family") == nil)
    }

    @Test
    func `resolveDefaultPolicy returns LastAssistantOpenerPolicy for known family`() {
        let tokenizer = TestTokenizer()
        let policy = resolveDefaultPolicy(
            modelID: "Qwen/Qwen3.5-9B-Instruct", tokenizer: tokenizer)
        #expect(policy is LastAssistantOpenerPolicy)
    }

    @Test
    func `resolveDefaultPolicy falls back to IdentityPolicy for unknown family`() {
        let tokenizer = TestTokenizer()
        let policy = resolveDefaultPolicy(
            modelID: "meta-llama/Llama-3.2-3B", tokenizer: tokenizer)
        #expect(policy is IdentityPolicy)
    }

    @Test
    func `resolveDefaultPolicy falls back to IdentityPolicy when no tokenizer supplied`() {
        let policy = resolveDefaultPolicy(
            modelID: "Qwen/Qwen3.5-9B-Instruct", tokenizer: nil)
        #expect(policy is IdentityPolicy)
    }

    @Test
    func `resolveDefaultPolicy falls back to IdentityPolicy when no modelID supplied`() {
        let tokenizer = TestTokenizer()
        let policy = resolveDefaultPolicy(modelID: nil, tokenizer: tokenizer)
        #expect(policy is IdentityPolicy)
    }
}

// MARK: - PrefixKey

@Suite
struct PrefixKeyTests {

    @Test
    func `Equality requires all fields match`() {
        let a = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        let b = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        #expect(a == b)

        let differentLayers = PrefixKey(modelID: "x", layerCount: 16, kvHeadDim: 64, kvBits: nil)
        #expect(a != differentLayers)

        let differentBits = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: 4)
        #expect(a != differentBits)

        let differentModel = PrefixKey(modelID: "y", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        #expect(a != differentModel)
    }

    @Test
    func `Hashable produces consistent buckets`() {
        let a = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        let b = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        var set = Set<PrefixKey>()
        set.insert(a)
        set.insert(b)
        #expect(set.count == 1)
    }

    @Test
    func `Default formatVersion is 2`() {
        let key = PrefixKey(modelID: "x", layerCount: 1, kvHeadDim: 1)
        #expect(key.formatVersion == 2)
        #expect(PrefixKey.currentFormatVersion == 2)
    }

    @Test
    func `captureLayerIds nil means all-layers`() {
        let a = PrefixKey(modelID: "x", layerCount: 4, kvHeadDim: 1)
        let b = PrefixKey(modelID: "x", layerCount: 4, kvHeadDim: 1, captureLayerIds: nil)
        #expect(a == b)

        let c = PrefixKey(modelID: "x", layerCount: 4, kvHeadDim: 1, captureLayerIds: [0, 1])
        #expect(a != c)
    }
}

// MARK: - PrefixKVCache lookup / insert / eviction

@Suite
struct PrefixKVCacheTests {

    @Test
    func `Empty cache reports miss and no fingerprint reject`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let r = try cache.lookup(prefix: [1, 2, 3], key: makeKey())
        #expect(r == nil)
        #expect(cache.stats.misses == 1)
        #expect(cache.stats.hits == 0)
        #expect(cache.stats.fingerprintRejects == 0)
    }

    @Test
    func `Exact prefix match returns full snapshot, classifies as partialHit when suffix remains`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        try cache.insert(snap)

        let r = try cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(r != nil)
        #expect(r?.matchedLength == 3)
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.partialHits == 1)
        #expect(cache.stats.exactHits == 0)
        #expect(cache.stats.prefillTokensSaved == 3)
    }

    @Test
    func `Full-cover hit (matchedLen == request length) is an exactHit`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        try cache.insert(snap)
        _ = try cache.lookup(prefix: [1, 2, 3], key: makeKey())
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.partialHits == 0)
        #expect(cache.stats.exactHits == 1)
    }

    @Test
    func `Longest-prefix selection picks the longest matching snapshot`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2]))
        try cache.insert(makeSnapshot(tokens: [1, 2, 3, 4]))
        try cache.insert(makeSnapshot(tokens: [1, 2, 3]))

        let r = try cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(r?.matchedLength == 4)
        #expect(r?.snapshot.tokens == [1, 2, 3, 4])
    }

    @Test
    func `Cross-key isolation rejects mismatched model ID and counts as fingerprint reject`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(model: "modelA", tokens: [1, 2, 3]))

        let r = try cache.lookup(
            prefix: [1, 2, 3, 4],
            key: PrefixKey(modelID: "modelB", layerCount: 4, kvHeadDim: 64))
        #expect(r == nil)
        #expect(cache.stats.misses == 1)
        #expect(cache.stats.fingerprintRejects == 1)
    }

    @Test
    func `Mismatching prefix returns nil`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2, 3]))

        let r = try cache.lookup(prefix: [9, 8, 7, 6], key: makeKey())
        #expect(r == nil)
        // Same-key miss is NOT a fingerprint reject.
        #expect(cache.stats.fingerprintRejects == 0)
    }

    @Test
    func `Snapshot longer than request returns nil (won't truncate)`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2, 3, 4, 5]))

        let r = try cache.lookup(prefix: [1, 2], key: makeKey())
        #expect(r == nil)
    }

    @Test
    func `Re-insert with same key replaces in place (no double-count)`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        try cache.insert(snap)
        let bytesAfterFirst = cache.stats.bytesUsed
        try cache.insert(snap)
        #expect(cache.count == 1)
        #expect(cache.stats.bytesUsed == bytesAfterFirst)
    }

    @Test
    func `LRU bump moves matched entry to MRU position (record default)`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let s1 = makeSnapshot(tokens: [1, 2])
        let s2 = makeSnapshot(tokens: [3, 4])
        let s3 = makeSnapshot(tokens: [5, 6])
        try cache.insert(s1)
        try cache.insert(s2)
        try cache.insert(s3)

        _ = try cache.lookup(prefix: [1, 2, 99], key: makeKey())

        let order = cache.entrySnapshot.map { $0.tokens }
        #expect(order == [[3, 4], [5, 6], [1, 2]])
    }

    @Test
    func `lookup(..., record: false) does not bump LRU or update stats`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let s1 = makeSnapshot(tokens: [1, 2])
        let s2 = makeSnapshot(tokens: [3, 4])
        try cache.insert(s1)
        try cache.insert(s2)

        // Non-recording lookup.
        _ = try cache.lookup(prefix: [1, 2], key: makeKey(), record: false)
        #expect(cache.stats.hits == 0)
        #expect(cache.stats.misses == 0)
        // LRU order unchanged: s1 still at index 0.
        let order = cache.entrySnapshot.map { $0.tokens }
        #expect(order == [[1, 2], [3, 4]])
    }

    @Test
    func `Entry-count cap evicts oldest first`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 2)
        try cache.insert(makeSnapshot(tokens: [1]))
        try cache.insert(makeSnapshot(tokens: [2]))
        try cache.insert(makeSnapshot(tokens: [3]))
        #expect(cache.count == 2)
        let surviving = Set(cache.entrySnapshot.map { $0.tokens.first! })
        #expect(surviving == [2, 3])
        #expect(cache.stats.evictions == 1)
        // Entry-count eviction doesn't count as byte-budget.
        #expect(cache.stats.byteBudgetEvictions == 0)
    }

    @Test
    func `Byte-budget eviction frees space for large insert and increments byteBudgetEvictions`() throws {
        let cache = PrefixKVCache(maxBytes: 8 * 1024, maxEntries: 100)
        try cache.insert(makeSnapshot(tokens: [1], bytesPerLayer: 1024))
        try cache.insert(makeSnapshot(tokens: [2], bytesPerLayer: 1024))
        #expect(cache.count == 2)

        try cache.insert(makeSnapshot(tokens: [3], bytesPerLayer: 1024))
        #expect(cache.count == 2)
        #expect(cache.stats.evictions == 1)
        #expect(cache.stats.byteBudgetEvictions == 1)
    }

    @Test
    func `Insert exceeding budget alone is skipped (skippedTooLong)`() throws {
        let cache = PrefixKVCache(maxBytes: 1024, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1], bytesPerLayer: 2048)
        try cache.insert(snap)
        #expect(cache.count == 0)
        #expect(cache.stats.skippedTooLong == 1)
        #expect(cache.stats.insertions == 0)
    }

    @Test
    func `Format-version mismatch throws on insert`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let staleKey = PrefixKey(
            modelID: "x", layerCount: 1, kvHeadDim: 1, formatVersion: 1)
        let snap = PrefixSnapshot(
            key: staleKey, tokens: [1, 2],
            layerStates: [LayerCacheState(
                kind: .standardUnbounded, tokenCount: 2,
                arrays: [MLXArray.zeros([4], dtype: .float16)])])
        #expect(throws: PrefixKVCacheError.self) {
            try cache.insert(snap)
        }
    }

    @Test
    func `Layer tokenCount mismatch throws snapshotInvariantViolation`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let key = PrefixKey(modelID: "x", layerCount: 1, kvHeadDim: 1)
        let snap = PrefixSnapshot(
            key: key, tokens: [1, 2, 3],
            layerStates: [LayerCacheState(
                kind: .standardUnbounded, tokenCount: 7,  // != 3
                arrays: [MLXArray.zeros([4], dtype: .float16)])])
        #expect(throws: PrefixKVCacheError.self) {
            try cache.insert(snap)
        }
    }

    @Test
    func `Empty donor-sharing layer (tokenCount 0, no arrays) is exempt from invariant`() throws {
        // Mimic Gemma 4's KV-sharing pattern: some layers carry empty
        // state because they share a donor's K/V. The invariant must
        // not reject these layers; the rest of the snapshot is still
        // valid and the shared layers re-bind to their donor on hydrate.
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let key = PrefixKey(modelID: "x", layerCount: 2, kvHeadDim: 1)
        let snap = PrefixSnapshot(
            key: key, tokens: [1, 2, 3],
            layerStates: [
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 3,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 0,
                    arrays: []),
            ])
        try cache.insert(snap)
        #expect(cache.count == 1)
        #expect(cache.stats.insertions == 1)
    }

    @Test
    func `SSM layer is exempt from tokenCount invariant`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let key = PrefixKey(modelID: "x", layerCount: 2, kvHeadDim: 1)
        // SSM layer reports a tokenCount that doesn't match the prompt
        // (cumulative recurrent state has no positional dimension).
        let snap = PrefixSnapshot(
            key: key, tokens: [1, 2, 3],
            layerStates: [
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 3,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
                LayerCacheState(
                    kind: .ssm, tokenCount: 99,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
            ])
        try cache.insert(snap)
        #expect(cache.count == 1)
    }

    @Test
    func `close() retires the cache and throws on every subsequent op`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1]))
        cache.close()
        #expect(cache.closed == true)

        #expect(throws: PrefixKVCacheError.self) {
            try cache.lookup(prefix: [1], key: makeKey())
        }
        #expect(throws: PrefixKVCacheError.self) {
            try cache.insert(makeSnapshot(tokens: [2]))
        }
        #expect(throws: PrefixKVCacheError.self) {
            try cache.clear()
        }
    }

    @Test
    func `close() is idempotent`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.close()
        cache.close()
        #expect(cache.closed == true)
    }

    @Test
    func `clear empties entries and resets bytes`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2]))
        try cache.insert(makeSnapshot(tokens: [3, 4]))
        try cache.clear()
        #expect(cache.count == 0)
        #expect(cache.stats.bytesUsed == 0)
        #expect(cache.stats.entryCount == 0)
        #expect(cache.stats.insertions == 2)
    }

    @Test
    func `resetStats keeps occupancy counts, zeros activity counters`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2]))
        _ = try cache.lookup(prefix: [1, 2], key: makeKey())
        _ = try cache.lookup(prefix: [9], key: makeKey())
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.misses == 1)

        cache.resetStats()
        #expect(cache.stats.hits == 0)
        #expect(cache.stats.misses == 0)
        #expect(cache.stats.insertions == 0)
        #expect(cache.stats.prefillTokensSaved == 0)
        #expect(cache.stats.bytesUsed > 0)
        #expect(cache.stats.entryCount == 1)
    }

    @Test
    func `hitRate computation is correct`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2]))
        _ = try cache.lookup(prefix: [1, 2], key: makeKey())
        _ = try cache.lookup(prefix: [1, 2], key: makeKey())
        _ = try cache.lookup(prefix: [9], key: makeKey())
        #expect(abs(cache.stats.hitRate - 2.0/3.0) < 1e-9)
    }

    @Test
    func `meanMatchedLength averages over hits only`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2]))
        try cache.insert(makeSnapshot(tokens: [1, 2, 3, 4]))
        _ = try cache.lookup(prefix: [1, 2, 9], key: makeKey())
        _ = try cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        _ = try cache.lookup(prefix: [9, 8], key: makeKey())

        #expect(cache.stats.hits == 2)
        #expect(cache.stats.totalMatchedTokens == 6)
        #expect(abs(cache.stats.meanMatchedLength - 3.0) < 1e-9)
    }

    @Test
    func `prefillTokensSaved accumulates across hits`() throws {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        try cache.insert(makeSnapshot(tokens: [1, 2, 3, 4]))
        _ = try cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())  // 4
        _ = try cache.lookup(prefix: [1, 2, 3, 4], key: makeKey())     // 4
        _ = try cache.lookup(prefix: [9], key: makeKey())              // miss
        #expect(cache.stats.prefillTokensSaved == 8)
    }
}
