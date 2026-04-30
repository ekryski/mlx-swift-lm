// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLMCommon
import Testing

// MARK: - NGramCache + registry tests (spec 016 phases 1-2)

@Suite
struct NGramCacheThresholdsTests {

    @Test
    func `Lax profile matches llama.cpp constants`() {
        let t = NGramCacheThresholds.lax
        #expect(t.minSampleSize == [2, 2, 1, 1])
        #expect(t.minPercent == [66, 50, 50, 50])
    }

    @Test
    func `Strict profile matches llama.cpp constants`() {
        let t = NGramCacheThresholds.strict
        #expect(t.minSampleSize == [4, 3, 2, 2])
        #expect(t.minPercent == [75, 66, 66, 66])
    }

    @Test
    func `Static-tier profile aliases strict`() {
        #expect(NGramCacheThresholds.staticTier == NGramCacheThresholds.strict)
    }

    @Test
    func `Threshold lookup uses last entry for sizes past array length`() {
        let t = NGramCacheThresholds.lax
        let small = t.thresholds(forNgramSize: 1)
        #expect(small.sampleSize == 2)
        #expect(small.percent == 66)

        let big = t.thresholds(forNgramSize: 10)
        #expect(big.sampleSize == 1)
        #expect(big.percent == 50)
    }
}

@Suite
struct NGramCacheTests {

    @Test
    func `Initial cache is empty across all tiers`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4)
        #expect(c.count(in: .context) == 0)
        #expect(c.count(in: .dynamic) == 0)
        #expect(c.count(in: .static) == 0)
    }

    @Test
    func `extendContext appends to context tier`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4)
        c.extendContext([1, 2, 3])
        c.extendContext([4, 5])
        #expect(c.tokens(in: .context) == [1, 2, 3, 4, 5])
        #expect(c.count(in: .context) == 5)
    }

    @Test
    func `resetContext clears context but preserves other tiers`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4)
        c.extendContext([1, 2, 3])
        c.commitToDynamic([10, 20])
        c.loadStatic([100, 200])

        c.resetContext()
        #expect(c.count(in: .context) == 0)
        #expect(c.tokens(in: .dynamic) == [10, 20])
        #expect(c.tokens(in: .static) == [100, 200])
    }

    @Test
    func `commitToDynamic appends and FIFO-truncates at maxTokens`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4, maxTokens: 5)
        c.commitToDynamic([1, 2, 3])
        c.commitToDynamic([4, 5, 6, 7])  // overflow by 2
        #expect(c.tokens(in: .dynamic) == [3, 4, 5, 6, 7])
        #expect(c.count(in: .dynamic) == 5)
    }

    @Test
    func `commitToDynamic with single overflowing batch truncates from head`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4, maxTokens: 3)
        c.commitToDynamic([1, 2, 3, 4, 5])
        #expect(c.tokens(in: .dynamic) == [3, 4, 5])
    }

    @Test
    func `loadStatic replaces existing static corpus`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4)
        c.loadStatic([1, 2, 3])
        c.loadStatic([10, 20])
        #expect(c.tokens(in: .static) == [10, 20])
    }

    @Test
    func `clearAll empties every tier`() {
        let c = NGramCache(modelID: "m", ngramRange: 2 ... 4)
        c.extendContext([1])
        c.commitToDynamic([2])
        c.loadStatic([3])
        c.clearAll()
        #expect(c.count(in: .context) == 0)
        #expect(c.count(in: .dynamic) == 0)
        #expect(c.count(in: .static) == 0)
    }
}

@Suite
struct NGramCacheRegistryTests {

    @Test
    func `Default registry is empty`() {
        let r = NGramCacheRegistry()
        #expect(r.count == 0)
    }

    @Test
    func `cache(for:) returns the same instance on repeated lookup`() {
        let r = NGramCacheRegistry()
        let a = r.cache(for: "modelA", ngramRange: 2 ... 4)
        let b = r.cache(for: "modelA", ngramRange: 2 ... 4)
        #expect(a === b)
        #expect(r.count == 1)
    }

    @Test
    func `cache(for:) creates separate instances per model ID`() {
        let r = NGramCacheRegistry()
        let a = r.cache(for: "modelA", ngramRange: 2 ... 4)
        let b = r.cache(for: "modelB", ngramRange: 2 ... 4)
        #expect(a !== b)
        #expect(r.count == 2)
    }

    @Test
    func `cache(for:) creates separate instances per ngram range`() {
        let r = NGramCacheRegistry()
        let a = r.cache(for: "modelA", ngramRange: 2 ... 4)
        let b = r.cache(for: "modelA", ngramRange: 3 ... 5)
        #expect(a !== b)
        #expect(r.count == 2)
    }

    @Test
    func `Dynamic-tier persists across cache lookups (the phase-1 win)`() {
        let r = NGramCacheRegistry()
        let cache1 = r.cache(for: "modelA", ngramRange: 2 ... 4)
        cache1.extendContext([1, 2, 3, 4])
        cache1.commitToDynamic(cache1.tokens(in: .context))

        // Simulate "next request" — different iterator, same registry,
        // same model. Lookup returns the same cache; dynamic tier
        // still has the prior turn's tokens.
        let cache2 = r.cache(for: "modelA", ngramRange: 2 ... 4)
        #expect(cache2 === cache1)
        #expect(cache2.tokens(in: .dynamic) == [1, 2, 3, 4])
    }

    @Test
    func `remove(modelID:ngramRange:) drops the registered cache`() {
        let r = NGramCacheRegistry()
        _ = r.cache(for: "modelA", ngramRange: 2 ... 4)
        #expect(r.count == 1)
        let removed = r.remove(modelID: "modelA", ngramRange: 2 ... 4)
        #expect(removed)
        #expect(r.count == 0)

        // Idempotent on the next call.
        let removedAgain = r.remove(modelID: "modelA", ngramRange: 2 ... 4)
        #expect(!removedAgain)
    }

    @Test
    func `clear empties the registry`() {
        let r = NGramCacheRegistry()
        _ = r.cache(for: "modelA", ngramRange: 2 ... 4)
        _ = r.cache(for: "modelB", ngramRange: 3 ... 5)
        r.clear()
        #expect(r.count == 0)
    }
}
