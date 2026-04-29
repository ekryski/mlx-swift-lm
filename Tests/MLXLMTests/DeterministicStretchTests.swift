// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLMCommon
import Testing

// MARK: - Deterministic-stretch acceleration tests (spec 022 phase 1)
//
// Pure-Swift; no model evaluation needed. Phase 1 covers:
//
//   1. `ChatTemplateGrammar` protocol behaviour:
//      - `NoOpChatGrammar` always returns nil.
//      - `TriggerTokenGrammar` fires on the configured trigger only.
//   2. `BigramTable` semantics:
//      - lookup hit / miss
//      - greedy chain walk (`bigramDraft`)
//      - empty / short recentTokens fall-through
//      - `build(fromFrequencies:)` admission threshold
//      - re-insert overwrites
//   3. `BigramKey` equality + hashable.

// MARK: - Chat template grammar

@Suite
struct ChatTemplateGrammarTests {

    private static let config = ChatTemplateConfig(
        turnOpenerToken: 100,
        turnCloserToken: 101,
        thinkBeginToken: 200,
        thinkEndToken: 201,
        channelMarkerToken: 300,
        messageMarkerToken: 301,
        newlineTokens: [10, 13])

    @Test
    func `NoOpChatGrammar always returns nil`() {
        let g = NoOpChatGrammar()
        let state = ChatTemplateState(
            phase: .assistantTurn,
            recentTokens: [1, 2, 3],
            chatTemplateConfig: Self.config)
        #expect(g.deterministicContinuation(afterToken: 100, state: state) == nil)
        #expect(g.deterministicContinuation(afterToken: 999, state: state) == nil)
    }

    @Test
    func `TriggerTokenGrammar fires only on configured trigger`() {
        let g = TriggerTokenGrammar(triggerToken: 300, continuation: [50, 51, 52])
        let state = ChatTemplateState(
            phase: .channelMarker,
            recentTokens: [1, 2, 300],
            chatTemplateConfig: Self.config)
        #expect(g.deterministicContinuation(afterToken: 300, state: state) == [50, 51, 52])
        #expect(g.deterministicContinuation(afterToken: 299, state: state) == nil)
        #expect(g.deterministicContinuation(afterToken: 0, state: state) == nil)
    }

    @Test
    func `ChatTemplatePhase equality`() {
        #expect(ChatTemplatePhase.thinking == .thinking)
        #expect(ChatTemplatePhase.assistantTurn != .channelMarker)
    }
}

// MARK: - Bigram table

@Suite
struct BigramTableTests {

    @Test
    func `Empty table returns nil for any lookup`() {
        let t = BigramTable()
        #expect(t.confidentNext(prev: 1, current: 2) == nil)
        #expect(t.entryCount == 0)
    }

    @Test
    func `Insert and lookup round-trip`() {
        var t = BigramTable(admissionThreshold: 0.5)
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        #expect(t.confidentNext(prev: 1, current: 2)?.nextToken == 3)
        #expect(t.entryCount == 1)
    }

    @Test
    func `Re-insert overwrites existing entry`() {
        var t = BigramTable(admissionThreshold: 0.5)
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 99, confidence: 0.97))
        #expect(t.entryCount == 1)
        #expect(t.confidentNext(prev: 1, current: 2)?.nextToken == 99)
    }

    @Test
    func `bigramDraft walks chain greedily up to maxK`() {
        var t = BigramTable(admissionThreshold: 0.5)
        // Chain: 1, 2 → 3 → 4 → 5
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        t.insert(prev: 2, current: 3, entry: BigramEntry(nextToken: 4, confidence: 0.99))
        t.insert(prev: 3, current: 4, entry: BigramEntry(nextToken: 5, confidence: 0.99))

        let draft = t.bigramDraft(maxK: 5, recentTokens: [99, 1, 2])
        #expect(draft == [3, 4, 5])
    }

    @Test
    func `bigramDraft stops when chain breaks (no confident next)`() {
        var t = BigramTable(admissionThreshold: 0.5)
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        // No entry for (2, 3); chain breaks after one step.
        let draft = t.bigramDraft(maxK: 5, recentTokens: [1, 2])
        #expect(draft == [3])
    }

    @Test
    func `bigramDraft caps at maxK`() {
        var t = BigramTable(admissionThreshold: 0.5)
        // Self-loop: prev=1, current=1, next=1 forever.
        t.insert(prev: 1, current: 1, entry: BigramEntry(nextToken: 1, confidence: 0.99))
        let draft = t.bigramDraft(maxK: 4, recentTokens: [1, 1])
        #expect(draft == [1, 1, 1, 1])
    }

    @Test
    func `bigramDraft returns empty when recentTokens is too short`() {
        var t = BigramTable(admissionThreshold: 0.5)
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        #expect(t.bigramDraft(maxK: 4, recentTokens: []) == [])
        #expect(t.bigramDraft(maxK: 4, recentTokens: [1]) == [])
    }

    @Test
    func `bigramDraft returns empty when maxK is zero`() {
        var t = BigramTable(admissionThreshold: 0.5)
        t.insert(prev: 1, current: 2, entry: BigramEntry(nextToken: 3, confidence: 0.99))
        #expect(t.bigramDraft(maxK: 0, recentTokens: [1, 2]) == [])
    }

    @Test
    func `build admits entries above threshold, drops below`() {
        let frequencies: [BigramKey: [Int: Int]] = [
            BigramKey(prev: 1, current: 2): [3: 95, 4: 5],   // 95% → admit
            BigramKey(prev: 5, current: 6): [7: 70, 8: 30],  // 70% → drop @ 0.95
            BigramKey(prev: 9, current: 10): [11: 100],      // 100% → admit
        ]
        let t = BigramTable.build(fromFrequencies: frequencies, threshold: 0.95)
        #expect(t.entryCount == 2)
        #expect(t.confidentNext(prev: 1, current: 2)?.nextToken == 3)
        #expect(t.confidentNext(prev: 9, current: 10)?.nextToken == 11)
        #expect(t.confidentNext(prev: 5, current: 6) == nil)
    }

    @Test
    func `build picks the top-1 successor by count`() {
        let frequencies: [BigramKey: [Int: Int]] = [
            BigramKey(prev: 1, current: 2): [3: 50, 4: 95, 5: 5],  // 4 wins
        ]
        let t = BigramTable.build(fromFrequencies: frequencies, threshold: 0.5)
        #expect(t.confidentNext(prev: 1, current: 2)?.nextToken == 4)
    }

    @Test
    func `build with all-empty frequency map produces empty table`() {
        let frequencies: [BigramKey: [Int: Int]] = [:]
        let t = BigramTable.build(fromFrequencies: frequencies)
        #expect(t.entryCount == 0)
    }

    @Test
    func `build skips zero-count entries (defensive)`() {
        let frequencies: [BigramKey: [Int: Int]] = [
            BigramKey(prev: 1, current: 2): [:],  // empty inner map
        ]
        let t = BigramTable.build(fromFrequencies: frequencies)
        #expect(t.entryCount == 0)
    }

    @Test
    func `BigramKey equality + hashable`() {
        let a = BigramKey(prev: 1, current: 2)
        let b = BigramKey(prev: 1, current: 2)
        let c = BigramKey(prev: 2, current: 1)
        #expect(a == b)
        #expect(a != c)
        var s = Set<BigramKey>()
        s.insert(a)
        s.insert(b)
        #expect(s.count == 1)
    }
}
