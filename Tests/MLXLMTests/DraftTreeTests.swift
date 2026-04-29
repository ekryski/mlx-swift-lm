// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLMCommon
import Testing

// MARK: - Draft tree tests (spec 014 phase 1 scaffold)
//
// Pure-Swift; no MLX evaluation. Phase 1 covers:
//
//   1. `DraftTree` invariants + builders (linear, topKByLinear).
//   2. Ancestor + rootChildren accessors.
//   3. `buildTreeAttentionMask` — causal + self + ancestor + y-column.
//   4. `buildTreePositionIDs` — depth-aware position IDs.
//   5. `acceptDraftTree` — longest-matching-path walk in scenarios:
//      no match, partial match, full match, multi-branch with one
//      matching, descend-then-stop.

@Suite
struct DraftTreeBuilderTests {

    @Test
    func `Empty tree`() {
        let t = DraftTree.empty
        #expect(t.nodeCount == 0)
        #expect(t.tokens == [])
        #expect(t.rootChildren == [])
    }

    @Test
    func `Linear tree of length 4 has chained parents and depths 1-4`() {
        let t = DraftTree.linear(tokens: [10, 20, 30, 40])
        #expect(t.nodeCount == 4)
        #expect(t.parent == [-1, 0, 1, 2])
        #expect(t.depth == [1, 2, 3, 4])
        #expect(t.rootChildren == [0])
    }

    @Test
    func `topKByLinear with K=2, D=3 produces 2 branches`() {
        let t = DraftTree.topKByLinear(chains: [[10, 11, 12], [20, 21, 22]])
        #expect(t.nodeCount == 6)
        #expect(t.tokens == [10, 11, 12, 20, 21, 22])
        // Branch 1: indices 0,1,2; parents -1,0,1; depths 1,2,3.
        // Branch 2: indices 3,4,5; parents -1,3,4; depths 1,2,3.
        #expect(t.parent == [-1, 0, 1, -1, 3, 4])
        #expect(t.depth == [1, 2, 3, 1, 2, 3])
        #expect(t.rootChildren == [0, 3])
    }

    @Test
    func `topKByLinear with empty chains is empty`() {
        let t = DraftTree.topKByLinear(chains: [])
        #expect(t.nodeCount == 0)
    }

    @Test
    func `topKByLinear skips empty chains in the input`() {
        let t = DraftTree.topKByLinear(chains: [[1, 2], [], [3, 4]])
        #expect(t.tokens == [1, 2, 3, 4])
        #expect(t.parent == [-1, 0, -1, 2])
        #expect(t.rootChildren == [0, 2])
    }

    @Test
    func `Ancestors include the chain back to root`() {
        let t = DraftTree.linear(tokens: [10, 20, 30, 40])
        #expect(t.ancestors(of: 0) == [])
        #expect(t.ancestors(of: 3) == [0, 1, 2])
    }

    @Test
    func `Ancestors of unrelated branch are disjoint`() {
        let t = DraftTree.topKByLinear(chains: [[10, 11], [20, 21]])
        #expect(t.ancestors(of: 1) == [0])  // branch 1's leaf
        #expect(t.ancestors(of: 3) == [2])  // branch 2's leaf
    }
}

// MARK: - Mask construction

@Suite
struct TreeAttentionMaskTests {

    private static func mask(_ flat: [Bool], _ T: Int, _ row: Int, _ col: Int) -> Bool {
        flat[row * T + col]
    }

    @Test
    func `Empty tree mask is single self-attention slot`() {
        let m = buildTreeAttentionMask(.empty)
        #expect(m == [true])
    }

    @Test
    func `Linear tree mask is purely causal`() {
        let t = DraftTree.linear(tokens: [10, 20, 30])
        let m = buildTreeAttentionMask(t)
        let T = 4  // 1 + 3 nodes
        #expect(m.count == T * T)
        // Each row i should attend to all j <= i.
        for i in 0 ..< T {
            for j in 0 ..< T {
                #expect(Self.mask(m, T, i, j) == (j <= i))
            }
        }
    }

    @Test
    func `Two-branch tree mask isolates siblings`() {
        // y → branch A (depth 1, then 2), branch B (depth 1, then 2).
        let t = DraftTree.topKByLinear(chains: [[10, 11], [20, 21]])
        let m = buildTreeAttentionMask(t)
        let T = 5  // y + 4 nodes
        // Position layout:
        //   0: y
        //   1: A0 (parent y, depth 1)
        //   2: A1 (parent A0, depth 2)
        //   3: B0 (parent y, depth 1)
        //   4: B1 (parent B0, depth 2)

        // y sees only itself.
        for j in 1 ..< T { #expect(!Self.mask(m, T, 0, j)) }
        #expect(Self.mask(m, T, 0, 0))

        // A0 sees y + self.
        #expect(Self.mask(m, T, 1, 0))
        #expect(Self.mask(m, T, 1, 1))
        for j in 2 ..< T { #expect(!Self.mask(m, T, 1, j)) }

        // A1 sees y + A0 + self; not B0/B1.
        #expect(Self.mask(m, T, 2, 0))
        #expect(Self.mask(m, T, 2, 1))
        #expect(Self.mask(m, T, 2, 2))
        #expect(!Self.mask(m, T, 2, 3))
        #expect(!Self.mask(m, T, 2, 4))

        // B0 sees y + self; not A0/A1.
        #expect(Self.mask(m, T, 3, 0))
        #expect(!Self.mask(m, T, 3, 1))
        #expect(!Self.mask(m, T, 3, 2))
        #expect(Self.mask(m, T, 3, 3))
        #expect(!Self.mask(m, T, 3, 4))

        // B1 sees y + B0 + self; not A0/A1.
        #expect(Self.mask(m, T, 4, 0))
        #expect(!Self.mask(m, T, 4, 1))
        #expect(!Self.mask(m, T, 4, 2))
        #expect(Self.mask(m, T, 4, 3))
        #expect(Self.mask(m, T, 4, 4))
    }
}

// MARK: - Position IDs

@Suite
struct TreePositionIDsTests {

    @Test
    func `Linear tree position IDs are consecutive`() {
        let t = DraftTree.linear(tokens: [10, 20, 30])
        let ids = buildTreePositionIDs(t, previousOffset: 100)
        #expect(ids == [100, 101, 102, 103])
    }

    @Test
    func `Two-branch tree gives siblings the same position ID`() {
        let t = DraftTree.topKByLinear(chains: [[10, 11], [20, 21]])
        let ids = buildTreePositionIDs(t, previousOffset: 50)
        // y at 50; A0 / B0 at 51 (siblings, both depth 1); A1 / B1 at
        // 52 (depth 2). DFS layout: y, A0, A1, B0, B1.
        #expect(ids == [50, 51, 52, 51, 52])
    }

    @Test
    func `Empty tree position IDs are just the previousOffset`() {
        let ids = buildTreePositionIDs(.empty, previousOffset: 42)
        #expect(ids == [42])
    }
}

// MARK: - Acceptance walk

@Suite
struct AcceptDraftTreeTests {

    @Test
    func `Empty tree accepts zero, bonus is verify position 0`() {
        let r = acceptDraftTree(.empty, targetArgmax: [99])
        #expect(r.acceptedCount == 0)
        #expect(r.bonusVerifyIndex == 0)
    }

    @Test
    func `Linear full match accepts every node`() {
        let t = DraftTree.linear(tokens: [10, 20, 30])
        // targetArgmax[0]=10 (predicts after y), [1]=20 (after node 0),
        // [2]=30 (after node 1), [3]=99 (after node 2; bonus).
        let r = acceptDraftTree(t, targetArgmax: [10, 20, 30, 99])
        #expect(r.acceptedNodeIndices == [0, 1, 2])
        #expect(r.bonusVerifyIndex == 3)  // 1 + last accepted
    }

    @Test
    func `Linear partial match stops at first mismatch`() {
        let t = DraftTree.linear(tokens: [10, 20, 30])
        // Match 10, 20; argmax-after-20 wants 99 but draft has 30.
        let r = acceptDraftTree(t, targetArgmax: [10, 20, 99, 0])
        #expect(r.acceptedNodeIndices == [0, 1])
        #expect(r.bonusVerifyIndex == 2)  // 1 + 1
    }

    @Test
    func `Linear no-match accepts zero and bonus is 0`() {
        let t = DraftTree.linear(tokens: [10, 20, 30])
        let r = acceptDraftTree(t, targetArgmax: [99, 0, 0, 0])
        #expect(r.acceptedNodeIndices == [])
        #expect(r.bonusVerifyIndex == 0)
    }

    @Test
    func `Two-branch tree picks the matching branch only`() {
        // Branch A: [10, 11]; Branch B: [20, 21].
        let t = DraftTree.topKByLinear(chains: [[10, 11], [20, 21]])
        // Layout (DFS): A0=10, A1=11, B0=20, B1=21.
        // targetArgmax: y → 20 (matches B0), then after B0 → 21
        // (matches B1), then after B1 → 99 (bonus).
        let r = acceptDraftTree(t, targetArgmax: [20, 0, 0, 21, 99])
        #expect(r.acceptedNodeIndices == [2, 3])  // B0, B1
        #expect(r.bonusVerifyIndex == 4)  // 1 + 3
    }

    @Test
    func `Two-branch tree descends matching branch then stops at sibling`() {
        // A: [10, 11]; B: [20, 21]. Match A0 then mismatch on A1.
        let t = DraftTree.topKByLinear(chains: [[10, 11], [20, 21]])
        let r = acceptDraftTree(t, targetArgmax: [10, 99, 0, 0, 0])
        #expect(r.acceptedNodeIndices == [0])
        #expect(r.bonusVerifyIndex == 1)
    }
}
