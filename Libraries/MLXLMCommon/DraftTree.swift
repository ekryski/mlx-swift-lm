// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Draft tree representation (spec 014 phase 1 scaffold)
//
// Tree-attention speculative decoding generalises the linear draft —
// instead of one chain of K candidate tokens, the iterator drafts a
// **tree** with multiple branches at each level. The verifier processes
// the tree in one forward pass via a custom attention mask: each node
// attends only to its ancestors, so siblings are alternative
// completions of the same prefix.
//
// Phase 1 (this file): pure-Swift data structure + mask / position-id
// builders + longest-matching-path acceptance walk. The MLX-side
// attention mask construction and the iterator wiring land in phase 2;
// per-model position-id plumbing lands in phase 3. Phase 1 is fully
// unit-testable without GPU.

/// Flat-DFS tree of drafted candidate tokens.
///
/// The tree's root is the implicit `y` slot at verify-input position 0
/// (the most recently committed token). `tokens[i]` is node `i`'s
/// candidate token; `parent[i]` is its parent's index in `tokens`, or
/// `-1` if `i` is a root child (parent = the implicit `y`). `depth[i]`
/// is the distance from the root, equal to `parent[i] >= 0 ?
/// depth[parent[i]] + 1 : 1`.
///
/// Invariants enforced at construction:
///   - `tokens.count == parent.count == depth.count`.
///   - `parent[i] < i` for all `i` (DFS order — parents come before
///     children).
///   - `depth[i] == 1` if `parent[i] == -1`, else `depth[parent[i]] + 1`.
public struct DraftTree: Equatable, Sendable {
    public let tokens: [Int]
    public let parent: [Int]
    public let depth: [Int]

    public init(tokens: [Int], parent: [Int], depth: [Int]) {
        precondition(
            tokens.count == parent.count && tokens.count == depth.count,
            "DraftTree arrays must have the same count "
                + "(tokens=\(tokens.count), parent=\(parent.count), depth=\(depth.count))")
        for i in 0 ..< tokens.count {
            let p = parent[i]
            precondition(p >= -1 && p < i,
                "parent[\(i)]=\(p) must be in [-1, \(i)) (DFS order)")
            let expected = (p == -1) ? 1 : depth[p] + 1
            precondition(depth[i] == expected,
                "depth[\(i)]=\(depth[i]) inconsistent with parent (expected \(expected))")
        }
        self.tokens = tokens
        self.parent = parent
        self.depth = depth
    }

    /// Number of tree nodes (excluding the implicit root).
    public var nodeCount: Int { tokens.count }

    /// Empty tree — equivalent to "no draft this round".
    public static let empty = DraftTree(tokens: [], parent: [], depth: [])

    /// Build a linear chain of length `D`. parent[i] = i - 1, depth[i] =
    /// i + 1. Equivalent to today's `linear` shape — kept here for tests
    /// and as a building block for `topK_root_x_linear`.
    public static func linear(tokens: [Int]) -> DraftTree {
        var parent = [Int](repeating: -1, count: tokens.count)
        var depth = [Int](repeating: 0, count: tokens.count)
        for i in 0 ..< tokens.count {
            parent[i] = i - 1
            depth[i] = i + 1
        }
        return DraftTree(tokens: tokens, parent: parent, depth: depth)
    }

    /// Build a `topK × linear-D` tree: K root branches (each carrying
    /// the first token from the corresponding chain), each extended
    /// into a linear chain of depth `D - 1`.
    ///
    /// - Parameter chains: K independent draft chains. Each chain's
    ///   first token becomes a root child; the rest extend that branch
    ///   linearly. Empty chains are skipped.
    public static func topKByLinear(chains: [[Int]]) -> DraftTree {
        var tokens: [Int] = []
        var parent: [Int] = []
        var depth: [Int] = []
        for chain in chains where !chain.isEmpty {
            // The first token in the chain is a root child (parent = -1,
            // depth = 1).
            let rootIndex = tokens.count
            tokens.append(chain[0])
            parent.append(-1)
            depth.append(1)
            // Subsequent tokens form a linear chain off that root child.
            for j in 1 ..< chain.count {
                let prev = tokens.count - 1
                tokens.append(chain[j])
                parent.append(prev)
                depth.append(depth[prev] + 1)
            }
            // (Loose end — `rootIndex` not consumed; intent is to make
            // the chain-start position obvious to readers tracing the
            // construction.)
            _ = rootIndex
        }
        return DraftTree(tokens: tokens, parent: parent, depth: depth)
    }

    /// Children-of-each-node lookup, materialised on demand. O(N) to
    /// build; callers can cache it across uses.
    public func children() -> [[Int]] {
        var result = [[Int]](repeating: [], count: nodeCount)
        var rootChildren: [Int] = []
        for i in 0 ..< nodeCount {
            if parent[i] == -1 {
                rootChildren.append(i)
            } else {
                result[parent[i]].append(i)
            }
        }
        // Append a synthetic "root entry" by storing rootChildren in a
        // companion accessor — keep the array shape `nodeCount` so
        // index `i` always means "children of node i".
        return result + [rootChildren]
    }

    /// Indices of nodes whose parent is the implicit root (`y`).
    public var rootChildren: [Int] {
        (0 ..< nodeCount).filter { parent[$0] == -1 }
    }

    /// Set of all ancestors of node `i` (excluding `i` itself, including
    /// the implicit root represented by `-1`). Returned as a
    /// `Set<Int>` for fast mask-construction lookups; the `-1` element
    /// is omitted from the set (the mask builder treats `y` separately).
    public func ancestors(of i: Int) -> Set<Int> {
        precondition(i >= 0 && i < nodeCount, "node index out of range")
        var result: Set<Int> = []
        var p = parent[i]
        while p != -1 {
            result.insert(p)
            p = parent[p]
        }
        return result
    }
}

// MARK: - Tree shape policies (spec 014 §6)

/// Available tree-shape policies. Phase 1 ships the data structures for
/// all three; the iterator currently only uses `linear`. Phase 2 wires
/// `topKByLinear`; phase 3 wires `bifurcatingOnTightMargin`.
public enum DraftTreeShape: Equatable, Sendable {
    /// One path of depth `D`. Today's behaviour — no tree, equivalent
    /// to the existing linear draft.
    case linear(depth: Int)

    /// `K` root branches × linear continuation of depth `D - 1` per
    /// branch. Total nodes: `K * D`.
    case topKByLinear(k: Int, depth: Int)

    /// Linear path of depth `D` with sibling-of-top-2 inserted at
    /// tight-margin positions. Total nodes ≤ `2 * D`.
    case bifurcatingOnTightMargin(depth: Int, marginEpsilon: Float)
}

// MARK: - Attention mask construction

/// Build the tree-attention mask in pure Swift, returning a flat
/// `[T, T]` Bool array (row-major) where `T = 1 + tree.nodeCount`.
/// Position 0 is the implicit `y` slot; positions `1..<T` are tree
/// nodes in DFS order.
///
/// Mask convention: `mask[i, j] == true` means position `i`'s query
/// may attend to position `j`'s key. The mask combines:
///   - Causal: `i` attends to `j` only if `j <= i`.
///   - Self: `mask[i, i] = true`.
///   - Tree: tree nodes attend to `y` (column 0) plus all their
///     ancestors plus self. Sibling positions are NOT visible to each
///     other.
///   - `y` self-attends (`mask[0, 0] = true`).
///
/// The iterator materialises this as an `MLXArray` of dtype `bool`
/// before the verify forward; the bool-vs-additive-mask conversion is
/// MLX-side and not part of this helper.
public func buildTreeAttentionMask(_ tree: DraftTree) -> [Bool] {
    let T = 1 + tree.nodeCount
    var mask = [Bool](repeating: false, count: T * T)
    // y self-attends.
    mask[0] = true
    for i in 0 ..< tree.nodeCount {
        let row = (i + 1) * T
        // Tree node attends to y.
        mask[row + 0] = true
        // Self-attention.
        mask[row + (i + 1)] = true
        // Ancestors.
        var p = tree.parent[i]
        while p != -1 {
            mask[row + (p + 1)] = true
            p = tree.parent[p]
        }
    }
    return mask
}

/// Position IDs for the tree's verify input. Position 0 is `y`'s
/// position (`previousOffset`); position `i + 1` is `previousOffset +
/// depth[i]`. Siblings share their parent's `depth + 1` so RoPE rotates
/// them as alternative completions of the same prefix.
public func buildTreePositionIDs(_ tree: DraftTree, previousOffset: Int) -> [Int] {
    var ids = [Int](repeating: 0, count: 1 + tree.nodeCount)
    ids[0] = previousOffset
    for i in 0 ..< tree.nodeCount {
        ids[i + 1] = previousOffset + tree.depth[i]
    }
    return ids
}

// MARK: - Acceptance walk (longest-matching-path)

/// Result of accepting along the longest matching root-to-leaf path.
public struct DraftTreeAcceptance: Equatable, Sendable {
    /// Depth-ordered indices of accepted nodes (each index is into
    /// `tree.tokens`). Empty when no root child matches.
    public let acceptedNodeIndices: [Int]

    /// Bonus token verify-input position — the index in
    /// `[y] + tree.tokens` whose target argmax is the next emitted
    /// token. Equals `0` when nothing accepts (emit `targetArgmax[0]`);
    /// equals `1 + lastAccepted` when accepting deeper.
    public let bonusVerifyIndex: Int

    public var acceptedCount: Int { acceptedNodeIndices.count }
}

/// Walk the tree to find the longest path whose tokens match the
/// target's argmax at each verify position.
///
/// The contract:
///   - `targetArgmax[0]` is what should follow `y`. Look for a root
///     child whose token equals it; if none, accept zero, bonus is
///     position 0.
///   - For each accepted node, look at `targetArgmax[1 + acceptedIdx]`
///     — that's what should follow this node. Find a child whose
///     token equals it; descend.
///   - Stop when no child matches (accept ends here) or the path
///     reaches a leaf (full accept).
///
/// - Parameter tree: the drafted tree.
/// - Parameter targetArgmax: per-verify-position target argmax —
///   shape `[1 + tree.nodeCount]`. `targetArgmax[0]` is the
///   prediction at `y`'s position; `targetArgmax[i + 1]` is the
///   prediction at tree node `i`'s position.
public func acceptDraftTree(
    _ tree: DraftTree,
    targetArgmax: [Int]
) -> DraftTreeAcceptance {
    precondition(
        targetArgmax.count == 1 + tree.nodeCount,
        "targetArgmax must have 1 + tree.nodeCount entries "
            + "(got \(targetArgmax.count), expected \(1 + tree.nodeCount))")

    // Build a children map keyed by token: children-of-node-i has the
    // shape `[Int: Int]` mapping `token -> node index`. Built lazily —
    // most accept walks visit only a small fraction of nodes.
    func childrenOf(_ nodeIdx: Int) -> [Int] {
        if nodeIdx == -1 { return tree.rootChildren }
        return (0 ..< tree.nodeCount).filter { tree.parent[$0] == nodeIdx }
    }

    var accepted: [Int] = []
    var current: Int = -1  // -1 = implicit root (y)
    while true {
        let targetIdx = (current == -1) ? 0 : (1 + current)
        let want = targetArgmax[targetIdx]
        let kids = childrenOf(current)
        guard let pick = kids.first(where: { tree.tokens[$0] == want }) else {
            break
        }
        accepted.append(pick)
        current = pick
    }

    let bonusIdx = (accepted.isEmpty) ? 0 : (1 + accepted.last!)
    return DraftTreeAcceptance(
        acceptedNodeIndices: accepted,
        bonusVerifyIndex: bonusIdx)
}
