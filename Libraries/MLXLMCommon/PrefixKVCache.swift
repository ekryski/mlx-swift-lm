// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Cross-request prefix KV cache (spec 017 phase 1 — in-memory only)
//
// Multi-turn chat and agentic workloads send a prompt whose **prefix is
// identical** to the previous turn's prompt. Today every turn re-runs
// prefill over that prefix; the prefix cache snapshots the target's KV
// state at a stable prefix boundary so the next turn can hydrate from
// the snapshot and prefill only the suffix.
//
// **Phase 1 scope** (this file): in-memory LRU cache with token-exact
// prefix matching, byte-budget eviction, and a configurable
// ``StablePrefixPolicy``. The on-disk persistence layer (phase 4) is
// deferred. The actual cache-type-specific `serialise` / `hydrate`
// methods land in phase 1B once we've decided which caches to support
// first; the snapshot type here uses an opaque `[MLXArray]` payload that
// hydrate-time code interprets per-layer.

/// Identity key for a prefix snapshot. Snapshots whose `PrefixKey`
/// doesn't match the current target's are not interchangeable — model
/// configuration changes (layer count, head dims, KV bit-width) make
/// snapshot KV state semantically different.
public struct PrefixKey: Hashable, Sendable {
    /// Stable model identifier (HuggingFace ID or local model directory
    /// hash). The cache uses this to refuse cross-model snapshot reuse.
    public let modelID: String

    /// Total layers in the target stack. Snapshots from a different
    /// layer count are unusable.
    public let layerCount: Int

    /// Per-layer KV head dimension. Catches cases where two checkpoints
    /// share a model ID but were quantised with different head configs.
    public let kvHeadDim: Int

    /// KV bit-width (`nil` for fp16, otherwise the cache quantization
    /// bits). Snapshots from quantised vs. unquantised state can't
    /// hydrate into each other without re-quantisation work that's
    /// outside this cache's scope.
    public let kvBits: Int?

    public init(modelID: String, layerCount: Int, kvHeadDim: Int, kvBits: Int? = nil) {
        precondition(layerCount >= 1, "layerCount must be >= 1 (got \(layerCount))")
        precondition(kvHeadDim >= 1, "kvHeadDim must be >= 1 (got \(kvHeadDim))")
        if let bits = kvBits {
            precondition(bits >= 1 && bits <= 16,
                "kvBits must be in [1, 16] (got \(bits))")
        }
        self.modelID = modelID
        self.layerCount = layerCount
        self.kvHeadDim = kvHeadDim
        self.kvBits = kvBits
    }
}

/// Per-layer cache state captured in a snapshot. Phase 1 stores the
/// layer's `state` array opaquely (the `KVCache.state` accessor); phase
/// 1B will add typed variants per concrete cache class so hydration can
/// validate shape + dtype.
public struct LayerCacheState: @unchecked Sendable {
    /// Number of tokens this layer's state represents. For trim-able
    /// caches this is the cache offset; for hybrid caches it's the
    /// position-stamp at snapshot time.
    public let tokenCount: Int

    /// Opaque per-layer arrays — typically `[K, V]` for attention layers,
    /// `[hiddenState, convState]` or similar for SSM layers. The hydrate
    /// path matches by index, not by shape — caller's responsibility to
    /// align with the target cache's `state` layout.
    public let arrays: [MLXArray]

    public init(tokenCount: Int, arrays: [MLXArray]) {
        precondition(tokenCount >= 0, "tokenCount must be >= 0 (got \(tokenCount))")
        self.tokenCount = tokenCount
        self.arrays = arrays
    }

    /// Sum of all backing arrays' sizes in bytes. The cache uses this for
    /// LRU-budget accounting; not a contract on hydration cost.
    public var byteSize: Int {
        arrays.reduce(0) { $0 + $1.nbytes }
    }
}

/// One snapshot in the prefix cache. Captures the target's KV state at
/// the end of a stable prefix prefill.
public struct PrefixSnapshot: @unchecked Sendable {
    public let key: PrefixKey
    public let tokens: [Int]
    public let layerStates: [LayerCacheState]

    /// Optional final hidden state — DFlash uses it to warm-start its
    /// draft; baseline iterators ignore it. Stored as `[1, 1, H]` if
    /// present.
    public let lastHidden: MLXArray?

    public let createdAt: Date

    public init(
        key: PrefixKey,
        tokens: [Int],
        layerStates: [LayerCacheState],
        lastHidden: MLXArray? = nil,
        createdAt: Date = Date()
    ) {
        precondition(layerStates.count == key.layerCount,
            "layerStates.count (\(layerStates.count)) must equal "
                + "key.layerCount (\(key.layerCount))")
        self.key = key
        self.tokens = tokens
        self.layerStates = layerStates
        self.lastHidden = lastHidden
        self.createdAt = createdAt
    }

    /// Total bytes held by this snapshot — sum of per-layer byte sizes
    /// plus optional last-hidden. Drives the LRU budget.
    public var byteSize: Int {
        var b = layerStates.reduce(0) { $0 + $1.byteSize }
        if let lh = lastHidden { b += lh.nbytes }
        return b
    }
}

/// Result of a prefix-cache lookup. `matchedLength == 0` is a miss;
/// `> 0` is a hit, and the iterator should hydrate from `snapshot` then
/// prefill only over `tokens[matchedLength...]`.
public struct PrefixCacheLookupResult: @unchecked Sendable {
    public let matchedLength: Int
    public let snapshot: PrefixSnapshot

    public init(matchedLength: Int, snapshot: PrefixSnapshot) {
        precondition(matchedLength > 0, "lookup result must have matchedLength > 0")
        self.matchedLength = matchedLength
        self.snapshot = snapshot
    }
}

/// Telemetry for the prefix cache. Bench harness reads this once per
/// request to surface a `[PREFIX-CACHE]` line.
public struct PrefixCacheStats: Equatable, Sendable {
    public var hits: Int
    public var misses: Int
    public var partialHits: Int
    public var insertions: Int
    public var evictions: Int
    public var bytesUsed: Int
    public var entryCount: Int

    /// Sum of matched lengths on hits — used to compute mean matched
    /// length in benchmark reports.
    public var totalMatchedTokens: Int

    public init(
        hits: Int = 0, misses: Int = 0, partialHits: Int = 0,
        insertions: Int = 0, evictions: Int = 0,
        bytesUsed: Int = 0, entryCount: Int = 0,
        totalMatchedTokens: Int = 0
    ) {
        self.hits = hits
        self.misses = misses
        self.partialHits = partialHits
        self.insertions = insertions
        self.evictions = evictions
        self.bytesUsed = bytesUsed
        self.entryCount = entryCount
        self.totalMatchedTokens = totalMatchedTokens
    }

    public var hitRate: Double {
        let total = hits + misses
        guard total > 0 else { return 0 }
        return Double(hits) / Double(total)
    }

    public var meanMatchedLength: Double {
        guard hits > 0 else { return 0 }
        return Double(totalMatchedTokens) / Double(hits)
    }
}

/// In-memory LRU prefix KV cache.
///
/// Lookup is **longest token-prefix match**: among snapshots whose
/// `PrefixKey` matches and whose tokens form a prefix of the request
/// prompt, return the one with the longest matched length.
///
/// Eviction is byte-budgeted LRU: on insert, while
/// `bytesUsed + newSnap.byteSize > maxBytes`, evict the oldest entry.
/// `maxEntries` is a secondary cap — useful for tests and for capping
/// metadata overhead even when individual snapshots are tiny.
///
/// Thread safety: phase 1 is **not** thread-safe. Callers wrap accesses
/// with their own queue / lock when sharing across requests; the
/// `PrefixKVCache.shared` global instance is intended for single-iterator
/// use, not concurrent serving. Phase 4 will add an actor wrapper if/when
/// needed.
///
/// `@unchecked Sendable` for the shared global — the contract is documented
/// above and Phase 4 will add proper actor isolation when concurrent
/// serving lands.
public final class PrefixKVCache: @unchecked Sendable {

    /// Process-wide shared cache. Apps that want a single cache across
    /// all requests use this; tests construct local instances.
    public static let shared = PrefixKVCache()

    /// Hard byte budget. Default 8 GiB — same as the spec's design
    /// section.
    public let maxBytes: Int

    /// Hard entry-count cap. Default 4 — see spec rationale.
    public let maxEntries: Int

    /// Stable-prefix policy used at insert time to decide where to trim
    /// the snapshot before storing it. Lookup uses the full cached
    /// prefix; the trimming happens on the producer side.
    public let stablePrefixPolicy: StablePrefixPolicy

    /// Insertion-ordered storage. Most-recently-used = last index.
    /// LRU eviction pops `entries.first`. We use an array because at
    /// `maxEntries = 4` the linear scan is faster than dictionary
    /// hashing and gives byte-stable iteration order.
    private var entries: [PrefixSnapshot] = []

    private(set) public var stats: PrefixCacheStats = PrefixCacheStats()

    public init(
        maxBytes: Int = 8 * 1024 * 1024 * 1024,
        maxEntries: Int = 4,
        stablePrefixPolicy: StablePrefixPolicy = IdentityPolicy()
    ) {
        precondition(maxBytes > 0, "maxBytes must be > 0 (got \(maxBytes))")
        precondition(maxEntries >= 1, "maxEntries must be >= 1 (got \(maxEntries))")
        self.maxBytes = maxBytes
        self.maxEntries = maxEntries
        self.stablePrefixPolicy = stablePrefixPolicy
    }

    /// Look up the longest-matching snapshot for a request.
    ///
    /// - Returns: nil on miss; `(matchedLength, snapshot)` on hit. On
    ///   hit, the snapshot is moved to most-recently-used in the LRU
    ///   order. `matchedLength` is the length of the snapshot's full
    ///   prefix (token-exact match), not a partial match — phase 1 only
    ///   considers byte-equal prefix matches.
    public func lookup(prefix tokens: [Int], key: PrefixKey) -> PrefixCacheLookupResult? {
        var bestIndex: Int? = nil
        var bestLen = 0
        for (i, snap) in entries.enumerated() {
            guard snap.key == key else { continue }
            guard snap.tokens.count <= tokens.count else { continue }
            // Token-exact prefix match.
            let matches = snap.tokens.elementsEqual(tokens.prefix(snap.tokens.count))
            if matches && snap.tokens.count > bestLen {
                bestLen = snap.tokens.count
                bestIndex = i
            }
        }
        guard let idx = bestIndex else {
            stats.misses += 1
            return nil
        }
        // Partial-hit accounting: the snapshot covered some but not all
        // of the request — i.e. there's a non-empty suffix to prefill.
        // Full-hit (matchedLen == request length) is logically a "hit
        // with nothing left to prefill", which still counts as a hit
        // here.
        if bestLen < tokens.count { stats.partialHits += 1 }
        stats.hits += 1
        stats.totalMatchedTokens += bestLen

        // LRU bump: move the matched entry to the back of the list.
        let snap = entries.remove(at: idx)
        entries.append(snap)
        return PrefixCacheLookupResult(matchedLength: bestLen, snapshot: snap)
    }

    /// Insert a snapshot. Evicts oldest entries until budget allows.
    /// Replaces any existing snapshot whose key + tokens are exactly
    /// equal — same prefix from the same model is the same snapshot.
    public func insert(_ snapshot: PrefixSnapshot) {
        // Replace existing exact-match snapshot in place if present.
        if let existingIdx = entries.firstIndex(where: {
            $0.key == snapshot.key && $0.tokens == snapshot.tokens
        }) {
            stats.bytesUsed -= entries[existingIdx].byteSize
            entries.remove(at: existingIdx)
        }

        // Evict to fit the new snapshot.
        while !entries.isEmpty
            && (stats.bytesUsed + snapshot.byteSize > maxBytes
                || entries.count >= maxEntries) {
            let evicted = entries.removeFirst()
            stats.bytesUsed -= evicted.byteSize
            stats.evictions += 1
        }

        entries.append(snapshot)
        stats.bytesUsed += snapshot.byteSize
        stats.entryCount = entries.count
        stats.insertions += 1
    }

    /// Clear all entries. Stats are preserved (callers reset them
    /// independently).
    public func clear() {
        entries.removeAll()
        stats.bytesUsed = 0
        stats.entryCount = 0
    }

    /// Reset statistics counters without clearing entries. Useful at
    /// bench-run boundaries.
    public func resetStats() {
        let bytesUsed = stats.bytesUsed
        let entryCount = stats.entryCount
        stats = PrefixCacheStats(bytesUsed: bytesUsed, entryCount: entryCount)
    }

    /// Number of stored entries.
    public var count: Int { entries.count }

    /// Snapshot of entry order (oldest → newest), exposed for tests and
    /// diagnostics. Not for production callers — they should go through
    /// `lookup` / `insert`.
    public var entrySnapshot: [PrefixSnapshot] { entries }
}
