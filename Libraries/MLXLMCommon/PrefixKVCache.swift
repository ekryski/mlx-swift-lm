// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Cross-request prefix KV cache (spec 017)
//
// Multi-turn chat and agentic workloads send a prompt whose **prefix is
// identical** to the previous turn's prompt. Today every turn re-runs
// prefill over that prefix; the prefix cache snapshots the target's KV
// state at a stable prefix boundary so the next turn can hydrate from
// the snapshot and prefill only the suffix.
//
// **Phase 1**: in-memory LRU cache with token-exact prefix matching,
// byte-budget eviction, and a configurable ``StablePrefixPolicy``.
// **Phase 1B**: typed per-class `serialise()` / `hydrate(from:)` (see
// `KVCacheSerialisation.swift`) + `generate()` wiring (see
// `Evaluate.swift`).
// **Phase 2**: chat-aware ``LastAssistantOpenerPolicy``.
// **Phase 3**: hybrid-cache (SSM) snapshots via spec 020 state-replay.
// **Phase 4**: disk persistence (`PrefixKVCacheDisk.swift`).

// MARK: - Errors

/// Errors raised across the prefix-cache contract. Adopted from dflash-mlx
/// upstream's `4bc72c8` "fail-fast contract" pattern — every public method
/// on a retired cache throws ``closed``; insert with missing required
/// payloads throws a typed error rather than silently degrading.
public enum PrefixKVCacheError: Error, Equatable, Sendable {
    /// `lookup(...)` / `insert(...)` / `clear()` invoked after `close()`.
    case closed
    /// A DFlash-mode snapshot was inserted without the required
    /// `lastHidden` payload. Pure-attention / SSM snapshots don't need it,
    /// so this only fires when the caller explicitly opted in.
    case missingLogits
    /// On-disk (phase 4) or in-memory snapshot's `formatVersion` does not
    /// match this build's `PrefixKey.formatVersion`. Backward-incompatible
    /// schema bumps surface here.
    case formatVersionMismatch(expected: Int, found: Int)
    /// Attempted to snapshot a windowed cache whose rotating buffer has
    /// wrapped. dflash-mlx upstream's `target_cache_is_serializable(...)`
    /// returns `False` for the same case; we surface the same failure as a
    /// typed error so callers can opt out of insert at this boundary.
    case wrappedWindowedCache
    /// A snapshot invariant (FA cache offset == token-prefix length, or
    /// target hidden chunks truncated to prefix length) failed at insert /
    /// hydrate time. Mirrors dflash-mlx `463d722`.
    case snapshotInvariantViolation(String)
}

// MARK: - Identity key

/// Identity key for a prefix snapshot. Snapshots whose `PrefixKey`
/// doesn't match the current target's are not interchangeable — model
/// configuration changes (layer count, head dims, KV bit-width) make
/// snapshot KV state semantically different.
public struct PrefixKey: Hashable, Sendable {

    /// On-disk + cross-build schema version. Bumped 1 → 2 in spec 017
    /// post-`463d722` sync. Mismatch raises
    /// ``PrefixKVCacheError/formatVersionMismatch(expected:found:)``.
    public static let currentFormatVersion: Int = 2

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

    /// Layers whose state was captured. `nil` means "all layers".
    /// dflash-mlx's `DFlashPrefixKey.capture_layer_ids` (post-`463d722`) —
    /// reserved here so the snapshot wire format stays stable when we
    /// later cache a subset (e.g. only the layers DFlash needs).
    public let captureLayerIds: [Int]?

    /// Schema version. Defaults to ``currentFormatVersion``; older
    /// snapshots (e.g. on-disk from a prior build) carry the version they
    /// were written under so the loader can reject mismatches.
    public let formatVersion: Int

    public init(
        modelID: String,
        layerCount: Int,
        kvHeadDim: Int,
        kvBits: Int? = nil,
        captureLayerIds: [Int]? = nil,
        formatVersion: Int = PrefixKey.currentFormatVersion
    ) {
        precondition(layerCount >= 1, "layerCount must be >= 1 (got \(layerCount))")
        precondition(kvHeadDim >= 1, "kvHeadDim must be >= 1 (got \(kvHeadDim))")
        if let bits = kvBits {
            precondition(bits >= 1 && bits <= 16,
                "kvBits must be in [1, 16] (got \(bits))")
        }
        if let ids = captureLayerIds {
            precondition(ids.allSatisfy { $0 >= 0 && $0 < layerCount },
                "captureLayerIds must all be in [0, layerCount); got \(ids)")
        }
        precondition(formatVersion >= 1, "formatVersion must be >= 1 (got \(formatVersion))")
        self.modelID = modelID
        self.layerCount = layerCount
        self.kvHeadDim = kvHeadDim
        self.kvBits = kvBits
        self.captureLayerIds = captureLayerIds
        self.formatVersion = formatVersion
    }
}

// MARK: - Per-layer state

/// Per-layer cache state captured in a snapshot. Phase 1B introduces a
/// **kind** discriminator so the hydrate path validates shape + dtype
/// per concrete cache class. The arrays + metaState round-trip the
/// concrete cache's `state` / `metaState` accessors (see
/// `KVCacheSerialisation.swift` for the per-class implementations).
public struct LayerCacheState: @unchecked Sendable {
    /// Concrete cache class this state was captured from. Mirrors
    /// `KVStorageKind` but adds discriminators for windowed-vs-unbounded
    /// (the storage kind doesn't distinguish; the layer cache's
    /// `metaState` does).
    public enum Kind: Sendable, Equatable {
        /// Unbounded `StandardKVCache`.
        case standardUnbounded
        /// Windowed `StandardKVCache(maxSize:keep:)`.
        case standardWindowed(maxSize: Int, keep: Int)
        /// Group-quantized `AffineQuantizedKVCache`.
        case affineQuantized(bits: Int, groupSize: Int)
        /// TurboQuant compressed cache. Not currently snapshot-able when
        /// `isCompressed == true` — phase 1B captures raw mode only; the
        /// post-prefill compressed transition happens after the snapshot
        /// boundary.
        case turboCompressed(keyBits: Int, valueBits: Int)
        /// SSM / GatedDeltaNet state (spec 020 phase 5 path).
        case ssm
    }

    /// Concrete class this state corresponds to.
    public let kind: Kind

    /// Number of tokens this layer's state represents. For trim-able
    /// caches this is the cache offset; for hybrid caches it's the
    /// position-stamp at snapshot time.
    public let tokenCount: Int

    /// Per-layer raw arrays. Shape contract is per-`kind`:
    ///   - ``Kind/standardUnbounded`` / ``Kind/standardWindowed``: `[K, V]`
    ///   - ``Kind/affineQuantized``: `[kW, kS, kB?, vW, vS, vB?]`
    ///     (4 or 6 elements depending on biases).
    ///   - ``Kind/turboCompressed``: `[rawKeys, rawValues]` (pre-compression);
    ///     compressed snapshots are not currently captured.
    ///   - ``Kind/ssm``: `[convState, recurrentState]`.
    public let arrays: [MLXArray]

    /// String-typed metaState (e.g. window size / keep / step / offset /
    /// idx for StandardKVCache windowed). Round-trips through the
    /// concrete cache's `metaState` accessor.
    public let metaState: [String]

    public init(kind: Kind, tokenCount: Int, arrays: [MLXArray], metaState: [String] = []) {
        precondition(tokenCount >= 0, "tokenCount must be >= 0 (got \(tokenCount))")
        self.kind = kind
        self.tokenCount = tokenCount
        self.arrays = arrays
        self.metaState = metaState
    }

    /// Sum of all backing arrays' sizes in bytes. The cache uses this for
    /// LRU-budget accounting; not a contract on hydration cost.
    public var byteSize: Int {
        arrays.reduce(0) { $0 + $1.nbytes }
    }
}

// MARK: - Snapshot

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

// MARK: - Lookup result

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

// MARK: - Stats

/// Telemetry for the prefix cache. Bench harness reads this once per
/// request to surface a `[PREFIX-CACHE]` line.
public struct PrefixCacheStats: Equatable, Sendable {
    /// Hits where snapshot covered the **entire** request prompt.
    public var exactHits: Int
    /// Total hits (exact + partial). For backward compatibility with the
    /// phase-1 surface.
    public var hits: Int
    public var misses: Int
    public var partialHits: Int
    public var insertions: Int
    public var evictions: Int
    /// Evictions specifically caused by byte-budget pressure (vs.
    /// entry-count cap).
    public var byteBudgetEvictions: Int
    /// Snapshots rejected at insert because their byte size exceeded the
    /// configured budget on their own. dflash-mlx `prefix_l1.py`:
    /// `skipped_too_long`.
    public var skippedTooLong: Int
    /// Lookups that fell through key validation (model ID / format
    /// version / etc. mismatch).
    public var fingerprintRejects: Int
    /// Sum of token positions skipped via cache hits — i.e. tokens we
    /// did **not** have to re-prefill. Drives the "saved prefill tokens"
    /// bench metric.
    public var prefillTokensSaved: Int
    public var bytesUsed: Int
    public var entryCount: Int

    /// Sum of matched lengths on hits — used to compute mean matched
    /// length in benchmark reports.
    public var totalMatchedTokens: Int

    public init(
        hits: Int = 0, misses: Int = 0, partialHits: Int = 0,
        insertions: Int = 0, evictions: Int = 0,
        bytesUsed: Int = 0, entryCount: Int = 0,
        totalMatchedTokens: Int = 0,
        exactHits: Int = 0,
        byteBudgetEvictions: Int = 0,
        skippedTooLong: Int = 0,
        fingerprintRejects: Int = 0,
        prefillTokensSaved: Int = 0
    ) {
        self.hits = hits
        self.misses = misses
        self.partialHits = partialHits
        self.insertions = insertions
        self.evictions = evictions
        self.bytesUsed = bytesUsed
        self.entryCount = entryCount
        self.totalMatchedTokens = totalMatchedTokens
        self.exactHits = exactHits
        self.byteBudgetEvictions = byteBudgetEvictions
        self.skippedTooLong = skippedTooLong
        self.fingerprintRejects = fingerprintRejects
        self.prefillTokensSaved = prefillTokensSaved
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

// MARK: - Cache

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

    /// Set by ``close()``. Every public method throws ``PrefixKVCacheError/closed``
    /// after this transitions to `true`. Idempotent shutdown: calling
    /// `close()` twice is a no-op.
    private var isClosed: Bool = false

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
    /// - Parameters:
    ///   - tokens: the request's prompt tokens.
    ///   - key: the target's identity key. Lookups for a different key
    ///     are rejected at the bucket level.
    ///   - record: when `true` (default), increment stats counters and
    ///     bump the matched entry to the MRU slot. dflash-mlx upstream
    ///     `463d722` adds this flag so diagnostics / probing don't
    ///     pollute the LRU order or hit counts.
    /// - Returns: nil on miss; `(matchedLength, snapshot)` on hit.
    public func lookup(
        prefix tokens: [Int], key: PrefixKey, record: Bool = true
    ) throws -> PrefixCacheLookupResult? {
        if isClosed { throw PrefixKVCacheError.closed }

        var bestIndex: Int? = nil
        var bestLen = 0
        var sawAnyKey = false
        for (i, snap) in entries.enumerated() {
            if snap.key == key { sawAnyKey = true }
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
            if record {
                stats.misses += 1
                // If we saw entries but none matched the key, count as a
                // fingerprint reject (cache had snapshots, just for the
                // wrong target). Pure empty-bucket misses are not
                // fingerprint rejects.
                if !sawAnyKey && !entries.isEmpty {
                    stats.fingerprintRejects += 1
                }
            }
            return nil
        }
        if record {
            // Full-hit (matchedLen == request length) — call this an
            // exact hit. Partial-hit (matchedLen < request length) — still
            // a hit, but we have suffix left to prefill.
            if bestLen == tokens.count {
                stats.exactHits += 1
            } else {
                stats.partialHits += 1
            }
            stats.hits += 1
            stats.totalMatchedTokens += bestLen
            stats.prefillTokensSaved += bestLen
        }

        // LRU bump: move the matched entry to the back of the list.
        let snap = entries[idx]
        if record {
            entries.remove(at: idx)
            entries.append(snap)
        }
        return PrefixCacheLookupResult(matchedLength: bestLen, snapshot: snap)
    }

    /// Insert a snapshot. Evicts oldest entries until budget allows.
    /// Replaces any existing snapshot whose key + tokens are exactly
    /// equal — same prefix from the same model is the same snapshot,
    /// regardless of when it was captured.
    public func insert(_ snapshot: PrefixSnapshot) throws {
        if isClosed { throw PrefixKVCacheError.closed }

        // Invariant: format version must match this build.
        if snapshot.key.formatVersion != PrefixKey.currentFormatVersion {
            throw PrefixKVCacheError.formatVersionMismatch(
                expected: PrefixKey.currentFormatVersion,
                found: snapshot.key.formatVersion)
        }
        // Invariant: each layer's tokenCount must be consistent with the
        // snapshot's token-prefix length. Mirrors dflash-mlx `463d722`'s
        // "FA cache offset == token-prefix length" guard.
        //
        // Layers exempt from the check:
        //   - SSM layers (cumulative recurrent state, no positional dim).
        //   - **Donor-sharing layers** with empty state (Gemma 4's KV
        //     sharing: shared layers read K/V from a donor and don't
        //     store independent state — tokenCount=0 + empty arrays).
        //     At hydrate time the donor's state is what matters; the
        //     shared layer rebuilds its reference naturally.
        let promptLen = snapshot.tokens.count
        for (i, ls) in snapshot.layerStates.enumerated() {
            if case .ssm = ls.kind { continue }
            if ls.tokenCount == 0 && ls.arrays.isEmpty { continue }
            if ls.tokenCount != promptLen {
                throw PrefixKVCacheError.snapshotInvariantViolation(
                    "layer \(i) tokenCount (\(ls.tokenCount)) != prompt token count (\(promptLen))")
            }
        }

        // Reject snapshots that on their own exceed the budget. dflash-mlx
        // `prefix_l1.py`: `skipped_too_long`. Without this, a single
        // oversized insert would evict every other entry then still fail
        // to fit.
        if snapshot.byteSize > maxBytes {
            stats.skippedTooLong += 1
            return
        }

        // Replace existing exact-match snapshot in place if present.
        if let existingIdx = entries.firstIndex(where: {
            $0.key == snapshot.key && $0.tokens == snapshot.tokens
        }) {
            stats.bytesUsed -= entries[existingIdx].byteSize
            entries.remove(at: existingIdx)
        }

        // Evict to fit the new snapshot. byteBudgetEvictions tracks the
        // subset of evictions caused by byte pressure specifically (vs.
        // the entry-count cap).
        while !entries.isEmpty
            && (stats.bytesUsed + snapshot.byteSize > maxBytes
                || entries.count >= maxEntries) {
            let byteBudgetTriggered = stats.bytesUsed + snapshot.byteSize > maxBytes
            let evicted = entries.removeFirst()
            stats.bytesUsed -= evicted.byteSize
            stats.evictions += 1
            if byteBudgetTriggered { stats.byteBudgetEvictions += 1 }
        }

        entries.append(snapshot)
        stats.bytesUsed += snapshot.byteSize
        stats.entryCount = entries.count
        stats.insertions += 1
    }

    /// Clear all entries. Stats are preserved (callers reset them
    /// independently).
    public func clear() throws {
        if isClosed { throw PrefixKVCacheError.closed }
        entries.removeAll()
        stats.bytesUsed = 0
        stats.entryCount = 0
    }

    /// Retire this cache. After `close()`, every public method throws
    /// ``PrefixKVCacheError/closed``. Idempotent.
    public func close() {
        guard !isClosed else { return }
        isClosed = true
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

    /// Whether the cache has been retired via ``close()``.
    public var closed: Bool { isClosed }

    /// Snapshot of entry order (oldest → newest), exposed for tests and
    /// diagnostics. Not for production callers — they should go through
    /// `lookup` / `insert`.
    public var entrySnapshot: [PrefixSnapshot] { entries }
}
