// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Phase 4: disk persistence for the prefix KV cache (spec 017)
//
// L2 cache: snapshots are written to `~/.cache/mlx-swift-lm/prefix/`
// as `.safetensors` blobs plus a per-snapshot JSON manifest. On startup
// the cache walks the directory and lazily indexes everything; a lookup
// miss in the in-memory L1 falls through to L2 (sync read), and on hit
// the snapshot is promoted to L1.
//
// The disk schema:
//
//   <root>/
//     <fingerprint>/
//       index.json        — PrefixKey + tokens + token-hash + meta
//       arrays.safetensors — flattened layer arrays
//
// `<fingerprint>` is `sha256(modelID + ":" + first 16 token IDs + ":" +
// tokens.count)`. Truncating the hash input keeps the directory name
// short; the full token sequence + length is stored in `index.json` for
// disambiguation on lookup.
//
// Naming + layout match dflash-mlx upstream's `prefix_l2.py` enough
// that an audit can read both side-by-side. We do **not** mirror its
// async-writer queue here — Swift's `Task.detached` covers the same
// ground when callers want it.

/// On-disk index for a single snapshot. Bumped 1 → 2 in line with
/// ``PrefixKey/currentFormatVersion``.
private struct DiskSnapshotManifest: Codable {
    let formatVersion: Int
    let modelID: String
    let layerCount: Int
    let kvHeadDim: Int
    let kvBits: Int?
    let captureLayerIds: [Int]?
    let tokens: [Int]
    /// `[layer index] -> array count + metaState`
    let layers: [LayerManifest]
    let createdAt: Date
    let hasLastHidden: Bool

    struct LayerManifest: Codable {
        let kind: String           // serialised LayerCacheState.Kind discriminator
        let kindParam1: Int?       // varies by kind (e.g. maxSize / bits / keyBits)
        let kindParam2: Int?       // varies by kind (e.g. keep / groupSize / valueBits)
        let tokenCount: Int
        let arrayCount: Int
        let metaState: [String]
    }
}

/// Disk-persistence layer for ``PrefixKVCache``. Stateless: open a
/// fresh ``PrefixKVCacheDisk`` for each call site; instances don't hold
/// file descriptors.
///
/// Usage from ``PrefixKVCache``: after an L1 miss, call
/// ``lookup(prefix:key:)`` to check L2; on success, hydrate into a
/// fresh cache and (optionally) promote back into L1.
public final class PrefixKVCacheDisk: @unchecked Sendable {

    public static let defaultRoot: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("mlx-swift-lm", isDirectory: true)
            .appendingPathComponent("prefix", isDirectory: true)
    }()

    /// Process-wide shared disk cache rooted at ``defaultRoot``. Lazy.
    public static let shared = PrefixKVCacheDisk(root: PrefixKVCacheDisk.defaultRoot)

    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    /// Ensure the root directory exists. Called lazily on first
    /// write — `FileManager.createDirectory` is idempotent with
    /// `withIntermediateDirectories: true`.
    private func ensureRoot() throws {
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true, attributes: nil)
    }

    // MARK: - Fingerprint

    /// Stable filesystem-safe identifier for a (key, tokens) pair. We
    /// use a Foundation-based hash (FNV-1a 64-bit) — good enough for
    /// disk-layout disambiguation, no crypto requirement.
    private func fingerprint(_ key: PrefixKey, tokens: [Int]) -> String {
        var hash: UInt64 = 0xcbf29ce484222325  // FNV offset basis
        let prime: UInt64 = 0x100000001b3        // FNV prime
        func mix(_ value: UInt64) {
            hash ^= value
            hash &*= prime
        }
        for byte in key.modelID.utf8 {
            mix(UInt64(byte))
        }
        mix(UInt64(key.layerCount))
        mix(UInt64(key.kvHeadDim))
        if let bits = key.kvBits { mix(UInt64(bits)) } else { mix(0xffff) }
        mix(UInt64(tokens.count))
        // Mix the first 16 token IDs so two snapshots with the same
        // length but different prefixes don't collide in directory
        // names. Full-prefix equality is rechecked from the manifest.
        for i in 0 ..< min(16, tokens.count) {
            // Tokens are typically [0, vocab_size); fold negative IDs
            // (impossible in practice but cheap to defend against).
            mix(UInt64(bitPattern: Int64(tokens[i])))
        }
        return String(format: "%016llx", hash)
    }

    private func snapshotDir(_ key: PrefixKey, tokens: [Int]) -> URL {
        root.appendingPathComponent(fingerprint(key, tokens: tokens), isDirectory: true)
    }

    // MARK: - Write

    /// Persist a snapshot to disk. Replaces any existing snapshot with
    /// the same fingerprint. Idempotent.
    ///
    /// Backed by safetensors via the existing
    /// `MLX.save(arrays:metadata:url:)` helper. Layer arrays are
    /// flattened as `"<layer-idx>.<array-idx>"` keys.
    public func write(_ snapshot: PrefixSnapshot) throws {
        try ensureRoot()
        let dir = snapshotDir(snapshot.key, tokens: snapshot.tokens)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true, attributes: nil)

        // Manifest (JSON).
        let manifest = makeManifest(snapshot)
        let manifestURL = dir.appendingPathComponent("index.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)

        // Arrays (safetensors).
        var arrays: [String: MLXArray] = [:]
        for (i, layer) in snapshot.layerStates.enumerated() {
            for (j, arr) in layer.arrays.enumerated() {
                arrays["\(i).\(j)"] = arr
            }
        }
        if let lh = snapshot.lastHidden {
            arrays["lastHidden"] = lh
        }
        // No metadata needed on the safetensors side — the manifest
        // carries everything structural.
        let arraysURL = dir.appendingPathComponent("arrays.safetensors")
        try save(arrays: arrays, url: arraysURL)
    }

    // MARK: - Read

    /// Look up a snapshot whose tokens are a prefix of `requestTokens`.
    /// Returns the longest match (token-exact) across all on-disk
    /// snapshots that share the supplied ``PrefixKey``.
    ///
    /// This is a **synchronous** scan: phase 4 disk reads only happen
    /// on L1 miss, so the latency is amortised against the saved
    /// prefill. Callers that want async should wrap in `Task.detached`.
    ///
    /// - Throws: ``PrefixKVCacheError/formatVersionMismatch(expected:found:)``
    ///   when a disk snapshot's format version doesn't match this build.
    ///   Throws are non-fatal — caller may skip the snapshot and continue.
    public func lookup(
        prefix requestTokens: [Int], key: PrefixKey
    ) throws -> PrefixSnapshot? {
        let listing: [URL]
        do {
            listing = try FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        } catch CocoaError.fileReadNoSuchFile {
            return nil
        }

        var best: PrefixSnapshot? = nil
        for dir in listing {
            guard let snap = try? read(from: dir) else { continue }
            guard snap.key == key else { continue }
            guard snap.tokens.count <= requestTokens.count else { continue }
            if snap.tokens.elementsEqual(requestTokens.prefix(snap.tokens.count)) {
                if best == nil || snap.tokens.count > best!.tokens.count {
                    best = snap
                }
            }
        }
        return best
    }

    /// Read a single on-disk snapshot at `dir`. Throws on schema
    /// mismatch.
    public func read(from dir: URL) throws -> PrefixSnapshot {
        let manifestURL = dir.appendingPathComponent("index.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(DiskSnapshotManifest.self, from: manifestData)
        if manifest.formatVersion != PrefixKey.currentFormatVersion {
            throw PrefixKVCacheError.formatVersionMismatch(
                expected: PrefixKey.currentFormatVersion,
                found: manifest.formatVersion)
        }

        // Load arrays.
        let arraysURL = dir.appendingPathComponent("arrays.safetensors")
        let arrays = try loadArrays(url: arraysURL)

        // Reconstruct layer states.
        var layerStates: [LayerCacheState] = []
        layerStates.reserveCapacity(manifest.layers.count)
        for (i, lm) in manifest.layers.enumerated() {
            guard let kind = layerKind(from: lm) else {
                throw PrefixKVCacheError.snapshotInvariantViolation(
                    "layer \(i) has unknown kind '\(lm.kind)'")
            }
            var layerArrays: [MLXArray] = []
            for j in 0 ..< lm.arrayCount {
                guard let arr = arrays["\(i).\(j)"] else {
                    throw PrefixKVCacheError.snapshotInvariantViolation(
                        "layer \(i) array \(j) missing from safetensors")
                }
                layerArrays.append(arr)
            }
            layerStates.append(LayerCacheState(
                kind: kind, tokenCount: lm.tokenCount,
                arrays: layerArrays, metaState: lm.metaState))
        }

        let key = PrefixKey(
            modelID: manifest.modelID,
            layerCount: manifest.layerCount,
            kvHeadDim: manifest.kvHeadDim,
            kvBits: manifest.kvBits,
            captureLayerIds: manifest.captureLayerIds,
            formatVersion: manifest.formatVersion)

        return PrefixSnapshot(
            key: key,
            tokens: manifest.tokens,
            layerStates: layerStates,
            lastHidden: manifest.hasLastHidden ? arrays["lastHidden"] : nil,
            createdAt: manifest.createdAt)
    }

    /// Remove all on-disk snapshots. Use at bench-run boundaries or to
    /// reset corrupted state.
    public func clear() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Manifest <-> kind

    private func makeManifest(_ s: PrefixSnapshot) -> DiskSnapshotManifest {
        let layers: [DiskSnapshotManifest.LayerManifest] = s.layerStates.map { ls in
            let (kindStr, p1, p2) = encodeKind(ls.kind)
            return .init(
                kind: kindStr,
                kindParam1: p1,
                kindParam2: p2,
                tokenCount: ls.tokenCount,
                arrayCount: ls.arrays.count,
                metaState: ls.metaState)
        }
        return DiskSnapshotManifest(
            formatVersion: s.key.formatVersion,
            modelID: s.key.modelID,
            layerCount: s.key.layerCount,
            kvHeadDim: s.key.kvHeadDim,
            kvBits: s.key.kvBits,
            captureLayerIds: s.key.captureLayerIds,
            tokens: s.tokens,
            layers: layers,
            createdAt: s.createdAt,
            hasLastHidden: s.lastHidden != nil)
    }

    private func encodeKind(_ kind: LayerCacheState.Kind) -> (String, Int?, Int?) {
        switch kind {
        case .standardUnbounded: return ("standardUnbounded", nil, nil)
        case .standardWindowed(let m, let k): return ("standardWindowed", m, k)
        case .affineQuantized(let b, let g): return ("affineQuantized", b, g)
        case .turboCompressed(let k, let v): return ("turboCompressed", k, v)
        case .ssm: return ("ssm", nil, nil)
        }
    }

    private func layerKind(from lm: DiskSnapshotManifest.LayerManifest) -> LayerCacheState.Kind? {
        switch lm.kind {
        case "standardUnbounded":
            return .standardUnbounded
        case "standardWindowed":
            guard let m = lm.kindParam1, let k = lm.kindParam2 else { return nil }
            return .standardWindowed(maxSize: m, keep: k)
        case "affineQuantized":
            guard let b = lm.kindParam1, let g = lm.kindParam2 else { return nil }
            return .affineQuantized(bits: b, groupSize: g)
        case "turboCompressed":
            guard let k = lm.kindParam1, let v = lm.kindParam2 else { return nil }
            return .turboCompressed(keyBits: k, valueBits: v)
        case "ssm":
            return .ssm
        default:
            return nil
        }
    }
}
