// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Phase 4 disk persistence tests (spec 017)

private func makeTempDir() -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("prefix-cache-test-\(UUID().uuidString)", isDirectory: true)
    return url
}

private func makeKey(layers: Int = 2) -> PrefixKey {
    PrefixKey(modelID: "test-disk", layerCount: layers, kvHeadDim: 4)
}

private func makeSnap(tokens: [Int], layers: Int = 2) -> PrefixSnapshot {
    let key = makeKey(layers: layers)
    let layerStates = (0..<layers).map { _ in
        LayerCacheState(
            kind: .standardUnbounded, tokenCount: tokens.count,
            arrays: [MLXArray(0..<8).asType(.float16).reshaped([1, 1, 8])])
    }
    return PrefixSnapshot(key: key, tokens: tokens, layerStates: layerStates)
}

@Suite
struct PrefixKVCacheDiskTests {

    @Test
    func `write+lookup recovers snapshot byte-stable`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)
        let snap = makeSnap(tokens: [1, 2, 3])
        try disk.write(snap)

        let recovered = try disk.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(recovered != nil)
        #expect(recovered?.tokens == [1, 2, 3])
        #expect(recovered?.key == snap.key)
        #expect(recovered?.layerStates.count == 2)
    }

    @Test
    func `lookup returns nil when no snapshot in directory`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)
        let r = try disk.lookup(prefix: [1, 2, 3], key: makeKey())
        #expect(r == nil)
    }

    @Test
    func `idempotent write overwrites the previous snapshot at the same fingerprint`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)
        try disk.write(makeSnap(tokens: [1, 2, 3]))
        try disk.write(makeSnap(tokens: [1, 2, 3]))
        // Only one directory should exist (same fingerprint).
        let entries = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        #expect(entries.count == 1)
    }

    @Test
    func `lookup picks longest matching snapshot across multiple stored prefixes`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)
        try disk.write(makeSnap(tokens: [1, 2]))
        try disk.write(makeSnap(tokens: [1, 2, 3, 4]))
        try disk.write(makeSnap(tokens: [1, 2, 3]))

        let r = try disk.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(r?.tokens == [1, 2, 3, 4])
    }

    @Test
    func `lookup ignores snapshots from other modelIDs`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)

        let otherKey = PrefixKey(modelID: "other", layerCount: 2, kvHeadDim: 4)
        let otherSnap = PrefixSnapshot(
            key: otherKey, tokens: [1, 2, 3],
            layerStates: (0..<2).map { _ in
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 3,
                    arrays: [MLXArray.zeros([4], dtype: .float16)])
            })
        try disk.write(otherSnap)
        try disk.write(makeSnap(tokens: [1, 2, 3]))

        // Lookup with the test-disk key picks the test-disk snap.
        let r = try disk.lookup(prefix: [1, 2, 3, 4], key: makeKey())
        #expect(r != nil)
        #expect(r?.key.modelID == "test-disk")
    }

    @Test
    func `clear removes the entire root`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)
        try disk.write(makeSnap(tokens: [1, 2]))
        try disk.clear()
        let r = try disk.lookup(prefix: [1, 2], key: makeKey())
        #expect(r == nil)
    }

    @Test
    func `read throws on format version mismatch`() throws {
        let root = makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = PrefixKVCacheDisk(root: root)

        // Write a snap with a stale version baked into the key.
        let staleKey = PrefixKey(
            modelID: "test-disk", layerCount: 2, kvHeadDim: 4,
            formatVersion: 1)
        let snap = PrefixSnapshot(
            key: staleKey, tokens: [1, 2],
            layerStates: (0..<2).map { _ in
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 2,
                    arrays: [MLXArray.zeros([4], dtype: .float16)])
            })
        try disk.write(snap)

        // Find the directory we wrote into and read it back.
        let entries = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        #expect(entries.count == 1)
        let dir = entries[0]
        #expect(throws: PrefixKVCacheError.self) {
            _ = try disk.read(from: dir)
        }
        // lookup() returns nil because read() throws are caught internally.
        let r = try disk.lookup(prefix: [1, 2], key: makeKey())
        #expect(r == nil)
    }
}
