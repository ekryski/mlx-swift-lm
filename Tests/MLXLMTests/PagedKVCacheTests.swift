// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("PagedKVCache foundation")
struct PagedKVCacheTests {

    // MARK: - Round-trip identity

    @Test
    func `scatter then gather returns identical K and V`() throws {
        let cache = PagedKVCache(
            numBlocks: 8, blockSize: 4, numKVHeads: 2, headDim: 16, dtype: .float32
        )
        let allocator = BlockAllocator(numBlocks: 8)

        // Allocate enough blocks for 12 tokens (3 blocks of 4)
        let blocks = try allocator.allocate(3)
        cache.appendBlocks(blocks)

        // Synthetic K/V: [1, 2, 12, 16] — distinguishable per (head, token, dim)
        let T = 12
        let keys = makePattern(B: 1, H: 2, T: T, D: 16, salt: 1.0)
        let values = makePattern(B: 1, H: 2, T: T, D: 16, salt: 100.0)

        cache.scatter(keys: keys, values: values)
        let (kOut, vOut) = cache.gather()

        // Output shape and values must match input exactly
        #expect(kOut.shape == keys.shape)
        #expect(vOut.shape == values.shape)
        #expect(MLX.allClose(kOut, keys, atol: 1e-6).item(Bool.self))
        #expect(MLX.allClose(vOut, values, atol: 1e-6).item(Bool.self))

        allocator.free(blocks)
    }

    @Test
    func `partial last block is trimmed`() throws {
        // 1 block of size 4, write 3 tokens. Gather must return 3 not 4.
        let cache = PagedKVCache(
            numBlocks: 4, blockSize: 4, numKVHeads: 1, headDim: 8, dtype: .float32
        )
        let allocator = BlockAllocator(numBlocks: 4)
        cache.appendBlocks(try allocator.allocate(1))

        let keys = makePattern(B: 1, H: 1, T: 3, D: 8, salt: 1.0)
        let values = makePattern(B: 1, H: 1, T: 3, D: 8, salt: 100.0)

        cache.scatter(keys: keys, values: values)
        let (kOut, vOut) = cache.gather()

        #expect(kOut.dim(2) == 3)
        #expect(vOut.dim(2) == 3)
        #expect(MLX.allClose(kOut, keys, atol: 1e-6).item(Bool.self))
        #expect(MLX.allClose(vOut, values, atol: 1e-6).item(Bool.self))
    }

    @Test
    func `multiple scatter calls accumulate`() throws {
        // Scatter 3 then 5 then 4 tokens → gather 12 in original order.
        let cache = PagedKVCache(
            numBlocks: 8, blockSize: 4, numKVHeads: 2, headDim: 16, dtype: .float32
        )
        let allocator = BlockAllocator(numBlocks: 8)
        cache.appendBlocks(try allocator.allocate(3))

        let full = makePattern(B: 1, H: 2, T: 12, D: 16, salt: 1.0)
        let chunkSizes = [3, 5, 4]
        var startTok = 0
        for n in chunkSizes {
            let kChunk = full[0..., 0..., startTok ..< (startTok + n), 0...]
            let vChunk = full[0..., 0..., startTok ..< (startTok + n), 0...]
            cache.scatter(keys: kChunk, values: vChunk)
            startTok += n
        }
        let (kOut, _) = cache.gather()
        #expect(MLX.allClose(kOut, full, atol: 1e-6).item(Bool.self))
    }

    // MARK: - Forward equivalence vs StandardKVCache

    /// `PagedKVCache.update()` and `StandardKVCache.update()` must return
    /// element-identical `(K, V)` for the same input sequence — a model that
    /// swaps in `PagedKVCache` then produces the same tokens as the same
    /// model with `StandardKVCache`. Six chunks of varying length cross block
    /// boundaries multiple times.
    @Test
    func `update output matches StandardKVCache element-wise`() throws {
        let kvHeads = 4
        let headDim = 32
        let blockSize = 8

        // Allocate enough blocks for ~5 incremental updates of varying length.
        // Total tokens we'll push: 7 + 1 + 1 + 4 + 1 + 1 = 15  → 2 blocks
        let chunkSizes = [7, 1, 1, 4, 1, 1]
        let totalTokens = chunkSizes.reduce(0, +)
        let blocksNeeded = (totalTokens + blockSize - 1) / blockSize

        let paged = PagedKVCache(
            numBlocks: blocksNeeded + 1,
            blockSize: blockSize,
            numKVHeads: kvHeads,
            headDim: headDim,
            dtype: .float32,
        )
        let allocator = BlockAllocator(numBlocks: blocksNeeded + 1)
        paged.appendBlocks(try allocator.allocate(blocksNeeded))

        let simple = StandardKVCache()

        var startTok = 0
        for (idx, n) in chunkSizes.enumerated() {
            let kChunk = makePattern(B: 1, H: kvHeads, T: n, D: headDim, salt: Float(idx + 1))
            let vChunk = makePattern(B: 1, H: kvHeads, T: n, D: headDim, salt: Float(100 * (idx + 1)))

            // Grow paged block table if next write would cross boundary
            let needBlocks = (paged.offset + n + blockSize - 1) / blockSize
            if needBlocks > paged.blockTable.count {
                paged.appendBlocks(try allocator.allocate(needBlocks - paged.blockTable.count))
            }

            let (pK, pV) = paged.update(keys: kChunk, values: vChunk)
            let (sK, sV) = simple.update(keys: kChunk, values: vChunk)

            // Both should hold (startTok + n) tokens after this update
            #expect(pK.dim(2) == startTok + n, "step \(idx): paged token count")
            #expect(sK.dim(2) == startTok + n, "step \(idx): simple token count")
            #expect(pK.shape == sK.shape, "step \(idx): K shape mismatch")
            #expect(pV.shape == sV.shape, "step \(idx): V shape mismatch")
            #expect(MLX.allClose(pK, sK, atol: 1e-6).item(Bool.self),
                    "step \(idx): K values diverged from StandardKVCache")
            #expect(MLX.allClose(pV, sV, atol: 1e-6).item(Bool.self),
                    "step \(idx): V values diverged from StandardKVCache")

            startTok += n
        }

        // Bonus: peek() must return same as last update
        if let (peekK, peekV) = paged.peek(), let (sPeekK, sPeekV) = simple.peek() {
            #expect(MLX.allClose(peekK, sPeekK, atol: 1e-6).item(Bool.self), "peek K mismatch")
            #expect(MLX.allClose(peekV, sPeekV, atol: 1e-6).item(Bool.self), "peek V mismatch")
        }
    }

    // MARK: - Allocator

    @Test
    func `allocator hands out and reclaims blocks`() throws {
        let allocator = BlockAllocator(numBlocks: 4)
        #expect(allocator.freeCount == 4)

        let a = try allocator.allocate(2)
        #expect(allocator.allocatedCount == 2)
        #expect(allocator.refcount(of: a[0]) == 1)
        #expect(allocator.refcount(of: a[1]) == 1)

        allocator.free(a)
        #expect(allocator.freeCount == 4)
        #expect(allocator.refcount(of: a[0]) == 0)
    }

    @Test
    func `allocator throws on exhaustion`() throws {
        let allocator = BlockAllocator(numBlocks: 2)
        _ = try allocator.allocate(2)
        #expect(throws: AllocatorError.self) {
            _ = try allocator.allocate(1)
        }
    }

    @Test
    func `retain bumps refcount for prefix sharing`() throws {
        let allocator = BlockAllocator(numBlocks: 4)
        let a = try allocator.allocate(2)
        allocator.retain(a)
        #expect(allocator.refcount(of: a[0]) == 2)
        // First free — block stays owned (refcount drops to 1)
        allocator.free(a)
        #expect(allocator.allocatedCount == 2)
        #expect(allocator.refcount(of: a[0]) == 1)
        // Second free — block returns to pool
        allocator.free(a)
        #expect(allocator.allocatedCount == 0)
    }

    // MARK: - Helpers

    /// `[B, H, T, D]` array where each element is uniquely determined by its index
    /// (so any reordering bug shows up as a value mismatch).
    private func makePattern(B: Int, H: Int, T: Int, D: Int, salt: Float) -> MLXArray {
        let total = B * H * T * D
        let values = (0 ..< total).map { Float($0) * salt }
        return MLXArray(values).reshaped([B, H, T, D])
    }
}
