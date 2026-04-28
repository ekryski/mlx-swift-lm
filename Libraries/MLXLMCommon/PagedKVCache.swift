// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Per-layer paged KV cache. One instance per transformer layer.
///
/// Stores K and V in fixed-size blocks of `blockSize` tokens. Sequences
/// address blocks via a `blockTable` populated externally from a shared
/// `BlockAllocator`. The forward path here gathers blocks into a contiguous
/// `[1, kvHeads, T, headDim]` view and hands it to MLX SDPA — a Metal paged
/// kernel will replace `gather()` for the production decode path.
///
/// Block layout matches vLLM's MetalPagedKVCache exactly so a future kernel
/// port can read this storage directly:
///   keyBlocks:   [numBlocks, blockSize, numKVHeads, headDim]
///   valueBlocks: [numBlocks, blockSize, numKVHeads, headDim]
public class PagedKVCache: BaseKVCache {

    public let numBlocks: Int
    public let blockSize: Int
    public let numKVHeads: Int
    public let headDim: Int

    public internal(set) var keyBlocks: MLXArray
    public internal(set) var valueBlocks: MLXArray

    /// Block ids assigned to this cache, in logical token order.
    /// One request per cache instance — multi-tenancy is handled upstream.
    public internal(set) var blockTable: [Int]

    public override var maxSize: Int? { numBlocks * blockSize }

    public init(
        numBlocks: Int,
        blockSize: Int = 16,
        numKVHeads: Int,
        headDim: Int,
        dtype: DType = .bfloat16
    ) {
        precondition(numBlocks > 0, "numBlocks must be > 0")
        precondition(blockSize > 0, "blockSize must be > 0")
        precondition(numKVHeads > 0, "numKVHeads must be > 0")
        precondition(headDim > 0, "headDim must be > 0")

        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numKVHeads = numKVHeads
        self.headDim = headDim

        let shape = [numBlocks, blockSize, numKVHeads, headDim]
        self.keyBlocks = MLXArray.zeros(shape, dtype: dtype)
        self.valueBlocks = MLXArray.zeros(shape, dtype: dtype)
        self.blockTable = []

        super.init()
    }

    /// Append block ids from the allocator to this cache's table.
    public func appendBlocks(_ ids: [Int]) {
        precondition(ids.allSatisfy { $0 >= 0 && $0 < numBlocks }, "block id out of range")
        blockTable.append(contentsOf: ids)
    }

    /// Drop the last `n` block table entries. Block reclamation is the
    /// allocator's responsibility — this only severs the link.
    public func dropLastBlocks(_ n: Int) {
        let drop = Swift.min(n, blockTable.count)
        blockTable.removeLast(drop)
    }

    /// Scatter `[1, kvHeads, T, headDim]` K/V into block storage starting at
    /// the current `offset`. Caller must have already appended enough blocks
    /// to cover `offset + T` positions.
    public func scatter(keys: MLXArray, values: MLXArray) {
        precondition(keys.dim(0) == 1, "scatter requires B=1")
        precondition(keys.dim(1) == numKVHeads, "kv_heads mismatch on scatter")
        precondition(keys.dim(3) == headDim, "head_dim mismatch on scatter")

        let numNew = keys.dim(2)
        let startTok = offset

        for i in 0 ..< numNew {
            let absTok = startTok + i
            let blockIdx = absTok / blockSize
            let slotIdx = absTok % blockSize
            precondition(blockIdx < blockTable.count,
                "scatter past end of block table — caller must allocate blocks first")
            let physicalBlock = blockTable[blockIdx]

            let kSlice = keys[0..., 0..., i ..< (i + 1), 0...]
            let vSlice = values[0..., 0..., i ..< (i + 1), 0...]

            // Block layout is [block_size, kv_heads, head_dim] per block;
            // squeeze input from [1, kv_heads, 1, head_dim] to match.
            keyBlocks[physicalBlock, slotIdx, 0..., 0...] = kSlice.squeezed(axis: 0).squeezed(axis: 1)
            valueBlocks[physicalBlock, slotIdx, 0..., 0...] = vSlice.squeezed(axis: 0).squeezed(axis: 1)
        }

        offset += numNew
    }

    /// Reconstruct contiguous `[1, kvHeads, offset, headDim]` K/V from the
    /// blocks listed in `blockTable`. Output matches what `KVCacheSimple`
    /// would return for the same input sequence.
    public func gather() -> (MLXArray, MLXArray) {
        let tokenCount = offset
        let numUsedBlocks = (tokenCount + blockSize - 1) / blockSize
        precondition(numUsedBlocks <= blockTable.count, "block table too short for offset")

        let physicalIds = MLXArray(blockTable.prefix(numUsedBlocks).map { Int32($0) })

        let kGathered = keyBlocks[physicalIds, 0..., 0..., 0...]
        let vGathered = valueBlocks[physicalIds, 0..., 0..., 0...]

        let totalSlots = numUsedBlocks * blockSize
        let kFlat = kGathered.reshaped([1, totalSlots, numKVHeads, headDim])
        let vFlat = vGathered.reshaped([1, totalSlots, numKVHeads, headDim])

        let kTransposed = kFlat.transposed(0, 2, 1, 3)
        let vTransposed = vFlat.transposed(0, 2, 1, 3)

        // Trim padding from the partially-filled last block.
        let kTrimmed = kTransposed[0..., 0..., ..<tokenCount, 0...]
        let vTrimmed = vTransposed[0..., 0..., ..<tokenCount, 0...]

        return (kTrimmed, vTrimmed)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        scatter(keys: keys, values: values)
        return gather()
    }

    public override func peek() -> (MLXArray, MLXArray)? {
        guard offset > 0 else { return nil }
        return gather()
    }

    public override var state: [MLXArray] {
        get { [keyBlocks, valueBlocks] }
        set {
            guard newValue.count == 2 else {
                fatalError("PagedKVCache state requires 2 arrays (key_blocks, value_blocks)")
            }
            keyBlocks = newValue[0]
            valueBlocks = newValue[1]
        }
    }

    public override func innerState() -> [MLXArray] {
        [keyBlocks, valueBlocks]
    }

    public override var memoryBytes: Int {
        arrayBytes(keyBlocks) + arrayBytes(valueBlocks)
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimCount = Swift.min(n, offset)
        offset -= trimCount
        return trimCount
    }
}
