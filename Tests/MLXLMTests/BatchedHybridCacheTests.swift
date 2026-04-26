// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("BatchedHybridCache + BatchedMambaCache")
struct BatchedHybridCacheTests {

    // MARK: - BatchedMambaCache

    @Test
    func `addSlot increments active and zero-inits new slot`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 4, kernelMinusOne: 3, convDim: 8, Hv: 2, Dv: 4, Dk: 4
        )
        #expect(cache.active == 0)

        let slot0 = cache.addSlot()
        #expect(slot0 == 0)
        #expect(cache.active == 1)

        let slot1 = cache.addSlot()
        #expect(slot1 == 1)
        #expect(cache.active == 2)

        // Both newly-added slots must be exactly zero.
        let convSlot0 = cache.convState[0, 0..., 0...]
        let recSlot0 = cache.recState[0, 0..., 0..., 0...]
        #expect(convSlot0.sum().item(Float.self) == 0)
        #expect(recSlot0.sum().item(Float.self) == 0)
    }

    @Test
    func `addSlot wipes stale state after slot reuse`() throws {
        // The critical invariant from the design doc: if slot N is freed and
        // re-allocated later (via swap-from-end), the next addSlot must
        // zero-init it.
        let cache = BatchedMambaCache(
            maxBatch: 3, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2
        )

        cache.addSlot()  // 0
        cache.addSlot()  // 1
        cache.addSlot()  // 2

        // Stuff slot 1 with non-zero state to simulate active GDN history
        let convFill = MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 7.5
        let recFill = MLXArray.ones([1, 2, 2], dtype: .float32) * 3.25
        cache.convState[1, 0..., 0...] = convFill
        cache.recState[1, 0..., 0..., 0...] = recFill

        // Remove slot 0 → swap-from-end pulls slot 2 into 0; old slot 2 is
        // now garbage. active = 2.
        cache.removeSlot(0)
        #expect(cache.active == 2)

        // Stuff the (now garbage) tail position 2 to make sure the next
        // addSlot wipes it.
        cache.convState[2, 0..., 0...] =
            MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 11.0
        cache.recState[2, 0..., 0..., 0...] =
            MLXArray.ones([1, 2, 2], dtype: .float32) * 13.0

        // Re-allocate slot 2.
        let reused = cache.addSlot()
        #expect(reused == 2)

        let convReused = cache.convState[2, 0..., 0...]
        let recReused = cache.recState[2, 0..., 0..., 0...]
        #expect(convReused.sum().item(Float.self) == 0,
                "stale conv leaked into reused slot")
        #expect(recReused.sum().item(Float.self) == 0,
                "stale rec leaked into reused slot")
    }

    @Test
    func `removeSlot swaps from end and decrements active`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 3, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2
        )
        cache.addSlot()  // 0
        cache.addSlot()  // 1
        cache.addSlot()  // 2

        // Distinguishable per-slot fills
        for s in 0..<3 {
            let v = Float(s + 1)
            cache.convState[s, 0..., 0...] =
                MLXArray.ones([2, 4], dtype: cache.convState.dtype) * v
            cache.recState[s, 0..., 0..., 0...] =
                MLXArray.ones([1, 2, 2], dtype: .float32) * v
        }

        // Remove slot 0 → slot 2 swaps in
        cache.removeSlot(0)
        #expect(cache.active == 2)

        // Slot 0 should now hold the values that were at slot 2
        let convAt0 = cache.convState[0, 0..., 0...]
        let recAt0 = cache.recState[0, 0..., 0..., 0...]
        #expect(MLX.allClose(convAt0,
                             MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 3.0,
                             atol: 1e-6).item(Bool.self))
        #expect(MLX.allClose(recAt0,
                             MLXArray.ones([1, 2, 2], dtype: .float32) * 3.0,
                             atol: 1e-6).item(Bool.self))

        // Slot 1 untouched
        let convAt1 = cache.convState[1, 0..., 0...]
        #expect(MLX.allClose(convAt1,
                             MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 2.0,
                             atol: 1e-6).item(Bool.self))
    }

    @Test
    func `removeSlot of last slot is no-op data-wise`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 2, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2
        )
        cache.addSlot()  // 0
        cache.addSlot()  // 1
        cache.convState[0, 0..., 0...] =
            MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 5.0

        cache.removeSlot(1)  // last
        #expect(cache.active == 1)
        let convAt0 = cache.convState[0, 0..., 0...]
        #expect(MLX.allClose(convAt0,
                             MLXArray.ones([2, 4], dtype: cache.convState.dtype) * 5.0,
                             atol: 1e-6).item(Bool.self))
    }

    @Test
    func `slice returns prefix views over active slots`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 4, kernelMinusOne: 3, convDim: 8, Hv: 2, Dv: 4, Dk: 4
        )
        cache.addSlot()
        cache.addSlot()
        cache.addSlot()  // active = 3

        let (conv, rec) = cache.slice(active: cache.active)
        #expect(conv.shape == [3, 3, 8])
        #expect(rec.shape == [3, 2, 4, 4])
    }

    @Test
    func `writeback updates only active prefix`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 3, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2
        )
        cache.addSlot()
        cache.addSlot()  // active = 2

        let newConv = MLXArray.ones([2, 2, 4], dtype: cache.convState.dtype) * 9.0
        let newRec = MLXArray.ones([2, 1, 2, 2], dtype: .float32) * 4.0

        cache.writeback(conv: newConv, rec: newRec)

        // Active prefix updated
        let prefixConv = cache.convState[..<2, 0..., 0...]
        #expect(MLX.allClose(prefixConv, newConv, atol: 1e-6).item(Bool.self))
        let prefixRec = cache.recState[..<2, 0..., 0..., 0...]
        #expect(MLX.allClose(prefixRec, newRec, atol: 1e-6).item(Bool.self))
    }

    @Test
    func `recState is fp32 even when inputs are bf16`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 2, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2,
            dtype: .bfloat16
        )
        #expect(cache.recState.dtype == .float32)
        #expect(cache.convState.dtype == .bfloat16)
    }

    @Test
    func `reset zeros active count`() throws {
        let cache = BatchedMambaCache(
            maxBatch: 3, kernelMinusOne: 2, convDim: 4, Hv: 1, Dv: 2, Dk: 2
        )
        cache.addSlot()
        cache.addSlot()
        #expect(cache.active == 2)
        cache.reset()
        #expect(cache.active == 0)
    }

    // MARK: - BatchedHybridCache lockstep

    @Test
    func `addSlot keeps every layer in lockstep`() throws {
        let layers: [BatchedHybridCache.BatchedLayerCache] = [
            .gdn(BatchedMambaCache(
                maxBatch: 4, kernelMinusOne: 3, convDim: 8,
                Hv: 2, Dv: 4, Dk: 4)),
            .attention(BatchedKVCache(
                maxBatch: 4, kvHeads: 2, headDim: 16, maxSeq: 32,
                dtype: .bfloat16)),
            .gdn(BatchedMambaCache(
                maxBatch: 4, kernelMinusOne: 3, convDim: 8,
                Hv: 2, Dv: 4, Dk: 4)),
            .attention(BatchedKVCache(
                maxBatch: 4, kvHeads: 2, headDim: 16, maxSeq: 32,
                dtype: .bfloat16)),
        ]
        let hybrid = BatchedHybridCache(layers: layers)

        #expect(hybrid.active == 0)

        let s0 = hybrid.addSlot()
        #expect(s0 == 0)
        #expect(hybrid.active == 1)

        let s1 = hybrid.addSlot()
        #expect(s1 == 1)
        #expect(hybrid.active == 2)

        // Verify every layer reports the same active count
        for layer in hybrid.layers {
            switch layer {
            case .attention(let c): #expect(c.active == 2)
            case .gdn(let c): #expect(c.active == 2)
            }
        }
    }

    @Test
    func `removeSlot keeps every layer in lockstep`() throws {
        let layers: [BatchedHybridCache.BatchedLayerCache] = [
            .gdn(BatchedMambaCache(
                maxBatch: 3, kernelMinusOne: 2, convDim: 4,
                Hv: 1, Dv: 2, Dk: 2)),
            .attention(BatchedKVCache(
                maxBatch: 3, kvHeads: 1, headDim: 8, maxSeq: 16,
                dtype: .bfloat16)),
        ]
        let hybrid = BatchedHybridCache(layers: layers)

        hybrid.addSlot()
        hybrid.addSlot()
        hybrid.addSlot()
        #expect(hybrid.active == 3)

        hybrid.removeSlot(1)
        #expect(hybrid.active == 2)
        for layer in hybrid.layers {
            switch layer {
            case .attention(let c): #expect(c.active == 2)
            case .gdn(let c): #expect(c.active == 2)
            }
        }
    }

    @Test
    func `reset clears every layer`() throws {
        let layers: [BatchedHybridCache.BatchedLayerCache] = [
            .gdn(BatchedMambaCache(
                maxBatch: 2, kernelMinusOne: 2, convDim: 4,
                Hv: 1, Dv: 2, Dk: 2)),
            .attention(BatchedKVCache(
                maxBatch: 2, kvHeads: 1, headDim: 8, maxSeq: 16,
                dtype: .bfloat16)),
        ]
        let hybrid = BatchedHybridCache(layers: layers)
        hybrid.addSlot()
        hybrid.addSlot()
        #expect(hybrid.active == 2)
        hybrid.reset()
        #expect(hybrid.active == 0)
    }
}
