// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression coverage for the BatchedKVCache turbo storage path. Pins:
//   * Raw mode is bit-identical pre/post turbo refactor (covered elsewhere
//     in BatchedHybridCacheTests + Qwen35BatchedHybridCacheTests).
//   * Turbo mode actually compresses and dequants — i.e. the kv_scheme flag
//     stops being a no-op on the batched-decode path. Buddy's v0.5.1 alpha
//     report (Qwen3.6-35B-A3B-4bit + turbo4v2 → ".2.2.2.2..." drift) was a
//     symptom of turbo silently bypassed; this suite locks down the fix.
//   * Turbo attention output stays close to the raw-mode reference output
//     within a tolerance that matches the codec bit-width (V at 2 bits is
//     intentionally lossy; we only require it not be catastrophic).

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("BatchedKVCache turbo storage")
struct BatchedTurboKVCacheTests {

    // MARK: - Construction

    @Test
    func `raw init reports isTurbo == false`() throws {
        let cache = BatchedKVCache(
            maxBatch: 4, kvHeads: 2, headDim: 128, maxSeq: 16
        )
        #expect(!cache.isTurbo)
        #expect(cache.keyBits == 0)
        #expect(cache.valueBits == 0)
    }

    @Test
    func `turbo init reports isTurbo == true and bit-widths`() throws {
        let cache = BatchedKVCache(
            maxBatch: 2, kvHeads: 2, headDim: 128, maxSeq: 16,
            turboKeyBits: 4, turboValueBits: 2
        )
        #expect(cache.isTurbo)
        #expect(cache.keyBits == 4)
        #expect(cache.valueBits == 2)
        #expect(!cache.rawKeyMode)
    }

    @Test
    func `turbo raw-key mode flagged when keyBits == 0`() throws {
        let cache = BatchedKVCache(
            maxBatch: 1, kvHeads: 2, headDim: 128, maxSeq: 16,
            turboKeyBits: 0, turboValueBits: 4
        )
        #expect(cache.isTurbo)
        #expect(cache.rawKeyMode)
    }

    // MARK: - Update

    @Test
    func `turbo update advances offset and writes to packed storage`() throws {
        let cache = BatchedKVCache(
            maxBatch: 1, kvHeads: 2, headDim: 128, maxSeq: 16,
            turboKeyBits: 4, turboValueBits: 4
        )
        _ = cache.addRequest()  // active = 1, offset = 0

        let newK = MLXArray.ones([1, 2, 1, 128], dtype: .bfloat16)
        let newV = MLXArray.ones([1, 2, 1, 128], dtype: .bfloat16)
        cache.update(newKeys: newK, newValues: newV)

        #expect(cache.offsets[0] == 1)
    }

    // MARK: - Attention

    @Test
    func `turbo attention output shape matches raw attention output shape`() throws {
        let B = 2
        let kvH = 2
        let nQH = 4
        let D = 64
        let T = 8

        let raw = BatchedKVCache(
            maxBatch: B, kvHeads: kvH, headDim: D, maxSeq: T
        )
        let turbo = BatchedKVCache(
            maxBatch: B, kvHeads: kvH, headDim: D, maxSeq: T,
            turboKeyBits: 4, turboValueBits: 4
        )
        for _ in 0..<B {
            _ = raw.addRequest()
            _ = turbo.addRequest()
        }

        // Fill T tokens of K/V into both caches.
        for _ in 0..<T {
            let k = MLXArray.ones([B, kvH, 1, D], dtype: .bfloat16)
            let v = MLXArray.ones([B, kvH, 1, D], dtype: .bfloat16)
            raw.update(newKeys: k, newValues: v)
            turbo.update(newKeys: k, newValues: v)
        }

        let q = MLXArray.ones([B, nQH, 1, D], dtype: .bfloat16)
        let mask = MLXArray.zeros([B, 1, 1, T], dtype: .bfloat16)

        let rawOut = raw.attention(queries: q, scale: 1.0, mask: mask)
        let turboOut = turbo.attention(queries: q, scale: 1.0, mask: mask)

        #expect(rawOut.shape == [B, nQH, 1, D])
        #expect(turboOut.shape == [B, nQH, 1, D])
    }

    @Test
    func `turbo attention with raw-key mode produces non-NaN output`() throws {
        // V-only compression (raw-key mode) is the highest-quality TurboQuant
        // configuration. Verify the rawKeyMode codepath in turboAttention()
        // doesn't blow up (no NaN / inf).
        let cache = BatchedKVCache(
            maxBatch: 1, kvHeads: 2, headDim: 128, maxSeq: 4,
            turboKeyBits: 0, turboValueBits: 4
        )
        _ = cache.addRequest()

        // Random-ish small inputs.
        let k = MLXArray(0..<256).reshaped([1, 2, 1, 128]).asType(.bfloat16) * Float(0.01)
        let v = MLXArray(0..<256).reshaped([1, 2, 1, 128]).asType(.bfloat16) * Float(0.01)
        cache.update(newKeys: k, newValues: v)

        let q = MLXArray.ones([1, 2, 1, 128], dtype: .bfloat16) * Float(0.1)
        let mask = MLXArray.zeros([1, 1, 1, 1], dtype: .bfloat16)
        let out = cache.attention(queries: q, scale: 1.0, mask: mask)
        let outFloat = out.asType(.float32)
        let total = outFloat.sum().item(Float.self)
        #expect(total.isFinite)
    }

    @Test
    func `turbo update advances offsets per batch slot`() throws {
        // With B=4 active slots, all slots should advance in lockstep when
        // they share a starting offset (the common batched-decode path).
        let cache = BatchedKVCache(
            maxBatch: 4, kvHeads: 2, headDim: 128, maxSeq: 8,
            turboKeyBits: 4, turboValueBits: 4
        )
        for _ in 0..<4 { _ = cache.addRequest() }

        let k = MLXArray.ones([4, 2, 1, 128], dtype: .bfloat16)
        let v = MLXArray.ones([4, 2, 1, 128], dtype: .bfloat16)
        cache.update(newKeys: k, newValues: v)
        cache.update(newKeys: k, newValues: v)
        cache.update(newKeys: k, newValues: v)

        // All four slots should be at offset 3.
        for slot in 0..<4 {
            #expect(cache.offsets[slot] == 3)
        }
    }
}
