// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Flat batched KV cache: single pre-allocated tensor for B requests.
///
/// Instead of B separate StandardKVCache objects, stores all K/V in
/// `[B, kv_heads, max_seq, head_dim]`. Cache update and attention
/// are single batched operations — no per-request loops.
///
/// Two storage modes:
/// * **Raw** (default): keys/values held as fp16/bf16 tensors. Fastest decode,
///   no quantization overhead.
/// * **Turbo**: TurboQuant+ MSE codec — keys/values held as packed-byte
///   indices + per-token L2 norms in rotated codec space. Decode runs
///   bulk-dequant-first SDPA (mirrors `TurboQuantKVCache.compressedAttention`,
///   lines 1633–1652) so the kv_scheme flag set at engine create actually
///   takes effect on the batched-decode path. Without this, vllm-swift's
///   `--additional-config '{"kv_scheme":"turbo4v2"}'` was a no-op on Qwen3 /
///   Qwen3.5 / Qwen3.6 hybrid models — the batched cache silently held raw
///   fp16 K/V regardless.
public class BatchedKVCache {
    public let maxBatch: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let dtype: DType

    /// Current offset per request (how many tokens cached)
    public var offsets: [Int]

    /// K cache: `[B, kv_heads, max_seq, head_dim]` in raw mode.
    /// In turbo mode this is a tiny placeholder — `attention()` reads from
    /// packed storage instead. `keys.dtype` still answers correctly.
    public var keys: MLXArray
    /// V cache: `[B, kv_heads, max_seq, head_dim]` in raw mode.
    /// Same placeholder caveat as `keys` in turbo mode.
    public var values: MLXArray

    /// Active request count (first `active` slots are in use)
    public var active: Int = 0

    /// Cached mask — invalidated on cache update, shared across layers
    private var _cachedMask: MLXArray?
    private var _cachedMaxOff: Int = 0

    // MARK: - Turbo storage (nil in raw mode)

    /// True when this cache is operating in TurboQuant compressed mode.
    public let isTurbo: Bool
    /// Bit-width for keys (0 = raw-fp16 keys, value-only compression).
    /// Mirrors `TurboQuantKVCache.keyBits`.
    public let keyBits: Int
    /// Bit-width for values.
    public let valueBits: Int
    /// Raw-key mode: keys stay at fp16, only values are compressed. The
    /// single biggest TurboQuant+ quality finding — K precision dominates
    /// quality via softmax amplification, V compression is nearly free.
    public let rawKeyMode: Bool

    /// Packed K indices, `[maxBatch, kvHeads, maxSeq, packedW]` uint32.
    /// nil when `rawKeyMode` (keys live in `self.keys`) or in raw mode.
    private var keyPacked: MLXArray?
    /// Per-token K L2 norms (or norm-correction factors), `[maxBatch, kvHeads, maxSeq]` fp32.
    private var keyNorms: MLXArray?
    /// Packed V indices, `[maxBatch, kvHeads, maxSeq, packedW]` uint32.
    private var valPacked: MLXArray?
    private var valNorms: MLXArray?

    private let keyCodec: MSECodec?
    private let valueCodec: MSECodec?

    /// Initialize a raw-storage batched KV cache. Behaviour bit-identical to
    /// the pre-turbo version — pre-allocates `[maxBatch, kvHeads, maxSeq, headDim]`
    /// fp16/bf16 K and V tensors and writes new tokens at each slot's offset.
    public init(maxBatch: Int, kvHeads: Int, headDim: Int, maxSeq: Int = 2048,
                dtype: DType = .bfloat16) {
        self.maxBatch = maxBatch
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.offsets = Array(repeating: 0, count: maxBatch)

        self.keys = MLXArray.zeros([maxBatch, kvHeads, maxSeq, headDim], dtype: dtype)
        self.values = MLXArray.zeros([maxBatch, kvHeads, maxSeq, headDim], dtype: dtype)
        eval(self.keys, self.values)

        self.isTurbo = false
        self.keyBits = 0
        self.valueBits = 0
        self.rawKeyMode = false
        self.keyCodec = nil
        self.valueCodec = nil
    }

    /// Initialize a TurboQuant-compressed batched KV cache. Allocates packed
    /// indices + per-token norms instead of fp16 K/V tensors. `attention()`
    /// dispatches to a dequant-first SDPA path that pre-rotates queries,
    /// bulk-dequants K/V into the rotated codec space, runs MLXFast SDPA, and
    /// applies the inverse rotation to the output.
    ///
    /// - Parameters:
    ///   - turboKeyBits: 0 enables raw-fp16 keys (only values compressed —
    ///     the V-only compression mode that nearly preserves quality).
    ///   - turboValueBits: 2/3/4/8 — V codebook size = 2^bits.
    public init(maxBatch: Int, kvHeads: Int, headDim: Int, maxSeq: Int,
                dtype: DType = .bfloat16,
                turboKeyBits: Int, turboValueBits: Int, seed: UInt64 = 42) {
        self.maxBatch = maxBatch
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.offsets = Array(repeating: 0, count: maxBatch)
        self.isTurbo = true
        self.keyBits = turboKeyBits
        self.valueBits = turboValueBits
        self.rawKeyMode = (turboKeyBits == 0)

        // Codecs share boundaries/codebook tables across instances with the
        // same (dim, bits, seed). Reuses TurboQuantKVCache.getOrCreateCodec
        // would also share the rotation matrix — TODO follow-up to wire that
        // through. For now each batched cache builds its own; the cost is one
        // [headDim, headDim] matrix per cache instance (one per attention layer).
        self.valueCodec = MSECodec(dim: headDim, bits: turboValueBits, seed: seed &+ 1)
        if turboKeyBits > 0 {
            self.keyCodec = MSECodec(dim: headDim, bits: turboKeyBits, seed: seed)
        } else {
            self.keyCodec = nil
        }

        // Raw-key mode keeps a normal fp16 K cache so attention can pass
        // unmodified keys to SDPA without any dequant overhead. Otherwise
        // `keys`/`values` are tiny placeholders — turboAttention() reads from
        // packed storage instead.
        if rawKeyMode {
            self.keys = MLXArray.zeros([maxBatch, kvHeads, maxSeq, headDim], dtype: dtype)
        } else {
            self.keys = MLXArray.zeros([1, 1, 1, headDim], dtype: dtype)
        }
        self.values = MLXArray.zeros([1, 1, 1, headDim], dtype: dtype)

        // Allocate packed buffers + norms.
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: turboValueBits)
        self.valPacked = MLXArray.zeros([maxBatch, kvHeads, maxSeq, vpw], dtype: .uint32)
        self.valNorms = MLXArray.zeros([maxBatch, kvHeads, maxSeq])
        if turboKeyBits > 0 {
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: turboKeyBits)
            self.keyPacked = MLXArray.zeros([maxBatch, kvHeads, maxSeq, kpw], dtype: .uint32)
            self.keyNorms = MLXArray.zeros([maxBatch, kvHeads, maxSeq])
        }

        eval(self.keys, self.values, self.valPacked!, self.valNorms!)
        if let kp = self.keyPacked, let kn = self.keyNorms { eval(kp, kn) }
    }

    /// Add a new request. Returns batch slot index.
    public func addRequest() -> Int {
        let slot = active
        active += 1
        offsets[slot] = 0
        return slot
    }

    /// Set offset for a slot (after prefill)
    public func setOffset(_ slot: Int, _ offset: Int) {
        offsets[slot] = offset
    }

    /// Batched cache update: write new K/V for all active requests.
    /// newK, newV: [B_active, kv_heads, 1, head_dim]
    ///
    /// Fast path: when all requests have the same offset (common during
    /// continuous decode), uses single slice assignment — no loop.
    public func update(newKeys: MLXArray, newValues: MLXArray) {
        let B = active
        guard B > 0 else { return }
        _cachedMask = nil

        if isTurbo {
            turboUpdate(newKeys: newKeys, newValues: newValues, B: B)
            return
        }

        let allSameOffset = offsets[0..<B].allSatisfy { $0 == offsets[0] }

        if allSameOffset {
            let off = offsets[0]
            keys[..<B, 0..., off, 0...] = newKeys[0..., 0..., 0, 0...]
            values[..<B, 0..., off, 0...] = newValues[0..., 0..., 0, 0...]
            for i in 0..<B { offsets[i] = off + 1 }
        } else {
            // Different offsets — per-request write
            for i in 0..<B {
                let off = offsets[i]
                keys[i, 0..., off, 0...] = newKeys[i, 0..., 0, 0...]
                values[i, 0..., off, 0...] = newValues[i, 0..., 0, 0...]
                offsets[i] = off + 1
            }
        }
    }

    /// Run scaled-dot-product attention against the cached K/V for the
    /// active requests. Returns `[B_active, n_q_heads, L_q, head_dim]`.
    ///
    /// Raw mode mirrors the inline pattern Qwen3/Qwen3.5/3.6/Qwen3Next previously
    /// used (slice cache.keys/values, run MLXFast SDPA). Turbo mode does
    /// dequant-first SDPA in rotated codec space — pre-rotates queries with
    /// the codec's rotation matrix (preserving inner products under orthogonal
    /// rotation), bulk-dequants K/V via the existing `bulkDequantRotated`
    /// kernel into rotated fp16 tensors, runs SDPA, then applies the inverse
    /// rotation to the output. Same kernel chain as TurboQuantKVCache's
    /// dequant-first SDPA path, just with [B, ...] shaped buffers — the
    /// kernels were already batch-parallel, only the storage layer was
    /// missing.
    public func attention(
        queries: MLXArray, scale: Float, mask: MLXArray
    ) -> MLXArray {
        if isTurbo {
            return turboAttention(queries: queries, scale: scale, mask: mask)
        }
        let maxOffset = offsets[0..<active].max() ?? 0
        let allK = keys[..<active, 0..., ..<maxOffset, 0...]
        let allV = values[..<active, 0..., ..<maxOffset, 0...]
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: allK, values: allV,
            scale: scale, mask: .array(mask)
        )
    }

    /// Get cached K/V for all active requests up to their offsets.
    /// Returns (K, V, mask) for batched SDPA.
    /// K: [B, kv_heads, max_offset, head_dim]
    /// mask: [B, 1, 1, max_offset] with -inf for positions beyond each request's offset
    public func getCachedWithMask() -> (MLXArray, MLXArray, MLXArray) {
        let B = active
        let maxOff = offsets[0..<B].max() ?? 0

        let k = keys[..<B, 0..., ..<maxOff, 0...]
        let v = values[..<B, 0..., ..<maxOff, 0...]

        // Build mask: [B, 1, 1, maxOff] — fully vectorized, no loop
        let cacheDtype = k.dtype
        let positions = MLXArray(0..<maxOff).reshaped(1, maxOff)  // [1, maxOff]
        let offsetsArr = MLXArray(offsets[0..<B]).reshaped(B, 1)  // [B, 1]
        let valid = positions .< offsetsArr  // [B, maxOff] broadcast
        let mask = MLX.where(valid,
                             MLXArray(Float(0)).asType(cacheDtype),
                             MLXArray(Float(-1e9)).asType(cacheDtype))
            .reshaped(B, 1, 1, maxOff)

        return (k, v, mask)
    }

    /// Reset all slots
    public func reset() {
        active = 0
        for i in 0..<maxBatch { offsets[i] = 0 }
    }

    // MARK: - Turbo path

    /// Encode the new token's K/V into packed indices and write at each
    /// active slot's offset. Mirrors `TurboQuantKVCache.encodeNewToken` but
    /// for [B, kvHeads, 1, headDim] inputs — flattens the leading dims for
    /// the encode kernel (which takes [numRows, D]) and reshapes back.
    private func turboUpdate(newKeys: MLXArray, newValues: MLXArray, B: Int) {
        guard let valueCodec else { return }
        let H = kvHeads

        // V encode (always present — V is always quantized).
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)
        let flatV = newValues.reshaped([B * H, headDim])
        let (vPackedFlat, vNormsFlat) = encodeVectors(flatV, codec: valueCodec, bits: valueBits)
        let vPacked = vPackedFlat.reshaped([B, H, vpw])
        let vNormsArr = vNormsFlat.reshaped([B, H])

        // K encode — only when not raw-key mode.
        var kPacked: MLXArray? = nil
        var kNormsArr: MLXArray? = nil
        if !rawKeyMode, let keyCodec {
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatK = newKeys.reshaped([B * H, headDim])
            let (kPackedFlat, kNormsFlat) = encodeVectors(flatK, codec: keyCodec, bits: keyBits)
            kPacked = kPackedFlat.reshaped([B, H, kpw])
            kNormsArr = kNormsFlat.reshaped([B, H])
        }

        let allSameOffset = offsets[0..<B].allSatisfy { $0 == offsets[0] }

        if allSameOffset {
            let off = offsets[0]
            valPacked![..<B, 0..., off, 0...] = vPacked
            valNorms![..<B, 0..., off] = vNormsArr
            if rawKeyMode {
                keys[..<B, 0..., off, 0...] = newKeys[0..., 0..., 0, 0...]
            } else if let kp = kPacked, let kn = kNormsArr {
                keyPacked![..<B, 0..., off, 0...] = kp
                keyNorms![..<B, 0..., off] = kn
            }
            for i in 0..<B { offsets[i] = off + 1 }
        } else {
            for i in 0..<B {
                let off = offsets[i]
                valPacked![i, 0..., off, 0...] = vPacked[i, 0..., 0...]
                valNorms![i, 0..., off] = vNormsArr[i, 0...]
                if rawKeyMode {
                    keys[i, 0..., off, 0...] = newKeys[i, 0..., 0, 0...]
                } else if let kp = kPacked, let kn = kNormsArr {
                    keyPacked![i, 0..., off, 0...] = kp[i, 0..., 0...]
                    keyNorms![i, 0..., off] = kn[i, 0...]
                }
                offsets[i] = off + 1
            }
        }
    }

    /// Encode `[N, D]` vectors via the codec's WHT or dense fused kernel.
    /// Picks WHT when the codec was built with WHT enabled (power-of-2 dim).
    private func encodeVectors(
        _ vectors: MLXArray, codec: MSECodec, bits: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        if codec.useWHT, let signs = codec.whtSigns {
            return TurboQuantKernelOps.fusedEncodeWHT(
                input: vectors, whtSigns: signs, boundaries: codec.boundaries,
                codebook: codec.codebook, bits: bits, dim: headDim)
        } else {
            return TurboQuantKernelOps.fusedEncode(
                input: vectors, rotation: codec.rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: headDim)
        }
    }

    /// Dequant-first SDPA in rotated codec space.
    ///
    /// Math: an orthogonal rotation R applied to both Q and K preserves the
    /// score matrix, since `(RQ)·(RK)ᵀ = QKᵀ`. Applied also to V, the SDPA
    /// output ends up rotated; multiplying by the rotation matrix one last
    /// time recovers the model-native space output. So the SDPA itself runs
    /// in rotated space at no correctness cost — and we never have to write
    /// the dequant'd K/V back to a fp16 cache.
    private func turboAttention(
        queries: MLXArray, scale: Float, mask: MLXArray
    ) -> MLXArray {
        guard let valueCodec, let valPacked, let valNorms else {
            // Defensive — turbo mode without codec/packed buffers is a bug.
            return MLXArray.zeros(queries.shape, dtype: queries.dtype)
        }
        let B = active
        let maxOffset = offsets[0..<B].max() ?? 0

        // Pre-rotate queries with valueCodec.rotationT (folded with scale).
        // Single matmul per layer — same kernel as TurboQuantKVCache uses.
        let qDtype = queries.dtype
        let dt: DType = (qDtype == .bfloat16 || qDtype == .float16) ? qDtype : .bfloat16
        let qScaled = (matmul(queries, valueCodec.rotationT) * scale).asType(dt)

        // Bulk-dequant V into rotated fp16 — kernel is batch-parallel over
        // [B, kvHeads, T, dim], so a single dispatch covers all active slots.
        let vSlicedPacked = valPacked[..<B, 0..., ..<maxOffset, 0...]
        let vSlicedNorms = valNorms[..<B, 0..., ..<maxOffset]
        let vRot = TurboQuantKernelOps.bulkDequantRotated(
            packed: vSlicedPacked, norms: vSlicedNorms,
            codebook: valueCodec.codebook,
            tokenCount: maxOffset, bits: valueBits, dim: headDim, dtype: dt)

        // Keys: raw-mode keeps fp16 keys; turbo-mode dequants from packed.
        let kRot: MLXArray
        if rawKeyMode {
            // Keys are already in model-native space; rotate to match query
            // rotation. (Score preservation requires same rotation on Q and K.)
            let kRaw = keys[..<B, 0..., ..<maxOffset, 0...]
            kRot = matmul(kRaw, valueCodec.rotationT).asType(dt)
        } else if let keyCodec, let keyPacked, let keyNorms {
            // Subtle: K is encoded in keyCodec's rotated space, which differs
            // from valueCodec's rotated space (different seeds). For the
            // score-preservation trick to work we need both Q and K in the
            // SAME rotated frame. Quick fix: dequant K into model-native
            // space (apply keyCodec.rotation to undo), then re-rotate via
            // valueCodec.rotationT to land in V's frame. Two extra matmuls
            // per layer per decode step — not ideal but correctness > perf
            // for v0 of the batched-turbo path. Optimization in a follow-up:
            // share a single rotation matrix across K and V codecs.
            let kSlicedPacked = keyPacked[..<B, 0..., ..<maxOffset, 0...]
            let kSlicedNorms = keyNorms[..<B, 0..., ..<maxOffset]
            let kInKeyRotFrame = TurboQuantKernelOps.bulkDequantRotated(
                packed: kSlicedPacked, norms: kSlicedNorms,
                codebook: keyCodec.codebook,
                tokenCount: maxOffset, bits: keyBits, dim: headDim, dtype: dt)
            // Undo K's rotation, then apply V's rotation.
            let kModelNative = matmul(kInKeyRotFrame, keyCodec.rotation)
            kRot = matmul(kModelNative, valueCodec.rotationT).asType(dt)
        } else {
            return MLXArray.zeros(queries.shape, dtype: queries.dtype)
        }

        // SDPA in rotated space — qScaled already includes scale, pass 1.0.
        let rotOut = MLXFast.scaledDotProductAttention(
            queries: qScaled, keys: kRot, values: vRot,
            scale: 1.0, mask: .array(mask)
        )

        // Inverse rotation back to model-native space.
        var output = matmul(rotOut, valueCodec.rotation)
        if output.dtype != qDtype {
            output = output.asType(qDtype)
        }
        return output
    }
}
