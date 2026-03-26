// Copyright © 2026 Eric Kryski. TurboQuant KV cache compression.
//
// Implements TurboQuant (arXiv:2504.19874) for KV cache compression:
//   Algorithm 1 (MSE-optimal): rotation Π + scalar codebook quantization
//   Algorithm 2 (inner-product optimal): MSE at b-1 bits + QJL residual at 1 bit
//
// Keys use Algorithm 2 (unbiased inner products for Q·K attention scoring).
// Values use Algorithm 1 (MSE-only, best reconstruction for weighted sum).
//
// References:
//   - TurboQuant: https://arxiv.org/abs/2504.19874
//   - QJL: https://arxiv.org/abs/2406.03482
//   - PolarQuant: https://arxiv.org/abs/2502.02617

import Foundation
import MLX
import MLXNN

// MARK: - Codebook Generation

/// Optimal Lloyd-Max codebook centroids for Beta-distributed coordinates.
///
/// After random orthogonal rotation, each coordinate of a unit-sphere vector
/// follows Beta distribution f_X(x) ∝ (1-x²)^((d-3)/2) on [-1,1].
/// For large d, this converges to N(0, 1/d).
public enum TurboQuantCodebook {

    /// Pre-computed codebook centroids for common (dim, bits) pairs.
    /// Centroids are sorted ascending in [-1, 1].
    public static func codebook(dim: Int, bits: Int) -> MLXArray {
        let centroids = generateCentroids(dim: dim, bits: bits)
        return MLXArray(centroids)
    }

    /// Pre-computed codebook boundaries (midpoints between adjacent centroids).
    /// Used for fast quantization via comparison instead of argmin.
    public static func boundaries(dim: Int, bits: Int) -> MLXArray {
        let centroids = generateCentroids(dim: dim, bits: bits)
        var bounds = [Float]()
        for i in 0 ..< centroids.count - 1 {
            bounds.append((centroids[i] + centroids[i + 1]) / 2.0)
        }
        return MLXArray(bounds)
    }

    /// Generate codebook centroids via weighted k-means on Beta distribution.
    static func generateCentroids(dim: Int, bits: Int) -> [Float] {
        let levels = 1 << bits
        let gridSize = 32768
        let sigma = 1.0 / sqrt(Float(dim))

        // Generate grid points and PDF weights
        var grid = [Float](repeating: 0, count: gridSize)
        var weights = [Float](repeating: 0, count: gridSize)
        for i in 0 ..< gridSize {
            let x = -1.0 + 2.0 * Float(i) / Float(gridSize - 1)
            grid[i] = x
            // Beta PDF ∝ (1 - x²)^((d-3)/2), approximated by Gaussian for large d
            let exponent = Float(dim - 3) / 2.0
            let w = pow(max(1.0 - x * x, 1e-30), exponent)
            weights[i] = w
        }

        // Initialize centroids via quantiles
        let totalW = weights.reduce(0, +)
        var centroids = [Float](repeating: 0, count: levels)
        var cumW: Float = 0
        var ci = 0
        for i in 0 ..< gridSize {
            cumW += weights[i]
            let target = (Float(ci) + 0.5) / Float(levels) * totalW
            if cumW >= target && ci < levels {
                centroids[ci] = grid[i]
                ci += 1
            }
        }
        // Fill remaining
        while ci < levels {
            centroids[ci] = centroids[ci - 1] + sigma
            ci += 1
        }

        // K-means iterations
        for _ in 0 ..< 100 {
            var sums = [Float](repeating: 0, count: levels)
            var counts = [Float](repeating: 0, count: levels)
            for i in 0 ..< gridSize {
                var bestJ = 0
                var bestDist = Float.infinity
                for j in 0 ..< levels {
                    let d = abs(grid[i] - centroids[j])
                    if d < bestDist { bestDist = d; bestJ = j }
                }
                sums[bestJ] += grid[i] * weights[i]
                counts[bestJ] += weights[i]
            }
            for j in 0 ..< levels {
                if counts[j] > 0 { centroids[j] = sums[j] / counts[j] }
            }
        }

        return centroids.sorted()
    }
}

// MARK: - Rotation Matrix

/// Random orthogonal rotation matrix generation.
///
/// TurboQuant Algorithm 1 line 2: Π ∈ ℝ^(d×d) via QR decomposition
/// on random Gaussian matrix. Sign-corrected for determinism.
public enum TurboQuantRotation {

    /// Generate a deterministic random orthogonal rotation matrix.
    /// Uses QR decomposition on CPU (not yet GPU-supported in MLX).
    public static func rotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let gaussian = MLXRandom.normal([dim, dim], key: key)

        // QR on CPU (MLX GPU QR not supported yet)
        let (q, r) = MLXLinalg.qr(gaussian, stream: .cpu)
        let diagR = r.diagonal(stream: .cpu)
        let signs = sign(diagR, stream: .cpu)
        let result = q * expandedDimensions(signs, axis: 0)
        eval(result)
        return result
    }
}

// MARK: - Bit Packing

/// Efficient bit packing/unpacking for codebook indices.
public enum TurboQuantPacking {

    /// Number of uint32 words needed to pack `count` values at `bits` each.
    public static func packedWidth(count: Int, bits: Int) -> Int {
        (count * bits + 31) / 32
    }

    /// Pack b-bit indices into uint32 words.
    /// Input: [rows, count] as uint32 (values 0..2^bits-1)
    /// Output: [rows, packedWidth] as uint32
    public static func packLowBit(_ indices: MLXArray, bits: Int) -> MLXArray {
        let count = indices.dim(-1)
        let batchShape = Array(indices.shape.dropLast())
        let rows = batchShape.reduce(1, *)
        let flat = indices.reshaped([rows, count])
        let pw = packedWidth(count: count, bits: bits)
        let mask = UInt32((1 << bits) - 1)

        var wordArrays = [MLXArray]()
        for w in 0 ..< pw {
            var word = MLXArray.zeros([rows], dtype: .uint32)
            for d in 0 ..< count {
                let bitOffset = d * bits
                let wordIdx = bitOffset / 32
                let offset = bitOffset % 32
                let spill = offset + bits - 32

                if wordIdx == w {
                    let shifted = (flat[0..., d].asType(.uint32) & MLXArray(mask)) << MLXArray(UInt32(offset))
                    word = word | shifted
                }
                if spill > 0 && wordIdx + 1 == w {
                    let shifted = (flat[0..., d].asType(.uint32) & MLXArray(mask)) >> MLXArray(UInt32(bits - spill))
                    word = word | shifted
                }
            }
            wordArrays.append(expandedDimensions(word, axis: -1))
        }
        let packed = concatenated(wordArrays, axis: -1)  // [rows, pw]
        return packed.reshaped(batchShape + [pw])
    }

    /// Unpack b-bit indices from uint32 words.
    /// Input: [rows, packedWidth] as uint32
    /// Output: [rows, count] as uint32
    public static func unpackLowBit(_ packed: MLXArray, bits: Int, count: Int) -> MLXArray {
        let shape = packed.shape
        let batchShape = Array(shape.dropLast())
        let rows = batchShape.reduce(1, *)
        let flat = packed.reshaped([rows, -1])
        let mask = UInt32((1 << bits) - 1)

        var dimArrays = [MLXArray]()
        for d in 0 ..< count {
            let bitOffset = d * bits
            let wordIdx = bitOffset / 32
            let offset = bitOffset % 32
            let spill = offset + bits - 32

            var value = (flat[0..., wordIdx] >> MLXArray(UInt32(offset))) & MLXArray(mask)
            if spill > 0 {
                let high = (flat[0..., wordIdx + 1] << MLXArray(UInt32(bits - spill))) & MLXArray(mask)
                value = value | high
            }
            dimArrays.append(expandedDimensions(value, axis: -1))
        }
        let unpacked = concatenated(dimArrays, axis: -1)  // [rows, count]
        return unpacked.reshaped(batchShape + [count])
    }
}

// MARK: - MSE Codec (TurboQuant Algorithm 1)

/// State for MSE-quantized vectors.
public struct MSECodecState {
    public var norms: MLXArray       // [B, H, T] — original vector L2 norms
    public var packedIndices: MLXArray // [B, H, T, PackedWidth] — packed codebook indices
    public var tokenCount: Int
    public let dim: Int
    public let bits: Int
}

/// MSE-optimal codec per TurboQuant Algorithm 1.
///
/// QUANT: y ← Π·x, idx_j ← argmin|y_j - c_k|
/// DEQUANT: ỹ_j ← c_{idx_j}, x̃ ← Π^T · ỹ
public class MSECodec {
    public let dim: Int
    public let bits: Int
    public let seed: UInt64

    /// Codebook centroids [2^bits]
    public let codebook: MLXArray
    /// Codebook boundaries for fast quantization [2^bits - 1]
    public let boundaries: MLXArray
    /// Rotation matrix Π [dim, dim]
    public let rotation: MLXArray
    /// Π^T for inverse rotation
    public let rotationT: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.boundaries = TurboQuantCodebook.boundaries(dim: dim, bits: bits)
        self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
        self.rotationT = self.rotation.transposed()
    }

    /// Encode vectors (Algorithm 1 QUANT).
    /// Input: [B, H, T, D]
    /// Returns MSECodecState with norms and packed indices.
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        // Extract norms and normalize (paper assumes unit sphere; we store norms separately)
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // Rotate: y ← Π · x (Algorithm 1 line 5)
        let rotated = matmul(unit, rotationT)  // [B,H,T,D] @ [D,D] = [B,H,T,D]

        // Quantize via boundary comparison (fast, no broadcast)
        // searchsorted finds which bin each coordinate falls into
        let indices = boundaryQuantize(rotated)

        // Pack indices
        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        return MSECodecState(
            norms: norms,
            packedIndices: packed,
            tokenCount: vectors.dim(2),
            dim: dim,
            bits: bits
        )
    }

    /// Decode from state (Algorithm 1 DEQUANT).
    /// Returns: [B, H, T, D]
    public func decode(_ state: MSECodecState) -> MLXArray {
        // Unpack indices
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)

        // Codebook lookup: ỹ_j ← c_{idx_j} (Algorithm 1 line 9)
        let approx = codebook[indices]

        // Inverse rotate: x̃ ← Π^T · ỹ (Algorithm 1 line 10)
        let unrotated = matmul(approx, rotation)  // Π^T · ỹ = ỹ @ Π (since Π orthogonal)

        // Rescale by stored norms
        return expandedDimensions(state.norms, axis: -1) * unrotated
    }

    /// Pre-rotate queries for compressed-domain scoring.
    /// q' ← Π · q (once per query, reused for all cached keys)
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        return matmul(queries, rotationT)
    }

    /// Fast quantization via boundary comparison instead of argmin broadcast.
    /// boundaries = sorted midpoints between adjacent centroids.
    /// Returns uint32 indices in [0, 2^bits - 1].
    func boundaryQuantize(_ rotated: MLXArray) -> MLXArray {
        // For each coordinate, count how many boundaries it exceeds
        // This gives the codebook index directly
        let ndim = rotated.ndim
        let expanded = expandedDimensions(rotated, axis: -1)  // [..., D, 1]
        // Reshape boundaries to broadcast: [1, 1, ..., 1, numBoundaries]
        var bShape = [Int](repeating: 1, count: ndim + 1)
        bShape[ndim] = boundaries.count
        let b = boundaries.reshaped(bShape)
        let greater = (expanded .> b).asType(.uint32)         // compare against all boundaries
        let indices = greater.sum(axis: -1)                   // count exceeded = index
        return indices.asType(.uint32)
    }
}

// MARK: - TurboQuantKVCache

/// KV cache using TurboQuant compression.
///
/// Keys: Algorithm 2 (MSE at b-1 bits + QJL residual at 1 bit)
/// Values: Algorithm 1 (MSE at b bits)
///
/// During decode, attention reads packed data directly via Metal kernel.
/// During prefill, dequantizes for standard SDPA compatibility.
public class TurboQuantKVCache: BaseKVCache {

    /// Total bits per coordinate (e.g., 4 for "turbo4")
    public let bits: Int
    private let seed: UInt64

    // Codecs (lazy init on first update — need head dim)
    private var keyMSECodec: MSECodec?     // b-1 bits for keys
    private var valueMSECodec: MSECodec?   // b bits for values

    // QJL projection matrix for keys (lazy init)
    private var qjlProjection: MLXArray?   // [D, D] random Gaussian, orthogonalized
    private var qjlProjectionT: MLXArray?

    // Compressed key storage: MSE indices + QJL signs + norms
    private var keyPackedMSE: MLXArray?     // [B, H, allocSteps, KeyPackedWidth] uint32
    private var keyPackedQJL: MLXArray?     // [B, H, allocSteps, QJLPackedWidth] uint32
    private var keyNorms: MLXArray?         // [B, H, allocSteps] float32
    private var keyResidualNorms: MLXArray? // [B, H, allocSteps] float32

    // Compressed value storage: MSE indices + norms
    private var valPackedMSE: MLXArray?     // [B, H, allocSteps, ValPackedWidth] uint32
    private var valNorms: MLXArray?         // [B, H, allocSteps] float32

    private let step = 256
    private var allocatedSteps = 0

    public init(bits: Int = 4, seed: UInt64 = 42) {
        self.bits = bits
        self.seed = seed
        super.init()
    }

    override public var isTrimmable: Bool { true }

    /// Encode new K/V tokens into compressed storage.
    ///
    /// Keys: Algorithm 2 — MSE at (b-1) bits + QJL residual at 1 bit
    /// Values: Algorithm 1 — MSE at b bits
    ///
    /// Returns dequantized K/V for prefill SDPA. During decode (L=1),
    /// callers should use compressedAttention() instead.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        // Lazy codec init
        if keyMSECodec == nil {
            let keyBits = max(bits - 1, 1)  // Algorithm 2: b-1 bits for MSE stage
            keyMSECodec = MSECodec(dim: headDim, bits: keyBits, seed: seed)
            valueMSECodec = MSECodec(dim: headDim, bits: bits, seed: seed + 1)

            // QJL projection matrix (orthogonalized Gaussian, per QJL paper Section 4.1)
            let projSeed = seed + UInt64(headDim) * 2971 + 17
            qjlProjection = TurboQuantRotation.rotationMatrix(dim: headDim, seed: projSeed)
            qjlProjectionT = qjlProjection!.transposed()
        }
        guard let keyMSECodec, let valueMSECodec else {
            return (keys, values)
        }

        let keyBits = max(bits - 1, 1)
        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
        let qpw = TurboQuantPacking.packedWidth(count: headDim, bits: 1)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)

        // === Key Encode (Algorithm 2) ===
        // Step 1: MSE encode at b-1 bits
        let keyMSEState = keyMSECodec.encode(keys)

        // Step 2: Compute residual r = x - DEQUANT_mse(idx) (Algorithm 2 line 6)
        let keyMSERecon = keyMSECodec.decode(keyMSEState)
        let residual = keys - keyMSERecon

        // Step 3: QJL on residual — sign(S · r) (Algorithm 2 line 7)
        let residualNorms = sqrt((residual * residual).sum(axis: -1))
        let projected = matmul(residual, qjlProjectionT!)  // S · r
        let signs = (projected .>= MLXArray(Float(0.0))).asType(.uint32)
        let packedQJL = TurboQuantPacking.packLowBit(signs, bits: 1)

        // === Value Encode (Algorithm 1) ===
        let valMSEState = valueMSECodec.encode(values)

        // === Store compressed data ===
        if keyPackedMSE == nil || (prev + numSteps) > allocatedSteps {
            let newAlloc = ((prev + numSteps + step - 1) / step) * step

            let newKP = MLXArray.zeros([B, nKVHeads, newAlloc, kpw], dtype: .uint32)
            let newKQ = MLXArray.zeros([B, nKVHeads, newAlloc, qpw], dtype: .uint32)
            let newKN = MLXArray.zeros([B, nKVHeads, newAlloc])
            let newKRN = MLXArray.zeros([B, nKVHeads, newAlloc])
            let newVP = MLXArray.zeros([B, nKVHeads, newAlloc, vpw], dtype: .uint32)
            let newVN = MLXArray.zeros([B, nKVHeads, newAlloc])

            if let existing = keyPackedMSE, prev > 0 {
                newKP[0..., 0..., ..<prev, 0...] = existing[0..., 0..., ..<prev, 0...]
                newKQ[0..., 0..., ..<prev, 0...] = keyPackedQJL![0..., 0..., ..<prev, 0...]
                newKN[0..., 0..., ..<prev] = keyNorms![0..., 0..., ..<prev]
                newKRN[0..., 0..., ..<prev] = keyResidualNorms![0..., 0..., ..<prev]
                newVP[0..., 0..., ..<prev, 0...] = valPackedMSE![0..., 0..., ..<prev, 0...]
                newVN[0..., 0..., ..<prev] = valNorms![0..., 0..., ..<prev]
            }

            keyPackedMSE = newKP
            keyPackedQJL = newKQ
            keyNorms = newKN
            keyResidualNorms = newKRN
            valPackedMSE = newVP
            valNorms = newVN
            allocatedSteps = newAlloc
        }

        offset = prev + numSteps
        keyPackedMSE![0..., 0..., prev..<offset, 0...] = keyMSEState.packedIndices
        keyPackedQJL![0..., 0..., prev..<offset, 0...] = packedQJL
        keyNorms![0..., 0..., prev..<offset] = keyMSEState.norms
        keyResidualNorms![0..., 0..., prev..<offset] = residualNorms
        valPackedMSE![0..., 0..., prev..<offset, 0...] = valMSEState.packedIndices
        valNorms![0..., 0..., prev..<offset] = valMSEState.norms

        // Return dequantized ALL cached tokens for prefill SDPA
        let allKeyState = MSECodecState(
            norms: keyNorms![0..., 0..., ..<offset],
            packedIndices: keyPackedMSE![0..., 0..., ..<offset, 0...],
            tokenCount: offset, dim: headDim, bits: keyBits
        )
        let allValState = MSECodecState(
            norms: valNorms![0..., 0..., ..<offset],
            packedIndices: valPackedMSE![0..., 0..., ..<offset, 0...],
            tokenCount: offset, dim: headDim, bits: bits
        )
        let fullKeys = keyMSECodec.decode(allKeyState)
        let fullValues = valueMSECodec.decode(allValState)

        return (fullKeys, fullValues)
    }

    // MARK: - State / Trim

    override public var state: [MLXArray] {
        get {
            guard let kpm = keyPackedMSE, let kpq = keyPackedQJL,
                  let kn = keyNorms, let krn = keyResidualNorms,
                  let vpm = valPackedMSE, let vn = valNorms,
                  offset > 0 else { return [] }
            return [
                kpm[0..., 0..., ..<offset, 0...],
                kpq[0..., 0..., ..<offset, 0...],
                kn[0..., 0..., ..<offset],
                krn[0..., 0..., ..<offset],
                vpm[0..., 0..., ..<offset, 0...],
                vn[0..., 0..., ..<offset],
            ]
        }
        set {
            guard newValue.count == 6 else { return }
            keyPackedMSE = newValue[0]
            keyPackedQJL = newValue[1]
            keyNorms = newValue[2]
            keyResidualNorms = newValue[3]
            valPackedMSE = newValue[4]
            valNorms = newValue[5]
            offset = newValue[0].dim(2)
            allocatedSteps = offset
        }
    }

    /// Encode ONLY new token into compressed storage (no dequant of full cache).
    /// Returns the number of tokens now in cache.
    public func encodeOnly(keys: MLXArray, values: MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        if keyMSECodec == nil {
            let keyBits = max(bits - 1, 1)
            keyMSECodec = MSECodec(dim: headDim, bits: keyBits, seed: seed)
            valueMSECodec = MSECodec(dim: headDim, bits: bits, seed: seed + 1)
            let projSeed = seed + UInt64(headDim) * 2971 + 17
            qjlProjection = TurboQuantRotation.rotationMatrix(dim: headDim, seed: projSeed)
            qjlProjectionT = qjlProjection!.transposed()
        }
        guard let keyMSECodec, let valueMSECodec else { return }

        let keyBits = max(bits - 1, 1)
        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
        let qpw = TurboQuantPacking.packedWidth(count: headDim, bits: 1)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)

        let keyMSEState = keyMSECodec.encode(keys)
        let keyMSERecon = keyMSECodec.decode(keyMSEState)
        let residual = keys - keyMSERecon
        let residualNorms = sqrt((residual * residual).sum(axis: -1))
        let projected = matmul(residual, qjlProjectionT!)
        let signs = (projected .>= MLXArray(Float(0.0))).asType(.uint32)
        let packedQJL = TurboQuantPacking.packLowBit(signs, bits: 1)
        let valMSEState = valueMSECodec.encode(values)

        if keyPackedMSE == nil || (prev + numSteps) > allocatedSteps {
            let newAlloc = ((prev + numSteps + step - 1) / step) * step
            let newKP = MLXArray.zeros([B, nKVHeads, newAlloc, kpw], dtype: .uint32)
            let newKQ = MLXArray.zeros([B, nKVHeads, newAlloc, qpw], dtype: .uint32)
            let newKN = MLXArray.zeros([B, nKVHeads, newAlloc])
            let newKRN = MLXArray.zeros([B, nKVHeads, newAlloc])
            let newVP = MLXArray.zeros([B, nKVHeads, newAlloc, vpw], dtype: .uint32)
            let newVN = MLXArray.zeros([B, nKVHeads, newAlloc])
            if let existing = keyPackedMSE, prev > 0 {
                newKP[0..., 0..., ..<prev, 0...] = existing[0..., 0..., ..<prev, 0...]
                newKQ[0..., 0..., ..<prev, 0...] = keyPackedQJL![0..., 0..., ..<prev, 0...]
                newKN[0..., 0..., ..<prev] = keyNorms![0..., 0..., ..<prev]
                newKRN[0..., 0..., ..<prev] = keyResidualNorms![0..., 0..., ..<prev]
                newVP[0..., 0..., ..<prev, 0...] = valPackedMSE![0..., 0..., ..<prev, 0...]
                newVN[0..., 0..., ..<prev] = valNorms![0..., 0..., ..<prev]
            }
            keyPackedMSE = newKP; keyPackedQJL = newKQ; keyNorms = newKN
            keyResidualNorms = newKRN; valPackedMSE = newVP; valNorms = newVN
            allocatedSteps = newAlloc
        }

        offset = prev + numSteps
        keyPackedMSE![0..., 0..., prev..<offset, 0...] = keyMSEState.packedIndices
        keyPackedQJL![0..., 0..., prev..<offset, 0...] = packedQJL
        keyNorms![0..., 0..., prev..<offset] = keyMSEState.norms
        keyResidualNorms![0..., 0..., prev..<offset] = residualNorms
        valPackedMSE![0..., 0..., prev..<offset, 0...] = valMSEState.packedIndices
        valNorms![0..., 0..., prev..<offset] = valMSEState.norms
    }

    /// Compressed-domain attention via Metal kernels.
    ///
    /// 1. Encode new token (only the new 1 token, not all cached)
    /// 2. Pre-rotate query
    /// 3. Metal score kernel on ALL compressed tokens
    /// 4. Softmax
    /// 5. Metal value kernel + inverse rotation
    ///
    /// Key: NO dequantization of full cache. Only the encode of 1 new token + Metal scoring.
    public func compressedAttention(
        queries: MLXArray,
        keys newKeys: MLXArray,
        values newValues: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let headDim = newKeys.dim(-1)
        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let nKVHeads = newKeys.dim(1)
        let L = queries.dim(2)
        let nRepeats = nQHeads / nKVHeads

        // Step 1: Encode ONLY the new token(s) — no full cache dequant!
        encodeOnly(keys: newKeys, values: newValues)

        guard let keyMSECodec, let valueMSECodec else {
            return queries  // fallback
        }

        let tokenCount = offset
        let keyBits = max(bits - 1, 1)

        // Step 2: Pre-rotate query
        let qRot = keyMSECodec.prepareQueries(queries) * MLXArray(scale)

        // Step 3: Metal score kernel
        let flatQ = qRot.reshaped([B * nQHeads * L, headDim])
        let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatKeyNorms = keyNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        var scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: flatQ,
            packed: flatKeyPacked,
            norms: flatKeyNorms,
            codebook: keyMSECodec.codebook,
            tokenCount: tokenCount,
            repeatCount: nRepeats,
            bits: keyBits,
            dim: headDim
        ).reshaped([B, nQHeads, L, tokenCount])

        // Step 4: Mask + softmax
        switch mask {
        case .causal:
            let (qL, kL) = (scores.dim(-2), scores.dim(-1))
            let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
            let kIndices = MLXArray(0 ..< kL)
            let causalMask = greaterEqual(
                expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
            scores = MLX.where(causalMask, scores, MLXArray(Float.leastNormalMagnitude))
        case .array(let maskArray):
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
            } else { scores = scores + maskArray }
        case .none: break
        default: break
        }

        let attnWeights = softmax(scores, axis: -1)

        // Step 5: Metal value kernel
        let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatValNorms = valNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: flatWeights,
            packed: flatValPacked,
            norms: flatValNorms,
            codebook: valueMSECodec.codebook,
            tokenCount: tokenCount,
            repeatCount: nRepeats,
            bits: bits,
            dim: headDim
        )

        // Step 6: Inverse rotation
        return matmul(
            rotatedOutput.reshaped([B, nQHeads, L, headDim]),
            valueMSECodec.rotation
        )
    }

    @discardableResult
    override public func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let trimCount = min(n, offset)
        offset -= trimCount
        if offset == 0 {
            keyPackedMSE = nil
            keyPackedQJL = nil
            keyNorms = nil
            keyResidualNorms = nil
            valPackedMSE = nil
            valNorms = nil
            allocatedSteps = 0
        }
        return trimCount
    }
}
