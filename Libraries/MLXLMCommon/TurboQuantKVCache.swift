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

    /// Generate a deterministic random orthogonal rotation matrix (dense, d×d).
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

    /// Generate WHT sign vector: random ±1 per dimension, length d.
    /// Used with Walsh-Hadamard Transform for O(d log d) rotation.
    public static func whtSigns(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        // Random bits → ±1
        // Generate random ±1 signs using uniform random
        let uniform = MLXRandom.uniform(low: 0, high: 1, [dim], key: key)
        let signs = MLX.where(uniform .> MLXArray(Float(0.5)), MLXArray(Float(1.0)), MLXArray(Float(-1.0)))
        eval(signs)
        return signs
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

    /// Whether to use WHT (power-of-2 dim) or dense rotation
    public let useWHT: Bool
    /// WHT sign vector [dim] — for O(d log d) rotation (power-of-2 dims)
    public let whtSigns: MLXArray?
    /// Dense rotation matrix Π [dim, dim] — fallback for non-power-of-2
    public let rotation: MLXArray
    /// Π^T for inverse rotation
    public let rotationT: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.boundaries = TurboQuantCodebook.boundaries(dim: dim, bits: bits)

        // Use WHT for power-of-2 dims (O(d log d)), dense matmul otherwise (O(d²))
        let isPowerOf2 = dim > 0 && (dim & (dim - 1)) == 0
        self.useWHT = isPowerOf2 && dim <= 1024
        if useWHT {
            self.whtSigns = TurboQuantRotation.whtSigns(dim: dim, seed: seed)
            // Still need dense rotation for prepareQueries (matmul path)
            // and for decode (inverse WHT). Store signs as "rotation" for kernel.
            self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotationT = self.rotation.transposed()
        } else {
            self.whtSigns = nil
            self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotationT = self.rotation.transposed()
        }
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
    ///
    /// Note: queries use dense matmul even when encode uses WHT.
    /// This is fine because query rotation runs once per decode step (not per layer).
    /// The encode kernel uses WHT because it runs once per layer × 64 layers.
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

/// KV cache using TurboQuant compression with two-phase architecture:
///
/// **Phase 1 — Prefill** (L>1): Store raw K/V like KVCacheSimple. Zero overhead.
/// **Transition**: On first decode call, compress entire raw cache in one batch.
/// **Phase 2 — Decode** (L=1): Encode 1 new token. Metal kernel scores against
///   all compressed tokens. Zero dequantization.
///
/// Keys: Algorithm 2 (MSE at b-1 bits + QJL residual at 1 bit)
/// Values: Algorithm 1 (MSE at b bits)
public class TurboQuantKVCache: BaseKVCache {

    // Profiling accumulators (static so they accumulate across all layers)
    nonisolated(unsafe) static var profileEncodeMs: Double = 0
    nonisolated(unsafe) static var profileScoreMs: Double = 0
    nonisolated(unsafe) static var profileValueMs: Double = 0
    nonisolated(unsafe) static var profileRotateMs: Double = 0
    nonisolated(unsafe) static var profileOtherMs: Double = 0
    nonisolated(unsafe) static var profileCount: Int = 0

    /// Print and reset profiling stats.
    public static func printProfile() {
        guard profileCount > 0 else { return }
        let total = profileEncodeMs + profileScoreMs + profileValueMs + profileRotateMs + profileOtherMs
        let perToken = total / Double(profileCount)
        print("[TURBO-PROFILE] \(profileCount) decode steps across all layers:")
        print("[TURBO-PROFILE]   encode:  \(String(format: "%.1f", profileEncodeMs))ms (\(String(format: "%.0f", profileEncodeMs / total * 100))%)")
        print("[TURBO-PROFILE]   score:   \(String(format: "%.1f", profileScoreMs))ms (\(String(format: "%.0f", profileScoreMs / total * 100))%)")
        print("[TURBO-PROFILE]   value:   \(String(format: "%.1f", profileValueMs))ms (\(String(format: "%.0f", profileValueMs / total * 100))%)")
        print("[TURBO-PROFILE]   rotate:  \(String(format: "%.1f", profileRotateMs))ms (\(String(format: "%.0f", profileRotateMs / total * 100))%)")
        print("[TURBO-PROFILE]   other:   \(String(format: "%.1f", profileOtherMs))ms (\(String(format: "%.0f", profileOtherMs / total * 100))%)")
        print("[TURBO-PROFILE]   total:   \(String(format: "%.1f", total))ms (\(String(format: "%.2f", perToken))ms/step)")
        profileEncodeMs = 0; profileScoreMs = 0; profileValueMs = 0
        profileRotateMs = 0; profileOtherMs = 0; profileCount = 0
    }

    public let bits: Int
    private let seed: UInt64

    // Codecs (lazy init)
    private var keyMSECodec: MSECodec?   // b bits for keys (MSE-only, no QJL — per Tom Turney's finding)
    private var valueMSECodec: MSECodec? // b bits for values

    // Phase 1: Raw K/V storage (like KVCacheSimple) — used during prefill
    private var rawKeys: MLXArray?       // [B, H, allocSteps, D]
    private var rawValues: MLXArray?     // [B, H, allocSteps, D]
    private var rawAllocSteps = 0

    // Phase 2: Compressed storage — used during decode
    // MSE-only: packed indices + norms (no QJL — simpler, same quality)
    private var keyPackedMSE: MLXArray?
    private var keyNorms: MLXArray?
    private var valPackedMSE: MLXArray?
    private var valNorms: MLXArray?
    private var compressedAllocSteps = 0

    /// Whether we've transitioned from raw → compressed
    public private(set) var isCompressed = false

    private let step = 256

    public init(bits: Int = 4, seed: UInt64 = 42) {
        self.bits = bits
        self.seed = seed
        super.init()
    }

    override public var isTrimmable: Bool { true }

    /// Initialize codecs if needed.
    private func ensureCodecs(headDim: Int) {
        guard keyMSECodec == nil else { return }
        // MSE-only for both keys and values — no QJL (per Tom Turney: same quality, simpler)
        keyMSECodec = MSECodec(dim: headDim, bits: bits, seed: seed)
        valueMSECodec = MSECodec(dim: headDim, bits: bits, seed: seed + 1)
    }

    // MARK: - Phase 1: Raw Prefill

    /// Prefill update: store raw K/V, return raw. Zero encoding overhead.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let H = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        // Ensure raw storage is large enough
        if rawKeys == nil || (prev + numSteps) > rawAllocSteps {
            let newAlloc = ((prev + numSteps + step - 1) / step) * step
            let newK = MLXArray.zeros([B, H, newAlloc, headDim])
            let newV = MLXArray.zeros([B, H, newAlloc, headDim])
            if let existing = rawKeys, prev > 0 {
                newK[0..., 0..., ..<prev, 0...] = existing[0..., 0..., ..<prev, 0...]
                newV[0..., 0..., ..<prev, 0...] = rawValues![0..., 0..., ..<prev, 0...]
            }
            rawKeys = newK
            rawValues = newV
            rawAllocSteps = newAlloc
        }

        offset = prev + numSteps
        rawKeys![0..., 0..., prev..<offset, 0...] = keys
        rawValues![0..., 0..., prev..<offset, 0...] = values

        return (
            rawKeys![0..., 0..., ..<offset, 0...],
            rawValues![0..., 0..., ..<offset, 0...]
        )
    }

    // MARK: - Transition: Compress Raw Cache

    /// Compress the entire raw K/V cache into packed format in one batch.
    /// Called once when transitioning from prefill to decode.
    private func compressRawCache() {
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }

        let allKeys = rk[0..., 0..., ..<offset, 0...]
        let allValues = rv[0..., 0..., ..<offset, 0...]
        let headDim = allKeys.dim(-1)

        ensureCodecs(headDim: headDim)
        guard let keyMSECodec, let valueMSECodec else { return }

        let B = allKeys.dim(0)
        let H = allKeys.dim(1)
        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)

        // Batch encode ALL tokens at once — single dispatch per codec
        let keyMSEState = keyMSECodec.encode(allKeys)
        let valMSEState = valueMSECodec.encode(allValues)

        // Allocate compressed storage
        let allocSteps = ((offset + step - 1) / step) * step
        keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
        keyNorms = MLXArray.zeros([B, H, allocSteps])
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valNorms = MLXArray.zeros([B, H, allocSteps])
        compressedAllocSteps = allocSteps

        // Write batch-compressed data
        keyPackedMSE![0..., 0..., ..<offset, 0...] = keyMSEState.packedIndices
        keyNorms![0..., 0..., ..<offset] = keyMSEState.norms
        valPackedMSE![0..., 0..., ..<offset, 0...] = valMSEState.packedIndices
        valNorms![0..., 0..., ..<offset] = valMSEState.norms

        // Free raw storage
        rawKeys = nil
        rawValues = nil
        rawAllocSteps = 0
        isCompressed = true
    }

    // MARK: - Phase 2: Compressed Decode

    /// Encode a single new token into compressed storage using fused Metal kernel.
    private func encodeNewToken(keys: MLXArray, values: MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let H = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        guard let keyMSECodec, let valueMSECodec else { return }

        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: bits)

        // Fused Metal encode: norm + rotate + quantize + pack in 1 dispatch per codec
        let flatKeys = keys.reshaped([B * H * numSteps, headDim])
        let flatVals = values.reshaped([B * H * numSteps, headDim])

        let (keyPacked, keyNormsNew) = TurboQuantKernelOps.fusedEncode(
            input: flatKeys, rotation: keyMSECodec.rotation,
            boundaries: keyMSECodec.boundaries, bits: bits, dim: headDim
        )
        let (valPacked, valNormsNew) = TurboQuantKernelOps.fusedEncode(
            input: flatVals, rotation: valueMSECodec.rotation,
            boundaries: valueMSECodec.boundaries, bits: bits, dim: headDim
        )

        // Reshape back to [B, H, T, ...]
        let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
        let keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])

        // Grow compressed storage if needed
        if (prev + numSteps) > compressedAllocSteps {
            let newAlloc = ((prev + numSteps + step - 1) / step) * step
            let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
            let newKN = MLXArray.zeros([B, H, newAlloc])
            let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
            let newVN = MLXArray.zeros([B, H, newAlloc])
            if prev > 0 {
                newKP[0..., 0..., ..<prev, 0...] = keyPackedMSE![0..., 0..., ..<prev, 0...]
                newKN[0..., 0..., ..<prev] = keyNorms![0..., 0..., ..<prev]
                newVP[0..., 0..., ..<prev, 0...] = valPackedMSE![0..., 0..., ..<prev, 0...]
                newVN[0..., 0..., ..<prev] = valNorms![0..., 0..., ..<prev]
            }
            keyPackedMSE = newKP; keyNorms = newKN
            valPackedMSE = newVP; valNorms = newVN
            compressedAllocSteps = newAlloc
        }

        offset = prev + numSteps
        keyPackedMSE![0..., 0..., prev..<offset, 0...] = keyPackedShaped
        keyNorms![0..., 0..., prev..<offset] = keyNormsShaped
        valPackedMSE![0..., 0..., prev..<offset, 0...] = valPackedShaped
        valNorms![0..., 0..., prev..<offset] = valNormsShaped
    }

    /// Compressed-domain attention via Metal kernels.
    ///
    /// On first call: compresses raw prefill cache in one batch.
    /// Then: encode 1 new token → Metal score kernel → softmax → Metal value kernel.
    /// Zero dequantization at any point.
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

        // Transition: compress raw cache on first decode call
        if !isCompressed {
            print("[TURBO-DEBUG] compressRawCache() at offset=\(offset)")
            compressRawCache()
        }

        let profiling = ProcessInfo.processInfo.environment["SAM_TURBO_PROFILE"] == "1"
        var t0 = Date()

        // Phase A: Encode new token
        encodeNewToken(keys: newKeys, values: newValues)
        if profiling { eval(keyPackedMSE!, valPackedMSE!); let t1 = Date(); Self.profileEncodeMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

        guard let keyMSECodec, let valueMSECodec else {
            return queries
        }

        let tokenCount = offset
        let keyBits = bits

        // Phase B: Pre-rotate query + Metal score kernel
        let qRot = keyMSECodec.prepareQueries(queries) * MLXArray(scale)
        let flatQ = qRot.reshaped([B * nQHeads * L, headDim])
        let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatKeyNorms = keyNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        var scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: flatQ, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: keyMSECodec.codebook, tokenCount: tokenCount,
            repeatCount: nRepeats, bits: keyBits, dim: headDim
        ).reshaped([B, nQHeads, L, tokenCount])
        if profiling { eval(scores); let t1 = Date(); Self.profileScoreMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

        // Mask + softmax
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
        if profiling { eval(attnWeights); let t1 = Date(); Self.profileOtherMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

        // Phase C: Metal value kernel
        let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatValNorms = valNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
            codebook: valueMSECodec.codebook, tokenCount: tokenCount,
            repeatCount: nRepeats, bits: bits, dim: headDim
        )
        if profiling { eval(rotatedOutput); let t1 = Date(); Self.profileValueMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

        // Phase D: Inverse rotation
        let output = matmul(
            rotatedOutput.reshaped([B, nQHeads, L, headDim]),
            valueMSECodec.rotation
        )
        if profiling { eval(output); let t1 = Date(); Self.profileRotateMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

        Self.profileCount += 1
        return output
    }

    // MARK: - State / Trim

    override public var state: [MLXArray] {
        get {
            if isCompressed {
                guard let kpm = keyPackedMSE, let kn = keyNorms,
                      let vpm = valPackedMSE, let vn = valNorms,
                      offset > 0 else { return [] }
                return [
                    kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset],
                    vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                ]
            } else {
                guard let rk = rawKeys, let rv = rawValues, offset > 0 else { return [] }
                return [rk[0..., 0..., ..<offset, 0...], rv[0..., 0..., ..<offset, 0...]]
            }
        }
        set {
            if newValue.count == 4 {
                // Compressed state: [keyPacked, keyNorms, valPacked, valNorms]
                keyPackedMSE = newValue[0]; keyNorms = newValue[1]
                valPackedMSE = newValue[2]; valNorms = newValue[3]
                offset = newValue[0].dim(2)
                compressedAllocSteps = offset
                isCompressed = true
            } else if newValue.count == 2 {
                // Raw state
                rawKeys = newValue[0]; rawValues = newValue[1]
                offset = newValue[0].dim(2)
                rawAllocSteps = offset
                isCompressed = false
            }
        }
    }

    @discardableResult
    override public func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }
        let trimCount = min(n, offset)
        offset -= trimCount
        if offset == 0 {
            rawKeys = nil; rawValues = nil; rawAllocSteps = 0
            keyPackedMSE = nil; keyNorms = nil
            valPackedMSE = nil; valNorms = nil
            compressedAllocSteps = 0; isCompressed = false
        }
        return trimCount
    }
}
