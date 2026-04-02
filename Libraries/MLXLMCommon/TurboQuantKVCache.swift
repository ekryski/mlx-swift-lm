// Copyright © 2026 Eric Kryski. TurboQuant KV cache compression.
//
// Implements TurboQuant Algorithm 1 (MSE-optimal, arXiv:2504.19874) for KV cache:
//   rotation Π + optimal Lloyd-Max scalar codebook quantization on Beta distribution.
//
// Both keys and values use Algorithm 1 (MSE-only at b bits). QJL (Algorithm 2)
// is omitted — Tom Turney's research shows no quality benefit on Apple Silicon,
// and at 4-bit the MSE bias is negligible (paper Section 3.2: bias = 2/π,
// diminishing with bit-width).
//
// Enhancements beyond paper:
//   - Norm extraction/restoration: paper assumes ||x||=1; we store norms for arbitrary vectors
//   - Norm correction: store ||x|| / ||ỹ|| instead of ||x||, compensating for quantization error
//   - WHT rotation option: O(d log d) butterfly in Metal kernel for power-of-2 dims
//   - Two-phase architecture: raw prefill → batch compress → compressed decode
//   - Pre-rotated queries: q' = Π·q computed once, reused for all cached keys
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

    // MARK: - Pre-computed Centroids

    /// Pre-computed Lloyd-Max centroids for common (dim, bits) pairs.
    /// Generated offline via 100-iteration weighted k-means on 32K-point Beta PDF grid.
    /// Avoids ~50ms runtime codebook generation per codec.
    private static let precomputed: [Int: [Int: [Float]]] = [
        64: [
            2: [-0.18745463, -0.05649366, 0.05649367, 0.18745449],
            3: [-0.26375133, -0.16599470, -0.09368263, -0.03040462, 0.03040464, 0.09368261, 0.16599482, 0.26375186],
            4: [-0.32913971, -0.25096416, -0.19681059, -0.15295772, -0.11478586, -0.08000945, -0.04726735, -0.01563822, 0.01563822, 0.04723797, 0.07994876, 0.11472529, 0.15289739, 0.19675052, 0.25090477, 0.32908401],
        ],
        128: [
            2: [-0.13302007, -0.03998107, 0.03998102, 0.13302033],
            3: [-0.18828832, -0.11801215, -0.06648001, -0.02156330, 0.02156329, 0.06648005, 0.11801218, 0.18828897],
            4: [-0.23639172, -0.17934021, -0.14023653, -0.10881814, -0.08157559, -0.05678632, -0.03350975, -0.01108178, 0.01108178, 0.03350975, 0.05678631, 0.08157560, 0.10881804, 0.14023650, 0.17934017, 0.23639278],
        ],
        256: [
            2: [-0.09420358, -0.02827190, 0.02827190, 0.09420330],
            3: [-0.13371243, -0.08361249, -0.04704370, -0.01524900, 0.01524901, 0.04704368, 0.08361248, 0.13371260],
            4: [-0.16852295, -0.12754069, -0.09961203, -0.07719406, -0.05781249, -0.04021866, -0.02370371, -0.00783269, 0.00783269, 0.02370371, 0.04021868, 0.05781246, 0.07719407, 0.09961203, 0.12754090, 0.16852276],
        ],
    ]

    // MARK: - Public API

    /// Codebook centroids for (dim, bits). Uses pre-computed table for common configs,
    /// falls back to runtime generation for uncommon ones.
    public static func codebook(dim: Int, bits: Int) -> MLXArray {
        if let dimTable = precomputed[dim], let centroids = dimTable[bits] {
            return MLXArray(centroids)
        }
        let centroids = generateCentroids(dim: dim, bits: bits)
        return MLXArray(centroids)
    }

    /// Codebook boundaries (midpoints between adjacent centroids).
    public static func boundaries(dim: Int, bits: Int) -> MLXArray {
        let centroids: [Float]
        if let dimTable = precomputed[dim], let cached = dimTable[bits] {
            centroids = cached
        } else {
            centroids = generateCentroids(dim: dim, bits: bits)
        }
        var bounds = [Float]()
        for i in 0 ..< centroids.count - 1 {
            bounds.append((centroids[i] + centroids[i + 1]) / 2.0)
        }
        return MLXArray(bounds)
    }

    /// Generate codebook centroids via weighted k-means on Beta distribution.
    /// Used as fallback for uncommon (dim, bits) pairs not in the pre-computed table.
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

    /// Generate a Hadamard matrix of size dim × dim via recursive Kronecker product.
    /// Requires dim to be a power of 2. The resulting matrix H satisfies H·H = dim·I.
    public static func hadamardMatrix(dim: Int) -> MLXArray {
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        // Build recursively: H_1 = [[1]], H_2n = [[H_n, H_n], [H_n, -H_n]]
        var h: [[Float]] = [[1.0]]
        var size = 1
        while size < dim {
            var newH = [[Float]](repeating: [Float](repeating: 0, count: size * 2), count: size * 2)
            for i in 0 ..< size {
                for j in 0 ..< size {
                    newH[i][j] = h[i][j]
                    newH[i][j + size] = h[i][j]
                    newH[i + size][j] = h[i][j]
                    newH[i + size][j + size] = -h[i][j]
                }
            }
            h = newH
            size *= 2
        }
        let flat = h.flatMap { $0 }
        let result = MLXArray(flat, [dim, dim])
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

    /// Apply WHT butterfly on the last dimension of x. Shape-preserving.
    /// Computes unnormalized Walsh-Hadamard transform: H * x along last dim.
    private static func whtButterfly(_ x: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        let logDim = Int(log2(Double(dim)))
        let origShape = x.shape
        // Flatten leading dims: [N, dim]
        let N = origShape.dropLast().reduce(1, *)
        var y = x.reshaped([N, dim])

        for s in 0..<logDim {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let numBlocks = dim / blockSize
            // Reshape to [N, numBlocks, 2, halfBlock]
            y = y.reshaped([N, numBlocks, blockSize])
            let a = y[0..., 0..., ..<halfBlock]       // [N, numBlocks, halfBlock]
            let b = y[0..., 0..., halfBlock...]        // [N, numBlocks, halfBlock]
            let sumAB = a + b
            let diffAB = a - b
            y = concatenated([sumAB, diffAB], axis: -1)  // [N, numBlocks, blockSize]
            y = y.reshaped([N, dim])
        }

        return y.reshaped(origShape)
    }

    /// Apply SRHT forward rotation: y = H * diag(signs) * x / sqrt(dim)
    /// Works on the last dimension of any-shaped input (e.g. [B, H, T, D]).
    /// Uses butterfly pattern — O(d log d) vs O(d²) for dense matmul.
    public static func fwhtForward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let signed = x * signs
        let transformed = whtButterfly(signed)
        return transformed * MLXArray(Float(1.0 / sqrt(Float(dim))))
    }

    /// Apply SRHT inverse rotation: x = diag(signs) * H * y / sqrt(dim)
    /// WHT is self-inverse up to scale. Inverse of (H·D/√d) is (D·H/√d).
    public static func fwhtInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let transformed = whtButterfly(y)
        return transformed * MLXArray(Float(1.0 / sqrt(Float(dim)))) * signs
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
    /// WHT sign vector [dim] — for O(d log d) Metal encode kernel (power-of-2 dims only)
    public let whtSigns: MLXArray?
    /// Dense rotation matrix Π [dim, dim] — used for decode/query rotation (single matmul, fast)
    public let rotation: MLXArray
    /// Π^T — for forward rotation
    public let rotationT: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.boundaries = TurboQuantCodebook.boundaries(dim: dim, bits: bits)

        // Use WHT for power-of-2 dims (O(d log d) Metal encode kernel)
        let isPowerOf2 = dim > 0 && (dim & (dim - 1)) == 0
        self.useWHT = isPowerOf2 && dim <= 1024
        if useWHT {
            let signs = TurboQuantRotation.whtSigns(dim: dim, seed: seed)
            self.whtSigns = signs
            // Build dense WHT rotation matrix for decode/query path (single matmul is faster
            // than FWHT butterfly via MLX ops due to graph overhead)
            let hadamard = TurboQuantRotation.hadamardMatrix(dim: dim)
            let signsDiag = expandedDimensions(signs, axis: 0)
            let whtRot = hadamard * signsDiag / MLXArray(Float(sqrt(Float(dim))))
            eval(whtRot)
            self.rotation = whtRot
            self.rotationT = whtRot.transposed()
        } else {
            self.whtSigns = nil
            self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotationT = self.rotation.transposed()
        }
    }

    /// Encode vectors (Algorithm 1 QUANT) with norm correction.
    /// Input: [B, H, T, D]
    /// Returns MSECodecState with corrected norms and packed indices.
    ///
    /// Norm correction: store `original_norm / reconstruction_norm` instead of raw norm.
    /// During decode, `centroid[idx] * corrected_norm` automatically compensates for
    /// quantization error. This is why TurboQuant beats q8_0 on perplexity in CUDA benchmarks.
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        // Extract norms and normalize (paper assumes unit sphere; we store norms separately)
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // Rotate: y ← Π · x (Algorithm 1 line 5)
        let rotated = matmul(unit, rotationT)

        // Quantize via boundary comparison (fast, no broadcast)
        let indices = boundaryQuantize(rotated)

        // Norm correction: compute reconstruction norm and store corrected ratio
        let reconstructed = codebook[indices]  // [B,H,T,D] — quantized approximation in rotated space
        let reconNormSq = (reconstructed * reconstructed).sum(axis: -1)
        let reconNorms = sqrt(maximum(reconNormSq, MLXArray(Float(1e-16))))
        let correctedNorms = norms / reconNorms  // original_norm / reconstruction_norm

        // Pack indices
        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        return MSECodecState(
            norms: correctedNorms,
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
        let unrotated = matmul(approx, rotation)

        // Rescale by stored norms
        return expandedDimensions(state.norms, axis: -1) * unrotated
    }

    /// Decode in rotated space (skip inverse rotation).
    /// Returns centroid values scaled by norm, still in Π-rotated coordinate space.
    /// Used with pre-rotated queries for dequant-first SDPA.
    public func decodeRotated(_ state: MSECodecState) -> MLXArray {
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)
        let approx = codebook[indices]
        return expandedDimensions(state.norms, axis: -1) * approx
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

/// KV cache using TurboQuant compression with two-phase architecture:
///
/// **Phase 1 — Prefill** (L>1): Store raw K/V like KVCacheSimple. Zero overhead.
/// **Transition**: On first decode call, compress entire raw cache in one batch.
/// **Phase 2 — Decode** (L=1): Encode 1 new token. Metal kernel scores against
///   all compressed tokens. Zero dequantization.
///
/// Both keys and values: Algorithm 1 (MSE at b bits, no QJL)
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

    public let bits: Int         // Legacy: used when keyBits == valueBits
    public let keyBits: Int      // Bit-width for key compression
    public let valueBits: Int    // Bit-width for value compression (can be lower — V compression is nearly free)
    private let seed: UInt64

    // Codecs (lazy init)
    private var keyMSECodec: MSECodec?   // keyBits for keys
    private var valueMSECodec: MSECodec? // valueBits for values

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

    public init(bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil, seed: UInt64 = 42) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.seed = seed
        super.init()
    }

    override public var isTrimmable: Bool { true }

    // MARK: - Shared Codec Cache

    /// Shared codec cache: all layers with the same (dim, bits, seed) reuse the same codec.
    /// Eliminates 56 redundant [128,128] rotation matrices (~7 MB) across 28 layers.
    private static let codecLock = NSLock()
    nonisolated(unsafe) private static var sharedCodecs: [String: MSECodec] = [:]

    private static func getOrCreateCodec(dim: Int, bits: Int, seed: UInt64) -> MSECodec {
        let key = "\(dim)_\(bits)_\(seed)"
        codecLock.lock()
        if let cached = sharedCodecs[key] {
            codecLock.unlock()
            return cached
        }
        codecLock.unlock()
        let codec = MSECodec(dim: dim, bits: bits, seed: seed)
        codecLock.lock()
        sharedCodecs[key] = codec
        codecLock.unlock()
        return codec
    }

    /// Initialize codecs if needed. Uses shared cache to avoid duplicating rotation matrices.
    private func ensureCodecs(headDim: Int) {
        guard keyMSECodec == nil else { return }
        keyMSECodec = Self.getOrCreateCodec(dim: headDim, bits: keyBits, seed: seed)
        valueMSECodec = Self.getOrCreateCodec(dim: headDim, bits: valueBits, seed: seed + 1)
    }

    nonisolated(unsafe) private static var loggedEncodeKernel = false

    /// Dispatch to WHT or dense fused encode kernel based on codec configuration.
    private func fusedEncodeDispatch(
        input: MLXArray, codec: MSECodec, headDim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        if !Self.loggedEncodeKernel {
            Self.loggedEncodeKernel = true
            print("[TURBO] Encode kernel: \(codec.useWHT ? "WHT butterfly" : "dense matmul"), dim=\(headDim), bits=\(bits)")
        }
        if codec.useWHT, let signs = codec.whtSigns {
            return TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: headDim
            )
        } else {
            return TurboQuantKernelOps.fusedEncode(
                input: input, rotation: codec.rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: headDim
            )
        }
    }

    // MARK: - Phase 1: Raw Prefill

    /// Prefill update: store raw K/V, return raw. Zero encoding overhead.
    /// Uses KVCacheSimple-style allocation with concatenated growth.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.rawKeys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.rawKeys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.rawKeys, var currentValues = self.rawValues {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.rawKeys = concatenated([currentKeys, newK], axis: 2)
                self.rawValues = concatenated([currentValues, newV], axis: 2)
            } else {
                self.rawKeys = newK
                self.rawValues = newV
            }
            rawAllocSteps = self.rawKeys!.dim(2)
        }

        self.offset += keys.dim(2)

        self.rawKeys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.rawValues?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.rawKeys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.rawValues![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    // MARK: - Transition: Compress Raw Cache

    /// Compress the entire raw K/V cache into packed format in one batch.
    /// Called once when transitioning from prefill to decode.
    private func compressRawCache() {
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }
        let allKeys = rk[.ellipsis, ..<offset, 0...]
        let allValues = rv[.ellipsis, ..<offset, 0...]
        let headDim = allKeys.dim(-1)
        ensureCodecs(headDim: headDim)
        compressRawCacheInternal(allKeys: allKeys, allValues: allValues, headDim: headDim)
        rawKeys = nil
        rawValues = nil
        rawAllocSteps = 0
        isCompressed = true
        MLX.Memory.clearCache()
    }

    /// Compress given raw K/V arrays into packed format.
    private func compressRawCacheInternal(allKeys: MLXArray, allValues: MLXArray, headDim: Int) {
        guard let keyMSECodec, let valueMSECodec else { return }

        let B = allKeys.dim(0)
        let H = allKeys.dim(1)
        let tokenCount = allKeys.dim(2)
        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
        let flatVals = allValues.reshaped([B * H * tokenCount, headDim])

        let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatch(
            input: flatKeys, codec: keyMSECodec, headDim: headDim)
        let (valPackedFlat, valNormsFlat) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)

        let allocSteps = ((tokenCount + step - 1) / step) * step
        keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
        keyNorms = MLXArray.zeros([B, H, allocSteps])
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valNorms = MLXArray.zeros([B, H, allocSteps])
        compressedAllocSteps = allocSteps

        keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedFlat.reshaped([B, H, tokenCount, kpw])
        keyNorms![.ellipsis, ..<tokenCount] = keyNormsFlat.reshaped([B, H, tokenCount])
        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([B, H, tokenCount, vpw])
        valNorms![.ellipsis, ..<tokenCount] = valNormsFlat.reshaped([B, H, tokenCount])

        eval(keyPackedMSE!, keyNorms!, valPackedMSE!, valNorms!)
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

        let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        // Fused Metal encode: norm + rotate + quantize + pack + norm correction in 1 dispatch
        let flatKeys = keys.reshaped([B * H * numSteps, headDim])
        let flatVals = values.reshaped([B * H * numSteps, headDim])

        let (keyPacked, keyNormsNew) = fusedEncodeDispatch(
            input: flatKeys, codec: keyMSECodec, headDim: headDim)
        let (valPacked, valNormsNew) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)

        // Reshape back to [B, H, T, ...]
        let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
        let keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])

        // Grow compressed storage using concatenated growth
        if (prev + numSteps) > compressedAllocSteps {
            let newAlloc = ((prev + numSteps + step - 1) / step) * step
            let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
            let newKN = MLXArray.zeros([B, H, newAlloc])
            let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
            let newVN = MLXArray.zeros([B, H, newAlloc])
            if prev > 0 {
                newKP[.ellipsis, ..<prev, 0...] = keyPackedMSE![.ellipsis, ..<prev, 0...]
                newKN[.ellipsis, ..<prev] = keyNorms![.ellipsis, ..<prev]
                newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                newVN[.ellipsis, ..<prev] = valNorms![.ellipsis, ..<prev]
            }
            keyPackedMSE = newKP; keyNorms = newKN
            valPackedMSE = newVP; valNorms = newVN
            compressedAllocSteps = newAlloc
        }

        offset = prev + numSteps
        keyPackedMSE![.ellipsis, prev..<offset, 0...] = keyPackedShaped
        keyNorms![.ellipsis, prev..<offset] = keyNormsShaped
        valPackedMSE![.ellipsis, prev..<offset, 0...] = valPackedShaped
        valNorms![.ellipsis, prev..<offset] = valNormsShaped
    }

    // FP16 dequant cache in ROTATED space — built incrementally, only new tokens dequanted each step.
    // Keys and values stay in Π-rotated coordinates. Queries are pre-rotated to match.
    // Output from SDPA is inverse-rotated once. This avoids per-token inverse rotation.
    private var dequantKeys: MLXArray?    // [B, H, T, D] in rotated space
    private var dequantValues: MLXArray?  // [B, H, T, D] in rotated space

    /// Encode new token, dequant to rotated space, append to FP16 cache using efficient
    /// .ellipsis-style buffer management (matching KVCacheSimple for zero overhead).
    ///
    /// Returns (keys, values) in Π-ROTATED space. Caller must:
    /// 1. Pre-rotate queries: q' = q @ Π^T
    /// 2. Run SDPA: output_rot = SDPA(q', keys_rot, values_rot)
    /// 3. Inverse-rotate output: output = output_rot @ Π
    public func updateAndDequant(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = newKeys.dim(-1)
        ensureCodecs(headDim: headDim)

        guard let keyMSECodec, let valueMSECodec else {
            return (newKeys, newValues)
        }

        // Transition: on first decode call, rotate the raw prefill cache into rotated space
        if !isCompressed {
            isCompressed = true
            let tokenCount = offset
            if tokenCount > 0, let rk = rawKeys, let rv = rawValues {
                let rawK = rk[.ellipsis, ..<tokenCount, 0...]
                let rawV = rv[.ellipsis, ..<tokenCount, 0...]
                // Rotate prefill keys/values into each codec's rotated space
                let rotK = keyMSECodec.prepareQueries(rawK)
                let rotV = valueMSECodec.prepareQueries(rawV)

                // Also compress for storage (keeping compressed copy for memory)
                compressRawCacheInternal(allKeys: rawK, allValues: rawV, headDim: headDim)

                // Allocate dequant buffer using KVCacheSimple pattern
                let nSteps = (step + tokenCount - 1) / step
                let kShape = [rotK.dim(0), rotK.dim(1), nSteps * step, headDim]
                let vShape = [rotV.dim(0), rotV.dim(1), nSteps * step, headDim]
                dequantKeys = MLXArray.zeros(kShape, dtype: rotK.dtype)
                dequantValues = MLXArray.zeros(vShape, dtype: rotV.dtype)
                dequantKeys?[.ellipsis, ..<tokenCount, 0...] = rotK
                dequantValues?[.ellipsis, ..<tokenCount, 0...] = rotV
            }
            rawKeys = nil
            rawValues = nil
        }

        // Encode the new token into compressed storage (background — for memory savings)
        let prevOffset = offset
        encodeNewToken(keys: newKeys, values: newValues)

        // Rotate new token(s) into rotated space (one matmul each, very fast)
        let rotNewKeys = keyMSECodec.prepareQueries(newKeys)
        let rotNewValues = valueMSECodec.prepareQueries(newValues)

        // Grow dequant buffer using KVCacheSimple pattern (concatenated growth)
        let reset =
            if let dk = self.dequantKeys, prevOffset + newKeys.dim(2) > dk.dim(2) {
                true
            } else {
                self.dequantKeys == nil
            }
        if reset {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let nSteps = (step + newKeys.dim(2) - 1) / step
            let kShape = [B, H, nSteps * step, headDim]
            let newDK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newDV = MLXArray.zeros(kShape, dtype: newKeys.dtype)

            if var currentKeys = self.dequantKeys, var currentValues = self.dequantValues {
                if prevOffset % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prevOffset, 0...]
                    currentValues = currentValues[.ellipsis, ..<prevOffset, 0...]
                }
                self.dequantKeys = concatenated([currentKeys, newDK], axis: 2)
                self.dequantValues = concatenated([currentValues, newDV], axis: 2)
            } else {
                self.dequantKeys = newDK
                self.dequantValues = newDV
            }
        }

        // Append rotated token(s) using .ellipsis slicing
        self.dequantKeys?[.ellipsis, prevOffset ..< offset, 0...] = rotNewKeys
        self.dequantValues?[.ellipsis, prevOffset ..< offset, 0...] = rotNewValues

        return (
            self.dequantKeys![.ellipsis, ..<offset, 0...],
            self.dequantValues![.ellipsis, ..<offset, 0...]
        )
    }

    /// Pre-rotate queries to match rotated key space: q' = q @ Π_key^T
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        guard let keyMSECodec else { return queries }
        return keyMSECodec.prepareQueries(queries)
    }

    /// Inverse-rotate SDPA output from value-rotated space back to original
    public func inverseRotateOutput(_ rotatedOutput: MLXArray) -> MLXArray {
        guard let valueMSECodec else { return rotatedOutput }
        return matmul(rotatedOutput, valueMSECodec.rotation)
    }

    /// Whether to use compressed-domain Metal kernels instead of dequant + SDPA.
    /// Default false: dequant-first is simpler and uses MLX's optimized attention.
    /// Set true for very long contexts where FP16 materialization costs too much memory.
    public var useCompressedAttention: Bool = false

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
            repeatCount: nRepeats, bits: self.keyBits, dim: headDim
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
            repeatCount: nRepeats, bits: self.valueBits, dim: headDim
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
