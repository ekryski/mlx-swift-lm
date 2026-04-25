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
// VERIFIED: No QJL, residual quantization, or random projection correction exists
// in this codebase. TurboQuant+ research proved QJL hurts autoregressive generation —
// random projection variance compounds across decode steps. MSE-only is correct.
//
// Enhancements beyond paper:
//   - Norm extraction/restoration: paper assumes ||x||=1; we store norms for arbitrary vectors
//   - Norm correction: store ||x|| / ||ỹ|| for dense rotation path (WHT skips — orthogonal preserves norms)
//   - WHT rotation option: O(d log d) butterfly in Metal kernel for power-of-2 dims
//   - Two-phase architecture: raw prefill → batch compress → compressed decode
//   - Pre-rotated queries: q' = Π·q computed once, reused for all cached keys
//   - Asymmetric K/V bit-widths: K precision dominates quality (softmax amplification),
//     V compression is nearly free (linear averaging). Use "turbo4v2" for 4-bit K + 2-bit V.
//   - Boundary layer protection: first/last N attention layers stay FP16
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

    /// Lloyd-Max optimal centroids for N(0, 1/sqrt(dim)) — matches llama.cpp.
    /// These are the battle-tested values from ggml-metal.metal, scaled per dimension.
    /// The reference values are for d=128; other dims are scaled by sqrt(128/dim).
    ///
    /// Source: ggml/src/ggml-metal/ggml-metal.metal (turbo_centroids_*bit)
    private static let referenceCentroids128: [Int: [Float]] = [
        2: [-0.133462, -0.039994, 0.039994, 0.133462],
        3: [-0.190685, -0.117832, -0.065717, -0.021460,
             0.021460,  0.065717,  0.117832,  0.190685],
        4: [-0.173926, -0.117195, -0.089527, -0.068756,
            -0.051262, -0.035597, -0.020989, -0.006938,
             0.006938,  0.020989,  0.035597,  0.051262,
             0.068756,  0.089527,  0.117195,  0.173926],
    ]

    /// Reference midpoints (boundaries) for d=128 — matches llama.cpp.
    private static let referenceMidpoints128: [Int: [Float]] = [
        2: [-0.086728, 0.0, 0.086728],
        3: [-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259],
        4: [-0.145560, -0.103361, -0.079142, -0.060009,
            -0.043430, -0.028293, -0.013963,  0.000000,
             0.013963,  0.028293,  0.043430,  0.060009,
             0.079142,  0.103361,  0.145560],
    ]

    /// Scale reference centroids from d=128 to target dim.
    /// N(0, 1/sqrt(dim)) centroids scale as sqrt(128/dim) relative to d=128.
    private static func scaledCentroids(dim: Int, bits: Int) -> [Float]? {
        guard let ref = referenceCentroids128[bits] else { return nil }
        let scale = Float(sqrt(128.0 / Double(dim)))
        return ref.map { $0 * scale }
    }

    private static func scaledMidpoints(dim: Int, bits: Int) -> [Float]? {
        guard let ref = referenceMidpoints128[bits] else { return nil }
        let scale = Float(sqrt(128.0 / Double(dim)))
        return ref.map { $0 * scale }
    }

    // Legacy pre-computed table — kept for non-standard dims that need runtime generation
    nonisolated(unsafe) private static var precomputed: [Int: [Int: [Float]]] = [:]

    /// Lock for thread-safe lazy population of precomputed centroids.
    private static let centroidLock = NSLock()

    /// Dims that should be lazily pre-populated (non-power-of-2 dims used by real models).
    /// These fall back to dense rotation path since WHT requires power-of-2, but still
    /// benefit from cached centroids to avoid ~50ms runtime k-means per codec init.
    ///
    /// - 80: Qwen3-4B (head_dim=80)
    /// - 96: Various smaller models
    private static let lazyDims: [Int] = [80, 96]
    private static let lazyBits: [Int] = [2, 3, 4, 8]

    /// Ensure centroids for a given dim are populated. Thread-safe, generates once.
    private static func ensureCentroidsPopulated(dim: Int) {
        centroidLock.lock()
        let exists = precomputed[dim] != nil
        centroidLock.unlock()
        guard !exists else { return }

        // Generate all bit-widths for this dim
        var dimTable: [Int: [Float]] = [:]
        for bits in lazyBits {
            dimTable[bits] = generateCentroids(dim: dim, bits: bits)
        }

        centroidLock.lock()
        // Double-check after lock (another thread may have populated)
        if precomputed[dim] == nil {
            precomputed[dim] = dimTable
        }
        centroidLock.unlock()
    }

    // MARK: - Public API

    /// Codebook centroids for (dim, bits). Uses llama.cpp reference values (scaled from d=128)
    /// for standard configs, falls back to runtime Beta-distribution generation for exotic dims.
    public static func codebook(dim: Int, bits: Int) -> MLXArray {
        // Use llama.cpp reference centroids (scaled from d=128)
        if let scaled = scaledCentroids(dim: dim, bits: bits) {
            return MLXArray(scaled)
        }
        // Fallback for unsupported bit-widths: runtime generation
        let centroids = generateCentroids(dim: dim, bits: bits)
        return MLXArray(centroids)
    }

    /// Codebook boundaries (midpoints between adjacent centroids).
    /// Uses llama.cpp reference midpoints when available.
    public static func boundaries(dim: Int, bits: Int) -> MLXArray {
        // Use pre-computed reference midpoints from llama.cpp
        if let scaled = scaledMidpoints(dim: dim, bits: bits) {
            return MLXArray(scaled)
        }
        // Fallback: compute from centroids
        let centroids = scaledCentroids(dim: dim, bits: bits) ?? generateCentroids(dim: dim, bits: bits)
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
        return result
    }

    /// Generate WHT sign vector: random ±1 per dimension, length d.
    /// Used with Walsh-Hadamard Transform for O(d log d) rotation.
    public static func whtSigns(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        // Random bits → ±1
        // Generate random ±1 signs using uniform random
        let uniform = MLXRandom.uniform(low: 0, high: 1, [dim], key: key)
        let signs = MLX.where(uniform .> Float(0.5), Float(1.0), Float(-1.0))
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
        let invSqrtDim = MLXArray(1.0 / sqrt(Float(dim)), dtype: x.dtype)
        return transformed * invSqrtDim
    }

    /// Apply SRHT inverse rotation: x = diag(signs) * H * y / sqrt(dim)
    /// WHT is self-inverse up to scale. Inverse of (H·D/√d) is (D·H/√d).
    public static func fwhtInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let transformed = whtButterfly(y)
        let invSqrtDim = MLXArray(1.0 / sqrt(Float(dim)), dtype: y.dtype)
        return transformed * invSqrtDim * signs
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
            let whtRot = hadamard * signsDiag / Float(sqrt(Float(dim)))
            // Convert to bf16 — rotation matrices are computed in fp32 (QR, Hadamard)
            // but must match model dtype to avoid promoting bf16 inputs through matmul.
            // No eval() here — premature eval forces materialization of unrelated lazy
            // ops (e.g., GDN state) which can lock in fp32 intermediates.
            self.rotation = whtRot.asType(.bfloat16)
            self.rotationT = self.rotation.transposed()
        } else {
            self.whtSigns = nil
            let rot = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotation = rot.asType(.bfloat16)
            self.rotationT = self.rotation.transposed()
        }
    }

    /// Encode vectors (Algorithm 1 QUANT) with optional norm correction.
    /// Input: [B, H, T, D]
    /// Returns MSECodecState with norms and packed indices.
    ///
    /// WHT path: stores raw norms directly. WHT is an orthogonal transform that preserves
    /// norms, so reconstruction_norm ≈ original_norm (within floating point error).
    /// Skipping norm correction saves one codebook lookup, one norm computation, and one
    /// division per encoded vector.
    ///
    /// Dense rotation path: stores `original_norm / reconstruction_norm` (norm correction).
    /// This compensates for quantization error in the non-orthogonal rotation case.
    ///
    // TODO: Dual half-block scales (TQ4_1S format optimization)
    //
    // Our llama.cpp TQ4_1S format uses dual half-block scales (d0 for elements 0-15,
    // d1 for elements 16-31) instead of a single per-block L2 norm. After WHT rotation,
    // the two halves of a 32-element block can have very different energy distributions,
    // so a single norm under-scales one half and over-scales the other.
    //
    // Dual scales reduce MSE by ~15-25% in our testing (see TurboQuant+ paper, Section 4.2).
    //
    // Implementing this requires changes to:
    //   1. MSECodecState packing format — store two norms per block instead of one
    //   2. This encode path — compute half-block norms separately:
    //        d0 = ||rotated[..., :16]||₂,  d1 = ||rotated[..., 16:]||₂
    //   3. Metal dequant kernels — use d0/d1 during reconstruction
    //   4. TurboFlash attention kernels — weighted dequant with two scales per block
    //
    // Too invasive for this PR, but high-value follow-up for accuracy-sensitive models.
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        // Extract norms and normalize (paper assumes unit sphere; we store norms separately)
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, Float(1e-8))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // Rotate: y ← Π · x (Algorithm 1 line 5)
        let rotated = matmul(unit, rotationT)

        // Quantize via boundary comparison (fast, no broadcast)
        let indices = boundaryQuantize(rotated)

        let storedNorms: MLXArray
        if useWHT {
            // WHT fast path: orthogonal transform preserves norms, skip correction
            storedNorms = norms
        } else {
            // Dense rotation path: norm correction compensates for quantization error
            let reconstructed = codebook[indices]  // [B,H,T,D] — quantized approximation in rotated space
            let reconNormSq = (reconstructed * reconstructed).sum(axis: -1)
            let reconNorms = sqrt(maximum(reconNormSq, Float(1e-16)))
            storedNorms = norms / reconNorms  // original_norm / reconstruction_norm
        }

        // Pack indices
        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        return MSECodecState(
            norms: storedNorms,
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

    // Profiling accumulators (static so they accumulate across all layers).
    // Enabled by MLX_BENCH_PROFILE=3 (forces eval per phase — invasive).
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
    public let keyBits: Int      // Bit-width for key compression (0 = raw FP16, no compression)
    public let valueBits: Int    // Bit-width for value compression (can be lower — V compression is nearly free)
    private let seed: UInt64

    /// Raw-K mode: keys stay at FP16 (uncompressed) while only values are TurboQuant compressed.
    /// This is the single biggest quality finding from TurboQuant+ — K precision dominates
    /// quality via softmax amplification, V compression is nearly free (linear averaging).
    /// Enabled when keyBits == 0.
    public let rawKeyMode: Bool

    /// Sliding window support: when set, compressed buffers rotate at this size.
    /// Matches RotatingKVCache semantics — oldest tokens evicted when full.
    /// nil = unbounded growth (full attention layers).
    private let rotatingMaxSize: Int?
    /// Current write index within the rotating buffer (wraps at rotatingMaxSize).
    private var rotatingIdx: Int = 0

    /// Last returned K/V from compressedAttention — for KV sharing (Gemma 4 donor layers).
    /// Keys are raw FP16 (rawKeyMode) or dequanted; values are dequanted.
    public var lastReturnedKeys: MLXArray?
    public var lastReturnedValues: MLXArray?

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

    /// Compressed storage write position — independent from `offset` (which tracks
    /// total tokens for RoPE/masks). This prevents desync between dequant and
    /// compressed buffers when batch encoding runs on a different schedule.
    private var compressedWriteOffset = 0

    /// Pre-allocation step size for buffer growth. Larger values reduce resize frequency
    /// at the cost of upfront memory. At step=1024, a 16K context only resizes 16 times
    /// (vs 64 times at step=256), eliminating 75% of allocation + copy overhead.
    private let step: Int

    public override var maxSize: Int? { rotatingMaxSize }

    public init(bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil, step: Int = 1024, seed: UInt64 = 42, maxSize: Int? = nil) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.rawKeyMode = (keyBits ?? bits) == 0
        self.seed = seed
        self.step = step
        self.rotatingMaxSize = maxSize
        super.init()
    }

    override public var isTrimmable: Bool { true }

    /// Load raw K/V data from a prefilled cache (e.g., KVCacheSimple or RotatingKVCache).
    /// TurboQuantKVCache will compress these on first decode token.
    /// Keys/values should be shape [B, H, T, D] in temporal order.
    public func loadRawKV(keys: MLXArray, values: MLXArray, originalOffset: Int? = nil) {
        self.rawKeys = keys
        self.rawValues = values
        // Cap offset at buffer size — for rotating caches, originalOffset may exceed
        // the buffer (it tracks total tokens seen, not buffer position).
        // Using buffer size ensures updateAndDequant/compressRawCache don't over-slice.
        let bufferTokens = keys.dim(2)
        self.offset = min(originalOffset ?? bufferTokens, bufferTokens)
        self.rawAllocSteps = bufferTokens
    }

    // MARK: - Shared Codec Cache

    /// Shared codec cache: all layers with the same (dim, bits, seed) reuse the same codec.
    /// Eliminates 56 redundant [128,128] rotation matrices (~7 MB) across 28 layers.
    private static let codecLock = NSLock()
    nonisolated(unsafe) private static var sharedCodecs: [String: MSECodec] = [:]

    /// Clear the shared codec cache. Call between benchmark runs to avoid stale graph references.
    public static func clearCodecCache() {
        codecLock.lock()
        sharedCodecs.removeAll()
        codecLock.unlock()
    }

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
    /// In rawKeyMode, key codec is nil — keys stay at FP16, no rotation/quantization needed.
    private func ensureCodecs(headDim: Int) {
        guard valueMSECodec == nil else { return }
        if !rawKeyMode {
            keyMSECodec = Self.getOrCreateCodec(dim: headDim, bits: keyBits, seed: seed)
        }
        valueMSECodec = Self.getOrCreateCodec(dim: headDim, bits: valueBits, seed: seed + 1)
    }

    nonisolated(unsafe) private static var loggedEncodeKernel = false

    /// Dispatch to WHT or dense fused encode kernel based on codec configuration.
    private func fusedEncodeDispatch(
        input: MLXArray, codec: MSECodec, headDim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        if !Self.loggedEncodeKernel {
            Self.loggedEncodeKernel = true
            print("[TURBO] Encode kernel: \(codec.useWHT ? "WHT butterfly" : "dense matmul"), dim=\(headDim), bits=\(codec.bits)")
        }
        if codec.useWHT, let signs = codec.whtSigns {
            return TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: codec.bits, dim: headDim
            )
        } else {
            return TurboQuantKernelOps.fusedEncode(
                input: input, rotation: codec.rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: codec.bits, dim: headDim
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
    ///
    /// In rawKeyMode: only compress values. Keys stay as raw FP16 in rawKeys buffer.
    /// This is the highest-quality TurboQuant+ mode — K precision dominates quality.
    private func compressRawCache() {
        // Guard: skip if already compressed or no raw data
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }
        // Use actual buffer size, not offset (which may exceed buffer in rotating mode)
        let actualTokens = min(offset, rk.dim(2))
        let allKeys = rk[.ellipsis, ..<actualTokens, 0...]
        let allValues = rv[.ellipsis, ..<actualTokens, 0...]
        let headDim = allKeys.dim(-1)
        ensureCodecs(headDim: headDim)
        compressRawCacheInternal(allKeys: allKeys, allValues: allValues, headDim: headDim)
        if rawKeyMode {
            // Keep rawKeys alive — they're our FP16 key storage going forward
            // Only free rawValues since those are now compressed
            rawValues = nil
            // For rotating: expand rawKeys to maxSize so rotation can write in-place
            if let maxSz = rotatingMaxSize, rk.dim(2) < maxSz {
                let B = rk.dim(0), H = rk.dim(1), D = rk.dim(3)
                let expanded = MLXArray.zeros([B, H, maxSz, D], dtype: rk.dtype)
                expanded[.ellipsis, ..<actualTokens, 0...] = rk[.ellipsis, ..<actualTokens, 0...]
                rawKeys = expanded
                rawAllocSteps = maxSz
            }
        } else {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
        }
        isCompressed = true
        compressedWriteOffset = min(offset, rotatingMaxSize ?? offset)
        if rotatingMaxSize != nil {
            rotatingIdx = offset % (rotatingMaxSize ?? offset)
        }
        MLX.Memory.clearCache()
    }

    /// Compress given raw K/V arrays into packed format.
    ///
    /// In rawKeyMode: only compress values. Keys are not encoded — they stay as raw FP16
    /// in the rawKeys buffer. keyPackedMSE and keyNorms remain nil.
    private func compressRawCacheInternal(allKeys: MLXArray, allValues: MLXArray, headDim: Int) {
        guard let valueMSECodec else { return }

        let B = allKeys.dim(0)
        let H = allKeys.dim(1)
        let tokenCount = allKeys.dim(2)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        let flatVals = allValues.reshaped([B * H * tokenCount, headDim])
        let (valPackedFlat, valNormsFlat) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)

        // Pre-allocate to at least one step beyond current tokenCount to accommodate
        // the first decode token after compression. Without this, the buffer is exactly
        // tokenCount slots and the first encodeNewToken write overflows.
        let minAlloc = rotatingMaxSize ?? (tokenCount + step)
        let allocSteps = ((max(minAlloc, tokenCount + step) + step - 1) / step) * step
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valNorms = MLXArray.zeros([B, H, allocSteps])

        if !rawKeyMode {
            // Compress keys too (standard TurboQuant path)
            guard let keyMSECodec else { return }
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
            let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)

            keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
            keyNorms = MLXArray.zeros([B, H, allocSteps])

            keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedFlat.reshaped([B, H, tokenCount, kpw])
            keyNorms![.ellipsis, ..<tokenCount] = keyNormsFlat.reshaped([B, H, tokenCount])

        }

        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([B, H, tokenCount, vpw])
        valNorms![.ellipsis, ..<tokenCount] = valNormsFlat.reshaped([B, H, tokenCount])

        // Debug: validate encode output
        if ProcessInfo.processInfo.environment["TQ_DEBUG"] == "1" {
            eval(valPackedFlat, valNormsFlat)
            let vnHasNaN = MLX.isNaN(valNormsFlat).any().item(Bool.self)
            let vnMax = valNormsFlat.max().item(Float.self)
            let vnMin = valNormsFlat.min().item(Float.self)
            let vpMax = valPackedFlat.max().item(UInt32.self)
            let levels = UInt32(1 << valueBits)
            print("[TQ-ENCODE-V] bits=\(valueBits) rows=\(B*H*tokenCount) vpw=\(vpw) vnHasNaN=\(vnHasNaN) vnRange=[\(vnMin),\(vnMax)] vpMax=\(vpMax) levels=\(levels)")
            // Check if any packed index exceeds codebook size
            if vpMax >= UInt32.max / 2 {
                print("[TQ-ENCODE-V] WARNING: packed values suspiciously large!")
            }
        }

        compressedAllocSteps = allocSteps

        if rawKeyMode {
            eval(valPackedMSE!, valNorms!)
        } else {
            eval(keyPackedMSE!, keyNorms!, valPackedMSE!, valNorms!)
        }
    }

    // MARK: - Phase 2: Compressed Decode

    /// Encode new token(s) into compressed storage using fused Metal kernel.
    ///
    /// When `rotatingMaxSize` is set, writes wrap around at `maxSize` — oldest
    /// tokens are overwritten in-place, matching RotatingKVCache semantics.
    /// Each token is independently compressed at its write position, so rotation
    /// doesn't corrupt quantization groups (unlike bulk conversion).
    private func encodeNewToken(keys: MLXArray, values: MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let H = keys.dim(1)
        let numSteps = keys.dim(2)

        guard let valueMSECodec else { return }

        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        // Encode values via fused Metal kernel
        let flatVals = values.reshaped([B * H * numSteps, headDim])
        let (valPacked, valNormsNew) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])

        // Determine write position — rotating or linear.
        // Offset is managed by the caller (compressedAttention).
        let writeIdx: Int
        if let maxSz = rotatingMaxSize {
            // Rotating mode: wrap write position within the fixed buffer
            if offset >= maxSz {
                writeIdx = rotatingIdx
                rotatingIdx = (rotatingIdx + numSteps) % maxSz
            } else {
                // Still filling up — linear write
                writeIdx = offset
                rotatingIdx = offset + numSteps
            }
        } else {
            writeIdx = offset
        }

        if rawKeyMode {
            // Ensure buffers are allocated
            let targetSize = rotatingMaxSize ?? (writeIdx + numSteps)
            if writeIdx + numSteps > rawAllocSteps {
                let newAlloc = rotatingMaxSize ?? (((writeIdx + numSteps + step - 1) / step) * step)
                let newRK = MLXArray.zeros([B, H, newAlloc, headDim], dtype: keys.dtype)
                if writeIdx > 0, let rk = rawKeys {
                    let copyLen = min(writeIdx, rk.dim(2))
                    newRK[.ellipsis, ..<copyLen, 0...] = rk[.ellipsis, ..<copyLen, 0...]
                }
                rawKeys = newRK
                rawAllocSteps = newAlloc
            }
            if writeIdx + numSteps > compressedAllocSteps {
                let newAlloc = rotatingMaxSize ?? (((writeIdx + numSteps + step - 1) / step) * step)
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc])
                if writeIdx > 0 {
                    let copyLen = min(writeIdx, compressedAllocSteps)
                    newVP[.ellipsis, ..<copyLen, 0...] = valPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newVN[.ellipsis, ..<copyLen] = valNorms![.ellipsis, ..<copyLen]
                }
                valPackedMSE = newVP; valNorms = newVN
                compressedAllocSteps = newAlloc
            }

            rawKeys![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = keys
            valPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = valPackedShaped
            valNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = valNormsShaped
        } else {
            // Standard TurboQuant: encode both K and V
            guard let keyMSECodec else { return }

            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = keys.reshaped([B * H * numSteps, headDim])
            let (keyPacked, keyNormsNew) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)
            let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
            let keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])

            if writeIdx + numSteps > compressedAllocSteps {
                let newAlloc = rotatingMaxSize ?? (((writeIdx + numSteps + step - 1) / step) * step)
                let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
                let newKN = MLXArray.zeros([B, H, newAlloc])
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc])
                if writeIdx > 0 {
                    let copyLen = min(writeIdx, compressedAllocSteps)
                    newKP[.ellipsis, ..<copyLen, 0...] = keyPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newKN[.ellipsis, ..<copyLen] = keyNorms![.ellipsis, ..<copyLen]
                    newVP[.ellipsis, ..<copyLen, 0...] = valPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newVN[.ellipsis, ..<copyLen] = valNorms![.ellipsis, ..<copyLen]
                }
                keyPackedMSE = newKP; keyNorms = newKN
                valPackedMSE = newVP; valNorms = newVN
                compressedAllocSteps = newAlloc
            }

            keyPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = keyPackedShaped
            keyNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = keyNormsShaped
            valPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = valPackedShaped
            valNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = valNormsShaped
        }

        // [#87 fix] Advance offset so the caller's `compressedWriteOffset =
        // offset` line correctly captures the new write position. Without
        // this, every decode token writes to the same slot (overwriting the
        // prior) and slots beyond it stay zero-initialized — those zero
        // slots cause division-by-zero NaN in the dequant path during the
        // next attention pass.
        offset += numSteps
    }

    /// Batch-encode all pending raw tokens into compressed storage.
    // FP16 dequant cache in ROTATED space — built incrementally, only new tokens dequanted each step.
    // Keys and values stay in Π-rotated coordinates. Queries are pre-rotated to match.
    // Output from SDPA is inverse-rotated once. This avoids per-token inverse rotation.
    private var dequantKeys: MLXArray?    // [B, H, T, D] in rotated space
    private var dequantValues: MLXArray?  // [B, H, T, D] in rotated space

    /// Encode new token, dequant to rotated space, append to FP16 cache using efficient
    /// .ellipsis-style buffer management (matching KVCacheSimple for zero overhead).
    ///
    /// Returns (keys, values) in Π-ROTATED space. Caller must:
    /// 1. Pre-rotate queries: q' = q @ Π^T  (skip in rawKeyMode — queries stay raw)
    /// 2. Run SDPA: output_rot = SDPA(q', keys_rot, values_rot)
    /// 3. Inverse-rotate output: output = output_rot @ Π
    ///
    /// In rawKeyMode: keys are returned raw (unrotated). Values are in Π_value-rotated space.
    /// Caller must NOT pre-rotate queries for K scoring (raw keys → raw queries).
    /// Caller must still inverse-rotate the value component of the output.
    public func updateAndDequant(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = newKeys.dim(-1)
        ensureCodecs(headDim: headDim)

        guard let valueMSECodec else {
            return (newKeys, newValues)
        }

        // Transition: on first decode call, rotate the raw prefill cache into rotated space
        if !isCompressed {
            isCompressed = true
            // Use actual buffer size, not offset (which tracks total tokens for RoPE)
            let tokenCount = rotatingMaxSize.map { min(offset, $0) } ?? offset
            compressedWriteOffset = min(offset, rotatingMaxSize ?? offset)
            if tokenCount > 0, let rk = rawKeys, let rv = rawValues {
                let actualTokens = min(tokenCount, rk.dim(2))
                let rawK = rk[.ellipsis, ..<actualTokens, 0...]
                let rawV = rv[.ellipsis, ..<actualTokens, 0...]

                // Rotate values into codec's rotated space (keys stay raw in rawKeyMode)
                let rotV = valueMSECodec.prepareQueries(rawV)

                // Also compress for storage (keeping compressed copy for memory)
                compressRawCacheInternal(allKeys: rawK, allValues: rawV, headDim: headDim)

                if rawKeyMode {
                    // rawKeyMode: keys stay as-is in dequant buffer, values get rotated
                    let allocSize = rotatingMaxSize ?? actualTokens
                    let nSteps = (step + allocSize - 1) / step
                    let kShape = [rawK.dim(0), rawK.dim(1), nSteps * step, headDim]
                    let vShape = [rotV.dim(0), rotV.dim(1), nSteps * step, headDim]
                    dequantKeys = MLXArray.zeros(kShape, dtype: rawK.dtype)
                    dequantValues = MLXArray.zeros(vShape, dtype: rotV.dtype)
                    dequantKeys?[.ellipsis, ..<actualTokens, 0...] = rawK
                    dequantValues?[.ellipsis, ..<actualTokens, 0...] = rotV
                } else {
                    // Standard: rotate both keys and values
                    guard let keyMSECodec else { return (newKeys, newValues) }
                    let rotK = keyMSECodec.prepareQueries(rawK)

                    let allocSize = rotatingMaxSize ?? actualTokens
                    let nSteps = (step + allocSize - 1) / step
                    let kShape = [rotK.dim(0), rotK.dim(1), nSteps * step, headDim]
                    let vShape = [rotV.dim(0), rotV.dim(1), nSteps * step, headDim]
                    dequantKeys = MLXArray.zeros(kShape, dtype: rotK.dtype)
                    dequantValues = MLXArray.zeros(vShape, dtype: rotV.dtype)
                    dequantKeys?[.ellipsis, ..<actualTokens, 0...] = rotK
                    dequantValues?[.ellipsis, ..<actualTokens, 0...] = rotV
                }
            }
            if !rawKeyMode {
                rawKeys = nil
            }
            rawValues = nil
        }
        // Initialize rotating write index
        if let maxSz = rotatingMaxSize {
            let bufferTokens = min(offset, maxSz)
            rotatingIdx = bufferTokens % maxSz
        }

        // Dequant cache is primary for SDPA. Compressed encoding runs on separate
        // schedule using compressedWriteOffset to avoid desync.
        let prevOffset = offset
        offset = prevOffset + newKeys.dim(2)

        // Track compressed write position for accurate memoryBytes reporting
        compressedWriteOffset += newKeys.dim(2)

        // Rotate new token(s) into appropriate space
        let dequantNewKeys: MLXArray
        if rawKeyMode {
            // Raw-K mode: keys stay unrotated
            dequantNewKeys = newKeys
        } else {
            // Standard: rotate keys into key codec's rotated space
            guard let keyMSECodec else { return (newKeys, newValues) }
            dequantNewKeys = keyMSECodec.prepareQueries(newKeys)
        }
        let rotNewValues = valueMSECodec.prepareQueries(newValues)

        // Dequant buffer management — rotating or linear
        if let maxSz = rotatingMaxSize {
            // Rotating: write at rotatingIdx, return full buffer up to maxSize
            let writePos = (offset > maxSz) ? (rotatingIdx - newKeys.dim(2) + maxSz) % maxSz : prevOffset
            let bufferTokens = min(offset, maxSz)

            if self.dequantKeys == nil {
                // Should have been initialized in the transition block above
                let B = newKeys.dim(0)
                let H = newKeys.dim(1)
                let kShape = [B, H, maxSz, headDim]
                self.dequantKeys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
                self.dequantValues = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            }

            self.dequantKeys?[.ellipsis, writePos ..< (writePos + newKeys.dim(2)), 0...] = dequantNewKeys
            self.dequantValues?[.ellipsis, writePos ..< (writePos + newKeys.dim(2)), 0...] = rotNewValues

            let returnedKeys = self.dequantKeys![.ellipsis, ..<bufferTokens, 0...]
            let returnedValues = self.dequantValues![.ellipsis, ..<bufferTokens, 0...]

            self.lastReturnedKeys = returnedKeys
            self.lastReturnedValues = returnedValues
            return (returnedKeys, returnedValues)
        }

        // Linear (non-rotating) path
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

        self.dequantKeys?[.ellipsis, prevOffset ..< offset, 0...] = dequantNewKeys
        self.dequantValues?[.ellipsis, prevOffset ..< offset, 0...] = rotNewValues

        let returnedKeys = self.dequantKeys![.ellipsis, ..<offset, 0...]
        let returnedValues = self.dequantValues![.ellipsis, ..<offset, 0...]

        // Store for KV sharing (Gemma 4 donor layers)
        self.lastReturnedKeys = returnedKeys
        self.lastReturnedValues = returnedValues

        return (returnedKeys, returnedValues)
    }

    /// Pre-rotate queries to match rotated key space: q' = q @ Π_key^T
    /// In rawKeyMode: no-op, queries stay raw for standard Q*K matmul.
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        if rawKeyMode { return queries }
        guard let keyMSECodec else { return queries }
        return keyMSECodec.prepareQueries(queries)
    }

    /// Inverse-rotate SDPA output from value-rotated space back to original
    public func inverseRotateOutput(_ rotatedOutput: MLXArray) -> MLXArray {
        guard let valueMSECodec else { return rotatedOutput }
        return matmul(rotatedOutput, valueMSECodec.rotation)
    }

    /// Compressed-domain attention via Metal kernels.
    ///
    /// On first call: compresses raw prefill cache in one batch.
    /// Then: encode 1 new token → fused attention kernel → inverse rotation.
    ///
    /// For L=1 (decode): uses TurboFlashAttention — a single Metal dispatch that fuses
    /// Q×K scoring + online softmax + Attn×V aggregation. No intermediate score or weight
    /// arrays are materialized. This reduces 3 dispatches (score + softmax + value) to 1.
    ///
    /// For L>1 (prefill chunks): falls back to separate score → softmax → value kernels
    /// since causal masking across multiple query positions requires the full score matrix.
    ///
    /// In rawKeyMode: uses standard matmul for Q*K scoring (raw FP16 keys, no rotation),
    /// then compressed-domain Metal kernel for Attn*V (TurboQuant compressed values).
    /// TurboFlash is NOT used — it assumes both K and V are packed.
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


        // Level 3: per-phase TQ decode profiling (forces eval per phase — invasive)
        let profiling = (Int(ProcessInfo.processInfo.environment["MLX_BENCH_PROFILE"] ?? "0") ?? 0) >= 3
        var t0 = Date()

        // Phase A: Encode new token directly into compressed storage.
        offset += newKeys.dim(2)
        let savedOffset = offset
        offset = compressedWriteOffset
        encodeNewToken(keys: newKeys, values: newValues)
        compressedWriteOffset = offset
        offset = savedOffset
        if profiling { eval(valPackedMSE!, valNorms!); if let kp = keyPackedMSE, let kn = keyNorms { eval(kp, kn) }; let t1 = Date(); Self.profileEncodeMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }


        guard let valueMSECodec else {
            return queries
        }

        // For rotating caches, token count is capped at maxSize
        let tokenCount = rotatingMaxSize.map { min(offset, $0) } ?? offset

        // Debug: log on first few calls
        if ProcessInfo.processInfo.environment["TQ_DEBUG"] == "1" {
            print("[TQ-ATTN] offset=\(offset) tokenCount=\(tokenCount) compressedWriteOffset=\(compressedWriteOffset) rawKeyMode=\(rawKeyMode) L=\(L) B=\(B) nQH=\(nQHeads) nKVH=\(nKVHeads) repeat=\(nRepeats) dim=\(headDim)")
            if let vp = valPackedMSE { print("[TQ-ATTN]   valPacked=\(vp.shape)") }
            if let vn = valNorms { print("[TQ-ATTN]   valNorms=\(vn.shape)") }
            if let kp = keyPackedMSE { print("[TQ-ATTN]   keyPacked=\(kp.shape)") }
            if let kn = keyNorms {
                eval(kn)
                let knHasNaN = MLX.isNaN(kn[0..., 0..., ..<tokenCount]).any().item(Bool.self)
                let knMax = MLX.abs(kn[0..., 0..., ..<tokenCount]).max().item(Float.self)
                print("[TQ-ATTN]   keyNorms: hasNaN=\(knHasNaN) max=\(knMax)")
            }
        }

        // Shared V slicing (used by all paths)
        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatValNorms = valNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        let valRotation = valueMSECodec.rotation
        var output: MLXArray

        if rawKeyMode {
            // ═══ Raw-K + Compressed-V path ═══
            // Standard matmul for Q*K (raw FP16 keys, no rotation needed).
            // Compressed-domain Metal kernel for Attn*V weighted sum.
            if profiling { eval(valPackedMSE!); let t1 = Date(); Self.profileEncodeMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

            // Q*K scoring: standard matmul with raw FP16 keys
            guard let rk = rawKeys else { return queries }
            let allKeys = rk[.ellipsis, ..<tokenCount, 0...]  // [B, nKVHeads, T, D]

            // GQA: expand keys to match query heads
            let expandedKeys: MLXArray
            if nRepeats > 1 {
                // [B, nKVHeads, T, D] → [B, nKVHeads, 1, T, D] → [B, nKVHeads, nRepeats, T, D] → [B, nQHeads, T, D]
                let expanded = expandedDimensions(allKeys, axis: 2)
                let tiledKeys = MLX.tiled(expanded, repetitions: [1, 1, nRepeats, 1, 1])
                expandedKeys = tiledKeys.reshaped([B, nQHeads, tokenCount, headDim])
            } else {
                expandedKeys = allKeys
            }

            // scores = Q * K^T * scale  → [B, nQHeads, L, T]
            var scores = matmul(queries, expandedKeys.transposed(0, 1, 3, 2)) * scale
            if profiling { eval(scores); let t1 = Date(); Self.profileScoreMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

            // Mask + softmax
            switch mask {
            case .array(let maskArray):
                if maskArray.dtype == .bool {
                    scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
                } else { scores = scores + maskArray }
            case .causal:
                // Build causal mask manually
                let queryOffset = tokenCount - L
                let causalMask = MLXArray.tri(L, m: tokenCount, k: queryOffset, type: Bool.self)
                let expandedMask = expandedDimensions(expandedDimensions(causalMask, axis: 0), axis: 0)
                scores = MLX.where(expandedMask, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
            case .none: break
            default: break
            }

            let attnWeights = softmax(scores, axis: -1)
            if profiling { eval(attnWeights); let t1 = Date(); Self.profileOtherMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

            // Metal V kernel: compressed-domain weighted sum
            let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
            let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
                codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                repeatCount: nRepeats, bits: self.valueBits, dim: headDim
            )

            // Force eval of V kernel output before inverse rotation
            // (MLX lazy eval fusion with subsequent matmul produces incorrect results)
            eval(rotatedOutput)

            // Inverse value rotation
            output = matmul(
                rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                valueMSECodec.rotation
            )

            if profiling { eval(output); let t1 = Date(); Self.profileValueMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }
        } else {
            // ═══ Standard TurboQuant path: both K and V compressed ═══
            guard let keyMSECodec else { return queries }

            if profiling { eval(keyPackedMSE!, valPackedMSE!); let t1 = Date(); Self.profileEncodeMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

            // Pre-rotate query for compressed-domain K scoring
            let qRot = keyMSECodec.prepareQueries(queries) * scale
            let flatQ = qRot.reshaped([B * nQHeads * L, headDim])

            // K slicing
            let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
                .reshaped([B * nKVHeads, tokenCount, -1])
            let flatKeyNorms = keyNorms![0..., 0..., ..<tokenCount]
                .reshaped([B * nKVHeads, tokenCount])

            // TurboFlash fused kernel supports these (kb,vb) combos.
            // Anything else falls through to the separated score+softmax+value path.
            let hasTurboFlashKernel: Bool = {
                switch (keyBits, valueBits) {
                case (4,4), (4,2), (4,3), (3,2), (3,3), (8,4), (8,2), (8,8): return true
                default: return false
                }
            }()

            if L == 1 && hasTurboFlashKernel {
                // TurboFlashAttention path (decode, L=1)
                output = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: flatQ,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyMSECodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, repeatCount: nRepeats,
                    keyBits: self.keyBits, valueBits: self.valueBits, dim: headDim,
                    valRotation: valRotation
                ).reshaped([B, nQHeads, L, headDim])

                // Sync eval barrier required on small-nKVH shapes (#87).
                // Without it, MLX lazy-eval fusion of TurboFlash output with
                // downstream ops produces garbage on Qwen3.5 nKVH=2 shapes
                // (0.8B / 2B / 35B-A3B turbo4 / turbo4v2): model emits
                // `!!!!!` from decode token 1. Numerical A/B vs the
                // separated `mseScore + softmax + mseWeightedSum` path
                // confirms TurboFlash itself computes the right values
                // within fp32 noise — the bug is in MLX's fusion of those
                // values with subsequent residual/MLP/next-layer ops, not
                // in the Metal kernel.
                //
                // The barrier is gated on nKVH < 4 because a sync eval costs
                // ~40% decode tok/s on 4B (nKVH=4); shapes that don't trip
                // the fusion bug stay on the fast no-barrier path. AsyncEval
                // doesn't break the fusion enough — it must be a sync.
                // Dtype round-trip breaks MLX lazy-eval fusion (#87).
                // TurboFlash outputs f32; casting through the model's dtype
                // creates a graph boundary that prevents incorrect fusion with
                // downstream residual/MLP ops. Cost: ~3μs per layer vs sync
                // eval() which costs 40-60% decode throughput.
                let modelDtype = queries.dtype
                output = output.asType(modelDtype).asType(.float32).asType(modelDtype)
                if profiling {
                    let t1 = Date()
                    Self.profileValueMs += t1.timeIntervalSince(t0) * 1000  // flash+eval time
                    Self.profileCount += 1
                    if Self.profileCount % 50 == 0 {
                        let encMs = Self.profileEncodeMs / Double(Self.profileCount)
                        let flashMs = Self.profileValueMs / Double(Self.profileCount)
                        let totalMs = encMs + flashMs
                        print(String(format: "[TQ-PROFILE] %d steps: encode=%.2fms flash+eval=%.2fms total=%.2fms (%.0f tok/s per-layer)",
                            Self.profileCount, encMs, flashMs, totalMs, 1000.0/totalMs))
                    }
                    t0 = t1
                }
            } else if case .causal = mask, hasTurboFlashKernel {
                // Causal TurboFlashAttention path (prefill, L>1)
                let queryOffset = tokenCount - L
                output = TurboQuantKernelOps.turboFlashAttentionCausal(
                    rotatedQueries: flatQ,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyMSECodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, repeatCount: nRepeats,
                    keyBits: self.keyBits, valueBits: self.valueBits, dim: headDim,
                    queryChunkLength: L, queryOffset: queryOffset,
                    valRotation: valRotation
                ).reshaped([B, nQHeads, L, headDim])
                if profiling { eval(output); let t1 = Date(); Self.profileScoreMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }
            } else {
                // Separated path: Metal score kernel
                var scores = TurboQuantKernelOps.mseScore(
                    rotatedQueries: flatQ, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.keyBits, dim: headDim
                ).reshaped([B, nQHeads, L, tokenCount])

                if profiling { eval(scores); let t1 = Date(); Self.profileScoreMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

                // Mask + softmax
                switch mask {
                case .array(let maskArray):
                    if maskArray.dtype == .bool {
                        scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
                    } else { scores = scores + maskArray }
                case .none: break
                default: break
                }

                let attnWeights = softmax(scores, axis: -1)
                if profiling { eval(attnWeights); let t1 = Date(); Self.profileOtherMs += t1.timeIntervalSince(t0) * 1000; t0 = t1 }

                // Metal value kernel
                let flatWeights2 = attnWeights.reshaped([B * nQHeads * L, tokenCount])
                let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                    weights: flatWeights2, packed: flatValPacked, norms: flatValNorms,
                    codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.valueBits, dim: headDim
                )

                // Force eval before inverse rotation (same lazy eval fix as rawKeyMode)
                eval(rotatedOutput)

                // Inverse rotation
                output = matmul(
                    rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                    valueMSECodec.rotation
                )
            }
        }

        Self.profileCount += 1
        return output
    }

    // MARK: - Memory Reporting

    /// Actual memory footprint: compressed storage (packed indices + norms) for K and V,
    /// plus any raw FP16 buffers if still in prefill phase or rawKeyMode, plus dequant buffers.
    /// Does NOT include codec overhead (rotation matrices, codebooks) which is shared across layers.
    /// In rawKeyMode: rawKeys is always present (FP16 keys), no keyPackedMSE/keyNorms.
    override public var memoryBytes: Int {
        if !isCompressed {
            // Pre-compression: raw FP16 storage
            var total = 0
            if let rk = rawKeys { total += arrayBytes(rk) }
            if let rv = rawValues { total += arrayBytes(rv) }
            return total
        }
        // Report what compressed storage WOULD be for the current token count.
        let tokenCount = compressedWriteOffset
        guard tokenCount > 0, let rk = rawKeys ?? dequantKeys else { return 0 }
        print("[TQ-MEMBYTES] isCompressed=\(isCompressed) tokens=\(tokenCount) D=\(rk.dim(3)) rawKeyMode=\(rawKeyMode) keyBits=\(keyBits) valueBits=\(valueBits)")
        let B = rk.dim(0)
        let H = rk.dim(1)
        let D = rk.dim(3)
        var total = 0
        if rawKeyMode {
            // K stays fp16
            total += B * H * tokenCount * D * 2  // bfloat16
        } else {
            // K compressed: packed + norms
            let kpw = TurboQuantPacking.packedWidth(count: D, bits: keyBits)
            total += B * H * tokenCount * kpw * 4  // uint32
            total += B * H * tokenCount * 4  // float32 norms
        }
        // V compressed: packed + norms
        let vpw = TurboQuantPacking.packedWidth(count: D, bits: valueBits)
        total += B * H * tokenCount * vpw * 4  // uint32
        total += B * H * tokenCount * 4  // float32 norms
        return total
    }

    // MARK: - State / Trim

    override public var state: [MLXArray] {
        get {
            if isCompressed {
                if rawKeyMode {
                    // Raw-K mode compressed: [rawKeys, valPacked, valNorms]
                    guard let rk = rawKeys,
                          let vpm = valPackedMSE, let vn = valNorms,
                          offset > 0 else { return [] }
                    return [
                        rk[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                } else {
                    // Standard compressed: [keyPacked, keyNorms, valPacked, valNorms]
                    guard let kpm = keyPackedMSE, let kn = keyNorms,
                          let vpm = valPackedMSE, let vn = valNorms,
                          offset > 0 else { return [] }
                    return [
                        kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                }
            } else {
                guard let rk = rawKeys, let rv = rawValues, offset > 0 else { return [] }
                return [rk[0..., 0..., ..<offset, 0...], rv[0..., 0..., ..<offset, 0...]]
            }
        }
        set {
            if rawKeyMode && newValue.count == 3 {
                // Raw-K mode compressed state: [rawKeys, valPacked, valNorms]
                rawKeys = newValue[0]
                rawAllocSteps = newValue[0].dim(2)
                valPackedMSE = newValue[1]; valNorms = newValue[2]
                offset = newValue[0].dim(2)
                compressedAllocSteps = newValue[1].dim(2)
                isCompressed = true
            } else if newValue.count == 4 {
                // Standard compressed state: [keyPacked, keyNorms, valPacked, valNorms]
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
            dequantKeys = nil; dequantValues = nil
            compressedAllocSteps = 0; isCompressed = false
        }
        return trimCount
    }
}
