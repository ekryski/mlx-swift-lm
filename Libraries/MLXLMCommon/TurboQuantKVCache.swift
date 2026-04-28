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
        // 8-bit (256 levels) Lloyd-Max centroids for d=128. Computed offline
        // via the same generator as `generateCentroids(dim:bits:)` (gridSize
        // 32768, 100 k-means iterations on Beta(d) weights). Pre-baked here
        // to avoid the ~1–4s runtime cliff that 8-bit codec init was charging
        // to the first prefill window. See spec 010 Part A.
        8: [
            -0.321782, -0.280567, -0.254225, -0.234551,
            -0.218845, -0.205896, -0.194947, -0.185487,
            -0.177261, -0.170089, -0.163792, -0.158191,
            -0.153197, -0.148718, -0.144665, -0.140977,
            -0.137562, -0.134328, -0.131279, -0.128411,
            -0.125697, -0.123103, -0.120632, -0.118283,
            -0.116025, -0.113890, -0.111876, -0.109954,
            -0.108093, -0.106292, -0.104584, -0.102905,
            -0.101196, -0.099518, -0.097901, -0.096345,
            -0.094819, -0.093324, -0.091828, -0.090333,
            -0.088899, -0.087495, -0.086091, -0.084687,
            -0.083345, -0.082032, -0.080689, -0.079377,
            -0.078096, -0.076844, -0.075624, -0.074372,
            -0.073121, -0.071901, -0.070710, -0.069551,
            -0.068391, -0.067231, -0.066072, -0.064912,
            -0.063783, -0.062654, -0.061525, -0.060426,
            -0.059358, -0.058290, -0.057191, -0.056123,
            -0.055085, -0.054048, -0.053010, -0.051972,
            -0.050935, -0.049928, -0.048921, -0.047914,
            -0.046906, -0.045899, -0.044892, -0.043885,
            -0.042909, -0.041932, -0.040955, -0.040009,
            -0.039063, -0.038117, -0.037171, -0.036225,
            -0.035279, -0.034333, -0.033417, -0.032502,
            -0.031556, -0.030610, -0.029694, -0.028779,
            -0.027863, -0.026948, -0.026032, -0.025117,
            -0.024232, -0.023346, -0.022431, -0.021515,
            -0.020630, -0.019745, -0.018860, -0.017975,
            -0.017060, -0.016175, -0.015290, -0.014405,
            -0.013520, -0.012635, -0.011780, -0.010926,
            -0.010040, -0.009155, -0.008270, -0.007385,
            -0.006531, -0.005646, -0.004761, -0.003906,
            -0.003052, -0.002167, -0.001312, -0.000458,
             0.000458,  0.001343,  0.002197,  0.003052,
             0.003937,  0.004822,  0.005676,  0.006561,
             0.007446,  0.008301,  0.009186,  0.010071,
             0.010926,  0.011811,  0.012696,  0.013550,
             0.014435,  0.015320,  0.016205,  0.017090,
             0.017975,  0.018860,  0.019745,  0.020661,
             0.021546,  0.022431,  0.023346,  0.024262,
             0.025147,  0.026032,  0.026948,  0.027863,
             0.028779,  0.029694,  0.030610,  0.031556,
             0.032502,  0.033418,  0.034364,  0.035310,
             0.036225,  0.037171,  0.038148,  0.039094,
             0.040040,  0.040986,  0.041963,  0.042970,
             0.043946,  0.044923,  0.045930,  0.046937,
             0.047914,  0.048921,  0.049928,  0.050935,
             0.051972,  0.053010,  0.054048,  0.055085,
             0.056123,  0.057191,  0.058290,  0.059388,
             0.060456,  0.061555,  0.062684,  0.063813,
             0.064943,  0.066072,  0.067231,  0.068391,
             0.069551,  0.070741,  0.071931,  0.073152,
             0.074403,  0.075593,  0.076814,  0.078096,
             0.079408,  0.080720,  0.082032,  0.083345,
             0.084687,  0.086091,  0.087495,  0.088929,
             0.090394,  0.091859,  0.093323,  0.094819,
             0.096375,  0.097962,  0.099549,  0.101196,
             0.102875,  0.104584,  0.106353,  0.108154,
             0.109985,  0.111876,  0.113890,  0.116025,
             0.118283,  0.120632,  0.123103,  0.125697,
             0.128411,  0.131279,  0.134328,  0.137562,
             0.140977,  0.144665,  0.148718,  0.153197,
             0.158191,  0.163792,  0.170089,  0.177261,
             0.185487,  0.194947,  0.205897,  0.218845,
             0.234551,  0.254225,  0.280567,  0.321781,
        ],
    ]

    /// Reference midpoints (boundaries) for d=128 — matches llama.cpp.
    private static let referenceMidpoints128: [Int: [Float]] = [
        2: [-0.086728, 0.0, 0.086728],
        3: [-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259],
        4: [-0.145560, -0.103361, -0.079142, -0.060009,
            -0.043430, -0.028293, -0.013963,  0.000000,
             0.013963,  0.028293,  0.043430,  0.060009,
             0.079142,  0.103361,  0.145560],
        // 8-bit (255 boundaries between 256 centroids). Each entry is the
        // arithmetic midpoint of adjacent centroids; pre-baked alongside
        // the centroid table to skip the runtime fallback path.
        8: [
            -0.301175, -0.267396, -0.244388, -0.226698,
            -0.212371, -0.200422, -0.190217, -0.181374,
            -0.173675, -0.166941, -0.160992, -0.155694,
            -0.150958, -0.146692, -0.142821, -0.139269,
            -0.135945, -0.132803, -0.129845, -0.127054,
            -0.124400, -0.121868, -0.119458, -0.117154,
            -0.114958, -0.112883, -0.110915, -0.109023,
            -0.107193, -0.105438, -0.103745, -0.102051,
            -0.100357, -0.098709, -0.097123, -0.095582,
            -0.094071, -0.092576, -0.091080, -0.089616,
            -0.088197, -0.086793, -0.085389, -0.084016,
            -0.082688, -0.081361, -0.080033, -0.078736,
            -0.077470, -0.076234, -0.074998, -0.073747,
            -0.072511, -0.071305, -0.070131, -0.068971,
            -0.067811, -0.066651, -0.065492, -0.064347,
            -0.063218, -0.062089, -0.060975, -0.059892,
            -0.058824, -0.057740, -0.056657, -0.055604,
            -0.054566, -0.053529, -0.052491, -0.051454,
            -0.050431, -0.049424, -0.048417, -0.047410,
            -0.046403, -0.045396, -0.044389, -0.043397,
            -0.042420, -0.041444, -0.040482, -0.039536,
            -0.038590, -0.037644, -0.036698, -0.035752,
            -0.034806, -0.033875, -0.032960, -0.032029,
            -0.031083, -0.030152, -0.029236, -0.028321,
            -0.027405, -0.026490, -0.025574, -0.024674,
            -0.023789, -0.022889, -0.021973, -0.021073,
            -0.020188, -0.019303, -0.018418, -0.017517,
            -0.016617, -0.015732, -0.014847, -0.013962,
            -0.013077, -0.012207, -0.011353, -0.010483,
            -0.009598, -0.008713, -0.007828, -0.006958,
            -0.006088, -0.005203, -0.004334, -0.003479,
            -0.002609, -0.001740, -0.000885,  0.000000,
             0.000900,  0.001770,  0.002625,  0.003494,
             0.004379,  0.005249,  0.006119,  0.007004,
             0.007874,  0.008743,  0.009629,  0.010498,
             0.011368,  0.012253,  0.013123,  0.013993,
             0.014878,  0.015763,  0.016648,  0.017533,
             0.018418,  0.019303,  0.020203,  0.021103,
             0.021988,  0.022889,  0.023804,  0.024705,
             0.025590,  0.026490,  0.027405,  0.028321,
             0.029236,  0.030152,  0.031083,  0.032029,
             0.032960,  0.033891,  0.034837,  0.035767,
             0.036698,  0.037659,  0.038621,  0.039567,
             0.040513,  0.041474,  0.042466,  0.043458,
             0.044434,  0.045426,  0.046433,  0.047425,
             0.048417,  0.049424,  0.050431,  0.051454,
             0.052491,  0.053529,  0.054566,  0.055604,
             0.056657,  0.057740,  0.058839,  0.059922,
             0.061006,  0.062120,  0.063249,  0.064378,
             0.065507,  0.066651,  0.067811,  0.068971,
             0.070146,  0.071336,  0.072541,  0.073777,
             0.074998,  0.076204,  0.077455,  0.078752,
             0.080064,  0.081376,  0.082688,  0.084016,
             0.085389,  0.086793,  0.088212,  0.089661,
             0.091126,  0.092591,  0.094071,  0.095597,
             0.097168,  0.098755,  0.100373,  0.102036,
             0.103729,  0.105468,  0.107254,  0.109069,
             0.110931,  0.112883,  0.114958,  0.117154,
             0.119458,  0.121868,  0.124400,  0.127054,
             0.129845,  0.132803,  0.135945,  0.139269,
             0.142821,  0.146692,  0.150958,  0.155694,
             0.160992,  0.166941,  0.173675,  0.181374,
             0.190217,  0.200422,  0.212371,  0.226698,
             0.244388,  0.267396,  0.301174,
        ],
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

    /// Cache of scale → pre-scaled rotation matrices `(rotationT * scale)`.
    /// SDPA's per-step `scale` is constant per layer, so we hit-cache after
    /// the first lookup. Saves one elementwise multiply per decode step.
    private var scaledRotationTCache: [Float: MLXArray] = [:]

    /// Pre-rotate queries with attention scale folded into the rotation
    /// matrix. Equivalent to `prepareQueries(q) * scale` but uses a single
    /// matmul (vs matmul + multiply) — Tom Turney's optimization, PR #93.
    public func prepareQueriesScaled(_ queries: MLXArray, scale: Float) -> MLXArray {
        if let cached = scaledRotationTCache[scale] {
            return matmul(queries, cached)
        }
        let scaled = (rotationT * scale).asType(rotationT.dtype)
        // Force materialization of the scaled rotation matrix before storing
        // it in the cache, so subsequent cache hits matmul against real data
        // instead of a lazy graph node. `asyncEval` (vs sync `eval`) keeps the
        // CPU free to build the matmul graph below — the matmul chains on
        // `scaled` via MLX's dependency tracker, so the GPU work overlaps
        // with the rest of the layer's setup. Matches the codebase pattern in
        // `LLMModel.swift::prepare`, `Qwen35`, `NemotronH`.
        asyncEval(scaled)
        scaledRotationTCache[scale] = scaled
        return matmul(queries, scaled)
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

/// KV cache using TurboQuant (MSE-optimal) compression. Two attention paths:
///
/// **A — default (`useCompressedAttention = false`):** raw-FP16 cache + standard
/// `MLXFast.scaledDotProductAttention(... sinks:)`. The TurboQuant rotation Π
/// is bypassed at decode — SDPA is invariant to a fixed orthogonal rotation
/// applied to both Q and K (and the equivalent rotation around V), so rotating
/// the cache buys nothing for the fused-SDPA path while costing a transition
/// matmul + transient peak (raw + rotated copies + rotated dequant buffer all
/// live during transition) and a per-token rotation matmul. After this fix the
/// prefill→decode boundary keeps `rawKeys`/`rawValues` alive and decode appends
/// to them in place — semantically identical to `KVCacheSimple` while
/// preserving rotating-window and sliding-window semantics. Memory and decode
/// speed match `--kv none`. Sinks flow through SDPA natively, all `(kb,vb)`
/// combos work, no graph-fuser barriers.
///
/// Note: under A the compressed packed buffer is not built — `state` snapshots
/// round-trip only what was already in `rawKeys`/`rawValues`. For mid-decode
/// state preservation backed by compressed storage, opt into B.
///
/// **B — opt-in (`useCompressedAttention = true`):** compressed-domain Metal
/// kernels (`mseScore` + `mseWeightedSum`, or fused `turboFlashAttention`) read
/// the packed buffer directly — no FP16 workspace, memory ≈ `memoryBytes`.
/// Slower than A on alpha today on every model in the cross-bench (50-75%
/// regression vs A on Qwen, Nemotron) — left in place behind the gate for
/// future kernel work and as a compressed-state-snapshot escape hatch. Does
/// not support attention sinks.
///
/// Both K and V use Algorithm 1 (MSE at b bits, no QJL). See file header for
/// algorithmic details.
public class TurboQuantKVCache: BaseKVCache {

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

    /// When true (default), decode attention uses the compressed-domain
    /// fused-dequant + matrix-engine SDPA path — the B path. When false,
    /// decode uses the raw-FP16 cache + standard
    /// `MLXFast.scaledDotProductAttention` — the A path. B is the default
    /// because the dequant+SDPA pipeline (shipped in `c5ca7a3`) closes most
    /// of the historic A/B gap while preserving the compressed cache's
    /// memory savings. A path remains available via:
    ///   - `useCompressedAttention: false` on the constructor, or
    ///   - `TURBO_COMPRESSED_ATTENTION=0` env var (overrides the constructor).
    /// Sinks-using models (GPT-OSS family) auto-fallback to A in
    /// `AttentionUtils.attentionWithCacheUpdate` regardless of this flag,
    /// since the compressed-attention pass2 kernel doesn't yet incorporate
    /// sink-token logits in its online softmax.
    public var useCompressedAttention: Bool = true

    public init(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil,
        step: Int = 1024, seed: UInt64 = 42, maxSize: Int? = nil,
        useCompressedAttention: Bool = true,
        headDim: Int? = nil
    ) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.rawKeyMode = (keyBits ?? bits) == 0
        self.seed = seed
        self.step = step
        self.rotatingMaxSize = maxSize
        // `TURBO_COMPRESSED_ATTENTION` env var explicitly opts in or out of
        // the compressed-attention (B) path, overriding the constructor
        // default. `=0` forces A; `=1` forces B; unset uses the constructor
        // value (B by default). Useful when comparing decode tok/s before
        // and after a codec change without recompiling.
        let envOverride = ProcessInfo.processInfo.environment["TURBO_COMPRESSED_ATTENTION"]
        switch envOverride {
        case "0": self.useCompressedAttention = false
        case "1": self.useCompressedAttention = true
        default:  self.useCompressedAttention = useCompressedAttention
        }
        super.init()
        // Eager codec init when headDim is known. This pre-warms the MLX
        // rotation-matmul kernel JIT during model load instead of paying it
        // inside step(0)'s prefill→decode transition (which is bundled into
        // the bench prefill metric / user-visible TTFT). The codec itself is
        // shared across layers via `getOrCreateCodec`, so 22 layers all hit
        // the cache after the first one constructs and pre-warms.
        if let headDim {
            ensureCodecs(headDim: headDim)
        }
    }

    override public var isTrimmable: Bool { true }

    /// Load raw K/V data from a prefilled cache (e.g., KVCacheSimple or RotatingKVCache).
    /// TurboQuantKVCache will compress these on first decode token.
    /// Keys/values should be shape [B, H, T, D] in temporal order.
    ///
    /// `offset` is the absolute sequence position (used by RoPE on subsequent
    /// decode tokens) — preserved from `originalOffset` even when it exceeds
    /// the buffer length (rotating caches, where the buffer holds only the
    /// most recent windowSize tokens). Internal buffer slicing is gated by
    /// `min(offset, buffer.dim(2))` patterns elsewhere in this class.
    public func loadRawKV(keys: MLXArray, values: MLXArray, originalOffset: Int? = nil) {
        self.rawKeys = keys
        self.rawValues = values
        let bufferTokens = keys.dim(2)
        // Sequence/RoPE position — do NOT clamp to bufferTokens. RoPE on
        // subsequent decode tokens needs the absolute position.
        self.offset = originalOffset ?? bufferTokens
        self.rawAllocSteps = bufferTokens
    }

    // MARK: - Mask override (sliding-window aware)

    /// Sliding-window-aware mask. Mirrors `RotatingKVCache.makeMask` so that
    /// when this cache stands in for a sliding `RotatingKVCache`, the model's
    /// `makeAttentionMask(..., windowSize: ws)` call gets a proper windowed
    /// mask instead of `BaseKVCache`'s fallback (which ignores `windowSize`
    /// for `n > 1` and emits `.causal`).
    ///
    /// Without this override:
    /// - GPT-OSS sliding layers fall back to a non-windowed causal mask after
    ///   prefill→turbo conversion, which is wrong whenever `cappedOffset + n
    ///   > windowSize` (causes attention to read evicted/garbage slots).
    /// - For L=1 decode after the rotating buffer fills, no rolled mask is
    ///   produced — same failure mode `RotatingKVCache.makeMask` guards against.
    override public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            if let maxSz = rotatingMaxSize {
                let actualWindowSize = windowSize ?? maxSz
                let cappedOffset = min(maxSz - 1, offset)
                if cappedOffset + n > actualWindowSize || returnArray {
                    return .array(
                        createCausalMask(
                            n: n, offset: cappedOffset, windowSize: actualWindowSize))
                }
                return .causal
            }
            if returnArray || (windowSize != nil && n > windowSize!) {
                return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
            }
            return .causal
        }
        // n == 1
        guard let windowSize = windowSize, let maxSz = rotatingMaxSize else {
            return .none
        }
        // Rolled mask only needed when windowSize < maxSz (otherwise the buffer
        // and the window match — every cached slot is in-window).
        if offset >= windowSize, maxSz > windowSize {
            var currentIdx = rotatingIdx
            if currentIdx >= maxSz { currentIdx = 0 }
            let maskSize = offset < maxSz ? offset + 1 : maxSz
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            let rolledMask = roll(mask, shift: currentIdx + 1)
            return .array(rolledMask)
        }
        return .none
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
        // Pre-warm MLX kernel JIT for the rotation matmul. Without this, the
        // first call to `prepareQueries` (in updateAndDequant's prefill→decode
        // transition) pays a ~40–80ms JIT cost that lands inside TTFT — the
        // dominant turbo-vs-no-quant prefill gap at small contexts.
        //
        // Warm matmul kernels for shapes the rotation will actually see:
        //   - [B=1, H=*, T=1, D] (per-step decode rotation of one new token)
        //   - [B=1, H=*, T~prefill, D] (one-shot rotation of prefill batch
        //     during the prefill→decode transition)
        // MLX dispatches different specialized matmul kernels based on the
        // batch shape; warming both with eval at codec construction
        // (= model load time) shifts the JIT compilation cost out of TTFT.
        // The exact production shape isn't known at codec construction —
        // codecs are shared across cache instances — so we warm a small
        // representative batch shape that exercises the same kernel family.
        let warmupDecode = matmul(MLXArray.zeros([1, 1, 1, dim], dtype: .bfloat16), codec.rotation)
        let warmupPrefill = matmul(MLXArray.zeros([1, 8, 128, dim], dtype: .bfloat16), codec.rotation)
        eval(warmupDecode, warmupPrefill)
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
    ///
    /// When `rotatingMaxSize` is set (sliding-window layers), the buffer is
    /// trimmed to the most recent `maxSz` tokens after each update — mirroring
    /// `RotatingKVCache.updateConcat` semantics. `offset` still tracks the
    /// absolute sequence position (for RoPE), while the buffer holds at most
    /// `maxSz` temporally-ordered tokens.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let S = keys.dim(2)
        let previous = self.offset

        if let maxSz = rotatingMaxSize {
            // Sliding-window prefill: concat-then-trim. Stays correct across
            // multi-chunk prefill where each chunk extends past the window.
            let combinedK: MLXArray
            let combinedV: MLXArray
            if let rk = rawKeys, let rv = rawValues {
                let bufferTokens = min(previous, rk.dim(2))
                let existingK = rk[.ellipsis, ..<bufferTokens, 0...]
                let existingV = rv[.ellipsis, ..<bufferTokens, 0...]
                combinedK = concatenated([existingK, keys], axis: 2)
                combinedV = concatenated([existingV, values], axis: 2)
            } else {
                combinedK = keys
                combinedV = values
            }
            let total = combinedK.dim(2)
            if total > maxSz {
                rawKeys = combinedK[.ellipsis, (total - maxSz)..., 0...]
                rawValues = combinedV[.ellipsis, (total - maxSz)..., 0...]
            } else {
                rawKeys = combinedK
                rawValues = combinedV
            }
            rawAllocSteps = rawKeys!.dim(2)
            self.offset = previous + S
            return (rawKeys!, rawValues!)
        }

        // Non-rotating path: KVCacheSimple-style growable buffer.
        let reset =
            if let currentKeys = self.rawKeys, (previous + S) > currentKeys.dim(2) {
                true
            } else {
                self.rawKeys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + S - 1) / step
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

        self.offset = previous + S

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
        if ProcessInfo.processInfo.environment["TURBO_DEBUG"] == "1" {
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

    // Kept for `memoryBytes` and `trim`/state references that survive from the
    // original α implementation. A path leaves these nil — decode appends live
    // in `rawKeys`/`rawValues`.
    private var dequantKeys: MLXArray?
    private var dequantValues: MLXArray?

    /// A decode path: append raw K/V to the prefill buffer in place and return
    /// the populated-prefix slices for `MLXFast.scaledDotProductAttention`.
    ///
    /// SDPA is invariant to a fixed orthogonal rotation Π applied to both Q and
    /// K (and the equivalent rotation around V), so the rotation that the
    /// previous α implementation was applying at decode bought nothing for the
    /// fused-SDPA path while costing:
    ///   - a one-shot batch-rotate of the entire raw prefill cache at the
    ///     prefill→decode transition (extra matmul + transient peak: raw +
    ///     rotated copies + rotated dequant buffer all live during transition),
    ///   - a per-token rotation matmul on every decode step.
    ///
    /// This implementation keeps `rawKeys`/`rawValues` alive across the
    /// transition and appends new decode tokens directly into them with
    /// rotating-window or linear (step-aligned grow) semantics — equivalent to
    /// `KVCacheSimple` at decode while preserving the rotating-buffer write
    /// order that sliding-window models need. Memory and decode tok/s match
    /// `--kv none`.
    ///
    /// Returns (keys, values) in original (unrotated) space; `prepareQueries`
    /// and `inverseRotateOutput` are no-ops to match.
    ///
    /// Note: under A no compressed packed buffer is built — `state` snapshots
    /// round-trip whatever is in `rawKeys`/`rawValues`. For mid-decode
    /// compressed-state preservation, opt into B (`useCompressedAttention =
    /// true`).
    public func updateAndDequant(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = newKeys.dim(-1)
        let prevOffset = offset
        let S = newKeys.dim(2)

        if !isCompressed {
            isCompressed = true
            compressedWriteOffset = min(offset, rotatingMaxSize ?? offset)
            if let maxSz = rotatingMaxSize {
                let bufferTokens = min(offset, maxSz)
                rotatingIdx = bufferTokens % maxSz
            }
        }

        offset = prevOffset + S
        // `compressedWriteOffset` advances symbolically so `memoryBytes` reports
        // what compressed storage would cost for the current token count.
        compressedWriteOffset += S

        // Rotating-window: fixed-size raw FP16 buffer with wrap-around writes.
        if let maxSz = rotatingMaxSize {
            if rotatingIdx >= maxSz {
                rotatingIdx = 0
            }
            let writePos = rotatingIdx
            rotatingIdx += S
            let bufferTokens = min(offset, maxSz)

            if self.rawKeys == nil {
                let B = newKeys.dim(0)
                let H = newKeys.dim(1)
                let kShape = [B, H, maxSz, headDim]
                self.rawKeys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
                self.rawValues = MLXArray.zeros(kShape, dtype: newValues.dtype)
            }

            self.rawKeys?[.ellipsis, writePos ..< (writePos + S), 0...] = newKeys
            self.rawValues?[.ellipsis, writePos ..< (writePos + S), 0...] = newValues

            let returnedKeys = self.rawKeys![.ellipsis, ..<bufferTokens, 0...]
            let returnedValues = self.rawValues![.ellipsis, ..<bufferTokens, 0...]
            self.lastReturnedKeys = returnedKeys
            self.lastReturnedValues = returnedValues
            return (returnedKeys, returnedValues)
        }

        // Linear (non-rotating): KVCacheSimple-style step-aligned growth.
        let reset =
            if let rk = self.rawKeys, prevOffset + S > rk.dim(2) {
                true
            } else {
                self.rawKeys == nil
            }
        if reset {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let nSteps = (step + S - 1) / step
            let kShape = [B, H, nSteps * step, headDim]
            let newRK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newRV = MLXArray.zeros(kShape, dtype: newValues.dtype)

            if var currentKeys = self.rawKeys, var currentValues = self.rawValues {
                if prevOffset % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prevOffset, 0...]
                    currentValues = currentValues[.ellipsis, ..<prevOffset, 0...]
                }
                self.rawKeys = concatenated([currentKeys, newRK], axis: 2)
                self.rawValues = concatenated([currentValues, newRV], axis: 2)
            } else {
                self.rawKeys = newRK
                self.rawValues = newRV
            }
            rawAllocSteps = self.rawKeys!.dim(2)
        }

        self.rawKeys?[.ellipsis, prevOffset ..< offset, 0...] = newKeys
        self.rawValues?[.ellipsis, prevOffset ..< offset, 0...] = newValues

        let returnedKeys = self.rawKeys![.ellipsis, ..<offset, 0...]
        let returnedValues = self.rawValues![.ellipsis, ..<offset, 0...]
        self.lastReturnedKeys = returnedKeys
        self.lastReturnedValues = returnedValues
        return (returnedKeys, returnedValues)
    }

    /// Pre-rotate queries.
    /// A path is a no-op — `updateAndDequant` returns raw K/V and SDPA is
    /// invariant to Π applied to both Q and K, so no Q rotation is needed.
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        return queries
    }

    /// Inverse-rotate SDPA output back to original V basis.
    /// A path is a no-op — V is returned in its original basis, so SDPA output
    /// is already in the right space.
    public func inverseRotateOutput(_ rotatedOutput: MLXArray) -> MLXArray {
        return rotatedOutput
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

        // Phase: encode new token into compressed storage.
        // Wrapped in `tqEncode` signpost interval — captured by Instruments
        // / xctrace under `MLX_BENCH_PROFILE=2`. Zero overhead when off.
        let encH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqEncode)
        offset += newKeys.dim(2)
        let savedOffset = offset
        offset = compressedWriteOffset
        encodeNewToken(keys: newKeys, values: newValues)
        compressedWriteOffset = offset
        offset = savedOffset
        BenchmarkSignpost.end(encH)

        guard let valueMSECodec else {
            return queries
        }

        // For rotating caches, token count is capped at maxSize
        let tokenCount = rotatingMaxSize.map { min(offset, $0) } ?? offset

        // Debug: log on first few calls
        if ProcessInfo.processInfo.environment["TURBO_DEBUG"] == "1" {
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
            let scH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqScore)
            var scores = matmul(queries, expandedKeys.transposed(0, 1, 3, 2)) * scale
            BenchmarkSignpost.end(scH)

            // Mask + softmax
            let smH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqSoftmax)
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
            BenchmarkSignpost.end(smH)

            // Metal V kernel: compressed-domain weighted sum
            let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
            let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
            let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
                codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                repeatCount: nRepeats, bits: self.valueBits, dim: headDim
            )
            BenchmarkSignpost.end(vH)

            // Inverse value rotation
            let rH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqRotate)
            output = matmul(
                rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                valueMSECodec.rotation
            )
            BenchmarkSignpost.end(rH)
        } else {
            // ═══ Standard TurboQuant path: both K and V compressed ═══
            guard let keyMSECodec else { return queries }

            // Pre-rotate query for compressed-domain K scoring. Scale is
            // folded into the cached rotation matrix to eliminate the extra
            // elementwise multiply (matmul + multiply → single matmul).
            let rH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqRotate)
            let qRot = keyMSECodec.prepareQueriesScaled(queries, scale: scale)
            let flatQ = qRot.reshaped([B * nQHeads * L, headDim])
            BenchmarkSignpost.end(rH)

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

            // Dequant-first SDPA path (default for decode L=1 when bits ∈
            // {2, 4, 8}). Bulk-dequant K/V to FP16 in rotated codec space
            // via a fused Metal kernel (8 dims/thread bit-unpack + codebook
            // gather + norm scale fused), then call MLXFast SDPA — same
            // matrix-engine kernel A path uses. Costs temporary FP16 K/V
            // (B*H*T*D*2 bytes per layer, freed after SDPA) but skips
            // TurboFlash's per-token bit-unpack inside the score loop.
            //
            // M1 Max sweep (turbo4v2 summarization vs TurboFlash):
            //   Qwen 0.8B 1k/4k/8k/16k/32k: +27 / +44 / +52 / +34 / +14 %
            //   Qwen 9B   1k/4k/8k/16k/32k: + 4 / +14 / +14 /  +9 / +18 %
            //   Nemotron 30B 1k/.../32k:   + 1 / + 7 / + 7 /  +7 / +15 %
            //
            // Override via `TURBO_DEQUANT_SDPA=0` to force TurboFlash for
            // A/B comparison or fallback if a regression is hit on an
            // untested config.
            let dequantEnv = ProcessInfo.processInfo.environment["TURBO_DEQUANT_SDPA"]
            let useDequantSDPA = (dequantEnv != "0")
            if L == 1
                && useDequantSDPA
                && (keyBits == 4 || keyBits == 8 || keyBits == 2)
                && (valueBits == 4 || valueBits == 8 || valueBits == 2) {
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
                // The precompiled `turbo_dequant_rotated` kernel only
                // instantiates `bfloat` and `half` outputs (mlx fork's
                // turbo_quant.metal). Some models (e.g. Gemma 4 26B-A4B)
                // run attention in fp32; clamp to bf16 for the dequant +
                // SDPA + rotation chain, then cast the final output back
                // to the model's dtype so downstream layers see the
                // expected precision.
                let originalDtype = queries.dtype
                let dt: DType = (originalDtype == .bfloat16 || originalDtype == .float16)
                    ? originalDtype : .bfloat16
                let qForSDPA = (qRot.dtype == dt) ? qRot : qRot.asType(dt)
                let kFP = TurboQuantKernelOps.bulkDequantRotated(
                    packed: keyPackedMSE![0..., 0..., ..<tokenCount, 0...],
                    norms: keyNorms![0..., 0..., ..<tokenCount],
                    codebook: keyMSECodec.codebook,
                    tokenCount: tokenCount, bits: keyBits, dim: headDim, dtype: dt)
                let vFP = TurboQuantKernelOps.bulkDequantRotated(
                    packed: valPackedMSE![0..., 0..., ..<tokenCount, 0...],
                    norms: valNorms![0..., 0..., ..<tokenCount],
                    codebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, bits: valueBits, dim: headDim, dtype: dt)
                // qRot already includes scale (prepareQueriesScaled); pass scale=1.0.
                let rotOut = MLXFast.scaledDotProductAttention(
                    queries: qForSDPA.reshaped([B, nQHeads, L, headDim]),
                    keys: kFP, values: vFP,
                    scale: 1.0, mask: .none)
                output = matmul(rotOut, valueMSECodec.rotation)
                if output.dtype != originalDtype {
                    output = output.asType(originalDtype)
                }
                BenchmarkSignpost.end(vH)
            } else if L == 1 && hasTurboFlashKernel {
                // TurboFlashAttention path (decode, L=1) — fuses score + softmax + value.
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
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

                BenchmarkSignpost.end(vH)
            } else if case .causal = mask, hasTurboFlashKernel {
                // Causal TurboFlashAttention path (prefill, L>1)
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
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
                BenchmarkSignpost.end(vH)
            } else {
                // Separated path: Metal score kernel
                let scH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqScore)
                var scores = TurboQuantKernelOps.mseScore(
                    rotatedQueries: flatQ, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.keyBits, dim: headDim
                ).reshaped([B, nQHeads, L, tokenCount])
                BenchmarkSignpost.end(scH)

                // Mask + softmax
                let smH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqSoftmax)
                switch mask {
                case .array(let maskArray):
                    if maskArray.dtype == .bool {
                        scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
                    } else { scores = scores + maskArray }
                case .none: break
                default: break
                }

                let attnWeights = softmax(scores, axis: -1)
                BenchmarkSignpost.end(smH)

                // Metal value kernel
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
                let flatWeights2 = attnWeights.reshaped([B * nQHeads * L, tokenCount])
                let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                    weights: flatWeights2, packed: flatValPacked, norms: flatValNorms,
                    codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.valueBits, dim: headDim
                )
                BenchmarkSignpost.end(vH)

                // Inverse rotation
                let irH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqRotate)
                output = matmul(
                    rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                    valueMSECodec.rotation
                )
                BenchmarkSignpost.end(irH)
            }
        }

        return output
    }

    // MARK: - Memory Reporting

    /// Actual memory footprint: compressed storage (packed indices + norms) for K and V,
    /// plus any raw FP16 buffers if still in prefill phase or rawKeyMode, plus dequant buffers.
    /// Does NOT include codec overhead (rotation matrices, codebooks) which is shared across layers.
    /// In rawKeyMode: `rawKeys` stays alive holding raw FP16 K; `valPackedMSE`/`valNorms` hold V.
    /// In standard mode after compression: `rawKeys`/`rawValues` are nilled out and the only
    /// storage is `keyPackedMSE` / `keyNorms` / `valPackedMSE` / `valNorms`.
    override public var memoryBytes: Int {
        if !isCompressed {
            // Pre-compression: raw FP16 storage
            var total = 0
            if let rk = rawKeys { total += arrayBytes(rk) }
            if let rv = rawValues { total += arrayBytes(rv) }
            return total
        }
        // Resolve [B, H, T, D] from whichever storage is live. `rawKeys` is
        // kept in rawKeyMode and during the first decode call; the packed
        // buffers exist post-compression in both modes.
        // Prefer the packed buffers since they're the authoritative
        // post-compression storage; fall back to `rawKeys` (rawKeyMode) and
        // finally to legacy `dequantKeys` for older state shapes.
        let shapeSrc: MLXArray? = keyPackedMSE ?? valPackedMSE ?? rawKeys ?? dequantKeys
        guard let rk = shapeSrc else { return 0 }
        let tokenCount = compressedWriteOffset
        guard tokenCount > 0 else { return 0 }
        let B = rk.dim(0)
        let H = rk.dim(1)
        // `keyPackedMSE` / `valPackedMSE` last dim is PackedWidth, not D.
        // Recover D from `rawKeys` when present (rawKeyMode), else infer
        // from the codec's `dim` property (passed in via the layer's K/V).
        // Conservatively use `keyMSECodec?.dim` / `valueMSECodec?.dim`.
        let D: Int = {
            if let rk = rawKeys { return rk.dim(3) }
            if let kc = keyMSECodec { return kc.dim }
            if let vc = valueMSECodec { return vc.dim }
            return rk.dim(3)
        }()
        var total = 0
        if rawKeyMode {
            // K stays fp16 in `rawKeys` (allocated to maxSize / step-aligned).
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
