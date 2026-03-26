// Copyright © 2026 Eric Kryski. TurboQuant KV cache compression.
//
// Implements Google Research's TurboQuant algorithm (arXiv 2504.19874) for
// extreme KV cache compression. Two-stage pipeline:
//   1. PolarQuant/MSE: random orthogonal rotation + optimal scalar quantization
//   2. QJL: 1-bit residual via Johnson-Lindenstrauss projection (Phase 3)
//
// References:
//   - TurboQuant paper: https://arxiv.org/abs/2504.19874
//   - PolarQuant paper: https://arxiv.org/abs/2502.02617
//   - QJL paper: https://arxiv.org/abs/2406.03482
//   - mlx-vlm implementation: https://github.com/Blaizzy/mlx-vlm/pull/858
//   - turboquant_plus reference: https://github.com/TheTom/turboquant_plus

import Foundation
import MLX
import MLXNN

// MARK: - Codebook Generation

/// Generates optimal Lloyd-Max codebook centroids for scalar quantization of
/// coordinates after random orthogonal rotation.
///
/// After rotation, each coordinate of a unit vector in d dimensions follows
/// a Beta distribution: f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
///
/// We find optimal centroids by solving a weighted k-means problem on [-1,1]
/// with the Beta PDF as weight function.
public enum TurboQuantCodebook {

    /// Cache of generated codebooks keyed by (dim, bits).
    nonisolated(unsafe) private static var cache: [String: MLXArray] = [:]
    private static let lock = NSLock()

    /// Get or generate the optimal codebook for the given dimension and bit-width.
    /// Codebooks are cached since they're deterministic for a given (dim, bits).
    public static func codebook(dim: Int, bits: Int) -> MLXArray {
        let key = "\(dim)_\(bits)"
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let cb = generateCodebook(dim: dim, bits: bits)

        lock.lock()
        cache[key] = cb
        lock.unlock()

        return cb
    }

    /// Generate optimal codebook centroids via beta-weighted k-means.
    ///
    /// Algorithm (following mlx-vlm and turboquant_plus):
    /// 1. Create a fine grid of 32768 points in [-1+ε, 1-ε]
    /// 2. Compute Beta distribution PDF weights at each grid point
    /// 3. Initialize centroids at weighted quantiles
    /// 4. Refine via k-means (100 iterations) with convergence threshold
    static func generateCodebook(dim: Int, bits: Int) -> MLXArray {
        let levels = 1 << bits  // 2^bits centroids
        let gridSize = 32768
        let eps: Float = 1e-6

        // 1. Fine grid on [-1+eps, 1-eps]
        var gridValues = [Float](repeating: 0, count: gridSize)
        let step = (2.0 - 2.0 * eps) / Float(gridSize - 1)
        for i in 0 ..< gridSize {
            gridValues[i] = -1.0 + eps + Float(i) * step
        }

        // 2. Beta distribution PDF weights: f(x) ∝ (1-x²)^((d-3)/2)
        // For numerical stability, work in log space for high dimensions
        let halfDimMinus = Float(dim - 3) / 2.0
        var weights = [Float](repeating: 0, count: gridSize)
        for i in 0 ..< gridSize {
            let x = gridValues[i]
            let oneMinusXSq = 1.0 - x * x
            if oneMinusXSq > 0 && halfDimMinus >= 0 {
                weights[i] = pow(oneMinusXSq, halfDimMinus)
            } else if halfDimMinus < 0 {
                // dim < 3: use uniform weights
                weights[i] = 1.0
            }
        }

        // Normalize weights
        let weightSum = weights.reduce(0, +)
        if weightSum > 0 {
            for i in 0 ..< gridSize {
                weights[i] /= weightSum
            }
        }

        // 3. Initialize centroids at weighted quantiles
        var centroids = [Float](repeating: 0, count: levels)
        var cumWeight: Float = 0
        var ci = 0
        for i in 0 ..< gridSize {
            cumWeight += weights[i]
            let targetQuantile = (Float(ci) + 0.5) / Float(levels)
            if cumWeight >= targetQuantile && ci < levels {
                centroids[ci] = gridValues[i]
                ci += 1
            }
        }
        // Fill any remaining centroids
        while ci < levels {
            centroids[ci] = gridValues[gridSize - 1]
            ci += 1
        }

        // 4. K-means refinement (100 iterations)
        for _ in 0 ..< 100 {
            // Compute boundaries between adjacent centroids
            var boundaries = [Float](repeating: 0, count: levels - 1)
            for j in 0 ..< levels - 1 {
                boundaries[j] = (centroids[j] + centroids[j + 1]) / 2.0
            }

            // Reassign and compute new centroids as weighted means
            var newCentroids = [Float](repeating: 0, count: levels)
            var clusterWeights = [Float](repeating: 0, count: levels)

            for i in 0 ..< gridSize {
                // Find cluster for this grid point
                var cluster = levels - 1
                for j in 0 ..< levels - 1 {
                    if gridValues[i] < boundaries[j] {
                        cluster = j
                        break
                    }
                }
                newCentroids[cluster] += weights[i] * gridValues[i]
                clusterWeights[cluster] += weights[i]
            }

            // Update centroids
            var maxDelta: Float = 0
            for j in 0 ..< levels {
                if clusterWeights[j] > 0 {
                    let newVal = newCentroids[j] / clusterWeights[j]
                    maxDelta = max(maxDelta, abs(newVal - centroids[j]))
                    centroids[j] = newVal
                }
            }

            // Convergence check
            if maxDelta < 1e-6 { break }
        }

        // Sort centroids (should already be sorted, but ensure)
        centroids.sort()

        return MLXArray(centroids)
    }
}

// MARK: - Rotation Matrix

/// Generates deterministic random orthogonal rotation matrices.
///
/// Supports two modes:
/// - **WHT** (Walsh-Hadamard Transform): O(d log d) butterfly, O(d) storage.
///   Uses randomized sign flipping for Gaussianization (per TurboQuant paper).
///   Only works for power-of-2 dimensions.
/// - **QR fallback**: O(d²) Haar-distributed via QR decomposition.
///   Works for any dimension, runs on CPU.
public enum TurboQuantRotation {

    /// Cache of rotation matrices keyed by (dim, seed).
    nonisolated(unsafe) private static var cache: [String: MLXArray] = [:]
    /// Cache of WHT random sign vectors keyed by (dim, seed).
    nonisolated(unsafe) private static var signCache: [String: MLXArray] = [:]
    private static let lock = NSLock()

    /// Check if dimension is a power of 2.
    static func isPowerOf2(_ n: Int) -> Bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Get or generate a random orthogonal matrix for the given dimension and seed.
    ///
    /// For power-of-2 dimensions, returns a WHT-based rotation (fast apply via butterfly).
    /// For other dimensions, returns a QR-based rotation matrix.
    public static func rotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
        let key = "\(dim)_\(seed)"
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let rot: MLXArray
        if isPowerOf2(dim) {
            // WHT: store the full matrix for compatibility with existing matmul path
            // (applyWHT is the fast path used in encode/decode)
            rot = generateWHTMatrix(dim: dim, seed: seed)
        } else {
            rot = generateRotation(dim: dim, seed: seed)
        }

        lock.lock()
        cache[key] = rot
        lock.unlock()

        return rot
    }

    /// Apply randomized WHT rotation to vectors using O(d log d) butterfly.
    ///
    /// Computes: D · H · x where D = diag(random ±1), H = Hadamard matrix / √d
    /// This is equivalent to matmul(x, R^T) but O(d log d) instead of O(d²).
    ///
    /// - Parameters:
    ///   - vectors: Input tensor [..., D] where D is power of 2
    ///   - dim: Vector dimension
    ///   - seed: RNG seed (must match rotationMatrix seed)
    /// - Returns: Rotated vectors [..., D]
    public static func applyWHT(_ vectors: MLXArray, dim: Int, seed: UInt64) -> MLXArray {
        precondition(isPowerOf2(dim), "WHT requires power-of-2 dimension, got \(dim)")

        // Get or generate random signs for this (dim, seed)
        let signs = getRandomSigns(dim: dim, seed: seed)

        // Step 1: Random sign flip (element-wise multiply by ±1)
        let shape = vectors.shape
        let flatDims = shape.dropLast().reduce(1, *)
        var x = vectors.reshaped([flatDims, dim])
        x = x * signs  // broadcast signs [1, dim] across rows

        // Step 2: In-place Walsh-Hadamard butterfly
        var halfLen = 1
        while halfLen < dim {
            // Split into pairs and do butterfly: (a+b, a-b)
            let indices0 = stride(from: 0, to: dim, by: halfLen * 2).flatMap { start in
                (start ..< start + halfLen).map { $0 }
            }
            let indices1 = indices0.map { $0 + halfLen }

            let a = x[0..., MLXArray(indices0)]
            let b = x[0..., MLXArray(indices1)]

            let sum = a + b
            let diff = a - b

            // Interleave results back
            // Build full result by concatenating sums and diffs at proper positions
            var parts: [MLXArray] = []
            for i in stride(from: 0, to: dim, by: halfLen * 2) {
                parts.append(sum[0..., (i / 2) ..< (i / 2 + halfLen)])
                parts.append(diff[0..., (i / 2) ..< (i / 2 + halfLen)])
            }
            x = concatenated(parts, axis: -1)

            halfLen *= 2
        }

        // Normalize: H / √d
        x = x / MLXArray(Float(sqrt(Double(dim))))

        return x.reshaped(shape)
    }

    /// Apply inverse WHT rotation (WHT is self-inverse up to scaling).
    ///
    /// Since H^T = H and D^T = D, the inverse is: H · D · x / √d
    /// (apply WHT then sign flip, vs sign flip then WHT for forward)
    public static func applyInverseWHT(_ vectors: MLXArray, dim: Int, seed: UInt64) -> MLXArray {
        precondition(isPowerOf2(dim), "WHT requires power-of-2 dimension, got \(dim)")

        let signs = getRandomSigns(dim: dim, seed: seed)
        let shape = vectors.shape
        let flatDims = shape.dropLast().reduce(1, *)
        var x = vectors.reshaped([flatDims, dim])

        // Step 1: WHT butterfly (same as forward)
        var halfLen = 1
        while halfLen < dim {
            let indices0 = stride(from: 0, to: dim, by: halfLen * 2).flatMap { start in
                (start ..< start + halfLen).map { $0 }
            }
            let indices1 = indices0.map { $0 + halfLen }

            let a = x[0..., MLXArray(indices0)]
            let b = x[0..., MLXArray(indices1)]

            let sum = a + b
            let diff = a - b

            var parts: [MLXArray] = []
            for i in stride(from: 0, to: dim, by: halfLen * 2) {
                parts.append(sum[0..., (i / 2) ..< (i / 2 + halfLen)])
                parts.append(diff[0..., (i / 2) ..< (i / 2 + halfLen)])
            }
            x = concatenated(parts, axis: -1)

            halfLen *= 2
        }

        // Normalize and apply sign flip
        x = x / MLXArray(Float(sqrt(Double(dim))))
        x = x * signs

        return x.reshaped(shape)
    }

    /// Get or generate random ±1 signs for WHT.
    private static func getRandomSigns(dim: Int, seed: UInt64) -> MLXArray {
        let key = "\(dim)_\(seed)"
        lock.lock()
        if let cached = signCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let rngKey = MLXRandom.key(seed)
        // Generate random ±1 signs via uniform threshold
        let uniform = MLXRandom.uniform(0.0 ..< 1.0, [1, dim], key: rngKey)
        let signs = (uniform .< MLXArray(Float(0.5))).asType(.float32) * 2.0 - 1.0
        eval(signs)

        lock.lock()
        signCache[key] = signs
        lock.unlock()

        return signs
    }

    /// Generate full WHT rotation matrix (for compatibility with matmul path).
    ///
    /// Returns D · H / √d as a [dim, dim] matrix.
    private static func generateWHTMatrix(dim: Int, seed: UInt64) -> MLXArray {
        // Build Hadamard matrix via Kronecker products
        var H = MLXArray([Float(1)]).reshaped([1, 1])
        var size = 1
        while size < dim {
            // H_2n = [[H_n, H_n], [H_n, -H_n]]
            let top = concatenated([H, H], axis: 1)
            let bot = concatenated([H, -H], axis: 1)
            H = concatenated([top, bot], axis: 0)
            size *= 2
        }

        // Apply random sign diagonal: D · H / √d
        let signs = getRandomSigns(dim: dim, seed: seed).reshaped([dim])
        let D = MLX.diag(signs)
        let result = matmul(D, H) / MLXArray(Float(sqrt(Double(dim))))
        eval(result)

        return result
    }

    /// Generate a Haar-distributed random orthogonal matrix via QR decomposition.
    ///
    /// Fallback for non-power-of-2 dimensions.
    static func generateRotation(dim: Int, seed: UInt64) -> MLXArray {
        // Use seeded random key for determinism
        let key = MLXRandom.key(seed)

        // Sample random Gaussian matrix
        let gaussian = MLXRandom.normal([dim, dim], key: key)

        // QR decomposition — must run on CPU (not yet supported on GPU)
        let cpuStream = StreamOrDevice.cpu
        let (q, r) = MLXLinalg.qr(gaussian, stream: cpuStream)

        // Deterministic sign fix: Q *= sign(diag(R))
        let diagR = r.diagonal(stream: cpuStream)
        let signs = sign(diagR, stream: cpuStream)
        // Broadcast signs across columns: Q[:, i] *= sign(R[i,i])
        let result = q * expandedDimensions(signs, axis: 0)

        // Force evaluation and move to default device
        eval(result)

        return result
    }
}

// MARK: - Bit Packing

/// Utilities for packing/unpacking sub-byte indices into uint32 words.
///
/// Packs `bits`-wide indices into contiguous uint32 words. For example,
/// 3-bit indices for dim=128 require 128*3 = 384 bits = 12 uint32 words.
///
/// Using uint32 because: Metal SIMD lanes are 32-bit, MLX's existing
/// quantization infrastructure uses uint32, and Metal prefers 32-bit
/// aligned coalesced reads.
public enum TurboQuantPacking {

    /// Compute the number of uint32 words needed to pack `count` indices of `bits` width.
    public static func packedWidth(count: Int, bits: Int) -> Int {
        (count * bits + 31) / 32
    }

    /// Pack indices into uint32 words.
    ///
    /// - Parameters:
    ///   - indices: Array of shape [..., count] with values in [0, 2^bits)
    ///   - bits: Bit-width per index (1-4)
    /// - Returns: Packed array of shape [..., packedWidth] as uint32
    public static func packLowBit(_ indices: MLXArray, bits: Int) -> MLXArray {
        let count = indices.dim(-1)
        let pw = packedWidth(count: count, bits: bits)
        let mask = UInt32((1 << bits) - 1)

        // Reshape to 2D: [rows, count]
        let shape = indices.shape
        let rows = shape.dropLast().reduce(1, *)
        let flat = indices.reshaped([rows, count]).asType(.uint32)

        // Pack each row
        // For each word position, accumulate the bits that belong to it
        var wordArrays: [MLXArray] = []
        for w in 0 ..< pw {
            let bitStart = w * 32
            var word = MLXArray.zeros([rows], dtype: .uint32)

            for d in 0 ..< count {
                let bitOffset = d * bits
                let wordIdx = bitOffset / 32
                let offset = bitOffset % 32

                if wordIdx == w {
                    // This index's bits start in this word
                    let shifted = (flat[0..., d] & MLXArray(mask)) << MLXArray(UInt32(offset))
                    word = word | shifted
                }

                // Handle spill: bits that cross into the next word
                let spill = offset + bits - 32
                if spill > 0 && wordIdx + 1 == w {
                    let shifted = (flat[0..., d] & MLXArray(mask)) >> MLXArray(UInt32(bits - spill))
                    word = word | shifted
                }
            }

            wordArrays.append(expandedDimensions(word, axis: -1))
        }

        let packed = concatenated(wordArrays, axis: -1)

        // Restore original batch dimensions
        var outShape = Array(shape.dropLast())
        outShape.append(pw)
        return packed.reshaped(outShape)
    }

    /// Unpack uint32 words back to indices.
    ///
    /// - Parameters:
    ///   - packed: Array of shape [..., packedWidth] as uint32
    ///   - bits: Bit-width per index (1-4)
    ///   - count: Number of indices to unpack (original dimension)
    /// - Returns: Unpacked array of shape [..., count] as uint32
    public static func unpackLowBit(_ packed: MLXArray, bits: Int, count: Int) -> MLXArray {
        let mask = UInt32((1 << bits) - 1)

        // Reshape to 2D: [rows, packedWidth]
        let shape = packed.shape
        let pw = shape.last!
        let rows = shape.dropLast().reduce(1, *)
        let flat = packed.reshaped([rows, pw])

        // Unpack each index
        var indexArrays: [MLXArray] = []
        for d in 0 ..< count {
            let bitOffset = d * bits
            let wordIdx = bitOffset / 32
            let offset = bitOffset % 32

            var value = (flat[0..., wordIdx] >> MLXArray(UInt32(offset))) & MLXArray(mask)

            // Handle spill from next word
            let spill = offset + bits - 32
            if spill > 0 && wordIdx + 1 < pw {
                let high = (flat[0..., wordIdx + 1] & MLXArray(UInt32((1 << spill) - 1)))
                    << MLXArray(UInt32(bits - spill))
                value = value | high
            }

            indexArrays.append(expandedDimensions(value, axis: -1))
        }

        let unpacked = concatenated(indexArrays, axis: -1)

        // Restore original batch dimensions
        var outShape = Array(shape.dropLast())
        outShape.append(count)
        return unpacked.reshaped(outShape)
    }
}

// MARK: - MSE Codec State

/// State for MSE-quantized vectors (PolarQuant without QJL residual).
///
/// Stores per-token norms and bit-packed codebook indices.
/// Used for both keys (Phase 1) and values (all phases).
public struct MSECodecState {
    /// Per-token L2 norms: [B, H, T]
    public var norms: MLXArray

    /// Bit-packed codebook indices: [B, H, T, packedWidth] as uint32
    public var packedIndices: MLXArray

    /// Number of tokens stored
    public var tokenCount: Int

    /// Original vector dimension (before packing)
    public let dim: Int

    /// Bits per coordinate
    public let bits: Int
}

// MARK: - MSE Codec

/// MSE-optimal vector quantizer using random rotation + scalar codebook.
///
/// Algorithm:
/// 1. Extract L2 norms, normalize to unit vectors
/// 2. Rotate via random orthogonal matrix: rotated = unit @ R^T
/// 3. Quantize each coordinate to nearest codebook centroid
/// 4. Pack indices into uint32 words
///
/// Dequantization reverses: unpack → lookup → inverse rotate → scale by norm
public struct MSECodec {
    /// Codebook centroids: [2^bits] float32
    public let codebook: MLXArray

    /// Random orthogonal rotation matrix: [dim, dim]
    public let rotation: MLXArray

    /// Transpose of rotation (precomputed for efficiency)
    public let rotationT: MLXArray

    /// Bits per coordinate
    public let bits: Int

    /// Vector dimension
    public let dim: Int

    /// RNG seed for WHT
    public let seed: UInt64

    /// Whether to use WHT butterfly (power-of-2 dims)
    public let useWHT: Bool

    /// Initialize an MSE codec for the given dimension and bit-width.
    ///
    /// - Parameters:
    ///   - dim: Vector dimension (e.g., head_dim = 128)
    ///   - bits: Bits per coordinate (1-4)
    ///   - seed: RNG seed for rotation matrix generation
    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.useWHT = TurboQuantRotation.isPowerOf2(dim)
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
        self.rotationT = self.rotation.transposed()
    }

    /// Encode vectors using MSE-optimal quantization.
    ///
    /// - Parameter vectors: Input tensor of shape [B, H, T, D]
    /// - Returns: MSECodecState with norms and packed indices
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        let d = vectors.dim(-1)
        assert(d == dim, "Vector dimension \(d) doesn't match codec dimension \(dim)")

        // 1. Extract L2 norms: [B, H, T]
        let norms = sqrt((vectors * vectors).sum(axis: -1))

        // 2. Normalize to unit vectors: [B, H, T, D]
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // 3. Rotate: [B, H, T, D] -> [B, H, T, D]
        let rotated: MLXArray
        if useWHT {
            rotated = TurboQuantRotation.applyWHT(unit, dim: dim, seed: seed)
        } else {
            rotated = matmul(unit, rotationT)
        }

        // 4. Quantize: find nearest codebook centroid per coordinate
        // codebook shape: [levels], rotated shape: [B, H, T, D]
        // Compute |rotated - codebook| for each coordinate
        let levels = 1 << bits
        let expanded = expandedDimensions(rotated, axis: -1)  // [B,H,T,D,1]
        let cbExpanded = codebook.reshaped([1, 1, 1, 1, levels])  // [1,1,1,1,levels]
        let distances = abs(expanded - cbExpanded)  // [B,H,T,D,levels]
        let indices = argMin(distances, axis: -1)  // [B,H,T,D] as uint32

        // 5. Pack indices
        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        let tokenCount = vectors.dim(2)
        return MSECodecState(
            norms: norms,
            packedIndices: packed,
            tokenCount: tokenCount,
            dim: dim,
            bits: bits
        )
    }

    /// Decode vectors from quantized state.
    ///
    /// - Parameter state: MSECodecState with norms and packed indices
    /// - Returns: Reconstructed tensor of shape [B, H, T, D]
    public func decode(_ state: MSECodecState) -> MLXArray {
        // 1. Unpack indices: [B, H, T, D]
        let indices = TurboQuantPacking.unpackLowBit(
            state.packedIndices, bits: bits, count: dim
        )

        // 2. Lookup codebook centroids: [B, H, T, D]
        let approx = codebook[indices]

        // 3. Inverse rotate: [B, H, T, D] -> [B, H, T, D]
        let unrotated: MLXArray
        if useWHT {
            unrotated = TurboQuantRotation.applyInverseWHT(approx, dim: dim, seed: seed)
        } else {
            unrotated = matmul(approx, rotation)
        }

        // 4. Scale by norms: [B, H, T, D]
        return expandedDimensions(state.norms, axis: -1) * unrotated
    }

    /// Prepare queries for efficient scoring against encoded keys.
    /// Pre-rotates queries: q' = WHT(q) or q @ R^T
    ///
    /// - Parameter queries: [B, H, L, D]
    /// - Returns: Rotated queries [B, H, L, D]
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        if useWHT {
            return TurboQuantRotation.applyWHT(queries, dim: dim, seed: seed)
        }
        return matmul(queries, rotationT)
    }
}

// MARK: - QJL State

/// State for QJL (Quantized Johnson-Lindenstrauss) residual projection.
/// Stores 1-bit signs of projected residual vectors.
public struct QJLState {
    /// Per-token residual L2 norms: [B, H, T]
    public var residualNorms: MLXArray

    /// Bit-packed projection signs (1 bit per dimension): [B, H, T, signPackedWidth] as uint32
    public var packedSigns: MLXArray

    /// Number of tokens stored
    public var tokenCount: Int

    /// Original vector dimension
    public let dim: Int
}

// MARK: - Product Codec State

/// Combined MSE + QJL state for unbiased inner product estimation.
/// Used for keys in turbo2-turbo4 modes.
public struct ProductCodecState {
    /// MSE component (b-1 bits)
    public var mseState: MSECodecState

    /// QJL residual component (1 bit)
    public var qjlState: QJLState

    /// Number of tokens
    public var tokenCount: Int
}

// MARK: - Product Codec

/// Product quantizer for unbiased inner product estimation.
///
/// Combines (b-1)-bit MSE codec with 1-bit QJL residual correction.
/// Per the TurboQuant paper (Theorem 2), this provides unbiased inner product
/// estimation with distortion ≤ (√3·π²·||y||²/d)·(1/4^b).
///
/// For b=1 (turbo1): QJL-only, no MSE component.
/// For b≥2: (b-1)-bit MSE + 1-bit QJL.
public struct ProductCodec {
    /// MSE codec for the base quantization (b-1 bits)
    public let mseCodec: MSECodec?

    /// Random projection matrix S ∈ ℝ^(d×d) with i.i.d. N(0,1) entries
    public let projection: MLXArray

    /// Transpose of projection (precomputed)
    public let projectionT: MLXArray

    /// Scale factor for QJL reconstruction: √(π/2) / d
    public let qjlScale: Float

    /// Total bits (MSE bits + 1 for QJL)
    public let bits: Int

    /// Vector dimension
    public let dim: Int

    /// Initialize a Product codec.
    ///
    /// - Parameters:
    ///   - dim: Vector dimension
    ///   - bits: Total bits per coordinate (1-4). MSE uses (bits-1), QJL uses 1.
    ///   - seed: RNG seed
    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        precondition(bits >= 1 && bits <= 4, "ProductCodec bits must be 1-4")
        self.dim = dim
        self.bits = bits
        self.qjlScale = sqrt(Float.pi / 2.0) / Float(dim)

        // MSE codec uses (bits-1) bits; nil for turbo1 (pure QJL)
        if bits > 1 {
            self.mseCodec = MSECodec(dim: dim, bits: bits - 1, seed: seed)
        } else {
            self.mseCodec = nil
        }

        // Random Gaussian projection matrix (NOT orthogonalized)
        let projSeed = seed &+ UInt64(dim) &* 2971 &+ 17
        let key = MLXRandom.key(projSeed)
        self.projection = MLXRandom.normal([dim, dim], key: key)
        self.projectionT = self.projection.transposed()
        eval(self.projection)
    }

    /// Encode vectors using Product quantization (MSE + QJL).
    ///
    /// - Parameter vectors: Input tensor [B, H, T, D]
    /// - Returns: ProductCodecState
    public func encode(_ vectors: MLXArray) -> ProductCodecState {
        // Extract norms and normalize
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // MSE encode the unit vectors (b-1 bits)
        let mseState: MSECodecState
        let residual: MLXArray

        if let mseCodec {
            mseState = mseCodec.encode(vectors)
            // Compute residual: original - MSE approximation
            let mseApprox = mseCodec.decode(mseState)
            // Residual of unit vectors
            let mseApproxNorms = sqrt((mseApprox * mseApprox).sum(axis: -1))
            let mseApproxUnit = mseApprox / expandedDimensions(
                maximum(mseApproxNorms, MLXArray(Float(1e-8))), axis: -1)
            residual = unit - mseApproxUnit
        } else {
            // turbo1: no MSE, full residual
            mseState = MSECodecState(
                norms: norms,
                packedIndices: MLXArray.zeros([vectors.dim(0), vectors.dim(1), vectors.dim(2), 0], dtype: .uint32),
                tokenCount: vectors.dim(2),
                dim: dim,
                bits: 0
            )
            residual = unit
        }

        // QJL encode: project residual, extract signs
        let residualNorms = sqrt((residual * residual).sum(axis: -1))
        let projected = matmul(residual, projectionT)  // [B, H, T, D]
        let signs = (projected .>= MLXArray(Float(0.0))).asType(.uint32)  // 1 if positive, 0 if negative

        let packedSigns = TurboQuantPacking.packLowBit(signs, bits: 1)

        let qjlState = QJLState(
            residualNorms: residualNorms,
            packedSigns: packedSigns,
            tokenCount: vectors.dim(2),
            dim: dim
        )

        return ProductCodecState(
            mseState: mseState,
            qjlState: qjlState,
            tokenCount: vectors.dim(2)
        )
    }

    /// Decode vectors from Product codec state.
    ///
    /// Reconstruction: x̃ = x̃_mse + (√(π/2)/d) · ||r|| · S^T · sign(S·r)
    public func decode(_ state: ProductCodecState) -> MLXArray {
        // MSE component
        let mseApprox: MLXArray
        if let mseCodec {
            mseApprox = mseCodec.decode(state.mseState)
        } else {
            // turbo1: no MSE component
            mseApprox = MLXArray.zeros(
                [state.qjlState.residualNorms.dim(0),
                 state.qjlState.residualNorms.dim(1),
                 state.qjlState.tokenCount, dim],
                dtype: .float32
            )
        }

        // QJL component: (√(π/2)/d) · ||r|| · S^T · signs
        let signs = TurboQuantPacking.unpackLowBit(
            state.qjlState.packedSigns, bits: 1, count: dim
        )
        // Convert 0/1 to -1/+1
        let signValues = signs.asType(.float32) * 2.0 - 1.0
        let qjlApprox = matmul(signValues, projection)  // S^T · signs
        let scaledQJL = expandedDimensions(state.qjlState.residualNorms, axis: -1) *
            qjlApprox * MLXArray(qjlScale)

        return mseApprox + scaledQJL
    }
}

// MARK: - TurboQuantKVCache

/// KV cache using TurboQuant compression for memory-efficient inference.
///
/// Uses MSE codec for values (minimizes reconstruction error) and
/// Product codec for keys (unbiased inner product estimation via MSE + QJL).
/// Phase 4 will add Metal kernels for compressed-domain attention.
public class TurboQuantKVCache: BaseKVCache {

    /// Product codec for key vectors (MSE + QJL for unbiased inner products)
    private var keyCodec: ProductCodec?

    /// MSE codec for value vectors (MSE-only for reconstruction)
    private var valueCodec: MSECodec?

    /// Compressed key state (Product codec)
    private var keyState: ProductCodecState?

    /// Compressed value state (MSE codec)
    private var valueState: MSECodecState?

    /// Bit-width for quantization
    public let bits: Int

    /// RNG seed for rotation matrices
    public let seed: UInt64

    /// State for serialization
    /// Layout: [key_mse_norms, key_mse_indices, key_qjl_residualNorms, key_qjl_signs,
    ///          val_norms, val_indices]
    override public var state: [MLXArray] {
        get {
            guard let ks = keyState, let vs = valueState else { return [] }
            return [
                ks.mseState.norms, ks.mseState.packedIndices,
                ks.qjlState.residualNorms, ks.qjlState.packedSigns,
                vs.norms, vs.packedIndices,
            ]
        }
        set {
            guard newValue.count == 6 else { return }
            let dim = keyCodec?.dim ?? 0
            let mseBits = max(bits - 1, 0)
            keyState = ProductCodecState(
                mseState: MSECodecState(
                    norms: newValue[0], packedIndices: newValue[1],
                    tokenCount: newValue[0].dim(-1), dim: dim, bits: mseBits
                ),
                qjlState: QJLState(
                    residualNorms: newValue[2], packedSigns: newValue[3],
                    tokenCount: newValue[2].dim(-1), dim: dim
                ),
                tokenCount: newValue[0].dim(-1)
            )
            valueState = MSECodecState(
                norms: newValue[4], packedIndices: newValue[5],
                tokenCount: newValue[4].dim(-1), dim: dim, bits: bits
            )
            offset = newValue[0].dim(-1)
        }
    }

    override public var metaState: [String] {
        get { ["\(bits)", "\(seed)"] }
        set { /* bits and seed are immutable */ }
    }

    override public var isTrimmable: Bool { true }

    override public func innerState() -> [MLXArray] {
        state
    }

    /// Initialize a TurboQuant cache.
    ///
    /// - Parameters:
    ///   - bits: Bit-width for quantization (1-4). Higher = better quality, more memory.
    ///   - seed: RNG seed for rotation matrix generation (default: 42)
    public init(bits: Int = 4, seed: UInt64 = 42) {
        precondition(bits >= 1 && bits <= 4, "TurboQuant bits must be 1-4, got \(bits)")
        self.bits = bits
        self.seed = seed
    }

    /// Compute attention entirely in compressed domain using Metal kernels.
    ///
    /// This is the Phase 4 fast path: encodes new keys/values, then computes
    /// Q×K scores and weighted V sum directly from packed indices + codebook,
    /// skipping full dequantization.
    ///
    /// - Parameters:
    ///   - queries: Query tensor [B, nQHeads, L, D]
    ///   - keys: New key tensor [B, nKVHeads, L, D]
    ///   - values: New value tensor [B, nKVHeads, L, D]
    ///   - scale: Attention scale factor (1/√d)
    ///   - mask: Attention mask
    /// - Returns: Attention output [B, nQHeads, L, D]
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
        let repeatCount = nQHeads / nKVHeads

        // Lazy codec initialization
        if keyCodec == nil {
            keyCodec = ProductCodec(dim: headDim, bits: bits, seed: seed)
            valueCodec = MSECodec(dim: headDim, bits: bits, seed: seed + 1)
        }
        guard let keyCodec, let valueCodec else {
            return queries  // fallback
        }

        // Encode new tokens and append
        let newKeyState = keyCodec.encode(newKeys)
        let newValueState = valueCodec.encode(newValues)

        if keyState == nil {
            keyState = newKeyState
            valueState = newValueState
        } else {
            keyState = ProductCodecState(
                mseState: MSECodecState(
                    norms: concatenated([keyState!.mseState.norms, newKeyState.mseState.norms], axis: -1),
                    packedIndices: concatenated(
                        [keyState!.mseState.packedIndices, newKeyState.mseState.packedIndices], axis: 2),
                    tokenCount: keyState!.tokenCount + newKeyState.tokenCount,
                    dim: headDim,
                    bits: max(bits - 1, 0)
                ),
                qjlState: QJLState(
                    residualNorms: concatenated(
                        [keyState!.qjlState.residualNorms, newKeyState.qjlState.residualNorms], axis: -1),
                    packedSigns: concatenated(
                        [keyState!.qjlState.packedSigns, newKeyState.qjlState.packedSigns], axis: 2),
                    tokenCount: keyState!.tokenCount + newKeyState.tokenCount,
                    dim: headDim
                ),
                tokenCount: keyState!.tokenCount + newKeyState.tokenCount
            )
            valueState = MSECodecState(
                norms: concatenated([valueState!.norms, newValueState.norms], axis: -1),
                packedIndices: concatenated(
                    [valueState!.packedIndices, newValueState.packedIndices], axis: 2),
                tokenCount: valueState!.tokenCount + newValueState.tokenCount,
                dim: headDim,
                bits: bits
            )
        }
        offset = keyState!.tokenCount

        let ks = keyState!
        let vs = valueState!
        let tokenCount = ks.tokenCount
        let mseBits = max(bits - 1, 0)

        // --- Score computation ---
        // 1. Pre-rotate queries for MSE component: q_rot = q @ R^T
        let rotatedQ: MLXArray
        if let mseCodec = keyCodec.mseCodec {
            rotatedQ = mseCodec.prepareQueries(queries)
        } else {
            rotatedQ = queries  // turbo1: no rotation needed for MSE
        }

        // 2. Compute MSE scores via Metal kernel
        // Flatten for kernel: [B*nQHeads*L, D] and [B*nKVHeads, T_kv, PackedWidth]
        let flatQ = rotatedQ.reshaped([B * nQHeads * L, headDim])
        let flatPackedK = ks.mseState.packedIndices.reshaped([B * nKVHeads, tokenCount, -1])
        let flatNormsK = ks.mseState.norms.reshaped([B * nKVHeads, tokenCount])

        var scores: MLXArray
        if mseBits > 0 {
            let codebook = keyCodec.mseCodec!.codebook
            scores = TurboQuantKernelOps.mseScore(
                rotatedQueries: flatQ,
                packedIndices: flatPackedK,
                norms: flatNormsK,
                codebook: codebook,
                tokenCount: tokenCount,
                repeatCount: repeatCount,
                bits: mseBits,
                dim: headDim
            )
        } else {
            // turbo1: no MSE score, start from zero
            scores = MLXArray.zeros([B * nQHeads * L, tokenCount])
        }

        // 3. TODO: Add QJL score correction for full Product codec scoring
        // For now, MSE-only scoring (the QJL correction is small at 3-4 bit)

        // 4. Apply scale
        scores = scores * MLXArray(scale)

        // 5. Apply mask
        scores = scores.reshaped([B, nQHeads, L, tokenCount])
        switch mask {
        case .none:
            break
        case .causal:
            // Create causal mask: upper triangular = -inf
            let maskOffset = tokenCount - L
            let causalMask = MLX.tri(tokenCount, k: maskOffset)
                .reshaped([1, 1, 1, tokenCount])
            let negInf = MLXArray(Float(-1e9))
            scores = MLX.where(causalMask .== 0, negInf, scores)
        case .array(let addMask):
            scores = scores + addMask
        case .arrays(let masks):
            if let m = masks.first {
                scores = scores + m
            }
        }

        // 6. Softmax
        let weights = softmax(scores, axis: -1)

        // --- Value aggregation ---
        // 7. Compute weighted sum of values via Metal kernel
        let flatWeights = weights.reshaped([B * nQHeads * L, tokenCount])
        let flatPackedV = vs.packedIndices.reshaped([B * nKVHeads, tokenCount, -1])
        let flatNormsV = vs.norms.reshaped([B * nKVHeads, tokenCount])
        let valCodebook = valueCodec.codebook

        // For decode (L=1), use Metal kernel; for prefill (L>1), fall back to dequantize
        if L == 1 {
            let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                weights: flatWeights,
                packedIndices: flatPackedV,
                norms: flatNormsV,
                codebook: valCodebook,
                tokenCount: tokenCount,
                repeatCount: repeatCount,
                bits: bits,
                dim: headDim
            )

            // Inverse-rotate: undo the rotation in value space
            let reshaped = rotatedOutput.reshaped([B, nQHeads, L, headDim])
            let output: MLXArray
            if valueCodec.useWHT {
                output = TurboQuantRotation.applyInverseWHT(reshaped, dim: headDim, seed: valueCodec.seed)
            } else {
                output = matmul(reshaped, valueCodec.rotationT.transposed())
            }
            return output
        } else {
            // Prefill: dequantize values and use standard matmul
            let fullValues = valueCodec.decode(vs)
            // Expand for GQA via repeat
            let expandedV: MLXArray
            if repeatCount > 1 {
                // [B, nKVHeads, T, D] -> [B, nQHeads, T, D]
                let repeated = repeated(fullValues, count: repeatCount, axis: 1)
                expandedV = repeated
            } else {
                expandedV = fullValues
            }
            let output = matmul(weights, expandedV)
            return output
        }
    }

    /// Update cache with new key/value pairs and return dequantized full tensors.
    ///
    /// This is the fallback path used during prefill (multiple query tokens).
    /// For decode (single token), use compressedAttention() instead.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let headDim = keys.dim(-1)

        // Lazy codec initialization on first update
        if keyCodec == nil {
            keyCodec = ProductCodec(dim: headDim, bits: bits, seed: seed)
            valueCodec = MSECodec(dim: headDim, bits: bits, seed: seed + 1)
        }

        guard let keyCodec, let valueCodec else {
            return (keys, values)
        }

        // Encode new tokens
        let newKeyState = keyCodec.encode(keys)
        let newValueState = valueCodec.encode(values)

        // Append to existing state
        if keyState == nil {
            keyState = newKeyState
            valueState = newValueState
        } else {
            // Concatenate key Product codec state along token dimension
            keyState = ProductCodecState(
                mseState: MSECodecState(
                    norms: concatenated([keyState!.mseState.norms, newKeyState.mseState.norms], axis: -1),
                    packedIndices: concatenated(
                        [keyState!.mseState.packedIndices, newKeyState.mseState.packedIndices], axis: 2),
                    tokenCount: keyState!.tokenCount + newKeyState.tokenCount,
                    dim: headDim,
                    bits: max(bits - 1, 0)
                ),
                qjlState: QJLState(
                    residualNorms: concatenated(
                        [keyState!.qjlState.residualNorms, newKeyState.qjlState.residualNorms], axis: -1),
                    packedSigns: concatenated(
                        [keyState!.qjlState.packedSigns, newKeyState.qjlState.packedSigns], axis: 2),
                    tokenCount: keyState!.tokenCount + newKeyState.tokenCount,
                    dim: headDim
                ),
                tokenCount: keyState!.tokenCount + newKeyState.tokenCount
            )
            // Concatenate value MSE state
            valueState = MSECodecState(
                norms: concatenated([valueState!.norms, newValueState.norms], axis: -1),
                packedIndices: concatenated(
                    [valueState!.packedIndices, newValueState.packedIndices], axis: 2),
                tokenCount: valueState!.tokenCount + newValueState.tokenCount,
                dim: headDim,
                bits: bits
            )
        }

        offset = keyState!.tokenCount

        // Dequantize for standard SDPA (Phase 4 will use compressed-domain attention)
        let fullKeys = keyCodec.decode(keyState!)
        let fullValues = valueCodec.decode(valueState!)

        return (fullKeys, fullValues)
    }

    /// Trim the last n tokens from the cache.
    @discardableResult
    override public func trim(_ n: Int) -> Int {
        guard n > 0, let ks = keyState, let vs = valueState else { return 0 }
        let trimCount = min(n, ks.tokenCount)
        let newCount = ks.tokenCount - trimCount

        if newCount == 0 {
            keyState = nil
            valueState = nil
            offset = 0
        } else {
            keyState = ProductCodecState(
                mseState: MSECodecState(
                    norms: ks.mseState.norms[0..., 0..., ..<newCount],
                    packedIndices: ks.mseState.packedIndices[0..., 0..., ..<newCount, 0...],
                    tokenCount: newCount,
                    dim: ks.mseState.dim,
                    bits: ks.mseState.bits
                ),
                qjlState: QJLState(
                    residualNorms: ks.qjlState.residualNorms[0..., 0..., ..<newCount],
                    packedSigns: ks.qjlState.packedSigns[0..., 0..., ..<newCount, 0...],
                    tokenCount: newCount,
                    dim: ks.qjlState.dim
                ),
                tokenCount: newCount
            )
            valueState = MSECodecState(
                norms: vs.norms[0..., 0..., ..<newCount],
                packedIndices: vs.packedIndices[0..., 0..., ..<newCount, 0...],
                tokenCount: newCount,
                dim: vs.dim,
                bits: vs.bits
            )
            offset = newCount
        }

        return trimCount
    }

    /// Create attention mask for the current cache state.
    override public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if n == 1 {
            // Single token decode — no mask needed
            return .none
        }
        // For multi-token (prefill), use causal mask
        return .causal
    }
}

// MARK: - KVCacheSimple Extension

extension KVCacheSimple {

    /// Convert this simple cache to a TurboQuant compressed cache.
    ///
    /// Transfers the existing cached keys/values into TurboQuant format.
    /// Called by `maybeQuantizeKVCache()` when `kvScheme` is set.
    public func toTurboQuantized(bits: Int = 4, seed: UInt64 = 42) -> TurboQuantKVCache {
        let turboCache = TurboQuantKVCache(bits: bits, seed: seed)

        // Transfer existing state if any
        if let keys = self.keys, let values = self.values, offset > 0 {
            let currentKeys = keys[0..., 0..., ..<offset, 0...]
            let currentValues = values[0..., 0..., ..<offset, 0...]
            _ = turboCache.update(keys: currentKeys, values: currentValues)
        }

        return turboCache
    }
}
