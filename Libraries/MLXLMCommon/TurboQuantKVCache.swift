// Copyright © 2026 Eric Kryski. KV cache compression for mlx-swift-lm.
//
// HONEST NAMING NOTE (2026-05-09):
// This file is named `TurboQuant*` but the algorithm it ships is NOT a
// faithful implementation of the TurboQuant paper (arXiv:2504.19874,
// ICLR 2026). It is a Frankenstein hybrid that fuses techniques from
// TurboQuant_mse, QuaRot, PolarQuant, and llama.cpp's k_quants codebook.
// In the companion paper we call this composite algorithm **GigaQuant**;
// see `papers/gigaquant-a-frankenstein-compression-algorithm.md` for the
// full prior-art map, divergence rationale, and open research directions.
//
// What this file ACTUALLY implements:
//
//   Pipeline: x  →  norm-extract  →  unit direction  →  random orthogonal
//             rotation Π  →  per-coord Lloyd-Max scalar quantization
//             (1D codebook, scaled by √(128/d))  →  packed indices + fp16 norm
//
//   Step-by-step divergences from the TurboQuant paper:
//
//   1. ROTATION — paper uses Gaussian-QR exclusively. We ship two paths:
//        a. Gaussian-QR (matches paper) for arbitrary head dims
//        b. SRHT = H·diag(±1)/√d via FWHT (QuaRot-style, arXiv:2404.00456)
//           for power-of-2 head dims up to 1024 — O(d log d) butterfly with
//           simd_shuffle_xor intra-SIMD stages + shared-memory cross-SIMD.
//      Rotation is fixed at codec init in both paths (QuaRot-style, not
//      SpinQuant-style learned — see arXiv:2405.16406 for the learned variant).
//
//   2. NORM/DIRECTION SPLIT — the paper treats this as a pre-step ("rescale
//      by L2"). We make it a first-class operation, following PolarQuant's
//      framing (arXiv:2502.02617): magnitude stored as fp16 scalar, direction
//      normalized to unit sphere before rotation. Plus a norm-correction
//      stored alongside (||x|| / ||ỹ||) for the dense-rotation path to
//      compensate for quantization error; WHT path skips this because
//      orthogonal SRHT preserves norms exactly.
//
//   3. CODEBOOK — paper derives an analytic per-coordinate Lloyd-Max
//      quantizer from the Beta distribution that rotated coordinates follow,
//      with closed-form values at each bit-width (paper Sec. 3.1; e.g. b=2:
//      {±0.453/√d, ±1.51/√d}). We instead use a *global 1D codebook* mined
//      from llama.cpp's k_quants tables at d=128, scaled by √(128/d) for
//      other head dims. The √(128/d) rescaling is a heuristic to approximate
//      the paper's 1/√d Beta-variance scaling. This is a real divergence: we
//      use one shared codebook across all coordinates rather than per-coord
//      quantizers as the paper specifies.
//
//   4. QJL SECOND STAGE — paper's TurboQuant_prod (Algorithm 2) adds a 1-bit
//      Quantized Johnson-Lindenstrauss transform on the quantization residual
//      to get an unbiased inner-product estimator. We omit it entirely.
//      This is an empirical decision, not just an engineering one: independent
//      benchmarking (Tom Turney's turbo4-resurrection write-up at
//      https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md,
//      March 2026) reports QJL-on regresses BOTH quality AND speed at long
//      context:
//        - PPL degrades from -0.28% at 2K context to +3.69% at 64K (clear
//          long-context degradation trend; "QJL eliminates the bias but
//          explodes the variance that softmax then amplifies")
//        - Decode regresses 76.84 vs 79.87 tok/s due to the extra residual-
//          projection dispatch and memory traffic
//        - NIAH 30/33 → 31/33 with QJL off
//      The bit-budget freed by removing QJL is better invested in more
//      centroids (16 instead of 8+QJL) — that's the path turbo4-resurrection
//      took and the path we ship. At 4-bit the residual MSE bias (2/π from
//      paper Sec. 3.2) is acceptable; trading bounded bias for unbounded
//      softmax-amplified variance was the wrong call for autoregressive KV.
//
//   5. ASYMMETRIC K/V — engineering addition. K precision dominates quality
//      (softmax amplifies a perturbation of ε in logit space to O(e^ε) in
//      attention weights); V precision matters less (V aggregation is a
//      linear weighted sum). Recipes: "turbo4v2" = 4-bit K + 2-bit V; the
//      production "q8_0-K + turbo4-V" = 8.5-bit K + 4.25-bit V (2.5× total).
//      Justification: Turney's asymmetric-kv-compression.md (March-April 2026,
//      benched across 7 models on Metal / CUDA / HIP / Vulkan). Catastrophic
//      counter-example: symmetric turbo3/turbo3 on Qwen2.5-7B produces PPL
//      3,556 vs asymmetric q8_0-K + turbo3-V at PPL 6.71. Independently
//      documented in KIVI (arXiv:2402.02750) and KVQuant (arXiv:2401.18079).
//      See: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md
//
//   6. BOUNDARY LAYERS — engineering addition. First N and last N attention
//      layers retain higher V precision (e.g. q8_0 V on layers {0, 1, N-2,
//      N-1}, turbo2 V elsewhere; K stays q8_0 throughout). Rationale: V
//      errors in boundary layers either affect every subsequent layer's
//      attention output (first layers) or directly distort the output
//      distribution (last layers); middle layers operate on abstracted
//      representations that absorb noise better. Empirical recipe from
//      Turney's layer-aware-v-compression.md ("LA-V7" / "Boundary V"). Same
//      per-layer-importance intuition as Q-Hitter (arXiv:2402.14905),
//      SqueezeAttention (arXiv:2404.04793), and KVQuant. CAVEAT: developed
//      on pure-attention models (phi-4, Qwen2.5); hybrid models like
//      Qwen3.5 with Gated Delta Net need a different layer-counting
//      convention or the protection mis-targets the wrong layers.
//      See: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md
//
//   7. PRE-ROTATED QUERIES — engineering addition. Compute q' = Π·q once per
//      layer; reuse for all cached keys. Avoids per-key inverse rotation;
//      attention scoring runs in the compressed/rotated domain.
//
//   8. TWO-PHASE ARCHITECTURE — engineering addition. Raw fp16 prefill buffer,
//      batch-compress at decode start, compressed decode thereafter. Hides
//      the encode cost in TTFT rather than per-token decode.
//
// VERIFIED INVARIANTS (do not regress):
//   - No QJL, no residual quantization, no random projection correction
//     anywhere in this codebase. Inner-product estimation is biased (the
//     2/π bias from paper Sec. 3.2) but the bias diminishes with bit-width
//     and is acceptable at 4-bit. Re-introducing QJL would hurt generation
//     quality — see TurboQuant_plus discussion.
//   - Rotation matrix is `public let` and never updated post-init.
//
// References:
//   - TurboQuant (paper):     https://arxiv.org/abs/2504.19874  (ICLR 2026)
//   - TurboQuant_plus (fork): https://github.com/TheTom/turboquant_plus
//   - turbo4-resurrection:    https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md
//   - asymmetric-kv:          https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md
//   - layer-aware-v:          https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md
//   - QuaRot (rotation):      https://arxiv.org/abs/2404.00456
//   - SpinQuant (learned rot):https://arxiv.org/abs/2405.16406
//   - PolarQuant (polar):     https://arxiv.org/abs/2502.02617
//   - QJL:                    https://arxiv.org/abs/2406.03482
//   - KIVI (asymm K/V):       https://arxiv.org/abs/2402.02750
//   - KVQuant (boundary):     https://arxiv.org/abs/2401.18079
//   - Q-Hitter:               https://arxiv.org/abs/2402.14905
//   - SqueezeAttention:       https://arxiv.org/abs/2404.04793
//   - QuIP# (adjacent):       https://arxiv.org/abs/2402.04396
//   - HIGGS (adjacent):       https://arxiv.org/abs/2411.17525
//   - llama.cpp k_quants:     https://github.com/ggml-org/llama.cpp
//   - GigaQuant write-up:     papers/gigaquant-a-frankenstein-compression-algorithm.md

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
    public var norms: MLXArray       // [B, H, T] — L2 norm of (centered) vector
    public var packedIndices: MLXArray // [B, H, T, PackedWidth] — packed codebook indices
    public var tokenCount: Int
    public let dim: Int
    public let bits: Int
    /// Optional per-vector DC offset. Present when the codec ran with
    /// `useBias: true`. `decode` adds `bias` back after the inverse
    /// rotation; `decodeRotated` adds `bias * rotatedOnes` in rotated
    /// space (since rotation is linear). Captures the mean the zero-mean
    /// Lloyd-Max codebook can't represent. See `testDualScaleMSEReduction`
    /// for the empirical case (9-22% MSE on structured K/V).
    public var bias: MLXArray?       // [B, H, T] — per-vector DC offset (optional)
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

    /// `1_vec @ rotationT`, precomputed once. A per-vector DC bias `b` in
    /// original space maps to `b * rotatedOnes[d]` in rotated space (since
    /// rotation is linear). Used by the rotated-space bias-aware dequant.
    public let rotatedOnes: MLXArray  // [1, dim]

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
        // Precompute `1_vec @ rotationT`. For WHT-rotated codecs this is a
        // function of the codec's WHT signs (not just `dim`), so it stays
        // cached on the codec instance — codecs are shared per
        // `(dim, bits, seed)` via `getOrCreateCodec`, so this is computed
        // once per unique codec for the lifetime of the process.
        self.rotatedOnes =
            matmul(MLXArray.ones([1, dim], dtype: .bfloat16), self.rotationT)
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
    // Dual-block scales (TQ4_1S-style) were considered but empirically only
    // give 1-8% MSE reduction on this codec (the WHT rotation flattens the
    // per-half-block energy asymmetry the scheme is designed to exploit).
    // The 9-22% gain from per-vector DC-bias correction below is the larger
    // structural win on real K/V; see `testDualScaleMSEReduction` in
    // `TurboQuantKernelTests`.
    public func encode(_ vectors: MLXArray, useBias: Bool = false) -> MSECodecState {
        // With `useBias`, subtract per-vector mean (head-dim axis) before
        // unit-norm + rotation. Centers the input so the zero-mean
        // Lloyd-Max codebook fits structured K/V (e.g. GPT-OSS's
        // `RMSNorm → Linear(bias=True)` projections).
        let centered: MLXArray
        let storedBias: MLXArray?
        if useBias {
            let bias = vectors.mean(axis: -1, keepDims: true)  // [B, H, T, 1]
            storedBias = bias.squeezed(axis: -1)               // [B, H, T]
            centered = vectors - bias
        } else {
            storedBias = nil
            centered = vectors
        }

        // Extract norms and normalize (paper assumes unit sphere; we store norms separately)
        let norms = sqrt((centered * centered).sum(axis: -1))
        let safeNorms = maximum(norms, Float(1e-8))
        let unit = centered / expandedDimensions(safeNorms, axis: -1)

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
            bits: bits,
            bias: storedBias
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

        // Rescale by stored norms; add DC bias back in original space.
        var recovered = expandedDimensions(state.norms, axis: -1) * unrotated
        if let bias = state.bias {
            recovered = recovered + expandedDimensions(bias, axis: -1)
        }
        return recovered
    }

    /// Decode in rotated space (skip inverse rotation).
    /// Returns centroid values scaled by norm, still in Π-rotated coordinate space.
    /// Used with pre-rotated queries for dequant-first SDPA.
    public func decodeRotated(_ state: MSECodecState) -> MLXArray {
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)
        let approx = codebook[indices]
        var recovered = expandedDimensions(state.norms, axis: -1) * approx
        // Bias-aware dequant in rotated space: `(c + b·1) @ R^T =
        // c @ R^T + b·rotatedOnes`. No extra matmul per token.
        if let bias = state.bias {
            recovered = recovered + expandedDimensions(bias, axis: -1) * rotatedOnes.asType(recovered.dtype)
        }
        return recovered
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

// MARK: - TurboQuantizedKVCache

/// KV cache backed by TurboQuant (MSE-optimal) compression.
///
/// Decode-time attention has two paths inside `compressedAttention`:
///
/// - **A path (default)** — TurboFlash. The `turboFlashAttention` Metal
///   kernel (or `turboFlashSDPAv` for sinks-using models) scores
///   directly against packed K/V and writes the weighted V in one
///   dispatch. No FP16 K/V materialisation.
///
/// - **B path (opt-in via `TURBO_DEQUANT_SDPA=1` env var or the
///   `useDequantSDPA: true` constructor arg)** — `bulkDequantRotated`
///   produces a transient FP16 K/V tensor each decode step, then
///   `MLXFast.scaledDotProductAttention` runs on it. The cache stays
///   compressed but working memory expands by `B*nKV*T*D*2` bytes per
///   layer per step. Also the path that consumes the stored bias term —
///   `useBias` forces the B path until TurboFlash kernels learn to
///   apply the per-vector DC bias themselves.
///
/// K and V follow Algorithm 1 from the TurboQuant paper (MSE at b bits,
/// no QJL). See the file header for the codec details.
public class TurboQuantizedKVCache: BaseKVCache {

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
    /// Matches StandardKVCache semantics — oldest tokens evicted when full.
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

    // Phase 1: Raw K/V storage (like StandardKVCache) — used during prefill
    private var rawKeys: MLXArray?       // [B, H, allocSteps, D]
    private var rawValues: MLXArray?     // [B, H, allocSteps, D]
    private var rawAllocSteps = 0

    // Phase 2: Compressed storage — used during decode
    // MSE-only: packed indices + norms (no QJL — simpler, same quality)
    private var keyPackedMSE: MLXArray?
    private var keyNorms: MLXArray?
    private var valPackedMSE: MLXArray?
    private var valNorms: MLXArray?
    // Per-vector DC offset, allocated when `useBias == true`. Shape
    // mirrors norms ([B, H, allocSteps]) — one fp32 per token-head.
    private var keyBias: MLXArray?
    private var valBias: MLXArray?
    private var compressedAllocSteps = 0

    /// Whether we've transitioned from raw → compressed
    public private(set) var isCompressed = false

    /// Compressed storage write position — independent from `offset` (which tracks
    /// total tokens for RoPE/masks). This prevents desync between dequant and
    /// compressed buffers when batch encoding runs on a different schedule.
    private var compressedWriteOffset = 0

    /// True once a write has overwritten a slot that previously held a
    /// different sequence position — i.e. the rotating ring buffer has
    /// completed a full cycle. From that point on, buffer position N no
    /// longer maps onto sequence position N: position N could hold any
    /// of `N`, `N + maxSize`, `N + 2·maxSize`, … depending on how many
    /// times the buffer has lapped.
    ///
    /// `trim(...)` *only* decrements `offset` — it does NOT undo
    /// rotation, so a cache that has wrapped during decode looks
    /// "in-range" by `offset` alone even though its buffer slots are
    /// scrambled. The prefix-cache snapshot path consults this flag to
    /// refuse a snapshot of a rotated buffer (issue surfaced when
    /// running 1K-then-8K summarisation on Gemma 4 26B-A4B —
    /// `sliding_window = 1024`, prompt = 1024 tokens, decode rotated
    /// position 0 of the buffer onto sequence token 1024+, corrupting
    /// the snapshot's first slots).
    public private(set) var hasWrappedRotatingBuffer = false

    /// Pre-allocation step size for buffer growth. Larger values reduce resize frequency
    /// at the cost of upfront memory. At step=1024, a 16K context only resizes 16 times
    /// (vs 64 times at step=256), eliminating 75% of allocation + copy overhead.
    private let step: Int

    public override var maxSize: Int? { rotatingMaxSize }

    /// Decode attention path selector for the L=1 case:
    ///
    /// - **A path (default, `useDequantSDPA == false`)** — TurboFlash. The
    ///   compressed-domain `turboFlashAttention` Metal kernel scores
    ///   directly against packed K/V; no per-layer FP16 K/V working
    ///   buffer. True compressed attention end-to-end.
    /// - **B path (opt-in, `useDequantSDPA == true`)** —
    ///   `bulkDequantRotated` materialises a transient FP16 K/V buffer
    ///   per decode step, then `MLXFast.scaledDotProductAttention` runs
    ///   on the matmul engine. The cache stays compressed but working
    ///   memory expands by `B*nKV*T*D*2` bytes per layer per step.
    ///   Faster on small / medium `headDim` shapes; required today by
    ///   the bias-correction path (TurboFlash kernels don't yet
    ///   consume the stored bias term — tracked as a follow-up).
    ///
    /// Env override: `TURBO_DEQUANT_SDPA=1` forces B globally; `=0`
    /// forces A; unset honours the constructor.
    public let useDequantSDPA: Bool

    /// When true, encode subtracts per-vector DC bias before
    /// quantisation; decode adds it back. Captures the mean offset that
    /// the zero-mean Lloyd-Max codebook can't represent — the dominant
    /// codec error source on K/V from `RMSNorm → Linear(bias=True)`.
    /// Default-off for general use, default-on for GPT-OSS-20B under
    /// turbo* schemes (see `GPTOSSModel.newCache(...)`). Env override:
    /// `TURBO_BIAS=1`/`=0`. See `testDualScaleMSEReduction` for the
    /// empirical comparison against the dual-scale alternative.
    ///
    /// Bias correction currently requires the B (dequant-SDPA) path —
    /// the TurboFlash kernels don't yet apply the stored bias term
    /// during attention. When `useBias == true`, the cache routes
    /// decode through dequant SDPA regardless of `useDequantSDPA`.
    public let useBias: Bool

    public init(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil,
        step: Int = 1024, seed: UInt64 = 42, maxSize: Int? = nil,
        useDequantSDPA: Bool = false,
        useBias: Bool = false,
        headDim: Int? = nil
    ) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.rawKeyMode = (keyBits ?? bits) == 0
        self.seed = seed
        self.step = step
        self.rotatingMaxSize = maxSize
        // Env overrides for A/B testing without recompiling. Each accepts
        // `0`/`1` to force the flag; unset uses the constructor default.
        let dequantEnv = ProcessInfo.processInfo.environment["TURBO_DEQUANT_SDPA"]
        switch dequantEnv {
        case "0": self.useDequantSDPA = false
        case "1": self.useDequantSDPA = true
        default:  self.useDequantSDPA = useDequantSDPA
        }
        let biasEnv = ProcessInfo.processInfo.environment["TURBO_BIAS"]
        switch biasEnv {
        case "0": self.useBias = false
        case "1": self.useBias = true
        default:  self.useBias = useBias
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

    /// Load raw K/V data from a prefilled cache (e.g., StandardKVCache or StandardKVCache).
    /// TurboQuantizedKVCache will compress these on first decode token.
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

    /// Sliding-window-aware mask. Mirrors `StandardKVCache.makeMask` so that
    /// when this cache stands in for a sliding `StandardKVCache`, the model's
    /// `makeAttentionMask(..., windowSize: ws)` call gets a proper windowed
    /// mask instead of `BaseKVCache`'s fallback (which ignores `windowSize`
    /// for `n > 1` and emits `.causal`).
    ///
    /// Without this override:
    /// - GPT-OSS sliding layers fall back to a non-windowed causal mask after
    ///   prefill→turbo conversion, which is wrong whenever `cappedOffset + n
    ///   > windowSize` (causes attention to read evicted/garbage slots).
    /// - For L=1 decode after the rotating buffer fills, no rolled mask is
    ///   produced — same failure mode `StandardKVCache.makeMask` guards against.
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
            let biasTag = useBias ? " +bias" : ""
            print("[TURBO] Encode kernel: \(codec.useWHT ? "WHT butterfly" : "dense matmul"), dim=\(headDim), bits=\(codec.bits)\(biasTag)")
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

    /// Wrap `fusedEncodeDispatch` with DC-bias correction. When `useBias`
    /// is enabled, subtracts the per-vector mean in Swift before handing
    /// the *centered* tensor to the Metal encode kernel and returns the
    /// stored bias alongside the existing (packed, norms) pair. When
    /// `useBias` is off, bias is `nil` and the centered tensor is the
    /// input itself — zero overhead vs the old path.
    private func fusedEncodeDispatchWithBias(
        input: MLXArray, codec: MSECodec, headDim: Int
    ) -> (packed: MLXArray, norms: MLXArray, bias: MLXArray?) {
        if !useBias {
            let (p, n) = fusedEncodeDispatch(input: input, codec: codec, headDim: headDim)
            return (p, n, nil)
        }
        // `input` is [N, D] flat. `keepDims: true` gives [N, 1] for the
        // squeeze step. We return bias as a [N] vector to match the
        // norms shape downstream consumers expect.
        let biasKeep = input.mean(axis: -1, keepDims: true)  // [N, 1]
        let centered = input - biasKeep
        let (packed, norms) = fusedEncodeDispatch(
            input: centered, codec: codec, headDim: headDim)
        return (packed, norms, biasKeep.squeezed(axis: -1))   // bias: [N]
    }

    // MARK: - Phase 1: Raw Prefill

    /// Prefill update: store raw K/V, return raw. Zero encoding overhead.
    /// Uses StandardKVCache-style allocation with concatenated growth.
    ///
    /// When `rotatingMaxSize` is set (sliding-window layers), the buffer is
    /// trimmed to the most recent `maxSz` tokens after each update — mirroring
    /// `StandardKVCache.updateConcat` semantics. `offset` still tracks the
    /// absolute sequence position (for RoPE), while the buffer holds at most
    /// `maxSz` temporally-ordered tokens.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let S = keys.dim(2)
        let previous = self.offset

        if let maxSz = rotatingMaxSize {
            // Sliding-window prefill: StandardKVCache-style retain-then-append.
            // Drop only `max(0, existingTokens - maxSz + 1)` of the oldest
            // existing tokens before appending the new chunk — NOT the full
            // overflow. This matches `StandardKVCache.updateConcat`'s
            // `trimSize = idx - maxCacheSize + 1` formula and is what makes
            // the returned buffer usable for the chunk's SDPA: every Q in
            // the chunk sees a K window of up to `maxSz` past tokens (the
            // overall buffer length is capped at `maxSz + S - 1` mid-prefill,
            // not `maxSz`). Final compaction to `maxSz` happens in
            // `compressRawCache()`.
            let combinedK: MLXArray
            let combinedV: MLXArray
            if let rk = rawKeys, let rv = rawValues {
                let bufferTokens = min(previous, rk.dim(2))
                let trimSize = max(0, bufferTokens - maxSz + 1)
                let keptK = rk[.ellipsis, trimSize ..< bufferTokens, 0...]
                let keptV = rv[.ellipsis, trimSize ..< bufferTokens, 0...]
                combinedK = concatenated([keptK, keys], axis: 2)
                combinedV = concatenated([keptV, values], axis: 2)
                if trimSize > 0 { hasWrappedRotatingBuffer = true }
            } else {
                combinedK = keys
                combinedV = values
            }
            rawKeys = combinedK
            rawValues = combinedV
            if combinedK.dim(2) > maxSz { hasWrappedRotatingBuffer = true }
            rawAllocSteps = rawKeys!.dim(2)
            self.offset = previous + S
            // Issue #185: KV-shared models (Gemma 4 E2B / E4B) read
            // `lastReturnedKeys` / `lastReturnedValues` after each donor
            // layer's update to thread the donor's K/V into shared
            // layers (matches `StandardKVCache.updateUnbounded`'s pattern).
            // Without this assignment, shared layers received nil arrays
            // and SDPA produced garbage logits.
            self.lastReturnedKeys = rawKeys
            self.lastReturnedValues = rawValues
            return (rawKeys!, rawValues!)
        }

        // Non-rotating path: StandardKVCache-style growable buffer.
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
        // Issue #185: see the rotating branch above. KV-shared models
        // depend on these being set after every donor `update()`.
        self.lastReturnedKeys = returnedKeys
        self.lastReturnedValues = returnedValues
        return (returnedKeys, returnedValues)
    }

    // MARK: - Transition: Compress Raw Cache

    /// Compress the entire raw K/V cache into packed format in one batch.
    /// Called once when transitioning from prefill to decode.
    ///
    /// In rawKeyMode: only compress values. Keys stay as raw FP16 in rawKeys buffer.
    /// This is the highest-quality TurboQuant+ mode — K precision dominates quality.
    ///
    /// `internal` so tests can drive the raw→compressed transition without
    /// going through a full attention call. Production callers should not
    /// invoke this directly — `compressedAttention(...)` triggers it on the
    /// first decode-step call.
    internal func compressRawCache() {
        // Guard: skip if already compressed or no raw data
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }
        // Rotating prefill (see `update(...)`) intentionally leaves the buffer
        // sized up to `maxSz + S - 1` so the last chunk's SDPA could see a
        // full window per query. Now that prefill is over, compact to the
        // **last** `maxSz` rows — the only ones decode will attend to under
        // a sliding window. Taking the prefix `..<maxSz` instead would feed
        // the model the *oldest* `maxSz` tokens (the trim-concat layout has
        // newest at the buffer's tail), which is exactly the regression
        // mode we saw mid-debug.
        let actualTokens: Int
        let allKeys: MLXArray
        let allValues: MLXArray
        if let maxSz = rotatingMaxSize, rk.dim(2) > maxSz {
            actualTokens = maxSz
            let bufLen = rk.dim(2)
            allKeys = rk[.ellipsis, (bufLen - maxSz)..., 0...]
            allValues = rv[.ellipsis, (bufLen - maxSz)..., 0...]
        } else {
            actualTokens = min(offset, rk.dim(2))
            allKeys = rk[.ellipsis, ..<actualTokens, 0...]
            allValues = rv[.ellipsis, ..<actualTokens, 0...]
        }
        let headDim = allKeys.dim(-1)
        ensureCodecs(headDim: headDim)
        compressRawCacheInternal(allKeys: allKeys, allValues: allValues, headDim: headDim)
        if rawKeyMode {
            // Keep rawKeys alive — they're our FP16 key storage going forward
            // Only free rawValues since those are now compressed
            rawValues = nil
            // For rotating: size rawKeys to exactly maxSize holding the same
            // `last maxSz` slice we just compressed. The pre-fix branch
            // expanded only when `rk.dim(2) < maxSz`; under the new
            // retain-then-append prefill (which may leave `rk.dim(2) >
            // maxSz`), the unconditional resize ensures decode reads
            // `rawKeys[..<maxSz]` and finds the most recent prefill window
            // rather than its oldest rows.
            if let maxSz = rotatingMaxSize {
                let B = allKeys.dim(0), H = allKeys.dim(1), D = allKeys.dim(3)
                let expanded = MLXArray.zeros([B, H, maxSz, D], dtype: allKeys.dtype)
                expanded[.ellipsis, ..<actualTokens, 0...] = allKeys
                rawKeys = expanded
                rawAllocSteps = maxSz
            }
            // Donor case in rawKeyMode: also need a raw V mirror so the
            // shared reader path can grab FP16 V via `lastReturnedValues`
            // without a per-step dequant. Sized identically to rawKeys.
            if isDonor, let maxSz = rotatingMaxSize {
                let B = allValues.dim(0), H = allValues.dim(1), D = allValues.dim(3)
                let expanded = MLXArray.zeros([B, H, maxSz, D], dtype: allValues.dtype)
                expanded[.ellipsis, ..<actualTokens, 0...] = allValues
                rawValues = expanded
            }
        } else if isDonor {
            // Donor case (non-rawKeyMode): retain raw K and V buffers as
            // FP16 mirrors of the compressed cache so the shared reader
            // path can read `lastReturnedKeys` / `lastReturnedValues`
            // without a per-step dequant. The mirror is initialised here
            // from the same slice we just compressed, then updated one
            // slot at a time inside `encodeNewToken` (cheap scatter
            // writes) to stay in lockstep with the compressed buffers.
            // Replaces the pre-fix per-decode-step full-prefix
            // `kCodec.decode(...)` / `valueMSECodec.decode(...)` refresh
            // that was costing ~45% of decode tok/s on Gemma 4 E2B / E4B
            // (measured: 26 → 48 tok/s when the refresh was bypassed in
            // a diagnostic build).
            //
            // Only kicks in for rotating (sliding-window) donor caches —
            // the common case for KV-sharing on Gemma 4. Non-rotating
            // donor caches (full-attention donors) fall through to the
            // old per-step dequant path until the mirror grows with
            // arbitrary buffer expansion (future work).
            if let maxSz = rotatingMaxSize {
                let B = allKeys.dim(0), H = allKeys.dim(1)
                let kD = allKeys.dim(3), vD = allValues.dim(3)
                let mirrorK = MLXArray.zeros([B, H, maxSz, kD], dtype: allKeys.dtype)
                mirrorK[.ellipsis, ..<actualTokens, 0...] = allKeys
                rawKeys = mirrorK
                let mirrorV = MLXArray.zeros([B, H, maxSz, vD], dtype: allValues.dtype)
                mirrorV[.ellipsis, ..<actualTokens, 0...] = allValues
                rawValues = mirrorV
                rawAllocSteps = maxSz
            } else {
                // Non-rotating donor — fall back to per-step dequant for now
                rawKeys = nil
                rawValues = nil
                rawAllocSteps = 0
            }
        } else {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
        }
        isCompressed = true
        compressedWriteOffset = min(offset, rotatingMaxSize ?? offset)
        if let maxSz = rotatingMaxSize {
            // Trim-concat prefill (`update(...)` rotating branch, lines ~1144)
            // leaves the raw buffer **time-ordered**: slot 0 = oldest of the
            // retained `maxSz` tokens, slot `maxSz-1` = newest. The next
            // decode write must therefore go to slot 0 (overwriting the
            // oldest), not to `offset % maxSz`, which assumes a pure
            // modular-rotation layout (slot 0 = position `maxSz`, etc.) the
            // trim-concat path does not produce.
            //
            // For the no-trim case (offset ≤ maxSz), the linear append
            // semantics still hold: `bufferTokens % maxSz == offset` when
            // `offset < maxSz`, and `== 0` at the maxSz boundary (where the
            // first wrap should target slot 0). `updateAndDequant`'s
            // transition already uses this form — keeping the two paths in
            // sync.
            let bufferTokens = min(offset, maxSz)
            rotatingIdx = bufferTokens % maxSz
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
        let (valPackedFlat, valNormsFlat, valBiasFlat) = fusedEncodeDispatchWithBias(
            input: flatVals, codec: valueMSECodec, headDim: headDim)

        // Pre-allocate to at least one step beyond current tokenCount to accommodate
        // the first decode token after compression. Without this, the buffer is exactly
        // tokenCount slots and the first encodeNewToken write overflows.
        let minAlloc = rotatingMaxSize ?? (tokenCount + step)
        let allocSteps = ((max(minAlloc, tokenCount + step) + step - 1) / step) * step
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valNorms = MLXArray.zeros([B, H, allocSteps])
        if useBias {
            valBias = MLXArray.zeros([B, H, allocSteps])
        }

        if !rawKeyMode {
            // Compress keys too (standard TurboQuant path)
            guard let keyMSECodec else { return }
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
            let (keyPackedFlat, keyNormsFlat, keyBiasFlat) = fusedEncodeDispatchWithBias(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)

            keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
            keyNorms = MLXArray.zeros([B, H, allocSteps])
            if useBias {
                keyBias = MLXArray.zeros([B, H, allocSteps])
            }

            keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedFlat.reshaped([B, H, tokenCount, kpw])
            keyNorms![.ellipsis, ..<tokenCount] = keyNormsFlat.reshaped([B, H, tokenCount])
            if useBias, let kb = keyBiasFlat {
                keyBias![.ellipsis, ..<tokenCount] = kb.reshaped([B, H, tokenCount])
            }
        }

        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([B, H, tokenCount, vpw])
        valNorms![.ellipsis, ..<tokenCount] = valNormsFlat.reshaped([B, H, tokenCount])
        if useBias, let vb = valBiasFlat {
            valBias![.ellipsis, ..<tokenCount] = vb.reshaped([B, H, tokenCount])
        }

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
    /// tokens are overwritten in-place, matching StandardKVCache semantics.
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
        let (valPacked, valNormsNew, valBiasNew) = fusedEncodeDispatchWithBias(
            input: flatVals, codec: valueMSECodec, headDim: headDim)
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])
        let valBiasShaped = valBiasNew?.reshaped([B, H, numSteps])

        // Determine write position — rotating or linear.
        // Offset is managed by the caller (compressedAttention).
        let writeIdx: Int
        if let maxSz = rotatingMaxSize {
            // Rotating mode: wrap write position within the fixed buffer
            if offset >= maxSz {
                writeIdx = rotatingIdx
                rotatingIdx = (rotatingIdx + numSteps) % maxSz
                hasWrappedRotatingBuffer = true
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
                let newVB: MLXArray? = useBias ? MLXArray.zeros([B, H, newAlloc]) : nil
                if writeIdx > 0 {
                    let copyLen = min(writeIdx, compressedAllocSteps)
                    newVP[.ellipsis, ..<copyLen, 0...] = valPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newVN[.ellipsis, ..<copyLen] = valNorms![.ellipsis, ..<copyLen]
                    if useBias, let vb = valBias {
                        newVB![.ellipsis, ..<copyLen] = vb[.ellipsis, ..<copyLen]
                    }
                }
                valPackedMSE = newVP; valNorms = newVN
                if useBias { valBias = newVB }
                compressedAllocSteps = newAlloc
            }

            rawKeys![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = keys
            valPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = valPackedShaped
            valNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = valNormsShaped
            if useBias, let vbShaped = valBiasShaped {
                valBias![.ellipsis, writeIdx..<(writeIdx + numSteps)] = vbShaped
            }
        } else {
            // Standard TurboQuant: encode both K and V
            guard let keyMSECodec else { return }

            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = keys.reshaped([B * H * numSteps, headDim])
            let (keyPacked, keyNormsNew, keyBiasNew) = fusedEncodeDispatchWithBias(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)
            let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
            let keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])
            let keyBiasShaped = keyBiasNew?.reshaped([B, H, numSteps])

            if writeIdx + numSteps > compressedAllocSteps {
                let newAlloc = rotatingMaxSize ?? (((writeIdx + numSteps + step - 1) / step) * step)
                let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
                let newKN = MLXArray.zeros([B, H, newAlloc])
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc])
                let newKB: MLXArray? = useBias ? MLXArray.zeros([B, H, newAlloc]) : nil
                let newVB: MLXArray? = useBias ? MLXArray.zeros([B, H, newAlloc]) : nil
                if writeIdx > 0 {
                    let copyLen = min(writeIdx, compressedAllocSteps)
                    newKP[.ellipsis, ..<copyLen, 0...] = keyPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newKN[.ellipsis, ..<copyLen] = keyNorms![.ellipsis, ..<copyLen]
                    newVP[.ellipsis, ..<copyLen, 0...] = valPackedMSE![.ellipsis, ..<copyLen, 0...]
                    newVN[.ellipsis, ..<copyLen] = valNorms![.ellipsis, ..<copyLen]
                    if useBias {
                        if let kb = keyBias {
                            newKB![.ellipsis, ..<copyLen] = kb[.ellipsis, ..<copyLen]
                        }
                        if let vb = valBias {
                            newVB![.ellipsis, ..<copyLen] = vb[.ellipsis, ..<copyLen]
                        }
                    }
                }
                keyPackedMSE = newKP; keyNorms = newKN
                valPackedMSE = newVP; valNorms = newVN
                if useBias { keyBias = newKB; valBias = newVB }
                compressedAllocSteps = newAlloc
            }

            keyPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = keyPackedShaped
            keyNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = keyNormsShaped
            valPackedMSE![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = valPackedShaped
            valNorms![.ellipsis, writeIdx..<(writeIdx + numSteps)] = valNormsShaped
            if useBias {
                if let kbShaped = keyBiasShaped {
                    keyBias![.ellipsis, writeIdx..<(writeIdx + numSteps)] = kbShaped
                }
                if let vbShaped = valBiasShaped {
                    valBias![.ellipsis, writeIdx..<(writeIdx + numSteps)] = vbShaped
                }
            }

            // Donor mirror (non-rawKeyMode rotating case): keep raw K/V in
            // lockstep with the compressed buffers so `lastReturnedKeys` /
            // `lastReturnedValues` slice cheaply without a per-step dequant.
            // Allocated in `compressRawCache` when `isDonor && rotatingMaxSize
            // != nil`. Cheap scatter writes — the dominant cost gone here is
            // the `kCodec.decode(...)` over the full active prefix that the
            // old `compressedAttention` refresh did every decode step.
            if isDonor, rawKeys != nil, rawValues != nil {
                rawKeys![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = keys
                rawValues![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = values
            }
        }

        // For `rawKeyMode` + donor, the raw V mirror also needs the per-step
        // append. `rawKeys` is already written above by the rawKeyMode branch
        // (line ~1450); rawValues mirror is initialised in `compressRawCache`
        // and updated here.
        if rawKeyMode, isDonor, let rawValuesBuf = rawValues {
            _ = rawValuesBuf  // capture
            rawValues![.ellipsis, writeIdx..<(writeIdx + numSteps), 0...] = values
        }

        // [#87 fix] Advance offset so the caller's `compressedWriteOffset =
        // offset` line correctly captures the new write position. Without
        // this, every decode token writes to the same slot (overwriting the
        // prior) and slots beyond it stay zero-initialized — those zero
        // slots cause division-by-zero NaN in the dequant path during the
        // next attention pass.
        offset += numSteps
    }

    /// Compressed-domain attention.
    ///
    /// First call: bulk-compresses the raw prefill cache. Subsequent
    /// calls: encode the new token → run a fused attention path → undo
    /// the V rotation.
    ///
    /// Decode (`L=1`) routing:
    /// - **A path (default)** — TurboFlash. `turboFlashAttention` (or
    ///   `turboFlashSDPAv` for sinks-using models, or
    ///   `turboFlashAttentionCausal` for L>1 prefill) scores directly
    ///   against packed K/V. No FP16 K/V materialisation.
    /// - **B path (opt-in)** — `bulkDequantRotated` materialises a
    ///   per-layer FP16 K/V buffer, then `MLXFast.scaledDotProductAttention`
    ///   runs on the matmul engine. Selected when `useDequantSDPA == true`
    ///   or `useBias == true` (bias correction currently requires the
    ///   B path — TurboFlash kernels don't yet consume the stored
    ///   bias term).
    ///
    /// Prefill (`L>1`) with `.causal` mask uses
    /// `turboFlashAttentionCausal`; otherwise falls back to the
    /// separated `mseScore → softmax → mseWeightedSum` kernels.
    ///
    /// `rawKeyMode` (raw FP16 keys + compressed values) bypasses the K
    /// codec: standard matmul for scoring, compressed-domain Metal kernel
    /// for `Attn · V`.
    public func compressedAttention(
        queries: MLXArray,
        keys newKeys: MLXArray,
        values newValues: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        sinks: MLXArray? = nil,
        windowSize: Int = -1
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

        // Refresh `lastReturnedKeys` / `lastReturnedValues` so KV-sharing
        // readers (Gemma 4 LLM-side `Gemma4ModelInner` threads each donor's
        // current K/V into shared-attention layers via these fields) see
        // the *current* cache state instead of the prefill snapshot that
        // `update(...)` left behind. Pre-fix: shared layers attended over
        // stale prefill-only K/V every decode step — produces clean output
        // on turn 1, accumulating repetition from turn 2 onward as the
        // attended K/V drifts further from what the donor itself saw.
        //
        // Gated on `isDonor` because the dequant ops, while lazy, do get
        // evaluated under MLX's normal forward-pass eval barriers
        // (Gemma 4 prefill's `eval(cache.innerState() + [logits])` from
        // issue #169 in particular). Caches not flagged as donors have no
        // shared readers, so the dequant work is pure overhead. Non-KV-
        // sharing models (Qwen 3.5 / Gemma 4 26B / 31B) leave `isDonor =
        // false` and pay zero on this path. KV-sharing models (Gemma 4
        // E2B / E4B) set `isDonor = true` on the sliding + full donors;
        // those caches do the dequant, others skip.
        if isDonor {
            let attendTokenCount = rotatingMaxSize.map { min(offset, $0) } ?? offset
            if attendTokenCount > 0 {
                // FP16 mirror path (rotating cache): rawKeys / rawValues are
                // kept in lockstep with the compressed buffers via the donor
                // branch of `encodeNewToken` + initialised in
                // `compressRawCache`. Slicing them is free — no dequant per
                // decode step. Measured impact: Gemma 4 E2B turbo4v2 8K
                // decode 26 → ? tok/s (was paying a full-prefix
                // `kCodec.decode(...)` × 2 (K + V) × N donor layers per
                // decode step under the pre-fix refresh).
                if let rk = rawKeys, let rv = rawValues {
                    self.lastReturnedKeys = rk[0..., 0..., ..<attendTokenCount, 0...]
                    self.lastReturnedValues = rv[0..., 0..., ..<attendTokenCount, 0...]
                } else {
                    // Fallback path — non-rotating donor caches (full-attention
                    // donors) and pre-1024 rawKeyMode paths still dequant on
                    // demand. The mirror init only kicks in when
                    // `rotatingMaxSize != nil`, so non-windowed donors land
                    // here. TODO: extend the mirror to growable buffers.
                    if rawKeyMode {
                        self.lastReturnedKeys = rawKeys?[0..., 0..., ..<attendTokenCount, 0...]
                    } else if let kCodec = keyMSECodec,
                        let kpm = keyPackedMSE, let kn = keyNorms
                    {
                        let kState = MSECodecState(
                            norms: kn[0..., 0..., ..<attendTokenCount],
                            packedIndices: kpm[0..., 0..., ..<attendTokenCount, 0...],
                            tokenCount: attendTokenCount, dim: kCodec.dim, bits: kCodec.bits)
                        self.lastReturnedKeys = kCodec.decode(kState)
                    }
                    if let vpm = valPackedMSE, let vn = valNorms {
                        let vState = MSECodecState(
                            norms: vn[0..., 0..., ..<attendTokenCount],
                            packedIndices: vpm[0..., 0..., ..<attendTokenCount, 0...],
                            tokenCount: attendTokenCount, dim: valueMSECodec.dim,
                            bits: valueMSECodec.bits)
                        self.lastReturnedValues = valueMSECodec.decode(vState)
                    }
                }
            }
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

            // Single-pass TurboFlash kernel with sinks + sliding window
            // (`turbo_flash_sdpa_v`). Same (kb,vb) coverage as
            // `turboFlashAttention` plus (2,2) / (8,3).
            let hasTurboFlashSDPAv: Bool = {
                switch (keyBits, valueBits) {
                case (4,4), (4,2), (4,3), (3,2), (3,3),
                     (8,2), (8,3), (8,4), (8,8), (2,2):
                    return true
                default: return false
                }
            }()

            // ─── B path (opt-in via `TURBO_DEQUANT_SDPA=1`) ───
            // Bulk-dequant K/V to FP16 + MLXFast SDPA. Trades a per-layer
            // FP16 working buffer (B*nKV*T*D*2 bytes per decode step) for
            // matrix-engine SDPA performance. Also the path that consumes
            // the stored `useBias` term — TurboFlash kernels don't yet
            // apply it during attention, so `useBias` forces this branch.
            //
            // Skipped when:
            //   - cache is a KV-sharing donor (already pays per-step FP16
            //     mirror updates for shared readers; layering another
            //     dequant doubles traffic for no gain).
            //   - bit widths fall outside `{2, 4, 8}` (no dequant kernel
            //     instantiation).
            let dequantEligible = L == 1 && !isDonor
                && (keyBits == 4 || keyBits == 8 || keyBits == 2)
                && (valueBits == 4 || valueBits == 8 || valueBits == 2)
            // Spec 043 Phase 4 — when sinks are present we route the bias
            // path through the new bias-aware `turbo_flash_sdpa_v` kernel
            // (A path) rather than dequant-first. Without sinks, bias still
            // forces B path because the non-sinks `turbo_flash_attention`
            // family doesn't take bias inputs (yet).
            let hasBiasAwareAPath =
                useBias && sinks != nil && L == 1 && hasTurboFlashSDPAv
            let runDequantSDPA = dequantEligible &&
                (useDequantSDPA || (useBias && !hasBiasAwareAPath))
            if runDequantSDPA {
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
                // The precompiled `turbo_dequant_rotated` kernel only
                // instantiates `bfloat` and `half` outputs. Some models
                // (e.g. Gemma 4 26B-A4B) run attention in fp32; clamp to
                // bf16 for the dequant + SDPA + rotation chain and cast
                // the final output back to the model's dtype.
                let originalDtype = queries.dtype
                let dt: DType = (originalDtype == .bfloat16 || originalDtype == .float16)
                    ? originalDtype : .bfloat16
                let qForSDPA = (qRot.dtype == dt) ? qRot : qRot.asType(dt)
                var kFP = TurboQuantKernelOps.bulkDequantRotated(
                    packed: keyPackedMSE![0..., 0..., ..<tokenCount, 0...],
                    norms: keyNorms![0..., 0..., ..<tokenCount],
                    codebook: keyMSECodec.codebook,
                    tokenCount: tokenCount, bits: keyBits, dim: headDim, dtype: dt)
                var vFP = TurboQuantKernelOps.bulkDequantRotated(
                    packed: valPackedMSE![0..., 0..., ..<tokenCount, 0...],
                    norms: valNorms![0..., 0..., ..<tokenCount],
                    codebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, bits: valueBits, dim: headDim, dtype: dt)
                // Bias-aware dequant in rotated space: the kernel emits
                // the centered rotated reconstruction (`codebook[idx] *
                // norm`); we add `b * rotatedOnes` to recover the rotated
                // reconstruction of the uncentered vector. See
                // `MSECodec.decodeRotated`.
                if useBias {
                    if let kb = keyBias {
                        let kbExp = expandedDimensions(
                            kb[0..., 0..., ..<tokenCount], axis: -1).asType(dt)
                        kFP = kFP + kbExp * keyMSECodec.rotatedOnes.asType(dt)
                    }
                    if let vb = valBias {
                        let vbExp = expandedDimensions(
                            vb[0..., 0..., ..<tokenCount], axis: -1).asType(dt)
                        vFP = vFP + vbExp * valueMSECodec.rotatedOnes.asType(dt)
                    }
                }
                // qRot already folds `scale` (prepareQueriesScaled).
                // Q and K both live in rotated codec space — the
                // rotation cancels in `Q · K`. V rotation is undone by
                // the post-matmul on `valueMSECodec.rotation`.
                let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
                    windowSize > 0 ? .slidingWindow(size: windowSize) : .none
                let rotOut = MLXFast.scaledDotProductAttention(
                    queries: qForSDPA.reshaped([B, nQHeads, L, headDim]),
                    keys: kFP, values: vFP,
                    scale: 1.0, mask: maskMode,
                    sinks: sinks)
                output = matmul(rotOut, valueMSECodec.rotation)
                if output.dtype != originalDtype {
                    output = output.asType(originalDtype)
                }
                BenchmarkSignpost.end(vH)
            } else if let sinksArr = sinks, L == 1, hasTurboFlashSDPAv {
                // ─── A path (sinks variant) ───
                // Single-pass TurboFlash kernel that folds sinks + sliding
                // window into its online softmax. Scores directly against
                // packed K/V; no FP16 working buffer.
                //
                // Spec 043 Phase 4 — when `useBias` is set, pass the
                // per-vector bias arrays + per-codec rotated_ones so the
                // kernel applies `b[t] * rotated_ones[d]` inside the
                // dequant prologue. Slices match the kernel's
                // `[nKV, tokenCount]` layout (collapse B + H, assuming
                // B=1 — matches the rest of this dispatch).
                let vH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.tqValue)
                let causal = (windowSize > 0)
                var kBiasArg: MLXArray? = nil
                var vBiasArg: MLXArray? = nil
                var kRotOnesArg: MLXArray? = nil
                var vRotOnesArg: MLXArray? = nil
                if useBias, let kb = keyBias, let vb = valBias {
                    kBiasArg = kb[0..., 0..., ..<tokenCount].reshaped([nKVHeads, tokenCount])
                    vBiasArg = vb[0..., 0..., ..<tokenCount].reshaped([nKVHeads, tokenCount])
                    kRotOnesArg = keyMSECodec.rotatedOnes.reshaped([headDim])
                    vRotOnesArg = valueMSECodec.rotatedOnes.reshaped([headDim])
                }
                let rotOut = TurboQuantKernelOps.turboFlashSDPAv(
                    rotatedQueries: flatQ,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyMSECodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, repeatCount: nRepeats,
                    keyBits: self.keyBits, valueBits: self.valueBits, dim: headDim,
                    sinks: sinksArr,
                    causal: causal,
                    windowSize: causal ? windowSize : -1,
                    valRotation: valRotation,
                    keyBias: kBiasArg,
                    valBias: vBiasArg,
                    keyRotatedOnes: kRotOnesArg,
                    valRotatedOnes: vRotOnesArg
                )
                BenchmarkSignpost.end(vH)
                output = rotOut.reshaped([B, nQHeads, L, headDim])
                if output.dtype != queries.dtype {
                    output = output.asType(queries.dtype)
                }
                return output
            } else if L == 1 && hasTurboFlashKernel {
                // ─── A path (no-sinks decode) ───
                // `turboFlashAttention` fuses score + online softmax +
                // value-weighted sum into one pass over packed K/V.
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
                // ─── A path (causal prefill, L>1) ───
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
        // Resolve [B, H, T, D] from whichever storage is live. Packed buffers
        // are authoritative post-compression; `rawKeys` is kept during
        // `rawKeyMode` and across the first decode call.
        let shapeSrc: MLXArray? = keyPackedMSE ?? valPackedMSE ?? rawKeys
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
                    // (+ [valBias] when `useBias=true`)
                    guard let rk = rawKeys,
                          let vpm = valPackedMSE, let vn = valNorms,
                          offset > 0 else { return [] }
                    var out: [MLXArray] = [
                        rk[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                    if useBias, let vb = valBias {
                        out.append(vb[0..., 0..., ..<offset])
                    }
                    return out
                } else {
                    // Standard compressed: [keyPacked, keyNorms, valPacked, valNorms]
                    // (+ [keyBias, valBias] when `useBias=true`)
                    guard let kpm = keyPackedMSE, let kn = keyNorms,
                          let vpm = valPackedMSE, let vn = valNorms,
                          offset > 0 else { return [] }
                    var out: [MLXArray] = [
                        kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                    if useBias, let kb = keyBias, let vb = valBias {
                        out.append(kb[0..., 0..., ..<offset])
                        out.append(vb[0..., 0..., ..<offset])
                    }
                    return out
                }
            } else {
                guard let rk = rawKeys, let rv = rawValues, offset > 0 else { return [] }
                return [rk[0..., 0..., ..<offset, 0...], rv[0..., 0..., ..<offset, 0...]]
            }
        }
        set {
            if rawKeyMode && (newValue.count == 3 || newValue.count == 4) {
                // Raw-K mode compressed state: [rawKeys, valPacked, valNorms]
                // (+ [valBias] when `useBias=true`)
                rawKeys = newValue[0]
                rawAllocSteps = newValue[0].dim(2)
                valPackedMSE = newValue[1]; valNorms = newValue[2]
                if newValue.count == 4 { valBias = newValue[3] }
                offset = newValue[0].dim(2)
                compressedAllocSteps = newValue[1].dim(2)
                isCompressed = true
            } else if newValue.count == 4 || newValue.count == 6 {
                // Standard compressed state: [keyPacked, keyNorms, valPacked, valNorms]
                // (+ [keyBias, valBias] when `useBias=true`)
                keyPackedMSE = newValue[0]; keyNorms = newValue[1]
                valPackedMSE = newValue[2]; valNorms = newValue[3]
                if newValue.count == 6 {
                    keyBias = newValue[4]; valBias = newValue[5]
                }
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
            keyBias = nil; valBias = nil
            compressedAllocSteps = 0; isCompressed = false
        }
        return trimCount
    }

    public override var storageKind: KVStorageKind {
        .turboCompressed(keyBits: keyBits, valueBits: valueBits)
    }

    // MARK: - Dequant compressed → raw (issue #185 / #197 prefix-cache fix)
    //
    // The prefix-cache snapshotter runs at stream end — by then a
    // TurboQuant cache that participated in `compressedAttention(...)`
    // has transitioned to compressed mode (`isCompressed == true`,
    // `rawKeys` / `rawValues` cleared). On the next request, the cache
    // is hydrated as compressed but the warm-turn suffix prefill calls
    // `update(...)` which expects `rawKeys` to be live — and overwrites
    // the hydrated state with a fresh zero buffer instead of appending
    // to it. The downstream SDPA then attends over zero-keys for the
    // prefix tokens and produces garbage.
    //
    // Two paths fix this:
    //   1. Dequant compressed K/V back to raw FP16 at snapshot time so
    //      every snapshot is raw (`isCompressed == false`, 2-array
    //      state). Hydrate then always loads into `rawKeys` /
    //      `rawValues`, and the warm-turn suffix prefill's `update(...)`
    //      concatenates against the hydrated raw buffer correctly.
    //      Cost: one round of TurboQuant dequant precision loss at
    //      snapshot. Acceptable for warm-turn TTFT use cases.
    //   2. Implement a compressed-suffix-prefill path in `update(...)`
    //      that encodes the new tokens via `fusedEncodeDispatch` and
    //      appends to `keyPackedMSE` / `valPackedMSE`. Lossless but
    //      adds an additional codepath through the hot prefill loop.
    //
    // This method implements path 1. The serialiser calls it before
    // snapshot when the cache is in compressed mode.
    //
    // Returns `(rawKeys, rawValues)` as `[B, H, offset, headDim]`
    // FP16 / bfloat16 arrays in *original* basis (already passed
    // through `decode()`'s `Π^T` inverse rotation).
    //
    // - Throws `PrefixKVCacheError.snapshotInvariantViolation` if the
    //   codec hasn't been built yet (compressed state without a
    //   matching codec is impossible in normal flow but we guard
    //   defensively).
    public func dequantToRaw() throws -> (MLXArray, MLXArray) {
        guard let vCodec = valueMSECodec else {
            throw PrefixKVCacheError.snapshotInvariantViolation(
                "TurboQuantizedKVCache.dequantToRaw: valueMSECodec is nil "
                    + "(compressed cache without an initialised codec)")
        }
        let tokens = offset
        guard tokens > 0 else {
            throw PrefixKVCacheError.snapshotInvariantViolation(
                "TurboQuantizedKVCache.dequantToRaw: offset == 0")
        }
        // Dequant values: codec.decode(state) returns
        // `[B, H, T, headDim]` in original basis (after Π^T).
        guard let vpm = valPackedMSE, let vn = valNorms else {
            throw PrefixKVCacheError.snapshotInvariantViolation(
                "TurboQuantizedKVCache.dequantToRaw: compressed V buffers missing")
        }
        let vState = MSECodecState(
            norms: vn[0..., 0..., ..<tokens],
            packedIndices: vpm[0..., 0..., ..<tokens, 0...],
            tokenCount: tokens, dim: vCodec.dim, bits: vCodec.bits)
        let valuesRaw = vCodec.decode(vState)

        // Dequant keys: rawKeyMode keeps K in FP16 (no codec needed);
        // standard mode dequants via keyMSECodec.
        let keysRaw: MLXArray
        if rawKeyMode {
            guard let rk = rawKeys else {
                throw PrefixKVCacheError.snapshotInvariantViolation(
                    "TurboQuantizedKVCache.dequantToRaw: rawKeyMode without rawKeys")
            }
            keysRaw = rk[0..., 0..., ..<tokens, 0...]
        } else {
            guard let kCodec = keyMSECodec else {
                throw PrefixKVCacheError.snapshotInvariantViolation(
                    "TurboQuantizedKVCache.dequantToRaw: keyMSECodec is nil")
            }
            guard let kpm = keyPackedMSE, let kn = keyNorms else {
                throw PrefixKVCacheError.snapshotInvariantViolation(
                    "TurboQuantizedKVCache.dequantToRaw: compressed K buffers missing")
            }
            let kState = MSECodecState(
                norms: kn[0..., 0..., ..<tokens],
                packedIndices: kpm[0..., 0..., ..<tokens, 0...],
                tokenCount: tokens, dim: kCodec.dim, bits: kCodec.bits)
            keysRaw = kCodec.decode(kState)
        }
        return (keysRaw, valuesRaw)
    }

    // MARK: - innerState (issue #185 fix)
    //
    // The default `innerState()` returns `[]`, which means the caller's
    // `eval(cache.innerState() + [logits])` barrier (Gemma 4 prepare,
    // issue #169) cannot commit pending K/V writes against a
    // `TurboQuantizedKVCache`. With KV-shared layers (Gemma 4 E2B / E4B),
    // shared layers consume the donor's K/V *across* the model graph;
    // un-committed prefill writes from the donor then surface as garbage
    // when the iterator advances to the first decode step.
    //
    // Override to expose every live raw / compressed buffer the cache
    // holds — the eval barrier walks the returned MLXArrays and commits
    // the in-place mutation graph behind each one. `compactMap` skips
    // nil fields, so the override is safe for any mix of pre /
    // post-compression and raw-key / standard mode.
    public override func innerState() -> [MLXArray] {
        [rawKeys, rawValues, keyPackedMSE, keyNorms, valPackedMSE, valNorms]
            .compactMap { $0 }
    }

    // MARK: - metaState (spec 017 phase 1B+ — issue #197)
    //
    // Round-tripped fields (10), in order:
    //   0 bits, 1 keyBits, 2 valueBits, 3 seed,
    //   4 rotatingMaxSize ("None" sentinel for nil),
    //   5 step, 6 offset, 7 isCompressed ("1"/"0"),
    //   8 rotatingIdx, 9 compressedWriteOffset.
    //
    // The first 6 entries (constructor params) are validated against the
    // target cache rather than written into it — `bits`, `keyBits`,
    // `valueBits`, `seed`, `rotatingMaxSize`, and `step` are stored on
    // `let` properties, so the caller must construct the hydrate target
    // with matching values. A mismatch is a configuration error and
    // fatals here, mirroring `StandardKVCache.metaState`'s strictness on
    // `maxSize` / `keep`.
    //
    // The remaining 4 entries are mutable runtime state. `state` setter
    // already restores `offset` + `isCompressed` + `*AllocSteps` from
    // the assigned-array shapes — `metaState` reasserts `offset` /
    // `isCompressed` (defence in depth) and adds `rotatingIdx` /
    // `compressedWriteOffset`, which the state setter cannot recover
    // from the arrays alone. `rawAllocSteps` / `compressedAllocSteps`
    // are intentionally not round-tripped — they must match the buffer
    // dims that the state setter just assigned (the snapshot's arrays
    // were sliced to `offset` for compactness, so `*AllocSteps == offset`
    // post-hydrate; subsequent `update(...)` grows the buffer as needed).
    // Order matters: callers assign `state` first, then `metaState`.
    public override var metaState: [String] {
        get {
            return [
                String(bits),
                String(keyBits),
                String(valueBits),
                String(seed),
                rotatingMaxSize.map(String.init) ?? "None",
                String(step),
                String(offset),
                isCompressed ? "1" : "0",
                String(rotatingIdx),
                String(compressedWriteOffset),
            ]
        }
        set {
            // No-op on empty / default metaState (lets the BaseKVCache [""]
            // default round-trip cleanly through unrelated callers).
            if newValue.isEmpty || (newValue.count == 1 && newValue[0].isEmpty) {
                return
            }
            guard newValue.count == 10 else {
                fatalError(
                    "TurboQuantizedKVCache metaState must have 10 values, got \(newValue.count)")
            }
            guard let snapBits = Int(newValue[0]),
                let snapKeyBits = Int(newValue[1]),
                let snapValueBits = Int(newValue[2]),
                let snapSeed = UInt64(newValue[3]),
                let snapStep = Int(newValue[5]),
                let snapOffset = Int(newValue[6]),
                let snapRotIdx = Int(newValue[8]),
                let snapCwo = Int(newValue[9])
            else {
                fatalError("TurboQuantizedKVCache metaState parse failure: \(newValue)")
            }
            // Constructor-param validation (matching `let`s on the cache).
            precondition(
                snapBits == bits,
                "TurboQuantizedKVCache bits mismatch: snapshot=\(snapBits) target=\(bits)")
            precondition(
                snapKeyBits == keyBits,
                "TurboQuantizedKVCache keyBits mismatch: snapshot=\(snapKeyBits) target=\(keyBits)")
            precondition(
                snapValueBits == valueBits,
                "TurboQuantizedKVCache valueBits mismatch: snapshot=\(snapValueBits) target=\(valueBits)"
            )
            precondition(
                snapSeed == seed,
                "TurboQuantizedKVCache seed mismatch: snapshot=\(snapSeed) target=\(seed)")
            precondition(
                snapStep == step,
                "TurboQuantizedKVCache step mismatch: snapshot=\(snapStep) target=\(step)")
            if newValue[4] == "None" {
                precondition(
                    rotatingMaxSize == nil,
                    "TurboQuantizedKVCache rotatingMaxSize mismatch: snapshot=nil target=\(rotatingMaxSize!)"
                )
            } else if let snapMaxSize = Int(newValue[4]) {
                precondition(
                    rotatingMaxSize == snapMaxSize,
                    "TurboQuantizedKVCache rotatingMaxSize mismatch: snapshot=\(snapMaxSize) target=\(rotatingMaxSize.map(String.init) ?? "nil")"
                )
            } else {
                fatalError(
                    "TurboQuantizedKVCache rotatingMaxSize unparsable: \(newValue[4])")
            }
            // Restore mutable runtime state. `state` setter already
            // populated `offset` / `isCompressed` / `*AllocSteps` from
            // the array shapes; reasserting offset + isCompressed is
            // defence-in-depth, and rotatingIdx / compressedWriteOffset
            // are recovered here because the state arrays alone don't
            // carry them.
            self.offset = snapOffset
            self.isCompressed = newValue[7] == "1"
            self.rotatingIdx = snapRotIdx
            self.compressedWriteOffset = snapCwo
        }
    }
}
