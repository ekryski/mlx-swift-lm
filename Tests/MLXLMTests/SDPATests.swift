import MLX
import MLXNN
import XCTest

/// Tests MLXFast.scaledDotProductAttention correctness against a naive
/// manual implementation for all Gemma4 head_dim × GQA configurations.
///
/// Background: Gemma4 26b and 31b (attention_k_eq_v=true) produce incoherent
/// output while E2B/E4B (attention_k_eq_v=false) work correctly.  The custom
/// mlx-swift fork adds Steel SDPA instantiations for BD=256 and BD=512 —
/// this test verifies those kernels produce the same result as a naive
/// matmul→softmax→matmul reference.
final class SDPATests: XCTestCase {

    // MARK: - Gemma4 E2B (WORKS — baseline)

    func testE2B_Sliding_Decode() throws {
        // E2B sliding: nHeads=8, nKVHeads=1, headDim=256, GQA=8
        try runSDPAComparison(
            B: 1, nHeads: 8, nKVHeads: 1, L: 1, S: 42, headDim: 256,
            scale: 1.0, useCausalMask: false, label: "E2B sliding decode")
    }

    func testE2B_Global_Decode() throws {
        // E2B global: nHeads=8, nKVHeads=1, headDim=512, GQA=8
        try runSDPAComparison(
            B: 1, nHeads: 8, nKVHeads: 1, L: 1, S: 42, headDim: 512,
            scale: 1.0, useCausalMask: false, label: "E2B global decode")
    }

    func testE2B_Sliding_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 8, nKVHeads: 1, L: 41, S: 41, headDim: 256,
            scale: 1.0, useCausalMask: true, label: "E2B sliding prefill")
    }

    func testE2B_Global_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 8, nKVHeads: 1, L: 41, S: 41, headDim: 512,
            scale: 1.0, useCausalMask: true, label: "E2B global prefill")
    }

    // MARK: - Gemma4 26b (BROKEN — attention_k_eq_v=true)

    func testGemma4_26b_Sliding_Decode() throws {
        // 26b sliding: nHeads=16, nKVHeads=8, headDim=256, GQA=2
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 8, L: 1, S: 42, headDim: 256,
            scale: 1.0, useCausalMask: false, label: "26b sliding decode")
    }

    func testGemma4_26b_Global_Decode() throws {
        // 26b global: nHeads=16, nKVHeads=2, headDim=512, GQA=8
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 1, S: 42, headDim: 512,
            scale: 1.0, useCausalMask: false, label: "26b global decode")
    }

    func testGemma4_26b_Sliding_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 8, L: 41, S: 41, headDim: 256,
            scale: 1.0, useCausalMask: true, label: "26b sliding prefill")
    }

    func testGemma4_26b_Global_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 41, S: 41, headDim: 512,
            scale: 1.0, useCausalMask: true, label: "26b global prefill")
    }

    // MARK: - Gemma4 31b (BROKEN — attention_k_eq_v=true)

    func testGemma4_31b_Sliding_Decode() throws {
        // 31b sliding: nHeads=32, nKVHeads=16, headDim=256, GQA=2
        try runSDPAComparison(
            B: 1, nHeads: 32, nKVHeads: 16, L: 1, S: 42, headDim: 256,
            scale: 1.0, useCausalMask: false, label: "31b sliding decode")
    }

    func testGemma4_31b_Global_Decode() throws {
        // 31b global: nHeads=32, nKVHeads=4, headDim=512, GQA=8
        try runSDPAComparison(
            B: 1, nHeads: 32, nKVHeads: 4, L: 1, S: 42, headDim: 512,
            scale: 1.0, useCausalMask: false, label: "31b global decode")
    }

    func testGemma4_31b_Sliding_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 32, nKVHeads: 16, L: 41, S: 41, headDim: 256,
            scale: 1.0, useCausalMask: true, label: "31b sliding prefill")
    }

    func testGemma4_31b_Global_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 32, nKVHeads: 4, L: 41, S: 41, headDim: 512,
            scale: 1.0, useCausalMask: true, label: "31b global prefill")
    }

    // MARK: - Longer sequences (stress test cache wrapping)

    func testGemma4_26b_Global_Decode_LongContext() throws {
        // After many decode steps (S=1024)
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 1, S: 1024, headDim: 512,
            scale: 1.0, useCausalMask: false, label: "26b global decode S=1024")
    }

    func testGemma4_26b_Global_Prefill_LongContext() throws {
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 512, S: 512, headDim: 512,
            scale: 1.0, useCausalMask: true, label: "26b global prefill L=512")
    }

    // MARK: - K=V sharing test (values aliased to keys)

    func testGemma4_26b_KEqV_Decode() throws {
        // Simulate attention_k_eq_v: V = vNorm(K_raw), K = kNorm(K_raw) + RoPE
        // This tests whether SDPA handles K != V correctly when they originate
        // from the same projection but have different norms applied
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 1, S: 42, headDim: 512,
            scale: 1.0, useCausalMask: false, label: "26b K=V aliased decode",
            keqv: true)
    }

    func testGemma4_26b_KEqV_Prefill() throws {
        try runSDPAComparison(
            B: 1, nHeads: 16, nKVHeads: 2, L: 41, S: 41, headDim: 512,
            scale: 1.0, useCausalMask: true, label: "26b K=V aliased prefill",
            keqv: true)
    }

    // MARK: - Bool mask vs causal mode comparison

    func testGemma4_26b_BoolMask_vs_Causal() throws {
        // Compare .causal mask mode with explicit bool mask — should be identical
        let B = 1, nHeads = 16, nKVHeads = 2, L = 41, S = 41, headDim = 512
        let scale: Float = 1.0

        MLXRandom.seed(42)
        let q = MLXRandom.normal([B, nHeads, L, headDim]).asType(.bfloat16)
        let k = MLXRandom.normal([B, nKVHeads, S, headDim]).asType(.bfloat16)
        let v = MLXRandom.normal([B, nKVHeads, S, headDim]).asType(.bfloat16)

        // SDPA with .causal
        let outCausal = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
        eval(outCausal)

        // SDPA with explicit bool mask
        let boolMask = MLXArray.tri(L, m: S, k: 0, dtype: .bool)
        let outBool = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(boolMask))
        eval(outBool)

        let diff = maxAbsDiff(outCausal, outBool)
        let mean = absMean(outCausal)
        let relDiff = mean > 0 ? diff / mean : diff
        print("[26b bool vs causal] max_diff=\(diff) rel=\(relDiff)")
        XCTAssertLessThan(relDiff, 1e-3, "Bool mask and .causal should produce identical results")
    }

    // MARK: - Helpers

    /// Naive reference SDPA: matmul → scale → mask → softmax → matmul
    /// All in float32 to avoid precision issues in the reference.
    private func naiveSDPA(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, mask: MLXArray? = nil
    ) -> MLXArray {
        let q = queries.asType(.float32)
        let k = keys.asType(.float32)
        let v = values.asType(.float32)

        let nHeads = q.dim(1)
        let nKVHeads = k.dim(1)
        let gqaFactor = nHeads / nKVHeads

        // Repeat K/V heads for GQA
        let kExpanded: MLXArray
        let vExpanded: MLXArray
        if gqaFactor > 1 {
            // [B, nKVHeads, S, D] -> [B, nKVHeads, 1, S, D] -> [B, nKVHeads, gqa, S, D] -> [B, nHeads, S, D]
            let kr = MLX.expandedDimensions(k, axis: 2)
            let vr = MLX.expandedDimensions(v, axis: 2)
            let kTiled = MLX.repeated(kr, count: gqaFactor, axis: 2)
            let vTiled = MLX.repeated(vr, count: gqaFactor, axis: 2)
            kExpanded = kTiled.reshaped(q.dim(0), nHeads, k.dim(2), k.dim(3))
            vExpanded = vTiled.reshaped(q.dim(0), nHeads, v.dim(2), v.dim(3))
        } else {
            kExpanded = k
            vExpanded = v
        }

        // scores = Q @ K^T * scale
        var scores = matmul(q, kExpanded.transposed(0, 1, 3, 2)) * scale

        // Apply mask
        if let mask {
            if mask.dtype == .bool {
                scores = MLX.where(mask, scores, MLXArray(Float(-1e9)))
            } else {
                scores = scores + mask
            }
        }

        // softmax
        let weights = softmax(scores, axis: -1)

        // output = weights @ V
        return matmul(weights, vExpanded)
    }

    private func runSDPAComparison(
        B: Int, nHeads: Int, nKVHeads: Int, L: Int, S: Int, headDim: Int,
        scale: Float, useCausalMask: Bool, label: String, keqv: Bool = false
    ) throws {
        // Deterministic random seed for reproducibility
        MLXRandom.seed(42)

        let q = MLXRandom.normal([B, nHeads, L, headDim]).asType(.bfloat16)
        var k = MLXRandom.normal([B, nKVHeads, S, headDim]).asType(.bfloat16)
        var v: MLXArray

        if keqv {
            // Simulate K=V: both derive from same projection, different norms
            let raw = MLXRandom.normal([B, nKVHeads, S, headDim]).asType(.bfloat16)
            // k_norm (RMSNorm with learned weight)
            k = MLXFast.rmsNorm(raw, weight: MLXArray.ones([headDim]).asType(.bfloat16), eps: 1e-6)
            // v_norm (RMSNorm without weight)
            v = MLXFast.rmsNorm(raw, weight: MLXArray.mlxNone, eps: 1e-6)
        } else {
            v = MLXRandom.normal([B, nKVHeads, S, headDim]).asType(.bfloat16)
        }

        eval(q, k, v)

        // Build mask
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        let naiveMask: MLXArray?
        if useCausalMask && L > 1 {
            mask = .causal
            // Build equivalent bool mask for naive implementation
            naiveMask = MLXArray.tri(L, m: S, k: 0, dtype: .bool)
        } else {
            mask = .none
            naiveMask = nil
        }

        // MLXFast SDPA (routes to Steel kernel for supported configs)
        let outFast = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        eval(outFast)

        // Naive reference (float32)
        let outNaive = naiveSDPA(queries: q, keys: k, values: v, scale: scale, mask: naiveMask)
        eval(outNaive)

        // Compare in float32
        let diff = maxAbsDiff(outFast, outNaive)
        let mean = absMean(outNaive)
        let relDiff = mean > 0 ? diff / mean : diff

        // Also compute per-head max diff to identify which heads diverge
        let fastF32 = outFast.asType(.float32)
        let naiveF32 = outNaive.asType(.float32)
        let perHeadDiff = abs(fastF32 - naiveF32).max(axes: [0, 2, 3])
        eval(perHeadDiff)
        let perHeadMax = perHeadDiff.asArray(Float.self)

        print("[\(label)] max_diff=\(String(format: "%.6f", diff)) rel=\(String(format: "%.6f", relDiff)) mean=\(String(format: "%.4f", mean))")
        print("  shapes: Q=\(q.shape) K=\(k.shape) V=\(v.shape) out=\(outFast.shape)")
        if nHeads <= 32 {
            let headStrs = perHeadMax.map { String(format: "%.4f", $0) }
            print("  per_head_max_diff: [\(headStrs.joined(separator: ", "))]")
        }

        // Print first few output values for visual comparison
        let fastFlat = fastF32.reshaped(-1)
        let naiveFlat = naiveF32.reshaped(-1)
        let n = min(8, fastFlat.dim(0))
        var fastVals = "  fast[0:\(n)]:  "
        var naiveVals = "  naive[0:\(n)]: "
        for i in 0..<n {
            fastVals += String(format: "%.4f ", fastFlat[i].item(Float.self))
            naiveVals += String(format: "%.4f ", naiveFlat[i].item(Float.self))
        }
        print(fastVals)
        print(naiveVals)

        // Thresholds: bf16 has ~3 decimal digits of precision.
        // For long sequences, accumulated softmax error can reach ~0.01.
        // Relative diff > 5% indicates a kernel bug, not just rounding.
        // bf16 accumulation error grows with sequence length.
        // S=1024 reaches ~2.1% relative error which is expected.
        let threshold: Float = S >= 512 ? 0.025 : (L > 1 ? 0.05 : 0.02)
        XCTAssertLessThan(
            relDiff, threshold,
            "[\(label)] SDPA relative error \(relDiff) exceeds threshold \(threshold). "
            + "Max abs diff: \(diff), mean: \(mean)")
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func absMean(_ a: MLXArray) -> Float {
        abs(a.asType(.float32)).mean().item(Float.self)
    }
}
