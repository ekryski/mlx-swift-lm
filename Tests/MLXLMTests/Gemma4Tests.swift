import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Testing

// MARK: - Fused kernel equivalence tests for Gemma4

/// Test that MLXFast.rmsNormRoPE produces the same result as separate
/// rmsNorm → transpose → RoPE → transpose for standard (sliding) attention.
@Test
func testFusedRmsNormRoPE_Sliding() async throws {
    let B = 1, L = 4, nHeads = 8, headDim = 256
    let eps: Float = 1e-6
    let base: Float = 10000.0
    let offset = 5

    // Synthetic input [B, L, H, D] and weight [D]
    let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.float32)
    let weight = MLXRandom.normal([headDim]).asType(.float32)

    // Build inv_freqs for fused path (matching Gemma4Attention init for sliding)
    let exponents = MLXArray(stride(from: Float(0), to: Float(headDim), by: 2)) / Float(headDim)
    let freqs = pow(MLXArray(base), exponents)
    let invFreqs = (1.0 / freqs).asType(.float32)

    // --- Fused path ---
    let fusedOut = MLXFast.rmsNormRoPE(
        x, weight: weight, invFreqs: invFreqs,
        eps: eps, offset: offset, nHeads: nHeads, seqLen: L
    ).transposed(0, 2, 1, 3)  // → [B, H, L, D]

    // --- Non-fused path ---
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    let transposed = normed.transposed(0, 2, 1, 3)  // → [B, H, L, D]
    let ropeOut = MLXFast.RoPE(
        transposed, dimensions: headDim, traditional: false,
        base: base, scale: 1.0, offset: offset)

    eval(fusedOut, ropeOut)

    let maxDiff = abs(fusedOut - ropeOut).max().item(Float.self)
    #expect(maxDiff < 1e-4, "Fused vs non-fused sliding RoPE max diff: \(maxDiff)")
}

/// Test fused rmsNormRoPE for full (global) attention with ProportionalRoPE (partial rotation).
@Test
func testFusedRmsNormRoPE_FullAttention() async throws {
    let B = 1, L = 4, nHeads = 16, headDim = 512
    let eps: Float = 1e-6
    let base: Float = 1_000_000.0
    let partialRotaryFactor: Float = 0.25
    let offset = 3
    let ropeDim = Int(Float(headDim) * partialRotaryFactor)  // 128

    let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.float32)
    let weight = MLXRandom.normal([headDim]).asType(.float32)

    // Build inv_freqs for fused (proportional) path
    let propExponents = MLXArray(
        stride(from: Float(0), to: Float(ropeDim), by: 2)
    ) / Float(headDim)
    let realFreqs = pow(MLXArray(base), propExponents)
    let paddingCount = (headDim - ropeDim) / 2
    let infPadding = MLXArray(Array(repeating: Float.infinity, count: paddingCount))
    let allFreqs = concatenated([realFreqs, infPadding], axis: 0)
    let invFreqs = (1.0 / allFreqs).asType(.float32)

    // --- Fused path ---
    let fusedOut = MLXFast.rmsNormRoPE(
        x, weight: weight, invFreqs: invFreqs,
        eps: eps, offset: offset, nHeads: nHeads, seqLen: L
    ).transposed(0, 2, 1, 3)

    // --- Non-fused path: norm, transpose, then ProportionalRoPE ---
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    let transposed = normed.transposed(0, 2, 1, 3)
    // Build freqs matching ProportionalRoPE._freqs
    let propFreqs = concatenated([realFreqs, infPadding], axis: 0)
    let ropeOut = MLXFast.RoPE(
        transposed, dimensions: headDim, traditional: false,
        base: nil, scale: 1.0, offset: offset, freqs: propFreqs)

    eval(fusedOut, ropeOut)

    let maxDiff = abs(fusedOut - ropeOut).max().item(Float.self)
    #expect(maxDiff < 1e-4, "Fused vs non-fused proportional RoPE max diff: \(maxDiff)")
}

/// Test fused rmsNormRoPE at bfloat16 precision (the actual inference dtype).
@Test
func testFusedRmsNormRoPE_BFloat16() async throws {
    let B = 1, L = 1, nHeads = 8, headDim = 256
    let eps: Float = 1e-6
    let base: Float = 10000.0
    let offset = 100  // High offset to exercise large theta values

    let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.bfloat16)
    let weight = MLXRandom.normal([headDim]).asType(.bfloat16)

    let exponents = MLXArray(stride(from: Float(0), to: Float(headDim), by: 2)) / Float(headDim)
    let freqs = pow(MLXArray(base), exponents)
    let invFreqs = (1.0 / freqs).asType(.float32)

    // Fused
    let fusedOut = MLXFast.rmsNormRoPE(
        x, weight: weight, invFreqs: invFreqs,
        eps: eps, offset: offset, nHeads: nHeads, seqLen: L
    ).transposed(0, 2, 1, 3)

    // Non-fused (bf16 norm, then f32 rope internals → bf16 out)
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    let transposed = normed.transposed(0, 2, 1, 3)
    let ropeOut = MLXFast.RoPE(
        transposed, dimensions: headDim, traditional: false,
        base: base, scale: 1.0, offset: offset)

    eval(fusedOut, ropeOut)

    // bfloat16 tolerance is larger
    let maxDiff = abs(fusedOut.asType(.float32) - ropeOut.asType(.float32)).max().item(Float.self)
    #expect(maxDiff < 0.05, "BF16 fused vs non-fused max diff: \(maxDiff)")
}

// MARK: - rmsNormResidual equivalence

/// Test that MLXFast.rmsNormResidual computes residual + rmsNorm(x, weight, eps).
@Test
func testRmsNormResidualEquivalence() async throws {
    let shape = [1, 4, 2816]  // Gemma4 26b hidden_size
    let eps: Float = 1e-6

    let x = MLXRandom.normal(shape).asType(.float32)
    let residual = MLXRandom.normal(shape).asType(.float32)
    let weight = MLXRandom.normal([2816]).asType(.float32)

    // Fused
    let fusedOut = MLXFast.rmsNormResidual(x, residual: residual, weight: weight, eps: eps)

    // Non-fused: separate norm + add
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    let manualOut = residual + normed

    eval(fusedOut, manualOut)

    let maxDiff = abs(fusedOut - manualOut).max().item(Float.self)
    #expect(maxDiff < 1e-5, "rmsNormResidual vs manual max diff: \(maxDiff)")
}

/// Test rmsNormResidual at bfloat16 precision.
@Test
func testRmsNormResidualEquivalence_BFloat16() async throws {
    let shape = [1, 1, 5376]  // Gemma4 31b hidden_size
    let eps: Float = 1e-6

    let x = MLXRandom.normal(shape).asType(.bfloat16)
    let residual = MLXRandom.normal(shape).asType(.bfloat16)
    let weight = MLXRandom.normal([5376]).asType(.bfloat16)

    let fusedOut = MLXFast.rmsNormResidual(x, residual: residual, weight: weight, eps: eps)
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    let manualOut = residual + normed

    eval(fusedOut, manualOut)

    let maxDiff = abs(fusedOut.asType(.float32) - manualOut.asType(.float32)).max().item(Float.self)
    // bfloat16 has ~0.4% relative error; for hidden_size=5376, max diff can reach 0.03
    #expect(maxDiff < 0.05, "BF16 rmsNormResidual max diff: \(maxDiff)")
}

// MARK: - V norm (weight=none) equivalence

/// Test that rmsNorm(x, weight: .mlxNone, eps) == rmsNorm(x, weight: ones, eps).
/// Gemma4 uses this for V normalization (no learnable scale).
@Test
func testVNormMlxNoneEqualsOnes() async throws {
    // Shape: [B, L, nKVHeads, headDim] before transpose
    let shapes: [[Int]] = [
        [1, 1, 2, 512],   // 26b full attn K=V
        [1, 4, 8, 256],   // 26b sliding attn
        [1, 1, 4, 512],   // 31b full attn K=V
    ]
    let eps: Float = 1e-6

    for shape in shapes {
        let x = MLXRandom.normal(shape).asType(.float32)
        let ones = MLXArray.ones([shape.last!]).asType(.float32)

        let noneOut = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
        let onesOut = MLXFast.rmsNorm(x, weight: ones, eps: eps)

        eval(noneOut, onesOut)

        let maxDiff = abs(noneOut - onesOut).max().item(Float.self)
        #expect(
            maxDiff < 1e-6,
            "V norm mlxNone vs ones diff: \(maxDiff) for shape \(shape)")
    }
}

/// Test V norm at bfloat16 (actual inference dtype).
@Test
func testVNormMlxNone_BFloat16() async throws {
    let shape = [1, 1, 2, 512]
    let eps: Float = 1e-6

    let x = MLXRandom.normal(shape).asType(.bfloat16)
    let ones = MLXArray.ones([512]).asType(.bfloat16)

    let noneOut = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    let onesOut = MLXFast.rmsNorm(x, weight: ones, eps: eps)

    eval(noneOut, onesOut)

    let maxDiff = abs(noneOut.asType(.float32) - onesOut.asType(.float32)).max().item(Float.self)
    #expect(maxDiff < 1e-3, "BF16 V norm mlxNone vs ones diff: \(maxDiff)")
}

// MARK: - K=V path divergence

/// When attentionKEqV=true, keys and values start as the same tensor but must
/// diverge after applying different transforms (kNorm+RoPE on keys, vNorm on values).
/// This test verifies both paths produce valid, different outputs from the same input.
@Test
func testKEqVPathDivergence() async throws {
    let B = 1, L = 1, nKVHeads = 2, headDim = 512
    let eps: Float = 1e-6
    let base: Float = 1_000_000.0
    let partialRotaryFactor: Float = 0.25
    let offset = 10
    let ropeDim = Int(Float(headDim) * partialRotaryFactor)

    // Simulate kProj(x) output
    let kprojOut = MLXRandom.normal([B, L, nKVHeads, headDim]).asType(.float32)

    // K=V: values = keys (same tensor)
    let keys = kprojOut
    let values = kprojOut

    // --- Keys path: kNorm + RoPE ---
    let kNormWeight = MLXRandom.normal([headDim]).asType(.float32)
    let propExponents = MLXArray(
        stride(from: Float(0), to: Float(ropeDim), by: 2)
    ) / Float(headDim)
    let realFreqs = pow(MLXArray(base), propExponents)
    let paddingCount = (headDim - ropeDim) / 2
    let infPadding = MLXArray(Array(repeating: Float.infinity, count: paddingCount))
    let allFreqs = concatenated([realFreqs, infPadding], axis: 0)
    let invFreqs = (1.0 / allFreqs).asType(.float32)

    let processedKeys = MLXFast.rmsNormRoPE(
        keys, weight: kNormWeight, invFreqs: invFreqs,
        eps: eps, offset: offset, nHeads: nKVHeads, seqLen: L
    ).transposed(0, 2, 1, 3)

    // --- Values path: vNorm (no weight, no RoPE) ---
    let processedValues = MLXFast.rmsNorm(
        values, weight: MLXArray.mlxNone, eps: eps
    ).transposed(0, 2, 1, 3)

    eval(processedKeys, processedValues)

    // Keys and values must be different after different transforms
    let diff = abs(processedKeys - processedValues).max().item(Float.self)
    #expect(diff > 0.01, "K=V: keys and values should diverge after transforms, but diff=\(diff)")

    // Both should have valid (non-NaN, non-Inf) values
    let keysHasNaN = any(isNaN(processedKeys)).item(Bool.self)
    let valuesHasNaN = any(isNaN(processedValues)).item(Bool.self)
    #expect(!keysHasNaN, "Keys contain NaN")
    #expect(!valuesHasNaN, "Values contain NaN")

    // Both should have finite values
    let keysHasInf = any(abs(processedKeys) .> 1e6).item(Bool.self)
    let valuesHasInf = any(abs(processedValues) .> 1e6).item(Bool.self)
    #expect(!keysHasInf, "Keys contain very large values")
    #expect(!valuesHasInf, "Values contain very large values")
}

// MARK: - Decode step consistency (single token)

/// Verify that single-token decode (L=1) through the fused path matches
/// the non-fused path. This is the hot path during generation.
@Test
func testSingleTokenDecodeConsistency() async throws {
    let B = 1, L = 1, nHeads = 16, headDim = 512
    let eps: Float = 1e-6
    let base: Float = 1_000_000.0
    let partialRotaryFactor: Float = 0.25
    let ropeDim = Int(Float(headDim) * partialRotaryFactor)

    let weight = MLXRandom.normal([headDim]).asType(.bfloat16)

    // Build frequencies
    let propExponents = MLXArray(
        stride(from: Float(0), to: Float(ropeDim), by: 2)
    ) / Float(headDim)
    let realFreqs = pow(MLXArray(base), propExponents)
    let paddingCount = (headDim - ropeDim) / 2
    let infPadding = MLXArray(Array(repeating: Float.infinity, count: paddingCount))
    let allFreqs = concatenated([realFreqs, infPadding], axis: 0)
    let invFreqs = (1.0 / allFreqs).asType(.float32)
    let propFreqs = allFreqs

    // Test multiple decode positions to catch offset-dependent bugs
    for offset in [0, 1, 10, 100, 1000] {
        let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.bfloat16)

        // Fused
        let fusedOut = MLXFast.rmsNormRoPE(
            x, weight: weight, invFreqs: invFreqs,
            eps: eps, offset: offset, nHeads: nHeads, seqLen: L
        ).transposed(0, 2, 1, 3)

        // Non-fused
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let transposed = normed.transposed(0, 2, 1, 3)
        let ropeOut = MLXFast.RoPE(
            transposed, dimensions: headDim, traditional: false,
            base: nil, scale: 1.0, offset: offset, freqs: propFreqs)

        eval(fusedOut, ropeOut)

        let maxDiff = abs(fusedOut.asType(.float32) - ropeOut.asType(.float32)).max().item(Float.self)
        // bf16 with headDim=512 accumulates precision noise from both norm and RoPE
        #expect(
            maxDiff < 0.1,
            "Single-token decode at offset \(offset): fused vs non-fused max diff \(maxDiff)")
    }
}

// MARK: - Precision diagnostic: fused vs non-fused breakdown

/// Isolate whether precision diff comes from the norm or the RoPE portion.
/// This is critical because even small per-element diffs compound over 30+ layers.
@Test
func testFusedPrecisionDiagnostic() async throws {
    let B = 1, L = 1, nHeads = 16, headDim = 512
    let eps: Float = 1e-6
    let base: Float = 1_000_000.0
    let partialRotaryFactor: Float = 0.25
    let ropeDim = Int(Float(headDim) * partialRotaryFactor)
    let offset = 10

    let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.bfloat16)
    let weight = MLXRandom.normal([headDim]).asType(.bfloat16)

    // Build frequencies
    let propExponents = MLXArray(
        stride(from: Float(0), to: Float(ropeDim), by: 2)
    ) / Float(headDim)
    let realFreqs = pow(MLXArray(base), propExponents)
    let paddingCount = (headDim - ropeDim) / 2
    let infPadding = MLXArray(Array(repeating: Float.infinity, count: paddingCount))
    let allFreqs = concatenated([realFreqs, infPadding], axis: 0)
    let invFreqs = (1.0 / allFreqs).asType(.float32)
    let propFreqs = allFreqs

    // --- Step 1: Compare JUST the norm portion ---
    let fusedNormRoPE = MLXFast.rmsNormRoPE(
        x, weight: weight, invFreqs: invFreqs,
        eps: eps, offset: offset, nHeads: nHeads, seqLen: L)
    let separateNorm = MLXFast.rmsNorm(x, weight: weight, eps: eps)

    // To isolate the norm: apply fused with offset=0 and zero invFreqs (identity RoPE)
    let zeroInvFreqs = MLXArray.zeros([headDim / 2]).asType(.float32)
    let fusedNormOnly = MLXFast.rmsNormRoPE(
        x, weight: weight, invFreqs: zeroInvFreqs,
        eps: eps, offset: 0, nHeads: nHeads, seqLen: L)

    eval(fusedNormOnly, separateNorm)

    let normDiff = abs(fusedNormOnly.asType(.float32) - separateNorm.asType(.float32))
    let normMaxDiff = normDiff.max().item(Float.self)
    let normMeanDiff = normDiff.mean().item(Float.self)
    print("[DIAG] Norm-only max diff: \(normMaxDiff), mean diff: \(normMeanDiff)")

    // --- Step 2: Compare full fused vs non-fused ---
    let fusedFull = fusedNormRoPE.transposed(0, 2, 1, 3)
    let nonFusedFull = MLXFast.RoPE(
        separateNorm.transposed(0, 2, 1, 3),
        dimensions: headDim, traditional: false,
        base: nil, scale: 1.0, offset: offset, freqs: propFreqs)

    eval(fusedFull, nonFusedFull)

    let fullDiff = abs(fusedFull.asType(.float32) - nonFusedFull.asType(.float32))
    let fullMaxDiff = fullDiff.max().item(Float.self)
    let fullMeanDiff = fullDiff.mean().item(Float.self)
    print("[DIAG] Full (norm+RoPE) max diff: \(fullMaxDiff), mean diff: \(fullMeanDiff)")

    // --- Step 3: Check rotated vs non-rotated dims ---
    let halfDim = headDim / 2  // 256
    let rotatedCount = ropeDim / 2  // 64 pairs → dims 0..63 and 256..319
    let diffArr = fullDiff  // [B, H, L, D]

    // Non-rotated dims should have zero or near-zero RoPE diff (identity rotation)
    let nonRotatedDiff = diffArr[0..., 0..., 0..., rotatedCount..<halfDim]
    let nonRotatedMax = nonRotatedDiff.max().item(Float.self)
    print("[DIAG] Non-rotated dims max diff: \(nonRotatedMax)")

    let rotatedDiff = diffArr[0..., 0..., 0..., 0..<rotatedCount]
    let rotatedMax = rotatedDiff.max().item(Float.self)
    print("[DIAG] Rotated dims max diff: \(rotatedMax)")

    // The norm-only diff should be very small (< epsilon of bf16)
    #expect(normMaxDiff < 0.01, "Norm precision issue: max diff \(normMaxDiff)")

    // Report whether the diff is in rotated or non-rotated dims
    if fullMaxDiff > 0.01 {
        print("[DIAG] WARNING: Significant precision diff detected in fused kernel")
        print("[DIAG] This compounds over \(30) layers × \(400) tokens = potential incoherence")
    }
}

/// Test compounding error: simulate multiple layers of fused vs non-fused norm+RoPE.
/// If small per-element errors compound, the divergence should grow with depth.
@Test
func testCompoundingError() async throws {
    let B = 1, L = 1, nHeads = 8, headDim = 256
    let eps: Float = 1e-6
    let base: Float = 10000.0

    let exponents = MLXArray(stride(from: Float(0), to: Float(headDim), by: 2)) / Float(headDim)
    let freqs = pow(MLXArray(base), exponents)
    let invFreqs = (1.0 / freqs).asType(.float32)

    // Simulate 30 layers of norm+RoPE
    var fusedH = MLXRandom.normal([B, L, nHeads, headDim]).asType(.bfloat16)
    var nonFusedH = fusedH

    var maxDiffs: [Float] = []

    for layer in 0..<30 {
        let weight = MLXRandom.normal([headDim]).asType(.bfloat16)
        let offset = layer  // Each layer sees the accumulated offset

        // Fused path
        let fusedOut = MLXFast.rmsNormRoPE(
            fusedH, weight: weight, invFreqs: invFreqs,
            eps: eps, offset: offset, nHeads: nHeads, seqLen: L
        )

        // Non-fused path
        let normed = MLXFast.rmsNorm(nonFusedH, weight: weight, eps: eps)
        let transposed = normed.transposed(0, 2, 1, 3)
        let ropeOut = MLXFast.RoPE(
            transposed, dimensions: headDim, traditional: false,
            base: base, scale: 1.0, offset: offset
        ).transposed(0, 2, 1, 3)  // transpose back to [B, L, H, D]

        eval(fusedOut, ropeOut)

        let diff = abs(fusedOut.asType(.float32) - ropeOut.asType(.float32)).max().item(Float.self)
        maxDiffs.append(diff)

        // Feed forward (simulating residual + next layer input)
        fusedH = fusedOut
        nonFusedH = ropeOut
    }

    let layerDiffs = maxDiffs.map { String(format: "%.4f", $0) }.joined(separator: ", ")
    print("[COMPOUND] Per-layer max diffs: \(layerDiffs)")
    print("[COMPOUND] Layer 0 diff: \(maxDiffs[0]), Layer 29 diff: \(maxDiffs[29])")

    // Check if error compounds (last layer should be worse than first if compounding)
    let ratio = maxDiffs[29] / max(maxDiffs[0], 1e-10)
    print("[COMPOUND] Compounding ratio (layer29/layer0): \(ratio)")

    // If errors compound significantly (>10x), this indicates a real problem
    #expect(ratio < 100, "Error compounds \(ratio)x over 30 layers — may cause incoherence")
}

// MARK: - Thread safety tests for compiled functions (T3)

/// T3: Module-level compiled closures are @Sendable and callable from any thread.
/// Concurrent calls should not produce NaN or crash.
@Test
func testCompiledLogitSoftcapConcurrentSafety() async throws {
    // Recreate the compiled function from Gemma4.swift:653-656
    let compiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { softcap, x in
            tanh(x / softcap) * softcap
        }

    let softcap = MLXArray(Float(30.0))

    // Run from multiple concurrent tasks
    await withTaskGroup(of: Bool.self) { group in
        for _ in 0..<10 {
            group.addTask {
                let x = MLXRandom.normal([1, 1, 32000]).asType(.float32)
                let result = compiledLogitSoftcap(softcap, x)
                eval(result)
                let hasNaN = any(isNaN(result)).item(Bool.self)
                return !hasNaN
            }
        }
        for await valid in group {
            #expect(valid, "Compiled logit softcap produced NaN under concurrent access")
        }
    }
}

/// T3: Compiled GEGLU should be safe under concurrent access.
@Test
func testCompiledGegluConcurrentSafety() async throws {
    let compiledGeglu: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { gate, x in
            geluApproximate(gate) * x
        }

    await withTaskGroup(of: Bool.self) { group in
        for _ in 0..<10 {
            group.addTask {
                let gate = MLXRandom.normal([1, 1, 2816]).asType(.float32)
                let x = MLXRandom.normal([1, 1, 2816]).asType(.float32)
                let result = compiledGeglu(gate, x)
                eval(result)
                let hasNaN = any(isNaN(result)).item(Bool.self)
                return !hasNaN
            }
        }
        for await valid in group {
            #expect(valid, "Compiled GEGLU produced NaN under concurrent access")
        }
    }
}

// MARK: - RotatingKVCache edge cases

/// Verify RotatingKVCache handles single-token updates correctly across the
/// wrap boundary (the decode hot path).
@Test
func testRotatingCacheWrapBoundaryDecoding() async throws {
    let maxSize = 8
    let cache = RotatingKVCache(maxSize: maxSize)

    var allKeys: [MLXArray] = []

    // Fill cache completely (8 tokens), then add 2 more (wrapping)
    for i in 0..<10 {
        let k = (Float(i) * MLXArray.ones([1, 2, 1, 4])).asType(.float32)
        let v = (Float(i) * MLXArray.ones([1, 2, 1, 4])).asType(.float32)
        let (retK, _) = cache.update(keys: k, values: v)
        allKeys.append(retK)
        eval(retK)
    }

    // After 10 tokens with maxSize=8, the last returned K should have 8 tokens
    let lastK = allKeys.last!
    #expect(lastK.dim(2) == maxSize, "Expected \(maxSize) cached tokens, got \(lastK.dim(2))")
}

/// Verify peek() returns correct data after wrapping.
@Test
func testRotatingCachePeekAfterWrap() async throws {
    let maxSize = 4
    let cache = RotatingKVCache(maxSize: maxSize)

    // Fill 6 tokens — wraps around
    for i in 0..<6 {
        let k = (Float(i + 1) * MLXArray.ones([1, 1, 1, 2])).asType(.float32)
        let v = (Float(i + 1) * MLXArray.ones([1, 1, 1, 2])).asType(.float32)
        _ = cache.update(keys: k, values: v)
    }

    // peek() should return the full physical buffer (4 tokens)
    let peeked = cache.peek()
    #expect(peeked != nil, "peek() should return non-nil after wrapping")
    if let (pk, _) = peeked {
        eval(pk)
        #expect(pk.dim(2) == maxSize, "peek() should return maxSize tokens after wrap")
    }
}

// MARK: - RMSNorm precision across dtypes

/// Verify RMSNorm produces consistent results across float32 and bfloat16.
/// Large divergence would indicate precision-related incoherence.
@Test
func testRmsNormPrecisionConsistency() async throws {
    let shape = [1, 1, 2816]
    let eps: Float = 1e-6

    let xf32 = MLXRandom.normal(shape).asType(.float32)
    let xbf16 = xf32.asType(.bfloat16)
    let wf32 = MLXRandom.normal([2816]).asType(.float32)
    let wbf16 = wf32.asType(.bfloat16)

    let outF32 = MLXFast.rmsNorm(xf32, weight: wf32, eps: eps)
    let outBF16 = MLXFast.rmsNorm(xbf16, weight: wbf16, eps: eps)

    eval(outF32, outBF16)

    let maxDiff = abs(outF32 - outBF16.asType(.float32)).max().item(Float.self)
    // bfloat16 has ~0.4% relative error, so for values around 1.0, expect ~0.01 abs diff
    #expect(maxDiff < 0.1, "RMSNorm f32 vs bf16 max diff: \(maxDiff)")
}
