import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Codebook Tests

@Suite("TurboQuant Codebook")
struct TurboQuantCodebookTests {

    @Test func codebookGeneration_3bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 3)
        #expect(cb.count == 8)
        let vals = cb.asArray(Float.self)
        for i in 0 ..< vals.count - 1 {
            #expect(vals[i] <= vals[i + 1], "Codebook not sorted")
        }
    }

    @Test func codebookGeneration_4bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 4)
        #expect(cb.count == 16)
        let vals = cb.asArray(Float.self)
        for v in vals {
            #expect(v >= -1.0 && v <= 1.0, "Centroid \(v) out of range")
        }
    }

    @Test func boundaryCount() {
        let b = TurboQuantCodebook.boundaries(dim: 128, bits: 3)
        #expect(b.count == 7)  // 2^3 - 1 boundaries for 8 centroids
    }

    @Test func codebookDeterminism() {
        let cb1 = TurboQuantCodebook.codebook(dim: 64, bits: 3)
        let cb2 = TurboQuantCodebook.codebook(dim: 64, bits: 3)
        let v1 = cb1.asArray(Float.self)
        let v2 = cb2.asArray(Float.self)
        #expect(v1 == v2, "Codebook should be deterministic")
    }
}

// MARK: - Rotation Tests

@Suite("TurboQuant Rotation")
struct TurboQuantRotationTests {

    @Test func rotationOrthogonality() {
        let dim = 64
        let R = TurboQuantRotation.rotationMatrix(dim: dim, seed: 42)
        #expect(R.shape == [dim, dim])
        let RRt = matmul(R, R.transposed())
        let identity = MLXArray.eye(dim)
        let diff = MLX.abs(RRt - identity)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-4, "R @ R^T differs from I by \(maxDiff)")
    }

    @Test func rotationDeterminism() {
        let R1 = TurboQuantRotation.rotationMatrix(dim: 32, seed: 123)
        let R2 = TurboQuantRotation.rotationMatrix(dim: 32, seed: 123)
        let diff = MLX.abs(R1 - R2).max().item(Float.self)
        #expect(diff < 1e-6, "Same seed should produce same rotation")
    }
}

// MARK: - Bit Packing Tests

@Suite("TurboQuant Bit Packing")
struct TurboQuantPackingTests {

    @Test func packedWidth() {
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 4) == 16)
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 3) == 12)
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 1) == 4)
    }

    @Test func packUnpack_3bit() {
        let indices = MLXArray([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4] as [UInt32]).reshaped([1, 12])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 3, count: 12)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "3-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_1bit() {
        let indices = MLXArray([0, 1, 0, 1, 1, 0, 1, 0] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 1)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 1, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "1-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_4bit() {
        let indices = MLXArray([0, 5, 10, 15, 3, 7, 11, 14] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 4)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 4, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "4-bit pack/unpack round-trip failed")
    }
}

// MARK: - MSE Codec Tests

@Suite("TurboQuant MSE Codec")
struct TurboQuantMSECodecTests {

    @Test func encodeDecodeRoundTrip_4bit() {
        let codec = MSECodec(dim: 32, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 32])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        #expect(state.tokenCount == 4)
        #expect(state.dim == 32)
        #expect(state.bits == 4)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        #expect(mse < 0.5, "4-bit MSE too high: \(mse)")
    }

    @Test func encodeDecodeRoundTrip_3bit() {
        let codec = MSECodec(dim: 32, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 32])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        #expect(mse < 1.0, "3-bit MSE too high: \(mse)")
    }

    @Test func cosineSimilarity_4bit() {
        let codec = MSECodec(dim: 64, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 8, 64])
        eval(vectors)

        let decoded = codec.decode(codec.encode(vectors))
        let dot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = dot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.95, "4-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func boundaryQuantizeMatchesArgmin() {
        // Verify boundary-based quantization gives same result as argmin
        let codec = MSECodec(dim: 16, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 4, 16])
        eval(vectors)

        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let unit = vectors / expandedDimensions(maximum(norms, MLXArray(Float(1e-8))), axis: -1)
        let rotated = matmul(unit, codec.rotationT)

        // Boundary quantize
        let boundaryIdx = codec.boundaryQuantize(rotated)

        // Argmin quantize (naive but correct)
        let expanded = expandedDimensions(rotated, axis: -1)
        let cb = codec.codebook.reshaped([1, 1, 1, 1, -1])
        let distances = MLX.abs(expanded - cb)
        let argminIdx = argMin(distances, axis: -1)

        let b = boundaryIdx.asArray(UInt32.self)
        let a = argminIdx.asArray(UInt32.self)
        #expect(b == a, "Boundary quantize should match argmin")
    }

    @Test func fwhtRoundTrip() {
        // FWHT forward → inverse should recover original vectors
        let signs = TurboQuantRotation.whtSigns(dim: 128, seed: 42)

        let vectors = MLXRandom.normal([2, 4, 8, 128])
        eval(vectors)

        let rotated = TurboQuantRotation.fwhtForward(vectors, signs: signs)
        let recovered = TurboQuantRotation.fwhtInverse(rotated, signs: signs)
        eval(recovered)

        let diff = MLX.abs(vectors - recovered).max().item(Float.self)
        #expect(diff < 1e-4, "FWHT round-trip max diff: \(diff)")
    }

    @Test func whtRotationOrthogonality() {
        // WHT rotation H*D/sqrt(d) should be orthogonal: Π·Π^T ≈ I
        let codec = MSECodec(dim: 128, bits: 3, seed: 42)
        #expect(codec.useWHT, "dim=128 should use WHT")

        let product = matmul(codec.rotation, codec.rotationT)
        let identity = MLXArray.identity(128)
        let diff = MLX.abs(product - identity).max().item(Float.self)
        #expect(diff < 1e-4, "WHT rotation should be orthogonal, max diff: \(diff)")
    }

    @Test func whtEncodeDecodeRoundTrip() {
        // Verify encode/decode with WHT rotation produces reasonable reconstruction
        let codec = MSECodec(dim: 128, bits: 4, seed: 42)
        #expect(codec.useWHT, "dim=128 should use WHT")

        let vectors = MLXRandom.normal([1, 1, 8, 128])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let cosDot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = cosDot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.90, "WHT 4-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func normCorrectionImprovesNormAccuracy() {
        // Norm correction ensures the reconstructed vector's L2 norm matches the original.
        // This improves attention score accuracy (dot products), which is what matters
        // for perplexity. Element-wise MSE may not improve, but norm accuracy should.
        let codec = MSECodec(dim: 128, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 64, 128])
        eval(vectors)

        // Original norms
        let origNorms = sqrt((vectors * vectors).sum(axis: -1))

        // Encode with norm correction (current implementation)
        let state = codec.encode(vectors)
        let decodedCorrected = codec.decode(state)
        let correctedNorms = sqrt((decodedCorrected * decodedCorrected).sum(axis: -1))

        // Norm error with correction
        let normErrorCorrected = MLX.abs(origNorms - correctedNorms).mean().item(Float.self)

        // Manually encode WITHOUT norm correction for comparison
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)
        let rotated = matmul(unit, codec.rotationT)
        let indices = codec.boundaryQuantize(rotated)
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let uncorrectedState = MSECodecState(
            norms: norms, packedIndices: packed, tokenCount: 64, dim: 128, bits: 3)
        let decodedUncorrected = codec.decode(uncorrectedState)
        let uncorrectedNorms = sqrt((decodedUncorrected * decodedUncorrected).sum(axis: -1))

        // Norm error without correction
        let normErrorUncorrected = MLX.abs(origNorms - uncorrectedNorms).mean().item(Float.self)

        #expect(normErrorCorrected < normErrorUncorrected,
            "Norm-corrected norm error (\(normErrorCorrected)) should be lower than uncorrected (\(normErrorUncorrected))")

        // Also verify dot product accuracy improves (key for attention scoring)
        // Dot product of original with itself = ||v||², so compare dot products
        let dotCorrected = (vectors * decodedCorrected).sum(axis: -1)
        let dotUncorrected = (vectors * decodedUncorrected).sum(axis: -1)
        let dotOriginal = (vectors * vectors).sum(axis: -1)
        let dotErrorCorrected = MLX.abs(dotOriginal - dotCorrected).mean().item(Float.self)
        let dotErrorUncorrected = MLX.abs(dotOriginal - dotUncorrected).mean().item(Float.self)

        #expect(dotErrorCorrected < dotErrorUncorrected,
            "Norm-corrected dot product error (\(dotErrorCorrected)) should be lower than uncorrected (\(dotErrorUncorrected))")
    }
}

// MARK: - KV Cache Tests

@Suite("TurboQuantKVCache")
struct TurboQuantKVCacheTests {

    @Test func cacheUpdate() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 4, 8, 64])
        let values = MLXRandom.normal([1, 4, 8, 64])
        eval(keys, values)

        let (outKeys, outValues) = cache.update(keys: keys, values: values)
        #expect(cache.offset == 8)
        #expect(outKeys.shape == [1, 4, 8, 64])
        #expect(outValues.shape == [1, 4, 8, 64])
    }

    @Test func cacheIncrementalUpdate() {
        let cache = TurboQuantKVCache(bits: 4)

        let k1 = MLXRandom.normal([1, 2, 4, 32])
        let v1 = MLXRandom.normal([1, 2, 4, 32])
        eval(k1, v1)
        let (_, _) = cache.update(keys: k1, values: v1)
        #expect(cache.offset == 4)

        let k2 = MLXRandom.normal([1, 2, 1, 32])
        let v2 = MLXRandom.normal([1, 2, 1, 32])
        eval(k2, v2)
        let (outK, _) = cache.update(keys: k2, values: v2)
        #expect(cache.offset == 5)
        #expect(outK.shape == [1, 2, 5, 32])
    }

    @Test func cacheTrim() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 2, 8, 32])
        let values = MLXRandom.normal([1, 2, 8, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.offset == 8)

        let trimmed = cache.trim(3)
        #expect(trimmed == 3)
        #expect(cache.offset == 5)
    }

    @Test func cacheState() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 2, 4, 32])
        let values = MLXRandom.normal([1, 2, 4, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)

        let state = cache.state
        // In raw phase (prefill): 2 arrays (rawKeys, rawValues)
        // In compressed phase: 4 arrays (keyPacked, keyNorms, valPacked, valNorms)
        #expect(state.count == 2 || state.count == 4, "State should have 2 or 4 arrays, got \(state.count)")
    }

    @Test func cacheIsTrimmable() {
        let cache = TurboQuantKVCache(bits: 4)
        #expect(cache.isTrimmable == true)
    }

    /// Regression test: asymmetric bits (e.g. turbo3v2 = 3-bit K, 2-bit V) must encode
    /// values with valueBits, not the legacy `bits` field. A mismatch causes packed width
    /// errors during compressRawCache → reshape.
    @Test func cacheAsymmetricCompression() {
        // turbo3v2 config: keyBits=3, valueBits=2
        let cache = TurboQuantKVCache(bits: 3, keyBits: 3, valueBits: 2)
        let B = 1
        let H = 4  // KV heads
        let T = 32  // prefill tokens
        let D = 128

        // Phase 1: Prefill (raw FP16 storage)
        let keys = MLXRandom.normal([B, H, T, D])
        let values = MLXRandom.normal([B, H, T, D])
        eval(keys, values)
        let (_, _) = cache.update(keys: keys, values: values)
        #expect(cache.offset == T)

        // Phase 2: First decode triggers compressRawCache()
        let newKey = MLXRandom.normal([B, H, 1, D])
        let newVal = MLXRandom.normal([B, H, 1, D])
        let queries = MLXRandom.normal([B, H * 2, 1, D])  // nQHeads = 2 * nKVHeads (GQA)
        eval(newKey, newVal, queries)

        let output = cache.compressedAttention(
            queries: queries, keys: newKey, values: newVal,
            scale: 1.0 / sqrt(Float(D))
        )
        eval(output)
        #expect(output.shape == [B, H * 2, 1, D])
        #expect(cache.isCompressed)

        // Phase 3: Several more decode steps to confirm incremental encode also works
        for _ in 0 ..< 5 {
            let dk = MLXRandom.normal([B, H, 1, D])
            let dv = MLXRandom.normal([B, H, 1, D])
            let dq = MLXRandom.normal([B, H * 2, 1, D])
            eval(dk, dv, dq)
            let out = cache.compressedAttention(
                queries: dq, keys: dk, values: dv,
                scale: 1.0 / sqrt(Float(D))
            )
            eval(out)
            #expect(out.shape == [B, H * 2, 1, D])
        }
        #expect(cache.offset == T + 6)
    }

    @Test func cacheAllBitWidths() {
        for bits in 2 ... 4 {
            let cache = TurboQuantKVCache(bits: bits)
            let keys = MLXRandom.normal([1, 2, 4, 32])
            let values = MLXRandom.normal([1, 2, 4, 32])
            eval(keys, values)
            let (outKeys, outValues) = cache.update(keys: keys, values: values)
            #expect(cache.offset == 4, "\(bits)-bit cache offset wrong")
            #expect(outKeys.shape == [1, 2, 4, 32], "\(bits)-bit key shape wrong")
            #expect(outValues.shape == [1, 2, 4, 32], "\(bits)-bit value shape wrong")
        }
    }
}

// MARK: - TurboFlashAttention Tests

@Suite("TurboFlashAttention")
struct TurboFlashAttentionTests {

    /// Validate that fused TurboFlashAttention produces the same output as
    /// separated Score → Softmax → Value kernels.
    @Test func flashMatchesSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 4
        let nQHeads = 8
        let nKVHeads = 4
        let tokenCount = 64
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        // Generate random KV cache: encode random vectors
        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        // Encode K and V
        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        // Generate random queries (pre-rotated and scaled)
        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // === Separated path: Score → Softmax → Value ===
        let scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: keyCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: keyBits, dim: dim)
        let attnWeights = softmax(scores, axis: -1)
        let separatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: attnWeights, packed: flatValPacked, norms: flatValNorms,
            codebook: valCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: valueBits, dim: dim)
        eval(separatedOutput)

        // === Fused path: TurboFlashAttention ===
        let fusedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)
        eval(fusedOutput)

        // Compare outputs — should match within floating point tolerance
        // Online softmax may have slightly different numerical behavior than
        // materialized softmax, so allow a small tolerance
        let diff = abs(separatedOutput - fusedOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        print("[TurboFlash] Separated output range: [\(separatedOutput.min().item(Float.self)), \(separatedOutput.max().item(Float.self))]")
        print("[TurboFlash] Fused output range: [\(fusedOutput.min().item(Float.self)), \(fusedOutput.max().item(Float.self))]")

        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
        #expect(meanDiff < 1e-4, "Mean diff \(meanDiff) exceeds tolerance 1e-4")
    }

    /// Test with asymmetric K/V bits (4-bit K, 2-bit V)
    @Test func flashAsymmetricBits() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 4
        let nKVHeads = 2
        let tokenCount = 32
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // Separated
        let scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: keyCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: keyBits, dim: dim)
        let attnWeights = softmax(scores, axis: -1)
        let separatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: attnWeights, packed: flatValPacked, norms: flatValNorms,
            codebook: valCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: valueBits, dim: dim)

        // Fused
        let fusedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)

        eval(separatedOutput, fusedOutput)

        let maxDiff = abs(separatedOutput - fusedOutput).max().item(Float.self)
        print("[TurboFlash Asymmetric 4K/2V] Max diff: \(maxDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance for asymmetric bits")
    }

    /// Microbenchmark: fused vs separated at various token counts
    @Test func microbenchFlashVsSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 24  // Qwen3.5-2B query heads
        let nKVHeads = 4  // Qwen3.5-2B KV heads
        let repeatCount = nQHeads / nKVHeads
        let iterations = 200
        let warmup = 50

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        for tokenCount in [128, 512, 1024, 2048, 4096, 8192] {
            let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
            let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
            eval(rawKeys, rawValues)

            let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
            let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: keyBits, dim: dim)
            let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatVals, whtSigns: valCodec.whtSigns!,
                boundaries: valCodec.boundaries, codebook: valCodec.codebook,
                bits: valueBits, dim: dim)

            let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
            let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
            let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
            let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
            let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
            let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

            let scale: Float = 1.0 / sqrt(Float(dim))
            let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
            eval(queries)

            // Warmup both paths
            for _ in 0..<warmup {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)

                let f = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyCodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valCodec.codebook,
                    tokenCount: tokenCount, repeatCount: repeatCount,
                    keyBits: keyBits, valueBits: valueBits, dim: dim)
                eval(f)
            }

            // Benchmark separated
            let startSep = Date()
            for _ in 0..<iterations {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let sepMs = Date().timeIntervalSince(startSep) * 1000 / Double(iterations)

            // Benchmark fused (default block size)
            let startFused = Date()
            for _ in 0..<iterations {
                let f = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyCodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valCodec.codebook,
                    tokenCount: tokenCount, repeatCount: repeatCount,
                    keyBits: keyBits, valueBits: valueBits, dim: dim)
                eval(f)
            }
            let fusedMs = Date().timeIntervalSince(startFused) * 1000 / Double(iterations)

            let speedup = sepMs / fusedMs
            print("[MICROBENCH] T=\(tokenCount): separated=\(String(format: "%.3f", sepMs))ms, fused(B=\(TurboQuantKernelOps.flashBlockSize))=\(String(format: "%.3f", fusedMs))ms, speedup=\(String(format: "%.2f", speedup))x")
        }
    }

    /// Block size sweep: find optimal block size for each token count
    @Test func microbenchBlockSizeSweep() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 24
        let nKVHeads = 4
        let repeatCount = nQHeads / nKVHeads
        let iterations = 200
        let warmup = 30

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let blockSizes = [32, 64, 128, 256, 512, 1024]
        let tokenCounts = [512, 1024, 2048, 4096, 8192]

        for tokenCount in tokenCounts {
            let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
            let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
            eval(rawKeys, rawValues)

            let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
            let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: keyBits, dim: dim)
            let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatVals, whtSigns: valCodec.whtSigns!,
                boundaries: valCodec.boundaries, codebook: valCodec.codebook,
                bits: valueBits, dim: dim)

            let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
            let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
            let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
            let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
            let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
            let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

            let scale: Float = 1.0 / sqrt(Float(dim))
            let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
            eval(queries)

            // Separated baseline
            for _ in 0..<warmup {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let startSep = Date()
            for _ in 0..<iterations {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let sepMs = Date().timeIntervalSince(startSep) * 1000 / Double(iterations)

            var results: [(Int, Double)] = []
            for bs in blockSizes where bs <= tokenCount {
                // Warmup
                for _ in 0..<warmup {
                    let f = TurboQuantKernelOps.turboFlashAttention(
                        rotatedQueries: queries,
                        keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                        keyCodebook: keyCodec.codebook,
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valCodec.codebook,
                        tokenCount: tokenCount, repeatCount: repeatCount,
                        keyBits: keyBits, valueBits: valueBits, dim: dim,
                        blockSize: bs)
                    eval(f)
                }

                let start = Date()
                for _ in 0..<iterations {
                    let f = TurboQuantKernelOps.turboFlashAttention(
                        rotatedQueries: queries,
                        keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                        keyCodebook: keyCodec.codebook,
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valCodec.codebook,
                        tokenCount: tokenCount, repeatCount: repeatCount,
                        keyBits: keyBits, valueBits: valueBits, dim: dim,
                        blockSize: bs)
                    eval(f)
                }
                let ms = Date().timeIntervalSince(start) * 1000 / Double(iterations)
                results.append((bs, ms))
            }

            let best = results.min(by: { $0.1 < $1.1 })!
            let resultStr = results.map { "B=\($0.0):\(String(format: "%.2f", $0.1))ms" }.joined(separator: "  ")
            print("[SWEEP] T=\(tokenCount): sep=\(String(format: "%.2f", sepMs))ms  \(resultStr)  BEST=B\(best.0)(\(String(format: "%.1f", sepMs/best.1))x)")
        }
    }

    /// Validate that causal TurboFlashAttention matches per-position reference.
    /// Computes reference by running non-causal flash on truncated KV for each query position.
    @Test func flashCausalMatchesSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 8
        let nKVHeads = 4
        let L = 4  // query chunk length
        let tokenCount = 32  // total KV cache length
        let repeatCount = nQHeads / nKVHeads
        let queryOffset = tokenCount - L  // queries cover positions 28..31

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        // Queries in [nQHeads, L, dim] layout, flattened to [nQHeads * L, dim]
        let queries = MLXRandom.normal([nQHeads, L, dim]) * MLXArray(scale)
        let flatQueries = queries.reshaped([nQHeads * L, dim])
        eval(flatQueries)

        // === Reference: compute per-position using non-causal flash on truncated KV ===
        // For each query position l, attend only to tokens 0...(queryOffset + l)
        var refOutputs: [MLXArray] = []
        for l in 0..<L {
            let visibleTokens = queryOffset + l + 1  // causal: can see up to and including position
            let truncKeyPacked = flatKeyPacked[0..., ..<visibleTokens, 0...]
            let truncKeyNorms = flatKeyNorms[0..., ..<visibleTokens]
            let truncValPacked = flatValPacked[0..., ..<visibleTokens, 0...]
            let truncValNorms = flatValNorms[0..., ..<visibleTokens]

            // Extract queries for position l across all heads: queries[:, l, :]
            let posQueries = queries[0..., l, 0...].reshaped([nQHeads, dim])

            let posOutput = TurboQuantKernelOps.turboFlashAttention(
                rotatedQueries: posQueries,
                keyPacked: truncKeyPacked, keyNorms: truncKeyNorms,
                keyCodebook: keyCodec.codebook,
                valPacked: truncValPacked, valNorms: truncValNorms,
                valCodebook: valCodec.codebook,
                tokenCount: visibleTokens, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim)
            refOutputs.append(posOutput)  // [nQHeads, dim]
        }
        // Stack and interleave to match [nQHeads * L, dim] layout from [nQHeads, L, dim]
        let refStacked = stacked(refOutputs, axis: 1)  // [nQHeads, L, dim]
        let refOutput = refStacked.reshaped([nQHeads * L, dim])
        eval(refOutput)

        // === Causal TurboFlashAttention ===
        let causalOutput = TurboQuantKernelOps.turboFlashAttentionCausal(
            rotatedQueries: flatQueries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            queryChunkLength: L, queryOffset: queryOffset)
        eval(causalOutput)

        let diff = abs(refOutput - causalOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash Causal] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
        #expect(meanDiff < 1e-4, "Mean diff \(meanDiff) exceeds tolerance 1e-4")
    }

    /// Validate that fused rotation in pass 2 matches separate matmul rotation.
    @Test func flashFusedRotationMatchesSeparate() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 8
        let nKVHeads = 4
        let tokenCount = 64
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // Without rotation fusion: get rotated output, then matmul
        let rotatedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)
        let separateRotOutput = matmul(rotatedOutput, valCodec.rotation)
        eval(separateRotOutput)

        // With rotation fusion: rotation applied in pass 2 kernel
        let fusedRotOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            valRotation: valCodec.rotation)
        eval(fusedRotOutput)

        let diff = abs(separateRotOutput - fusedRotOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash Fused Rotation] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
    }
}

// MARK: - Encode Kernel Microbenchmark

@Suite("TurboQuant Encode Microbench")
struct TurboQuantEncodeMicrobenchTests {

    /// Microbenchmark: Dense rotation fused encode kernel.
    /// Simulates the hot path: 1 token × nKVHeads encode calls per decode step.
    /// Runs many iterations to average out GPU scheduling noise.
    @Test func microbenchDenseEncode() {
        let dim = 128
        let bits = 4
        let nKVHeads = 4  // typical GQA KV head count for 27B
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        // Dense rotation — force non-WHT by using the QR rotation directly
        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)

        // Simulate single-token encode: [nKVHeads, dim] per call
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        // Warmup
        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        // Timed iterations
        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] Dense encode: \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        // Just a smoke test — the print output is what we care about
        #expect(perCall > 0)
    }

    /// Microbenchmark: WHT rotation fused encode kernel.
    @Test func microbenchWHTEncode() {
        let dim = 128
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=128 should use WHT")
            return
        }

        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        // Warmup
        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        // Timed iterations
        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] WHT encode:   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    /// Microbenchmark: Batch encode (prefill transition) — 512 tokens at once.
    @Test func microbenchBatchEncodeDense() {
        let dim = 128
        let bits = 4
        let batchSize = 512 * 4  // 512 tokens × 4 KV heads
        let iterations = 100
        let warmup = 10

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] Dense batch encode (512×4): \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    /// Microbenchmark: Batch encode WHT — 512 tokens at once.
    @Test func microbenchBatchEncodeWHT() {
        let dim = 128
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=128 should use WHT")
            return
        }
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] WHT batch encode dim=128 (512×4):   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    // --- dim=256 variants (Qwen3.5-27B full attention head_dim) ---

    @Test func microbenchDenseEncode256() {
        let dim = 256
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] Dense encode dim=256: \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    @Test func microbenchWHTEncode256() {
        let dim = 256
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=256 should use WHT")
            return
        }
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] WHT encode dim=256:   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    @Test func microbenchBatchEncodeDense256() {
        let dim = 256
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] Dense batch encode dim=256 (512×4): \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }

    @Test func microbenchBatchEncodeWHT256() {
        let dim = 256
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=256 should use WHT")
            return
        }
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print("[MICROBENCH] WHT batch encode dim=256 (512×4):   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call")
        #expect(perCall > 0)
    }
}

// MARK: - TurboQuant Diagnostic Tests (regression hunting)

@Suite("TurboQuantDiagnostics")
struct TurboQuantDiagnosticTests {

    /// KEY TEST: Compare compressedAttention output against reference SDPA.
    /// If TurboQuant produces garbled output, the error will be very large.
    /// Normal quantization error for 4-bit is ~0.01-0.1; garbled output gives >1.0.
    @Test func compressedAttentionMatchesReference() {
        let B = 1
        let nQHeads = 8
        let nKVHeads = 4
        let headDim = 128
        let prefillLen = 32
        let keyBits = 4
        let valueBits = 4

        // Create raw K/V for prefill (bf16 to match model dtype)
        let rawK = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        let rawV = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        eval(rawK, rawV)

        // === Reference path: standard SDPA with raw K/V ===
        let newQ = MLXRandom.normal([B, nQHeads, 1, headDim]).asType(.bfloat16)
        let newK = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
        let newV = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
        eval(newQ, newK, newV)

        let scale = 1.0 / sqrt(Float(headDim))
        let allK = concatenated([rawK, newK], axis: 2)
        let allV = concatenated([rawV, newV], axis: 2)
        let refOutput = MLXFast.scaledDotProductAttention(
            queries: newQ, keys: allK, values: allV,
            scale: scale, mask: .none
        )
        eval(refOutput)

        // === TurboQuant path: compress prefill then decode ===
        let cache = TurboQuantKVCache(bits: keyBits, keyBits: keyBits, valueBits: valueBits)
        let _ = cache.update(keys: rawK, values: rawV)

        let turboOutput = cache.compressedAttention(
            queries: newQ, keys: newK, values: newV,
            scale: scale, mask: .none
        )
        eval(turboOutput)

        // Compare
        let refFP = refOutput.asType(.float32)
        let turboFP = turboOutput.asType(.float32)
        let diff = abs(refFP - turboFP)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)
        let refRange = refFP.max().item(Float.self) - refFP.min().item(Float.self)

        print("[DIAGNOSTIC] compressedAttention vs reference SDPA:")
        print("[DIAGNOSTIC]   Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        print("[DIAGNOSTIC]   Ref range: \(refRange)")
        print("[DIAGNOSTIC]   Ref output shape: \(refOutput.shape), Turbo shape: \(turboOutput.shape)")
        print("[DIAGNOSTIC]   Relative error: \(maxDiff / max(refRange, 1e-6))")

        // Garbled output typically has maxDiff > 1.0.
        // Normal quantization error for 4-bit should be << 1.0.
        #expect(maxDiff < 1.0,
            "TurboQuant output is garbled (maxDiff=\(maxDiff)). Expected <1.0 for valid quantization.")
    }

    /// Test asymmetric turbo4v2 (4-bit K, 2-bit V) — the config used by Qwen3.5
    @Test func compressedAttentionAsymmetric4v2() {
        let B = 1
        let nQHeads = 32
        let nKVHeads = 8
        let headDim = 128
        let prefillLen = 64
        let keyBits = 4
        let valueBits = 2

        let rawK = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        let rawV = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        eval(rawK, rawV)

        let newQ = MLXRandom.normal([B, nQHeads, 1, headDim]).asType(.bfloat16)
        let newK = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
        let newV = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
        eval(newQ, newK, newV)

        let scale = 1.0 / sqrt(Float(headDim))

        // Reference
        let allK = concatenated([rawK, newK], axis: 2)
        let allV = concatenated([rawV, newV], axis: 2)
        let refOutput = MLXFast.scaledDotProductAttention(
            queries: newQ, keys: allK, values: allV,
            scale: scale, mask: .none
        )
        eval(refOutput)

        // TurboQuant
        let cache = TurboQuantKVCache(bits: keyBits, keyBits: keyBits, valueBits: valueBits)
        let _ = cache.update(keys: rawK, values: rawV)
        let turboOutput = cache.compressedAttention(
            queries: newQ, keys: newK, values: newV,
            scale: scale, mask: .none
        )
        eval(turboOutput)

        let refFP = refOutput.asType(.float32)
        let turboFP = turboOutput.asType(.float32)
        let maxDiff = abs(refFP - turboFP).max().item(Float.self)
        let meanDiff = mean(abs(refFP - turboFP)).item(Float.self)

        print("[DIAGNOSTIC] Asymmetric 4K/2V compressedAttention:")
        print("[DIAGNOSTIC]   Max diff: \(maxDiff), Mean diff: \(meanDiff)")

        #expect(maxDiff < 1.0,
            "TurboQuant 4v2 output is garbled (maxDiff=\(maxDiff)). Expected <1.0.")
    }

    /// Isolate the encode kernel: encode then manually decode and compare.
    /// Tests framework turboEncode independently of scoring.
    @Test func encodeDecodeRoundTrip() {
        let dim = 128
        let bits = 4
        let codec = MSECodec(dim: dim, bits: bits, seed: 42)

        // Random bf16 input vectors
        let input = MLXRandom.normal([8, dim]).asType(.bfloat16)
        eval(input)

        // Encode via framework dispatch
        let inputFP = input.asType(.float32)
        let (packed, norms) = TurboQuantKernelOps.fusedEncode(
            input: inputFP, rotation: codec.rotation,
            boundaries: codec.boundaries, codebook: codec.codebook,
            bits: bits, dim: dim)
        eval(packed, norms)

        // Manually decode: unpack indices → codebook lookup → inverse rotate → scale by norm
        let indices = TurboQuantPacking.unpackLowBit(packed, bits: bits, count: dim)
        let approx = codec.codebook[indices]  // [8, dim] — quantized unit vector in rotated space
        let unrotated = matmul(approx, codec.rotation)  // inverse rotate
        let reconstructed = expandedDimensions(norms, axis: -1) * unrotated
        eval(reconstructed)

        // Compare against original
        let diff = abs(inputFP - reconstructed)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)
        let inputRange = inputFP.max().item(Float.self) - inputFP.min().item(Float.self)

        print("[DIAGNOSTIC] Encode-decode round trip (dim=\(dim), bits=\(bits)):")
        print("[DIAGNOSTIC]   Max diff: \(maxDiff), Mean diff: \(meanDiff), Input range: \(inputRange)")

        // 4-bit quantization should have bounded error
        #expect(maxDiff < 2.0,
            "Encode round-trip error too large (\(maxDiff)). Encoder may be broken.")
    }

    /// Isolate the scoring kernel: compare framework turboScore against manual ops.
    @Test func scoreKernelMatchesOps() {
        let dim = 128
        let bits = 4
        let nKVHeads = 4
        let nQHeads = 8
        let tokenCount = 16
        let repeatCount = nQHeads / nKVHeads
        let codec = MSECodec(dim: dim, bits: bits, seed: 42)

        // Encode random keys
        let rawKeys = MLXRandom.normal([nKVHeads * tokenCount, dim])
        eval(rawKeys)
        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncode(
            input: rawKeys, rotation: codec.rotation,
            boundaries: codec.boundaries, codebook: codec.codebook,
            bits: bits, dim: dim)
        eval(keyPacked, keyNorms)

        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, -1])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])

        // Random pre-rotated queries (already in rotated space)
        let queries = MLXRandom.normal([nQHeads, dim])
        eval(queries)

        // Framework turboScore kernel
        let kernelScores = TurboQuantKernelOps.mseScore(
            rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: codec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: bits, dim: dim)
        eval(kernelScores)

        // Manual ops: dequant keys then matmul
        let indices = TurboQuantPacking.unpackLowBit(keyPacked, bits: bits, count: dim)
        let dequantKeys = codec.codebook[indices]  // [nKVHeads * tokenCount, dim]
        let dequantKeysReshaped = dequantKeys.reshaped([nKVHeads, tokenCount, dim])
        let dequantNorms = flatKeyNorms[0..., 0..., .newAxis]  // [nKVHeads, tokenCount, 1]
        let scaledKeys = dequantKeysReshaped * dequantNorms  // [nKVHeads, tokenCount, dim]

        // Expand keys for GQA
        var expandedKeys = expandedDimensions(scaledKeys, axis: 1)  // [nKVHeads, 1, tokenCount, dim]
        expandedKeys = MLX.tiled(expandedKeys, repetitions: [1, repeatCount, 1, 1])  // [nKVHeads, repeat, T, dim]
        expandedKeys = expandedKeys.reshaped([nQHeads, tokenCount, dim])

        // ops scores: Q @ K^T
        let opsScores = matmul(queries[0..., .newAxis, 0...], expandedKeys.transposed(0, 2, 1))
            .squeezed(axis: 1)  // [nQHeads, tokenCount]
        eval(opsScores)

        let diff = abs(kernelScores - opsScores)
        let maxDiff = diff.max().item(Float.self)

        print("[DIAGNOSTIC] Score kernel vs ops:")
        print("[DIAGNOSTIC]   Max diff: \(maxDiff)")
        print("[DIAGNOSTIC]   Kernel range: [\(kernelScores.min().item(Float.self)), \(kernelScores.max().item(Float.self))]")
        print("[DIAGNOSTIC]   Ops range: [\(opsScores.min().item(Float.self)), \(opsScores.max().item(Float.self))]")

        #expect(maxDiff < 0.01,
            "Score kernel mismatch (\(maxDiff)). Framework dispatch may be broken.")
    }

    /// Test multiple decode steps after compression to catch offset/buffer bugs.
    @Test func multiStepDecodeConsistency() {
        let B = 1
        let nQHeads = 8
        let nKVHeads = 4
        let headDim = 128
        let prefillLen = 16
        let keyBits = 4
        let valueBits = 4
        let decodeSteps = 10

        let rawK = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        let rawV = MLXRandom.normal([B, nKVHeads, prefillLen, headDim]).asType(.bfloat16)
        eval(rawK, rawV)

        let cache = TurboQuantKVCache(bits: keyBits, keyBits: keyBits, valueBits: valueBits)
        let _ = cache.update(keys: rawK, values: rawV)

        let scale = 1.0 / sqrt(Float(headDim))
        var prevOutput: MLXArray? = nil

        for step in 0 ..< decodeSteps {
            let q = MLXRandom.normal([B, nQHeads, 1, headDim]).asType(.bfloat16)
            let k = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
            let v = MLXRandom.normal([B, nKVHeads, 1, headDim]).asType(.bfloat16)
            eval(q, k, v)

            let output = cache.compressedAttention(
                queries: q, keys: k, values: v,
                scale: scale, mask: .none
            )
            eval(output)

            // Sanity checks
            #expect(output.shape == [B, nQHeads, 1, headDim],
                "Step \(step): wrong output shape \(output.shape)")
            #expect(cache.offset == prefillLen + step + 1,
                "Step \(step): wrong offset \(cache.offset), expected \(prefillLen + step + 1)")

            // Check output isn't NaN or Inf
            let hasNaN = any(isNaN(output)).item(Bool.self)
            let hasInf = any(abs(output) .> 1e6).item(Bool.self)
            #expect(!hasNaN, "Step \(step): output contains NaN")
            #expect(!hasInf, "Step \(step): output contains extreme values")

            // Check output varies between steps (not stuck)
            if let prev = prevOutput {
                let stepDiff = abs(output.asType(.float32) - prev.asType(.float32)).max().item(Float.self)
                #expect(stepDiff > 1e-6,
                    "Step \(step): output identical to previous step (stuck)")
            }
            prevOutput = output
        }
        print("[DIAGNOSTIC] Multi-step decode: \(decodeSteps) steps completed, offset=\(cache.offset)")
    }
}
