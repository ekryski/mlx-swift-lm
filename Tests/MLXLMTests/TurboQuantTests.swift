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
