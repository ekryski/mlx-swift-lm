import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Codebook Tests

@Suite("TurboQuant Codebook")
struct TurboQuantCodebookTests {

    @Test func codebookGeneration_1bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 1)
        #expect(cb.count == 2)
        // Centroids should be sorted
        let vals = cb.asArray(Float.self)
        #expect(vals[0] < vals[1])
        // Centroids should be in [-1, 1]
        #expect(vals[0] >= -1.0)
        #expect(vals[1] <= 1.0)
    }

    @Test func codebookGeneration_2bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 2)
        #expect(cb.count == 4)
        let vals = cb.asArray(Float.self)
        // Sorted
        for i in 0 ..< vals.count - 1 {
            #expect(vals[i] <= vals[i + 1], "Codebook not sorted at index \(i)")
        }
    }

    @Test func codebookGeneration_3bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 3)
        #expect(cb.count == 8)
    }

    @Test func codebookGeneration_4bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 4)
        #expect(cb.count == 16)
        let vals = cb.asArray(Float.self)
        // All within [-1, 1]
        for v in vals {
            #expect(v >= -1.0 && v <= 1.0, "Centroid \(v) out of range")
        }
    }

    @Test func codebookDeterminism() {
        let cb1 = TurboQuantCodebook.generateCodebook(dim: 64, bits: 3)
        let cb2 = TurboQuantCodebook.generateCodebook(dim: 64, bits: 3)
        let v1 = cb1.asArray(Float.self)
        let v2 = cb2.asArray(Float.self)
        #expect(v1 == v2, "Codebook generation should be deterministic")
    }

    @Test func codebookSymmetry() {
        // For symmetric distributions (Beta on [-1,1]), centroids should be
        // approximately symmetric around 0
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 2)
        let vals = cb.asArray(Float.self)
        // First + last should be approximately symmetric
        let sum = vals.first! + vals.last!
        #expect(abs(sum) < 0.1, "Centroids should be approximately symmetric, sum=\(sum)")
    }
}

// MARK: - Rotation Tests

@Suite("TurboQuant Rotation")
struct TurboQuantRotationTests {

    @Test func rotationOrthogonality() {
        let dim = 64
        let R = TurboQuantRotation.rotationMatrix(dim: dim, seed: 42)
        #expect(R.shape == [dim, dim])

        // R @ R^T should be identity
        let RRt = matmul(R, R.transposed())
        let identity = MLXArray.eye(dim)
        let diff = abs(RRt - identity)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-4, "R @ R^T differs from I by \(maxDiff)")
    }

    @Test func rotationDeterminism() {
        let R1 = TurboQuantRotation.generateRotation(dim: 32, seed: 123)
        let R2 = TurboQuantRotation.generateRotation(dim: 32, seed: 123)
        let diff = MLX.abs(R1 - R2).max().item(Float.self)
        #expect(diff < 1e-6, "Same seed should produce same rotation")
    }

    @Test func rotationDifferentSeeds() {
        let R1 = TurboQuantRotation.generateRotation(dim: 32, seed: 1)
        let R2 = TurboQuantRotation.generateRotation(dim: 32, seed: 2)
        let diff = MLX.abs(R1 - R2).max().item(Float.self)
        #expect(diff > 0.01, "Different seeds should produce different rotations")
    }
}

// MARK: - Bit Packing Tests

@Suite("TurboQuant Bit Packing")
struct TurboQuantPackingTests {

    @Test func packedWidth() {
        // 128 dims × 4 bits = 512 bits = 16 uint32 words
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 4) == 16)
        // 128 dims × 3 bits = 384 bits = 12 uint32 words
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 3) == 12)
        // 128 dims × 2 bits = 256 bits = 8 uint32 words
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 2) == 8)
        // 128 dims × 1 bit = 128 bits = 4 uint32 words
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 1) == 4)
    }

    @Test func packUnpack_1bit() {
        let indices = MLXArray([0, 1, 0, 1, 1, 0, 1, 0] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 1)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 1, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "1-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_2bit() {
        let indices = MLXArray([0, 1, 2, 3, 3, 2, 1, 0] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 2)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 2, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "2-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_3bit() {
        let indices = MLXArray([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4] as [UInt32]).reshaped([1, 12])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 3, count: 12)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "3-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_4bit() {
        let indices = MLXArray([0, 5, 10, 15, 3, 7, 11, 14] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 4)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 4, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "4-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_wordBoundary_3bit() {
        // 3-bit at dim 10: bit offset = 30, spans word boundary (30+3=33 > 32)
        // dim 11: bit offset = 33, also crosses into word 1
        let count = 16
        var values = [UInt32]()
        for i in 0 ..< count { values.append(UInt32(i % 8)) }
        let indices = MLXArray(values).reshaped([1, count])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 3, count: count)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "3-bit word boundary pack/unpack failed")
    }

    @Test func packUnpack_batched() {
        // Test with batch dimension: [2, 8]
        let indices = MLXArray([
            0, 1, 2, 3, 4, 5, 6, 7,
            7, 6, 5, 4, 3, 2, 1, 0,
        ] as [UInt32]).reshaped([2, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 3, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "Batched pack/unpack failed")
    }
}

// MARK: - MSE Codec Tests

@Suite("TurboQuant MSE Codec")
struct TurboQuantMSECodecTests {

    @Test func encodeDecodeRoundTrip_4bit() {
        let codec = MSECodec(dim: 16, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 16])  // [B=1, H=2, T=4, D=16]
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        #expect(state.tokenCount == 4)
        #expect(state.dim == 16)
        #expect(state.bits == 4)

        // Check MSE is bounded (paper: ≤ 0.009 for 4-bit)
        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        // Relax bound since our vectors aren't unit-sphere distributed
        #expect(mse < 0.5, "4-bit MSE too high: \(mse)")
    }

    @Test func encodeDecodeRoundTrip_3bit() {
        let codec = MSECodec(dim: 16, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 16])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        // 3-bit should have higher but still bounded MSE
        #expect(mse < 1.0, "3-bit MSE too high: \(mse)")
    }

    @Test func encodeDecodeRoundTrip_2bit() {
        let codec = MSECodec(dim: 16, bits: 2, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 16])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        #expect(mse < 2.0, "2-bit MSE too high: \(mse)")
    }

    @Test func cosineSimilarity_4bit() {
        let codec = MSECodec(dim: 64, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 8, 64])
        eval(vectors)

        let decoded = codec.decode(codec.encode(vectors))

        // Compute cosine similarity per vector
        let dot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = dot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.95, "4-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func cosineSimilarity_3bit() {
        let codec = MSECodec(dim: 64, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 8, 64])
        eval(vectors)

        let decoded = codec.decode(codec.encode(vectors))

        let dot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = dot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.85, "3-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func normPreservation() {
        let codec = MSECodec(dim: 32, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 4, 32])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        // Norms should be approximately preserved
        let origNorms = sqrt((vectors * vectors).sum(axis: -1))
        let decNorms = sqrt((decoded * decoded).sum(axis: -1))
        let normDiff = abs(origNorms - decNorms).mean().item(Float.self)
        let avgNorm = origNorms.mean().item(Float.self)

        #expect(
            normDiff / avgNorm < 0.1,
            "Norm preservation poor: diff=\(normDiff), avgNorm=\(avgNorm)")
    }

    @Test func prepareQueries() {
        let codec = MSECodec(dim: 32, bits: 4, seed: 42)
        let queries = MLXRandom.normal([1, 2, 1, 32])
        eval(queries)

        let prepared = codec.prepareQueries(queries)
        #expect(prepared.shape == queries.shape)

        // Prepared queries should have same norms (rotation preserves norms)
        let origNorm = sqrt((queries * queries).sum(axis: -1)).mean().item(Float.self)
        let prepNorm = sqrt((prepared * prepared).sum(axis: -1)).mean().item(Float.self)
        #expect(abs(origNorm - prepNorm) < 0.01, "Query rotation changed norms")
    }
}

// MARK: - TurboQuantKVCache Tests

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

        // First update: 4 tokens
        let k1 = MLXRandom.normal([1, 2, 4, 32])
        let v1 = MLXRandom.normal([1, 2, 4, 32])
        eval(k1, v1)
        let (out1k, _) = cache.update(keys: k1, values: v1)
        #expect(cache.offset == 4)
        #expect(out1k.shape == [1, 2, 4, 32])

        // Second update: 1 more token
        let k2 = MLXRandom.normal([1, 2, 1, 32])
        let v2 = MLXRandom.normal([1, 2, 1, 32])
        eval(k2, v2)
        let (out2k, _) = cache.update(keys: k2, values: v2)
        #expect(cache.offset == 5)
        // Output should contain all 5 tokens
        #expect(out2k.shape == [1, 2, 5, 32])
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

    @Test func cacheTrimAll() {
        let cache = TurboQuantKVCache(bits: 3)
        let keys = MLXRandom.normal([1, 2, 4, 32])
        let values = MLXRandom.normal([1, 2, 4, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)

        let trimmed = cache.trim(4)
        #expect(trimmed == 4)
        #expect(cache.offset == 0)
    }

    @Test func cacheTrimmable() {
        let cache = TurboQuantKVCache(bits: 4)
        #expect(cache.isTrimmable == true)
    }

    @Test func cacheState() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 2, 4, 32])
        let values = MLXRandom.normal([1, 2, 4, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)

        let state = cache.state
        #expect(state.count == 2, "State should have 2 arrays (dequantized keys, dequantized values)")
    }

    @Test func cacheAllBitWidths() {
        // Verify all supported bit widths work
        for bits in 1 ... 4 {
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
