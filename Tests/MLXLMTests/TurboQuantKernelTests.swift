import XCTest
import MLX
import MLXNN
@testable import MLXLMCommon

/// Validates TurboQuant Metal kernels against pure-MLX reference dequant.
/// Tests encode → Metal score/value → compare against encode → reference dequant → matmul.
final class TurboQuantKernelTests: XCTestCase {

    // MARK: - Helpers

    /// Reference dequant: unpack packed indices → codebook lookup → scale by norm.
    /// Returns [rows, dim] in rotated space.
    private func referenceDequant(
        packed: MLXArray, norms: MLXArray, codebook: MLXArray,
        bits: Int, dim: Int
    ) -> MLXArray {
        let mask = Int32((1 << bits) - 1)
        var allIndices: [MLXArray] = []
        for d in 0..<dim {
            let bitOffset = d * bits
            let wordIdx = bitOffset / 32
            let shift = bitOffset % 32
            var idx = (packed[0..., wordIdx...wordIdx] >> MLXArray(Int32(shift)))
            let spill = shift + bits - 32
            if spill > 0 {
                let nextWord = packed[0..., (wordIdx+1)...(wordIdx+1)]
                idx = idx | (nextWord << MLXArray(Int32(bits - spill)))
            }
            idx = idx & MLXArray(mask)
            allIndices.append(idx)
        }
        let indices = concatenated(allIndices, axis: -1)  // [rows, dim]
        let centroidValues = codebook[indices.asType(.int32)]
        let normsExpanded = norms.expandedDimensions(axis: -1)
        return centroidValues * normsExpanded  // [rows, dim] rotated
    }

    // MARK: - Tests

    /// Test turbo_score kernel: encode K, score Q*K via Metal, compare to reference.
    func testTurboScoreKernel() {
        for T in [64, 256, 1024, 2048, 4096] {
            for bits in [2, 3, 4] {
                let dim = 256
                let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
                let H = 4  // KV heads
                let nQHeads = 16  // query heads
                let repeatCount = nQHeads / H

                // Create codec
                let codec = MSECodec(dim: dim, bits: bits, seed: 42)

                // Random K data
                let key = MLXRandom.key(UInt64(T * 100 + bits))
                let rawK = MLXRandom.normal([H * T, dim], key: key).asType(.float32)

                // Encode via Metal
                let (packed, norms): (MLXArray, MLXArray)
                if codec.useWHT, let signs = codec.whtSigns {
                    (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
                        input: rawK, whtSigns: signs,
                        boundaries: codec.boundaries, codebook: codec.codebook,
                        bits: bits, dim: dim)
                } else {
                    (packed, norms) = TurboQuantKernelOps.fusedEncode(
                        input: rawK, rotation: codec.rotation,
                        boundaries: codec.boundaries, codebook: codec.codebook,
                        bits: bits, dim: dim)
                }
                let packedShaped = packed.reshaped([H, T, pw])
                let normsShaped = norms.reshaped([H, T])

                // Random query
                let q = MLXRandom.normal([nQHeads, dim], key: MLXRandom.key(42)).asType(.float32)
                let qRot = matmul(q, codec.rotationT)

                // Metal kernel score
                let metalScores = MLXFast.turboScore(
                    qRot, packed: packedShaped, norms: normsShaped,
                    codebook: codec.codebook,
                    tokenCount: T, repeatCount: repeatCount,
                    bits: bits, dim: dim)
                eval(metalScores)

                // Reference score: dequant K → matmul
                let dequantK = referenceDequant(
                    packed: packed, norms: norms, codebook: codec.codebook,
                    bits: bits, dim: dim)
                let dequantKShaped = dequantK.reshaped([H, T, dim])

                // GQA expand. Test fixes nQHeads=16, H=4 → repeatCount=4, so the
                // tile branch is always taken; if you parameterize H to equal
                // nQHeads later, restore the `if repeatCount > 1` guard.
                let exp = expandedDimensions(dequantKShaped, axis: 1)
                let tiled = MLX.tiled(exp, repetitions: [1, repeatCount, 1, 1])
                let kExpanded = tiled.reshaped([nQHeads, T, dim])

                let refScores = matmul(
                    qRot.expandedDimensions(axis: 1),  // [nQHeads, 1, dim]
                    kExpanded.transposed(0, 2, 1)       // [nQHeads, dim, T]
                ).squeezed(axis: 1)  // [nQHeads, T]

                // Scale by norms (reference handles this in dequant, Metal in kernel)
                eval(refScores)

                let metalFlat = metalScores.reshaped([-1])
                let refFlat = refScores.reshaped([-1])

                let metalHasNaN = MLX.isNaN(metalFlat).any().item(Bool.self)
                let refHasNaN = MLX.isNaN(refFlat).any().item(Bool.self)

                XCTAssertFalse(refHasNaN, "Reference has NaN at T=\(T) bits=\(bits)")
                XCTAssertFalse(metalHasNaN, "Metal turboScore has NaN at T=\(T) bits=\(bits)")

                if !metalHasNaN && !refHasNaN {
                    let diff = MLX.abs(metalFlat - refFlat)
                    let maxDiff = diff.max().item(Float.self)
                    let refMax = MLX.abs(refFlat).max().item(Float.self)
                    let relErr = refMax > 0 ? maxDiff / refMax : maxDiff
                    print("turboScore T=\(T) bits=\(bits): maxDiff=\(maxDiff) relErr=\(String(format: "%.4f", relErr))")
                    XCTAssertLessThan(relErr, 0.1, "turboScore relErr too high at T=\(T) bits=\(bits)")
                }
            }
        }
    }

    /// Test turbo_value kernel: encode V, weighted sum via Metal, compare to reference.
    func testTurboValueKernel() {
        for T in [64, 256, 1024, 2048, 4096] {
            for bits in [2, 3, 4] {
                let dim = 256
                let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
                let H = 4
                let nQHeads = 16
                let repeatCount = nQHeads / H

                let codec = MSECodec(dim: dim, bits: bits, seed: 43)

                // Random V data
                let rawV = MLXRandom.normal([H * T, dim], key: MLXRandom.key(UInt64(T + bits * 100))).asType(.float32)

                let (packed, norms): (MLXArray, MLXArray)
                if codec.useWHT, let signs = codec.whtSigns {
                    (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
                        input: rawV, whtSigns: signs,
                        boundaries: codec.boundaries, codebook: codec.codebook,
                        bits: bits, dim: dim)
                } else {
                    (packed, norms) = TurboQuantKernelOps.fusedEncode(
                        input: rawV, rotation: codec.rotation,
                        boundaries: codec.boundaries, codebook: codec.codebook,
                        bits: bits, dim: dim)
                }
                let packedShaped = packed.reshaped([H, T, pw])
                let normsShaped = norms.reshaped([H, T])

                // Random attention weights (softmax output)
                let rawWeights = MLXRandom.uniform(low: 0, high: 1, [nQHeads, T],
                    key: MLXRandom.key(77))
                let weights = rawWeights / rawWeights.sum(axis: -1, keepDims: true)

                // Metal kernel value
                let metalOutput = MLXFast.turboValue(
                    weights.reshaped([nQHeads, T]),
                    packed: packedShaped, norms: normsShaped,
                    codebook: codec.codebook,
                    tokenCount: T, repeatCount: repeatCount,
                    sparseThreshold: 0,
                    bits: bits, dim: dim)
                eval(metalOutput)

                // Reference: dequant V → matmul
                let dequantV = referenceDequant(
                    packed: packed, norms: norms, codebook: codec.codebook,
                    bits: bits, dim: dim)
                let dequantVShaped = dequantV.reshaped([H, T, dim])

                // GQA expand. Test fixes nQHeads=16, H=4 → repeatCount=4, so the
                // tile branch is always taken; if you parameterize H to equal
                // nQHeads later, restore the `if repeatCount > 1` guard.
                let exp = expandedDimensions(dequantVShaped, axis: 1)
                let tiled = MLX.tiled(exp, repetitions: [1, repeatCount, 1, 1])
                let vExpanded = tiled.reshaped([nQHeads, T, dim])

                let refOutput = matmul(
                    weights.expandedDimensions(axis: 1),  // [nQHeads, 1, T]
                    vExpanded                              // [nQHeads, T, dim]
                ).squeezed(axis: 1)  // [nQHeads, dim]
                eval(refOutput)

                let metalHasNaN = MLX.isNaN(metalOutput).any().item(Bool.self)
                let refHasNaN = MLX.isNaN(refOutput).any().item(Bool.self)

                XCTAssertFalse(refHasNaN, "Reference has NaN at T=\(T) bits=\(bits)")
                XCTAssertFalse(metalHasNaN, "Metal turboValue has NaN at T=\(T) bits=\(bits)")

                if !metalHasNaN && !refHasNaN {
                    let diff = MLX.abs(metalOutput.reshaped([-1]) - refOutput.reshaped([-1]))
                    let maxDiff = diff.max().item(Float.self)
                    let refMax = MLX.abs(refOutput).max().item(Float.self)
                    let relErr = refMax > 0 ? maxDiff / refMax : maxDiff
                    print("turboValue T=\(T) bits=\(bits): maxDiff=\(maxDiff) relErr=\(String(format: "%.4f", relErr))")
                    XCTAssertLessThan(relErr, 0.1, "turboValue relErr too high at T=\(T) bits=\(bits)")
                }
            }
        }
    }
}
