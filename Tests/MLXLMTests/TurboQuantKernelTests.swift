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

    /// Validates `MLXFast.turboFlashSDPAv(... sinks:)` against a
    /// dequant-then-SDPA reference on a GPT-OSS-shaped decode step. Both
    /// paths score in rotated codec space (`referenceDequant` returns
    /// rotated values) so this isolates the inline-dequant + online-softmax
    /// + sinks-fold composition from the codec rotation itself.
    func testTurboFlashSDPAvSinksSlidingWindow() {
        for (kb, vb) in [(4, 4), (4, 2), (3, 3), (8, 4)] {
            let B = 1, nQ = 8, nKV = 2, T = 64, D = 64
            let repeatCount = nQ / nKV
            let windowSize = 32
            let kpw = TurboQuantPacking.packedWidth(count: D, bits: kb)
            let vpw = TurboQuantPacking.packedWidth(count: D, bits: vb)

            let qRot = MLXRandom.normal([B * nQ, D],
                key: MLXRandom.key(UInt64(kb * 100 + vb))).asType(.float32)

            let kCodec = MSECodec(dim: D, bits: kb, seed: 13)
            let vCodec = MSECodec(dim: D, bits: vb, seed: 17)
            let rawK = MLXRandom.normal([nKV * T, D],
                key: MLXRandom.key(UInt64(kb * 31))).asType(.float32)
            let rawV = MLXRandom.normal([nKV * T, D],
                key: MLXRandom.key(UInt64(vb * 53))).asType(.float32)

            // Encode K
            let (kPackedFlat, kNormsFlat): (MLXArray, MLXArray)
            if kCodec.useWHT, let signs = kCodec.whtSigns {
                (kPackedFlat, kNormsFlat) = TurboQuantKernelOps.fusedEncodeWHT(
                    input: rawK, whtSigns: signs,
                    boundaries: kCodec.boundaries, codebook: kCodec.codebook,
                    bits: kb, dim: D)
            } else {
                (kPackedFlat, kNormsFlat) = TurboQuantKernelOps.fusedEncode(
                    input: rawK, rotation: kCodec.rotation,
                    boundaries: kCodec.boundaries, codebook: kCodec.codebook,
                    bits: kb, dim: D)
            }
            let kPacked = kPackedFlat.reshaped([nKV, T, kpw])
            let kNorms = kNormsFlat.reshaped([nKV, T])

            // Encode V
            let (vPackedFlat, vNormsFlat): (MLXArray, MLXArray)
            if vCodec.useWHT, let signs = vCodec.whtSigns {
                (vPackedFlat, vNormsFlat) = TurboQuantKernelOps.fusedEncodeWHT(
                    input: rawV, whtSigns: signs,
                    boundaries: vCodec.boundaries, codebook: vCodec.codebook,
                    bits: vb, dim: D)
            } else {
                (vPackedFlat, vNormsFlat) = TurboQuantKernelOps.fusedEncode(
                    input: rawV, rotation: vCodec.rotation,
                    boundaries: vCodec.boundaries, codebook: vCodec.codebook,
                    bits: vb, dim: D)
            }
            let vPacked = vPackedFlat.reshaped([nKV, T, vpw])
            let vNorms = vNormsFlat.reshaped([nKV, T])

            // Modest sink magnitude — measurable but doesn't dominate.
            let sinks = MLXRandom.normal([nQ], key: MLXRandom.key(UInt64(91))) * MLXArray(Float(0.5))

            let kernelOut = MLXFast.turboFlashSDPAv(
                queries: qRot,
                kPacked: kPacked, kNorms: kNorms, kCodebook: kCodec.codebook,
                vPacked: vPacked, vNorms: vNorms, vCodebook: vCodec.codebook,
                keyBits: kb, valueBits: vb, dim: D, repeatCount: repeatCount,
                sinks: sinks,
                causal: true,
                windowSize: windowSize
            )
            eval(kernelOut)

            // Reference: dequant K, dequant V, build score, softmax-with-sinks,
            // weighted sum. Both paths score in rotated codec space.
            let kDequantFlat = referenceDequant(
                packed: kPackedFlat, norms: kNormsFlat, codebook: kCodec.codebook,
                bits: kb, dim: D)  // [nKV*T, D]
            let vDequantFlat = referenceDequant(
                packed: vPackedFlat, norms: vNormsFlat, codebook: vCodec.codebook,
                bits: vb, dim: D)  // [nKV*T, D]
            let kRef = kDequantFlat.reshaped([nKV, T, D])
            let vRef = vDequantFlat.reshaped([nKV, T, D])

            // GQA expand to [nQ, T, D]
            let kExp = MLX.tiled(
                expandedDimensions(kRef, axis: 1),
                repetitions: [1, repeatCount, 1, 1]
            ).reshaped([nQ, T, D])
            let vExp = MLX.tiled(
                expandedDimensions(vRef, axis: 1),
                repetitions: [1, repeatCount, 1, 1]
            ).reshaped([nQ, T, D])

            // score = q @ K^T → [nQ, 1, T]
            let qVec = qRot.reshaped([nQ, 1, D])
            var scores = matmul(qVec, kExp.transposed(0, 2, 1))  // [nQ, 1, T]

            // Sliding window mask: keep i ∈ (T-1-ws, T-1]
            let positions = MLXArray(Int32(0)..<Int32(T))  // [T]
            let lower = Int32(T - 1 - windowSize)
            let keep = positions .> MLXArray(lower)  // [T] bool
            let keep4 = keep
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)  // [1, 1, T]
            scores = MLX.where(keep4, scores, MLXArray(-Float.infinity))

            // Softmax with sinks fold: numerically stable manual softmax.
            let maxScore = scores.max(axis: -1, keepDims: true)  // [nQ, 1, 1]
            let sinksReshaped = sinks.reshaped([nQ, 1, 1])
            let folded = MLX.maximum(maxScore, sinksReshaped)
            let expScores = MLX.exp(scores - folded)  // [nQ, 1, T]
            let expSinks = MLX.exp(sinksReshaped - folded)  // [nQ, 1, 1]
            let denom = expScores.sum(axis: -1, keepDims: true) + expSinks
            let weights = expScores / denom  // [nQ, 1, T]

            let refOut = matmul(weights, vExp).reshaped([nQ, D])  // [nQ, D]
            eval(refOut)

            // Kernel emits bfloat16; cast to float32 for the diff.
            let kernelF32 = kernelOut.asType(.float32)
            let diff = MLX.abs(kernelF32 - refOut)
            let maxDiff = diff.max().item(Float.self)
            let refMax = MLX.abs(refOut).max().item(Float.self)
            let relErr = refMax > 0 ? maxDiff / refMax : maxDiff
            let hasNaN = MLX.isNaN(kernelF32).any().item(Bool.self)
            let hasInf = MLX.isInf(kernelF32).any().item(Bool.self)

            print("turboFlashSDPAv kb=\(kb) vb=\(vb): maxDiff=\(maxDiff) relErr=\(String(format: "%.4f", relErr))")
            XCTAssertFalse(hasNaN, "kernel output NaN at kb=\(kb) vb=\(vb)")
            XCTAssertFalse(hasInf, "kernel output Inf at kb=\(kb) vb=\(vb)")
            // bf16 + online softmax tolerance — same bar the affine flash
            // SDPA tests use (~6e-2 worst-case for 4-bit codec on this shape).
            XCTAssertLessThan(relErr, 0.1,
                "turboFlashSDPAv relErr too high at kb=\(kb) vb=\(vb)")
        }
    }

    /// With `sinks` set to large-negative values (≈ no contribution),
    /// the new single-pass kernel must match the existing two-pass
    /// `turboFlashAttention` on the same packed K/V — catches
    /// score/softmax/value regressions independent of the sinks fold.
    func testTurboFlashSDPAvNoSinksMatchesTurboFlashAttention() {
        let kb = 4, vb = 4
        let B = 1, nQ = 8, nKV = 2, T = 64, D = 64
        let repeatCount = nQ / nKV
        let kpw = TurboQuantPacking.packedWidth(count: D, bits: kb)
        let vpw = TurboQuantPacking.packedWidth(count: D, bits: vb)

        let qRot = MLXRandom.normal([B * nQ, D],
            key: MLXRandom.key(701)).asType(.float32)

        let kCodec = MSECodec(dim: D, bits: kb, seed: 71)
        let vCodec = MSECodec(dim: D, bits: vb, seed: 73)
        let rawK = MLXRandom.normal([nKV * T, D],
            key: MLXRandom.key(81)).asType(.float32)
        let rawV = MLXRandom.normal([nKV * T, D],
            key: MLXRandom.key(83)).asType(.float32)

        let (kPackedFlat, kNormsFlat) = TurboQuantKernelOps.fusedEncodeWHT(
            input: rawK, whtSigns: kCodec.whtSigns!,
            boundaries: kCodec.boundaries, codebook: kCodec.codebook,
            bits: kb, dim: D)
        let (vPackedFlat, vNormsFlat) = TurboQuantKernelOps.fusedEncodeWHT(
            input: rawV, whtSigns: vCodec.whtSigns!,
            boundaries: vCodec.boundaries, codebook: vCodec.codebook,
            bits: vb, dim: D)
        let kPacked = kPackedFlat.reshaped([nKV, T, kpw])
        let kNorms = kNormsFlat.reshaped([nKV, T])
        let vPacked = vPackedFlat.reshaped([nKV, T, vpw])
        let vNorms = vNormsFlat.reshaped([nKV, T])

        // Reference: existing two-pass kernel, no sinks. Already validated.
        let outRef = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: qRot,
            keyPacked: kPacked, keyNorms: kNorms, keyCodebook: kCodec.codebook,
            valPacked: vPacked, valNorms: vNorms, valCodebook: vCodec.codebook,
            tokenCount: T, repeatCount: repeatCount,
            keyBits: kb, valueBits: vb, dim: D,
            valRotation: nil
        )

        // New single-pass kernel with sinks ≈ -INF (suppressed). The fold's
        // `exp(sink - new_max)` term goes to ~0, leaving the result equal to
        // the no-sinks path within fp32 precision.
        let suppressedSinks = MLXArray([Float](repeating: -1e30, count: nQ))
        let outBeta = MLXFast.turboFlashSDPAv(
            queries: qRot,
            kPacked: kPacked, kNorms: kNorms, kCodebook: kCodec.codebook,
            vPacked: vPacked, vNorms: vNorms, vCodebook: vCodec.codebook,
            keyBits: kb, valueBits: vb, dim: D, repeatCount: repeatCount,
            sinks: suppressedSinks,
            causal: false,
            windowSize: -1
        )
        eval(outRef, outBeta)

        let outRefF = outRef.asType(.float32)
        let outBetaF = outBeta.asType(.float32)
        let diff = MLX.abs(outRefF - outBetaF)
        let maxDiff = diff.max().item(Float.self)
        let refMag = MLX.abs(outRefF).max().item(Float.self)
        let relErr = refMag > 0 ? maxDiff / refMag : maxDiff
        let hasNaN = MLX.isNaN(outBetaF).any().item(Bool.self)

        print("turbo_flash_sdpa_v vs turboFlashAttention (sinks≈-INF): maxDiff=\(maxDiff) relErr=\(String(format: "%.4f", relErr))")
        XCTAssertFalse(hasNaN, "single-pass kernel output must not be NaN")
        // bf16 round-trip + fp32 reduction differences across two
        // independently-implemented kernels — same threshold as the
        // sinks test (0.05).
        XCTAssertLessThan(relErr, 0.05,
            "single-pass kernel (sinks≈-INF) must match turboFlashAttention; relErr=\(relErr)")

        // Sinks=0 probe: the sink contributes exp(0)=1 to the denominator,
        // which shifts the output magnitude slightly but not by ~30%. If
        // the output differs by more than ~30% relative to the no-sinks
        // path, the sinks handling has a systematic bug.
        let zeroSinks = MLXArray([Float](repeating: 0.0, count: nQ))
        let outBetaZero = MLXFast.turboFlashSDPAv(
            queries: qRot,
            kPacked: kPacked, kNorms: kNorms, kCodebook: kCodec.codebook,
            vPacked: vPacked, vNorms: vNorms, vCodebook: vCodec.codebook,
            keyBits: kb, valueBits: vb, dim: D, repeatCount: repeatCount,
            sinks: zeroSinks,
            causal: false,
            windowSize: -1
        )
        eval(outBetaZero)
        let outBetaZeroF = outBetaZero.asType(.float32)
        let diffZero = MLX.abs(outBetaZeroF - outRefF)
        let maxDiffZero = diffZero.max().item(Float.self)
        let relErrZero = refMag > 0 ? maxDiffZero / refMag : maxDiffZero
        print("turbo_flash_sdpa_v +sinks=0 vs turboFlashAttention: maxDiff=\(maxDiffZero) relErr=\(String(format: "%.4f", relErrZero))")
        XCTAssertLessThan(relErrZero, 0.3,
            "single-pass kernel (sinks=0) drifted too far from no-sinks: relErr=\(relErrZero)")

        // End-to-end equivalence WITH rotation. The existing path fuses
        // the inverse rotation into the pass2 kernel; the new single-pass wrapper
        // applies it as a separate `matmul(raw, valRotation)`. If both
        // are matmul(rotated_out, valRotation), the outputs must match
        // within bf16 precision.
        let valRotation = vCodec.rotation
        let outRefRotated = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: qRot,
            keyPacked: kPacked, keyNorms: kNorms, keyCodebook: kCodec.codebook,
            valPacked: vPacked, valNorms: vNorms, valCodebook: vCodec.codebook,
            tokenCount: T, repeatCount: repeatCount,
            keyBits: kb, valueBits: vb, dim: D,
            valRotation: valRotation
        )
        let outBetaRotated = TurboQuantKernelOps.turboFlashSDPAv(
            rotatedQueries: qRot,
            keyPacked: kPacked, keyNorms: kNorms, keyCodebook: kCodec.codebook,
            valPacked: vPacked, valNorms: vNorms, valCodebook: vCodec.codebook,
            tokenCount: T, repeatCount: repeatCount,
            keyBits: kb, valueBits: vb, dim: D,
            sinks: zeroSinks,
            causal: false,
            windowSize: -1,
            valRotation: valRotation
        )
        eval(outRefRotated, outBetaRotated)
        let diffRot = MLX.abs(outRefRotated.asType(.float32) - outBetaRotated.asType(.float32))
        let maxDiffRot = diffRot.max().item(Float.self)
        let refRotMag = MLX.abs(outRefRotated.asType(.float32)).max().item(Float.self)
        let relErrRot = refRotMag > 0 ? maxDiffRot / refRotMag : maxDiffRot
        print("turbo_flash_sdpa_v (sinks=0, +rot) vs turboFlashAttention (+rot): maxDiff=\(maxDiffRot) relErr=\(String(format: "%.4f", relErrRot))")
        XCTAssertLessThan(relErrRot, 0.05,
            "turbo_flash_sdpa_v (sinks=0) +rotation differs from turboFlashAttention +rotation: relErr=\(relErrRot)")

        // bf16 Q vs fp32 Q: the existing path and the new single-pass path both cast
        // queries to float32 inside the kernel, so the bf16 input should
        // produce ~identical output to the fp32 input. If not, something
        // in the cast chain is broken.
        let qRotBF16 = qRot.asType(.bfloat16)
        let outRefBF16 = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: qRotBF16,
            keyPacked: kPacked, keyNorms: kNorms, keyCodebook: kCodec.codebook,
            valPacked: vPacked, valNorms: vNorms, valCodebook: vCodec.codebook,
            tokenCount: T, repeatCount: repeatCount,
            keyBits: kb, valueBits: vb, dim: D,
            valRotation: valRotation
        )
        let outBetaBF16 = TurboQuantKernelOps.turboFlashSDPAv(
            rotatedQueries: qRotBF16,
            keyPacked: kPacked, keyNorms: kNorms, keyCodebook: kCodec.codebook,
            valPacked: vPacked, valNorms: vNorms, valCodebook: vCodec.codebook,
            tokenCount: T, repeatCount: repeatCount,
            keyBits: kb, valueBits: vb, dim: D,
            sinks: zeroSinks,
            causal: false,
            windowSize: -1,
            valRotation: valRotation
        )
        eval(outRefBF16, outBetaBF16)
        let diffBF16 = MLX.abs(outRefBF16.asType(.float32) - outBetaBF16.asType(.float32))
        let maxDiffBF16 = diffBF16.max().item(Float.self)
        let refMagBF16 = MLX.abs(outRefBF16.asType(.float32)).max().item(Float.self)
        print("turbo_flash_sdpa_v (BF16 Q, sinks=0, +rot) vs turboFlashAttention (BF16 Q, +rot): maxDiff=\(maxDiffBF16) refMag=\(refMagBF16)")
        XCTAssertLessThan(maxDiffBF16, 0.1,
            "turbo_flash_sdpa_v (BF16 Q, sinks=0) +rotation differs from turboFlashAttention BF16: maxDiff=\(maxDiffBF16)")
    }

    /// Empirical MSE comparison probe — single-scale vs dual-half-scale
    /// (TQ4_1S-style) vs DC-bias-correction on the MSE codec. Drove the
    /// session's decision to ship bias correction (9-22% MSE reduction on
    /// structured K/V) rather than dual-scale (1-8% — the WHT rotation
    /// flattens the per-half asymmetry dual-scale wants to exploit). Also
    /// asserts that `MSECodec.encode(useBias:)` matches the inline
    /// reference math.
    func testDualScaleMSEReduction() {
        let dim = 64
        let halfDim = dim / 2
        let N = 1024

        // Quantize helper that mirrors codec.boundaryQuantize but only for 2D input.
        func quantize2D(_ x: MLXArray, boundaries: MLXArray) -> MLXArray {
            let nBoundaries = boundaries.shape[0]
            let xExp = expandedDimensions(x, axis: -1)  // [N, D, 1]
            let bExp = boundaries.reshaped([1, 1, nBoundaries])  // [1, 1, B]
            let gt = (xExp .> bExp).asType(.int32)
            return gt.sum(axis: -1)  // [N, D]
        }

        // Three input distributions to probe dual-scale's behavior:
        //   (a) random unit — symmetric per-half on average, expect no gain
        //   (b) per-dim asymmetric mean — mimics K/V mean components
        let inputs: [(String, MLXArray)] = [
            ("normal_unit", {
                let v = MLXRandom.normal([N, dim], key: MLXRandom.key(11)).asType(.float32)
                return v
            }()),
            ("asymmetric_mean", {
                let firstMean = MLXArray.ones([1, halfDim], dtype: .float32) * MLXArray(Float(1.5))
                let secondMean = MLXArray.ones([1, halfDim], dtype: .float32) * MLXArray(Float(-0.4))
                let mean = concatenated([firstMean, secondMean], axis: -1)
                let noise = MLXRandom.normal([N, dim], key: MLXRandom.key(12)).asType(.float32)
                return mean + noise * MLXArray(Float(0.3))
            }())
        ]

        for bits in [2, 3, 4, 8] {
            let codec = MSECodec(dim: dim, bits: bits, seed: 42)
            let rotationT = codec.rotationT.asType(.float32)
            let rotation = codec.rotation.asType(.float32)
            let codebook = codec.codebook.asType(.float32)
            let boundaries = codec.boundaries

            for (label, v) in inputs {
                let totalNorm = sqrt((v * v).sum(axis: -1, keepDims: true))
                let safeNorm = maximum(totalNorm, MLXArray(Float(1e-8)))
                let unit = v / safeNorm
                let rotated = matmul(unit, rotationT)  // [N, dim], L2≈1

                // ─── Single-scale ───
                let idxSingle = quantize2D(rotated, boundaries: boundaries)
                let approxSingle = codebook[idxSingle]  // [N, dim]
                let recoveredVSingle = matmul(approxSingle, rotation) * totalNorm
                let mseSingle = ((v - recoveredVSingle) * (v - recoveredVSingle)).mean().item(Float.self)

                // ─── Dual-scale ───
                let firstHalf = rotated[0..., 0 ..< halfDim]
                let secondHalf = rotated[0..., halfDim ..< dim]
                let lowPartial = sqrt((firstHalf * firstHalf).sum(axis: -1, keepDims: true))
                let highPartial = sqrt((secondHalf * secondHalf).sum(axis: -1, keepDims: true))
                let sqrt2 = MLXArray(Float(sqrt(2.0)))
                let scaleLow = maximum(lowPartial * sqrt2, MLXArray(Float(1e-8)))
                let scaleHigh = maximum(highPartial * sqrt2, MLXArray(Float(1e-8)))
                let renormFirst = firstHalf / scaleLow
                let renormSecond = secondHalf / scaleHigh
                let renormed = concatenated([renormFirst, renormSecond], axis: -1)
                let idxDual = quantize2D(renormed, boundaries: boundaries)
                let approxDual = codebook[idxDual]
                let approxFirst = approxDual[0..., 0 ..< halfDim] * scaleLow
                let approxSecond = approxDual[0..., halfDim ..< dim] * scaleHigh
                let recoveredRotDual = concatenated([approxFirst, approxSecond], axis: -1)
                let recoveredVDual = matmul(recoveredRotDual, rotation) * totalNorm
                let mseDual = ((v - recoveredVDual) * (v - recoveredVDual)).mean().item(Float.self)

                let ratio = mseSingle > 0 ? mseDual / mseSingle : Float(1.0)
                let reduction = (1.0 - ratio) * 100.0
                print("[MSE] bits=\(bits) dist=\(label) single=\(String(format: "%.6f", mseSingle)) dual=\(String(format: "%.6f", mseDual)) ratio=\(String(format: "%.3f", ratio)) reduction=\(String(format: "%.1f%%", reduction))")

                // ─── Single-scale + per-vector bias correction ───
                // Subtract per-vector mean, encode the centered vector,
                // add mean back at decode time. Bias captures DC offset
                // that the Lloyd-Max codebook (zero-mean calibrated)
                // can't otherwise represent.
                let meanScalar = v.mean(axis: -1, keepDims: true)  // [N, 1]
                let vCentered = v - meanScalar
                let totalNormB = sqrt((vCentered * vCentered).sum(axis: -1, keepDims: true))
                let safeNormB = maximum(totalNormB, MLXArray(Float(1e-8)))
                let unitB = vCentered / safeNormB
                let rotatedB = matmul(unitB, rotationT)
                let idxBias = quantize2D(rotatedB, boundaries: boundaries)
                let approxBias = codebook[idxBias]
                let recoveredRotBias = approxBias
                let recoveredCenteredBias = matmul(recoveredRotBias, rotation) * totalNormB
                let recoveredVBias = recoveredCenteredBias + meanScalar
                let mseBias = ((v - recoveredVBias) * (v - recoveredVBias)).mean().item(Float.self)
                let ratioBias = mseSingle > 0 ? mseBias / mseSingle : Float(1.0)
                let reductionBias = (1.0 - ratioBias) * 100.0
                print("[MSE]                                bias=\(String(format: "%.6f", mseBias)) ratio=\(String(format: "%.3f", ratioBias)) reduction=\(String(format: "%.1f%%", reductionBias))")

                // Verify the integrated MSECodec.encode/decode bias path
                // matches the inline implementation above.
                let v4 = v.reshaped([1, 1, N, dim])
                let stateBias = codec.encode(v4, useBias: true)
                let recoveredBias4 = codec.decode(stateBias).reshaped([N, dim])
                let diffCodecBias = ((recoveredBias4 - recoveredVBias) * (recoveredBias4 - recoveredVBias)).mean().item(Float.self)
                XCTAssertLessThan(diffCodecBias, 1e-6,
                    "MSECodec.encode(useBias:) should match inline bias path; dist=\(label) bits=\(bits) diff=\(diffCodecBias)")

                // And bias=false matches the original (non-bias) encode.
                let stateNoBias = codec.encode(v4, useBias: false)
                XCTAssertNil(stateNoBias.bias)
            }
        }
    }
}
