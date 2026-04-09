import MLX
import MLXNN
import MLXLMCommon
import XCTest

final class FusedNormRoPETests: XCTestCase {

    func testFusedMatchesSeparate() throws {
        // Gemma4 sliding attention dimensions
        let B = 1, L = 4, nHeads = 16, headDim = 256
        let eps: Float = 1e-6
        let base: Float = 10000.0

        // Random input and weight
        let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.float16)
        let weight = MLXRandom.normal([headDim]).asType(.float16)

        // Compute invFreqs (same as Gemma4Attention init)
        let exponents = MLXArray(
            stride(from: Float(0), to: Float(headDim), by: 2)
        ) / Float(headDim)
        let freqs = pow(MLXArray(base), exponents)
        let invFreqs = 1.0 / freqs

        let offset = 42  // arbitrary position offset

        // === Separate path: RMSNorm then RoPE ===
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let transposed = normed.transposed(0, 2, 1, 3)  // [B, nHeads, L, headDim]
        let ropeResult = MLXFast.RoPE(
            transposed, dimensions: headDim, traditional: false,
            base: base, scale: 1.0, offset: offset)
        let separateOutput = ropeResult.transposed(0, 2, 1, 3)  // back to [B, L, nHeads, headDim]
        eval(separateOutput)

        // === Fused path ===
        let kernel = FusedNormRoPEKernel(invFreqs: invFreqs)
        let fusedOutput = kernel(
            x, weight: weight, eps: eps, offset: offset, nHeads: nHeads)
        eval(fusedOutput)

        // Compare
        let diff = abs(separateOutput.asType(.float32) - fusedOutput.asType(.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)

        print("Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        print("Separate[0,0,0,0..4]: \(separateOutput[0, 0, 0, 0..<4])")
        print("Fused[0,0,0,0..4]:    \(fusedOutput[0, 0, 0, 0..<4])")
        print("Separate shape: \(separateOutput.shape), Fused shape: \(fusedOutput.shape)")

        // Allow some tolerance for float16 accumulation differences
        XCTAssertLessThan(maxDiff, 0.1, "Max element diff too large: \(maxDiff)")
        XCTAssertLessThan(meanDiff, 0.01, "Mean element diff too large: \(meanDiff)")
    }

    func testTrivialCopy() throws {
        // Simplest possible kernel: just copy input to output
        let headDim = 256
        let halfDim = headDim / 2

        let x = MLXArray(Array(stride(from: Float(0), to: Float(headDim), by: 1)))
        eval(x)

        let copyKernel = MLXFast.metalKernel(
            name: "trivial_copy",
            inputNames: ["x"],
            outputNames: ["out"],
            source: """
                uint tid = thread_position_in_threadgroup.x;
                uint row = threadgroup_position_in_grid.x;
                uint idx1 = tid;
                uint idx2 = tid + halfDim;
                out[row * headDim + idx1] = x[row * headDim + idx1];
                out[row * headDim + idx2] = x[row * headDim + idx2];
                """
        )

        let results = copyKernel(
            [x.reshaped(1, headDim)],
            template: [("headDim", headDim), ("halfDim", halfDim)],
            grid: (halfDim, 1, 1),  // totalRows * halfDim threads
            threadGroup: (halfDim, 1, 1),
            outputShapes: [[1, headDim]],
            outputDTypes: [.float32]
        )
        eval(results[0])

        let output = results[0].reshaped(-1)
        let diff = abs(x - output).max().item(Float.self)
        print("Copy kernel output[0..8]: \(output[0..<8])")
        print("Copy kernel diff: \(diff)")
        XCTAssertLessThan(diff, 0.001, "Trivial copy failed!")
    }

    func testNormOnly() throws {
        // Test just the RMSNorm part (no rotation) with realistic head dim
        let B = 1, L = 1, nHeads = 1, headDim = 256
        let eps: Float = 1e-6

        let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.float16)
        let weight = MLXArray.ones([headDim]).asType(.float16)
        // Zero invFreqs = no rotation (identity)
        let invFreqs = MLXArray.zeros([headDim / 2]).asType(.float32)

        let kernel = FusedNormRoPEKernel(invFreqs: invFreqs)
        let fusedOutput = kernel(x, weight: weight, eps: eps, offset: 0, nHeads: nHeads)
        eval(fusedOutput)

        // Expected: RMSNorm(x) with weight=ones and no rotation
        // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16+25+36+49+64)/8) = sqrt(204/8) = sqrt(25.5)
        // normed = x / rms
        let expected = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        eval(expected)

        print("Input: \(x.reshaped(-1))")
        print("Expected (rmsNorm): \(expected.reshaped(-1))")
        print("Fused (no rotation): \(fusedOutput.reshaped(-1))")

        let diff = abs(expected.asType(.float32) - fusedOutput.asType(.float32)).max().item(Float.self)
        print("Max diff: \(diff)")
        XCTAssertLessThan(diff, 0.01, "Norm-only test failed with diff \(diff)")
    }

    func testFusedDecodeStep() throws {
        // Decode: L=1
        let B = 1, L = 1, nHeads = 16, headDim = 256
        let eps: Float = 1e-6
        let base: Float = 10000.0

        let x = MLXRandom.normal([B, L, nHeads, headDim]).asType(.float16)
        let weight = MLXRandom.normal([headDim]).asType(.float16)

        let exponents = MLXArray(
            stride(from: Float(0), to: Float(headDim), by: 2)
        ) / Float(headDim)
        let freqs = pow(MLXArray(base), exponents)
        let invFreqs = 1.0 / freqs

        let offset = 1000

        // Separate
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let transposed = normed.transposed(0, 2, 1, 3)
        let ropeResult = MLXFast.RoPE(
            transposed, dimensions: headDim, traditional: false,
            base: base, scale: 1.0, offset: offset)
        let separateOutput = ropeResult.transposed(0, 2, 1, 3)
        eval(separateOutput)

        // Fused
        let kernel = FusedNormRoPEKernel(invFreqs: invFreqs)
        let fusedOutput = kernel(
            x, weight: weight, eps: eps, offset: offset, nHeads: nHeads)
        eval(fusedOutput)

        let diff = abs(separateOutput.asType(.float32) - fusedOutput.asType(.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)

        print("Decode max diff: \(maxDiff), mean diff: \(meanDiff)")

        XCTAssertLessThan(maxDiff, 0.1)
        XCTAssertLessThan(meanDiff, 0.01)
    }
}
