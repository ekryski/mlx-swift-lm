import MLX
import MLXNN
import MLXLMCommon
import XCTest

final class FusedQGEMVTests: XCTestCase {

    func testFusedMatchesSeparate() throws {
        // Gemma4 Q projection dimensions: hidden=2816, output=4096 (16 heads × 256)
        let K = 2816
        let N = 4096
        let groupSize = 64
        let eps: Float = 1e-6

        // Random input and norm weight
        let x = MLXRandom.normal([1, K]).asType(.float16)
        let normWeight = MLXRandom.normal([K]).asType(.float16)

        // Create quantized weight matrix (simulate 4-bit quantization)
        let wFull = MLXRandom.normal([N, K]).asType(.float16)
        let (wQ, scales, biases) = MLX.quantized(wFull, groupSize: groupSize, bits: 4)
        eval(x, normWeight, wQ, scales, biases)

        // === Separate path: RMSNorm then quantized matmul ===
        let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let separateOutput = MLX.quantizedMatmul(
            normed, wQ, scales: scales, biases: biases!,
            transpose: true, groupSize: groupSize, bits: 4)
        eval(separateOutput)

        // === Fused path ===
        let fusedOutput = MLXFast.rmsNormQuantizedGEMV(
            x, normWeight: normWeight,
            w: wQ, scales: scales, biases: biases!,
            eps: eps, groupSize: groupSize)
        eval(fusedOutput)

        // Compare
        let diff = abs(separateOutput.asType(DType.float32) - fusedOutput.asType(DType.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)

        print("Separate shape: \(separateOutput.shape), Fused shape: \(fusedOutput.shape)")
        print("Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        print("Separate[0,0..4]: \(separateOutput[0, 0..<4])")
        print("Fused[0,0..4]:    \(fusedOutput[0, 0..<4])")

        // Allow tolerance for quantized matmul differences
        XCTAssertLessThan(maxDiff, 1.0, "Max diff too large: \(maxDiff)")
        XCTAssertLessThan(meanDiff, 0.1, "Mean diff too large: \(meanDiff)")
    }

    func testIdentityNorm() throws {
        // Test with weight=ones and no quantization effect — pure GEMV
        let K = 256
        let N = 32
        let groupSize = 64
        let eps: Float = 1e-6

        // Simple deterministic input
        let x = MLXArray(Array(stride(from: Float(0.01), through: Float(0.01) * Float(K), by: 0.01)).prefix(K).map { Float16($0) })
            .reshaped(1, K)
        let normWeight = MLXArray.ones([K]).asType(.float16)

        let wFull = MLXArray.ones([N, K]).asType(.float16) * 0.01
        let (wQ, scales, biases) = MLX.quantized(wFull, groupSize: groupSize, bits: 4)
        eval(x, normWeight, wQ, scales, biases)

        let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let separateOutput = MLX.quantizedMatmul(
            normed, wQ, scales: scales, biases: biases!,
            transpose: true, groupSize: groupSize, bits: 4)
        eval(separateOutput)

        let fusedOutput = MLXFast.rmsNormQuantizedGEMV(
            x, normWeight: normWeight,
            w: wQ, scales: scales, biases: biases!,
            eps: eps, groupSize: groupSize)
        eval(fusedOutput)

        print("Identity norm test:")
        print("  x[0..4]: \(x[0, 0..<4])")
        print("  normed[0..4]: \(normed[0, 0..<4])")
        print("  Separate[0,0..4]: \(separateOutput[0, 0..<4])")
        print("  Fused[0,0..4]:    \(fusedOutput[0, 0..<4])")

        let diff = abs(separateOutput.asType(DType.float32) - fusedOutput.asType(DType.float32))
        let maxDiff = diff.max().item(Float.self)
        print("  Max diff: \(maxDiff)")

        XCTAssertLessThan(maxDiff, 1.0, "Identity norm failed: \(maxDiff)")
    }

    func testSmallDimensions() throws {
        // K=256 matches block_size, should work. K < 256 triggers remainder-only path.
        let K = 256
        let N = 64
        let groupSize = 64
        let eps: Float = 1e-6

        let x = MLXRandom.normal([1, K]).asType(.float16)
        let normWeight = MLXArray.ones([K]).asType(.float16)

        let wFull = MLXRandom.normal([N, K]).asType(.float16)
        let (wQ, scales, biases) = MLX.quantized(wFull, groupSize: groupSize, bits: 4)
        eval(x, normWeight, wQ, scales, biases)

        let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let separateOutput = MLX.quantizedMatmul(
            normed, wQ, scales: scales, biases: biases!,
            transpose: true, groupSize: groupSize, bits: 4)
        eval(separateOutput)

        let fusedOutput = MLXFast.rmsNormQuantizedGEMV(
            x, normWeight: normWeight,
            w: wQ, scales: scales, biases: biases!,
            eps: eps, groupSize: groupSize)
        eval(fusedOutput)

        let diff = abs(separateOutput.asType(DType.float32) - fusedOutput.asType(DType.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)

        print("Small test: max diff=\(maxDiff), mean diff=\(meanDiff)")

        XCTAssertLessThan(maxDiff, 1.0)
        XCTAssertLessThan(meanDiff, 0.1)
    }
}
