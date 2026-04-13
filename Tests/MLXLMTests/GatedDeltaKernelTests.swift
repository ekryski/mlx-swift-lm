import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// Tests GDN framework kernel correctness against the ops fallback.
/// The ops fallback (gatedDeltaOps) uses pure MLX ops (matmul, etc.) and is
/// known-correct. The framework kernel uses a pre-compiled Metal dispatch.
final class GatedDeltaKernelTests: XCTestCase {

    /// Test GDN kernel for Qwen3.5-A3B config: Dk=192, Dv=128, Hk=4, Hv=4
    func testGDNKernelSmallSymmetric() throws {
        try runGDNComparison(B: 1, T: 1, Dk: 192, Dv: 128, Hk: 4, Hv: 4, label: "Qwen3.5-A3B (symmetric)")
    }

    /// Test GDN kernel for Qwen3.5-2B config: Dk=128, Dv=128, Hk=8, Hv=8
    func testGDNKernelMediumSymmetric() throws {
        try runGDNComparison(B: 1, T: 1, Dk: 128, Dv: 128, Hk: 8, Hv: 8, label: "Qwen3.5-2B (symmetric)")
    }

    /// Test GDN kernel for Qwen3.5-35B config: Dk=128, Dv=128, Hk=16, Hv=32
    /// THIS IS THE FAILING CASE — GQA with Hk ≠ Hv
    func testGDNKernelGQA() throws {
        try runGDNComparison(B: 1, T: 1, Dk: 128, Dv: 128, Hk: 16, Hv: 32, label: "Qwen3.5-35B (GQA Hk≠Hv)")
    }

    /// Test GDN kernel with T=4 (multi-step, used during chunked prefill)
    func testGDNKernelMultiStep() throws {
        try runGDNComparison(B: 1, T: 4, Dk: 128, Dv: 128, Hk: 16, Hv: 32, label: "Qwen3.5-35B T=4")
    }

    /// Test fused GDN kernel for Qwen3.5-35B config
    func testFusedGDNKernelGQA() throws {
        try runFusedGDNComparison(B: 1, T: 1, Dk: 128, Dv: 128, Hk: 16, Hv: 32, label: "Fused Qwen3.5-35B (GQA)")
    }

    // MARK: - Helpers

    private func runGDNComparison(
        B: Int, T: Int, Dk: Int, Dv: Int, Hk: Int, Hv: Int, label: String
    ) throws {
        // Create random inputs matching the kernel's expected shapes
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.bfloat16)
        let g = sigmoid(MLXRandom.normal([B, T, Hv])).asType(.bfloat16)  // decay in (0, 1)
        let beta = sigmoid(MLXRandom.normal([B, T, Hv])).asType(.bfloat16)  // beta in (0, 1)
        let state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01

        // Ops fallback (ground truth)
        let (yOps, stateOps) = gatedDeltaOps(
            q: q, k: k, v: v, g: g, beta: beta, state: state)
        eval(yOps, stateOps)

        // Framework kernel
        let (yKernel, stateKernel) = gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: state)
        eval(yKernel, stateKernel)

        // Compare outputs
        let yDiff = abs(yOps.asType(.float32) - yKernel.asType(.float32)).max().item(Float.self)
        let stateDiff = abs(stateOps.asType(.float32) - stateKernel.asType(.float32)).max().item(Float.self)
        let yMean = abs(yOps.asType(.float32)).mean().item(Float.self)
        let stateMean = abs(stateOps.asType(.float32)).mean().item(Float.self)
        let yRelDiff = yMean > 0 ? yDiff / yMean : yDiff
        let stateRelDiff = stateMean > 0 ? stateDiff / stateMean : stateDiff

        print("[\(label)] y max diff: \(yDiff) (rel: \(yRelDiff)), state max diff: \(stateDiff) (rel: \(stateRelDiff))")
        print("  y ops shape: \(yOps.shape), kernel shape: \(yKernel.shape)")
        print("  state ops shape: \(stateOps.shape), kernel shape: \(stateKernel.shape)")
        print("  y ops mean: \(yMean), state ops mean: \(stateMean)")

        // Print first few values for visual comparison
        let yOpsFlat = yOps.asType(.float32).reshaped(-1)
        let yKernelFlat = yKernel.asType(.float32).reshaped(-1)
        var yVals = "  y ops[0:8]:    "
        var yKVals = "  y kernel[0:8]: "
        for i in 0..<min(8, yOpsFlat.dim(0)) {
            yVals += String(format: "%.4f ", yOpsFlat[i].item(Float.self))
            yKVals += String(format: "%.4f ", yKernelFlat[i].item(Float.self))
        }
        print(yVals)
        print(yKVals)

        // bf16 tolerance: ~0.01 for accumulated errors across state update
        XCTAssertLessThan(yDiff, 0.1, "\(label): y output mismatch (max diff \(yDiff))")
        XCTAssertLessThan(stateDiff, 0.1, "\(label): state output mismatch (max diff \(stateDiff))")
    }

    private func runFusedGDNComparison(
        B: Int, T: Int, Dk: Int, Dv: Int, Hk: Int, Hv: Int, label: String
    ) throws {
        // Fused kernel takes raw q, k (before norm) + a, b, aLog, dtBias
        let qRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let kRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.bfloat16)
        let a = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
        let b = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
        let aLog = MLXRandom.normal([Hv]).asType(.bfloat16) - 2.0  // typically negative
        let dtBias = MLXRandom.normal([Hv]).asType(.bfloat16)
        let state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01

        // Compute g and beta manually (what the fused kernel does internally)
        let gManual = computeGatedDeltaG(aLog, a, dtBias)
        let betaManual = sigmoid(b)

        // Pre-normalize q and k (what the fused kernel does internally)
        let invScale = pow(Float(Dk), -0.5)
        let qNormed = MLXArray(pow(invScale, 2)).asType(qRaw.dtype)
            * MLXFast.rmsNorm(qRaw, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed = MLXArray(invScale).asType(kRaw.dtype)
            * MLXFast.rmsNorm(kRaw, weight: MLXArray.mlxNone, eps: 1e-6)

        // Ops fallback with pre-computed g, beta, normed q/k (ground truth)
        let (yOps, stateOps) = gatedDeltaOps(
            q: qNormed, k: kNormed, v: v, g: gManual, beta: betaManual, state: state)
        eval(yOps, stateOps)

        // Fused framework kernel
        let (yFused, stateFused) = fusedGatedDeltaUpdate(
            qRaw: qRaw, kRaw: kRaw, v: v,
            a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state)
        eval(yFused, stateFused)

        let yDiff = abs(yOps.asType(.float32) - yFused.asType(.float32)).max().item(Float.self)
        let stateDiff = abs(stateOps.asType(.float32) - stateFused.asType(.float32)).max().item(Float.self)

        print("[\(label)] y max diff: \(yDiff), state max diff: \(stateDiff)")

        XCTAssertLessThan(yDiff, 0.5, "\(label): y output mismatch (max diff \(yDiff))")
        XCTAssertLessThan(stateDiff, 0.5, "\(label): state output mismatch (max diff \(stateDiff))")
    }
}
