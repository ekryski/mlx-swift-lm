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

    // MARK: - Memory Leak Tests

    /// Verify the fused GDN C framework kernel doesn't leak memory when state
    /// is fed back in a loop (simulating decode steps).
    ///
    /// Root cause of the leak: GatedDeltaStep produces two outputs (y, state_out)
    /// via array::make_arrays sharing one primitive. In the generate loop, only y
    /// is in the token's dependency chain. If state_out is never included in an
    /// eval call, it's never detached — its primitive retains all inputs, including
    /// the previous state, creating an ever-growing chain.
    ///
    /// Fix: eval(state) after each step to ensure detach.
    func testFusedKernelMemoryStability() throws {
        let B = 1, Dk = 128, Dv = 128, Hk = 16, Hv = 16  // Qwen3.5 dense dims (available in metallib)
        let steps = 200

        // Create fixed inputs (only state changes each step)
        let qRaw = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let kRaw = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, 1, Hv, Dv]).asType(.bfloat16)
        let a = MLXRandom.normal([B, 1, Hv]).asType(.bfloat16)
        let b = MLXRandom.normal([B, 1, Hv]).asType(.bfloat16)
        let aLog = MLXRandom.normal([Hv]).asType(.bfloat16) - 2.0
        let dtBias = MLXRandom.normal([Hv]).asType(.bfloat16)
        var state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(qRaw, kRaw, v, a, b, aLog, dtBias, state)

        // Warmup
        for _ in 0..<10 {
            let outputs = MLXFast.gatedDeltaStepFused(
                qRaw: qRaw, kRaw: kRaw, v: v,
                a: a, bInput: b, aLog: aLog, dtBias: dtBias,
                state: state, T: 1, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            let y = outputs[0]
            state = outputs[1]
            eval(y, state)
        }

        MLX.Memory.clearCache()
        let baseline = MLX.Memory.activeMemory

        // Run loop: feed state back WITHOUT eval(state) — only eval(y)
        // This simulates the generate loop where only the token (which depends on y)
        // is eval'd, and state is a co-output that may not be detached.
        state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(state)
        MLX.Memory.clearCache()
        let baselineNoEval = MLX.Memory.activeMemory

        for step in 0..<steps {
            let outputs = MLXFast.gatedDeltaStepFused(
                qRaw: qRaw, kRaw: kRaw, v: v,
                a: a, bInput: b, aLog: aLog, dtBias: dtBias,
                state: state, T: 1, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            let y = outputs[0]
            state = outputs[1]
            // Only eval y (simulating asyncEval(token) which only reaches y)
            eval(y)

            if step % 50 == 49 {
                MLX.Memory.clearCache()
                let current = MLX.Memory.activeMemory
                let delta = Int64(current) - Int64(baselineNoEval)
                print("[FUSED-NO-EVAL] step=\(step+1) active=\(current/1024/1024)MB delta=\(delta/1024)KB")
            }
        }

        MLX.Memory.clearCache()
        let afterNoEval = MLX.Memory.activeMemory
        let growthNoEval = Int64(afterNoEval) - Int64(baselineNoEval)

        // Now run WITH eval(state) — this is the fix
        state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(state)
        MLX.Memory.clearCache()
        let baselineWithEval = MLX.Memory.activeMemory

        for step in 0..<steps {
            let outputs = MLXFast.gatedDeltaStepFused(
                qRaw: qRaw, kRaw: kRaw, v: v,
                a: a, bInput: b, aLog: aLog, dtBias: dtBias,
                state: state, T: 1, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            let y = outputs[0]
            state = outputs[1]
            // Eval BOTH y and state — ensures state is detached
            eval(y, state)

            if step % 50 == 49 {
                MLX.Memory.clearCache()
                let current = MLX.Memory.activeMemory
                let delta = Int64(current) - Int64(baselineWithEval)
                print("[FUSED-WITH-EVAL] step=\(step+1) active=\(current/1024/1024)MB delta=\(delta/1024)KB")
            }
        }

        MLX.Memory.clearCache()
        let afterWithEval = MLX.Memory.activeMemory
        let growthWithEval = Int64(afterWithEval) - Int64(baselineWithEval)

        print("[MEMORY] No eval(state): baseline=\(baselineNoEval/1024)KB growth=\(growthNoEval/1024)KB")
        print("[MEMORY] With eval(state): baseline=\(baselineWithEval/1024)KB growth=\(growthWithEval/1024)KB")

        // With eval(state), growth should be near zero (just the current state buffer)
        // Allow 2MB tolerance for allocator fragmentation
        let maxAllowedGrowthKB: Int64 = 2048
        XCTAssertLessThan(growthWithEval / 1024, maxAllowedGrowthKB,
            "With eval(state), memory should be stable. Growth: \(growthWithEval/1024)KB")
    }

    /// Control test: ops fallback should never leak regardless of eval pattern
    func testOpsKernelMemoryStability() throws {
        let B = 1, Dk = 128, Dv = 128, Hk = 16, Hv = 16
        let steps = 200

        let q = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let k = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, 1, Hv, Dv]).asType(.bfloat16)
        let g = sigmoid(MLXRandom.normal([B, 1, Hv])).asType(.bfloat16)
        let beta = sigmoid(MLXRandom.normal([B, 1, Hv])).asType(.bfloat16)
        var state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(q, k, v, g, beta, state)

        // Warmup
        for _ in 0..<10 {
            let (y, newState) = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
            state = newState
            eval(y, state)
        }

        MLX.Memory.clearCache()
        let baseline = MLX.Memory.activeMemory

        // Only eval y (NOT state) — ops should still be fine
        state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(state)
        MLX.Memory.clearCache()
        let baselineOps = MLX.Memory.activeMemory

        for step in 0..<steps {
            let (y, newState) = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
            state = newState
            eval(y)  // Only eval y, not state

            if step % 50 == 49 {
                MLX.Memory.clearCache()
                let current = MLX.Memory.activeMemory
                let delta = Int64(current) - Int64(baselineOps)
                print("[OPS-NO-EVAL] step=\(step+1) active=\(current/1024/1024)MB delta=\(delta/1024)KB")
            }
        }

        MLX.Memory.clearCache()
        let afterOps = MLX.Memory.activeMemory
        let growthOps = Int64(afterOps) - Int64(baselineOps)

        print("[MEMORY] Ops fallback: baseline=\(baselineOps/1024)KB growth=\(growthOps/1024)KB")

        // Ops should have minimal growth (each step produces independent arrays)
        let maxAllowedGrowthKB: Int64 = 4096
        XCTAssertLessThan(growthOps / 1024, maxAllowedGrowthKB,
            "Ops fallback should have stable memory. Growth: \(growthOps/1024)KB")
    }

    /// Test that the non-fused framework kernel also doesn't leak
    func testNonFusedKernelMemoryStability() throws {
        let B = 1, Dk = 128, Dv = 128, Hk = 16, Hv = 16
        let steps = 200

        let q = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let k = MLXRandom.normal([B, 1, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, 1, Hv, Dv]).asType(.bfloat16)
        let g = sigmoid(MLXRandom.normal([B, 1, Hv])).asType(.bfloat16)
        let beta = sigmoid(MLXRandom.normal([B, 1, Hv])).asType(.bfloat16)
        var state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(q, k, v, g, beta, state)

        // Warmup
        for _ in 0..<10 {
            let outputs = MLXFast.gatedDeltaStep(
                q: q, k: k, v: v, g: g, beta: beta, state: state,
                T: 1, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            state = outputs[1]
            eval(outputs[0], state)
        }

        state = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.bfloat16) * 0.01
        eval(state)
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.activeMemory

        for step in 0..<steps {
            let outputs = MLXFast.gatedDeltaStep(
                q: q, k: k, v: v, g: g, beta: beta, state: state,
                T: 1, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            let y = outputs[0]
            state = outputs[1]
            eval(y)  // Only eval y

            if step % 50 == 49 {
                MLX.Memory.clearCache()
                let current = MLX.Memory.activeMemory
                let delta = Int64(current) - Int64(baseline)
                print("[NONFUSED-NO-EVAL] step=\(step+1) active=\(current/1024/1024)MB delta=\(delta/1024)KB")
            }
        }

        MLX.Memory.clearCache()
        let after = MLX.Memory.activeMemory
        let growth = Int64(after) - Int64(baseline)

        print("[MEMORY] Non-fused kernel: baseline=\(baseline/1024)KB growth=\(growth/1024)KB")
    }
}
