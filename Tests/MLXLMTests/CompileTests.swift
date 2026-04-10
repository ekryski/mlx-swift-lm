import MLX
import MLXNN
import XCTest

final class CompileTests: XCTestCase {

    func testCompiledQuantizedLinear() throws {
        // Test that compile() works through QuantizedLinear (quantizedMatmul)
        let inputDim = 256
        let outputDim = 128

        // Create and quantize a linear layer
        let linear = Linear(inputDim, outputDim, bias: false)
        let qLinear = QuantizedLinear.from(linear: linear, groupSize: 64, bits: 4)
        eval(qLinear)

        let x = MLXRandom.normal([1, inputDim]).asType(.float16)
        eval(x)

        // Uncompiled path
        let uncompiled = qLinear(x)
        eval(uncompiled)

        // Compiled path — compile the quantized linear forward pass
        let compiledForward = compile(inputs: [qLinear], outputs: [], shapeless: true) {
            (arrays: [MLXArray]) -> [MLXArray] in
            [qLinear(arrays[0])]
        }

        let compiled = compiledForward([x])
        eval(compiled[0])

        let diff = abs(uncompiled.asType(DType.float32) - compiled[0].asType(DType.float32))
        let maxDiff = diff.max().item(Float.self)

        print("QuantizedLinear compile test: max diff = \(maxDiff)")
        XCTAssertLessThan(maxDiff, 0.001, "Compiled QuantizedLinear differs: \(maxDiff)")
    }

    func testCompiledMLP() throws {
        // Test compiling a full MLP (3 Linear layers + activation)
        let dim = 256
        let hidden = 128

        let gate = Linear(dim, hidden, bias: false)
        let up = Linear(dim, hidden, bias: false)
        let down = Linear(hidden, dim, bias: false)

        // Quantize all three
        let qGate = QuantizedLinear.from(linear: gate, groupSize: 64, bits: 4)
        let qUp = QuantizedLinear.from(linear: up, groupSize: 64, bits: 4)
        let qDown = QuantizedLinear.from(linear: down, groupSize: 64, bits: 4)
        eval(qGate, qUp, qDown)

        let x = MLXRandom.normal([1, dim]).asType(.float16)
        eval(x)

        // Uncompiled MLP
        let uncompiled = qDown(silu(qGate(x)) * qUp(x))
        eval(uncompiled)

        // Compiled MLP — wrap entire forward pass
        let compiledMLP = compile(
            inputs: [qGate, qUp, qDown], outputs: [], shapeless: true
        ) { (arrays: [MLXArray]) -> [MLXArray] in
            let inp = arrays[0]
            return [qDown(silu(qGate(inp)) * qUp(inp))]
        }

        let compiled = compiledMLP([x])
        eval(compiled[0])

        let diff = abs(uncompiled.asType(DType.float32) - compiled[0].asType(DType.float32))
        let maxDiff = diff.max().item(Float.self)

        print("Compiled MLP test: max diff = \(maxDiff)")
        print("  Uncompiled[0..4]: \(uncompiled[0, 0..<4])")
        print("  Compiled[0..4]:   \(compiled[0][0, 0..<4])")
        XCTAssertLessThan(maxDiff, 0.001, "Compiled MLP differs: \(maxDiff)")
    }

    func testCompiledMLPReuse() throws {
        // Test that the compiled function is reused across calls (same shapes)
        let dim = 256
        let hidden = 128

        let gate = Linear(dim, hidden, bias: false)
        let up = Linear(dim, hidden, bias: false)
        let down = Linear(hidden, dim, bias: false)
        let qGate = QuantizedLinear.from(linear: gate, groupSize: 64, bits: 4)
        let qUp = QuantizedLinear.from(linear: up, groupSize: 64, bits: 4)
        let qDown = QuantizedLinear.from(linear: down, groupSize: 64, bits: 4)
        eval(qGate, qUp, qDown)

        let compiledMLP = compile(
            inputs: [qGate, qUp, qDown], outputs: [], shapeless: true
        ) { (arrays: [MLXArray]) -> [MLXArray] in
            return [qDown(silu(qGate(arrays[0])) * qUp(arrays[0]))]
        }

        // Run 10 times with different inputs — should reuse compiled tape
        for i in 0..<10 {
            let x = MLXRandom.normal([1, dim]).asType(.float16)
            eval(x)
            let result = compiledMLP([x])
            eval(result[0])
        }

        // If we got here without crash, compile() reuse works
        print("Compiled MLP reuse: 10 calls succeeded")
    }
}
