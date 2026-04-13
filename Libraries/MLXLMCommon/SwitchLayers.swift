import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Sort threshold for MoE expert reordering. Default 128.
    /// Sorting reorders tokens by expert index for better gatherQuantizedMM cache locality.
    /// A/B testing shows: sort is critical at T >= 1024 (38-48% regression without it),
    /// but counterproductive at small sizes (25% overhead at T=128 with threshold=64).
    /// Threshold=128 avoids unnecessary sort at small batch while preserving prefill gains.
    /// Set MOE_SORT_THRESHOLD env var to override (0 = disable sorting entirely).
    static let sortThreshold: Int = {
        if let env = ProcessInfo.processInfo.environment["MOE_SORT_THRESHOLD"],
           let val = Int(env) { return val }
        return 128
    }()

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = Self.sortThreshold > 0 && indices.size >= Self.sortThreshold

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - Fused Gate Activation Metal Kernel

/// Activation types supported by the fused kernel.
public enum FusedGateActivationType: String {
    case silu
    case geluApproximate = "gelu_approx"
    /// GPT-OSS swiglu: clip(gate, max=7) * sigmoid(1.702 * clipped_gate) * (clip(up, -7, 7) + 1)
    case swiglu
}

/// Fused Metal kernel: split + activation + multiply in a single dispatch.
/// Input: gateUp [..., 2*H]  →  Output: activation(gate) * up  [..., H]
///
/// Eliminates 3 separate dispatches (split, activation, multiply) with one kernel.
private class FusedGateActivationKernel {

    static let shared = FusedGateActivationKernel()

    private var kernels: [String: MLXFast.MLXFastKernel] = [:]

    func kernel(for activationType: FusedGateActivationType, dtype: DType) -> MLXFast.MLXFastKernel
    {
        let key = "\(activationType.rawValue)_\(dtype)"
        if let cached = kernels[key] { return cached }

        let source: String
        switch activationType {
        case .silu:
            source = """
                uint idx = thread_position_in_grid.x;
                if (idx >= totalElements) return;
                uint row = idx / H;
                uint col = idx % H;
                float gate_f = static_cast<float>(gateUp[row * H * 2 + col]);
                float up_f = static_cast<float>(gateUp[row * H * 2 + col + H]);
                float activated = gate_f / (1.0f + exp(-gate_f));
                output[idx] = static_cast<T>(activated * up_f);
                """
        case .geluApproximate:
            source = """
                uint idx = thread_position_in_grid.x;
                if (idx >= totalElements) return;
                uint row = idx / H;
                uint col = idx % H;
                float gate_f = static_cast<float>(gateUp[row * H * 2 + col]);
                float up_f = static_cast<float>(gateUp[row * H * 2 + col + H]);
                float activated = 0.5f * gate_f * (1.0f + precise::tanh(0.7978845608f * (gate_f + 0.044715f * gate_f * gate_f * gate_f)));
                output[idx] = static_cast<T>(activated * up_f);
                """
        case .swiglu:
            // GPT-OSS swiglu: gate is xGlu (first half), up is xLinear (second half)
            // xGlu = clip(gate, max=7.0)
            // xLinear = clip(up, -7.0, 7.0)
            // result = xGlu * sigmoid(1.702 * xGlu) * (xLinear + 1.0)
            source = """
                uint idx = thread_position_in_grid.x;
                if (idx >= totalElements) return;
                uint row = idx / H;
                uint col = idx % H;
                float gate_f = static_cast<float>(gateUp[row * H * 2 + col]);
                float up_f = static_cast<float>(gateUp[row * H * 2 + col + H]);
                float x_glu = min(gate_f, 7.0f);
                float x_linear = clamp(up_f, -7.0f, 7.0f);
                float sig = 1.0f / (1.0f + exp(-1.702f * x_glu));
                output[idx] = static_cast<T>(x_glu * sig * (x_linear + 1.0f));
                """
        }

        let k = MLXFast.metalKernel(
            name: "fused_gate_act_\(key)",
            inputNames: ["gateUp"],
            outputNames: ["output"],
            source: source
        )
        kernels[key] = k
        return k
    }

    func callAsFunction(
        _ gateUp: MLXArray, hiddenDims: Int, activationType: FusedGateActivationType
    ) -> MLXArray {
        let k = kernel(for: activationType, dtype: gateUp.dtype)

        // gateUp shape: [..., 2*H]. Flatten leading dims.
        let shape = gateUp.shape
        let H = hiddenDims
        let leadingElements = shape.dropLast().reduce(1, *)
        let totalElements = leadingElements * H

        // Output shape: [..., H]
        var outShape = Array(shape.dropLast())
        outShape.append(H)

        let threadGroupSize = min(256, totalElements)
        let gridSize = (totalElements + threadGroupSize - 1) / threadGroupSize * threadGroupSize

        let results = k(
            [gateUp],
            template: [("T", gateUp.dtype), ("H", H), ("totalElements", totalElements)],
            grid: (gridSize, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [outShape],
            outputDTypes: [gateUp.dtype]
        )
        return results[0]
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant that stores gate_proj and up_proj as a single fused SwitchLinear.
///
/// Instead of two separate `gatherQuantizedMM` dispatches (one for gate, one for up),
/// this does a single dispatch with outputDims = 2 × hiddenDims, then splits the result.
/// This eliminates 1 Metal encoder dispatch per MoE block (40 blocks × 1 = 40 fewer
/// dispatches per token) and avoids intermediate buffer allocation.
///
/// The fused weight `gate_up_proj` comes directly from model files (e.g., Qwen3.5 stores
/// `experts.gate_up_proj` as a single tensor). Standard `SwitchGLU` splits it during
/// sanitization; this class keeps it fused.
///
/// Supports two activation modes:
/// - Single-argument (default): `activation(gate) * up` — for silu-gated models (Qwen3.5)
/// - Two-argument: `twoArgActivation(up, gate)` — for models with asymmetric activation
///   (GPT-OSS's swiglu with clipping/scaling)
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)?

    /// Warp decode activation type: 0=silu, 1=gelu_approx, 2=swiglu
    public var warpActivationType: Int = 0

    /// When set, uses the fused Metal kernel instead of split+activation+multiply.
    /// Disabled by default — benchmarks show MLX's native lazy eval already handles
    /// these 3 element-wise ops efficiently. The kernel saves <1% at decode sizes.
    /// Set MOE_FUSED_ACTIVATION=1 to enable for testing.
    private static let useFusedKernel: Bool = {
        ProcessInfo.processInfo.environment["MOE_FUSED_ACTIVATION"] == "1"
    }()

    private let fusedActivationType: FusedGateActivationType?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)? = nil,
        fusedActivationType: FusedGateActivationType? = nil,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.twoArgActivation = twoArgActivation
        self.fusedActivationType = Self.useFusedKernel ? fusedActivationType : nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Warp Decode: benchmarked at -20-25% regression vs gatherQuantizedMM.
    /// Disabled until framework primitives are upstreamed.
    private static let useWarpDecode: Bool = {
        false && ProcessInfo.processInfo.environment["WARP_MOE_DECODE"] == "1"
    }()

    /// Warp Decode: benchmarked at -20-25% regression vs gatherQuantizedMM.
    /// Disabled — framework primitives not yet upstreamed.
    /// See benchmarks/notes/warp-decode-moe-analysis-2026-04-12.md
    public func warpDecode(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray
    ) -> MLXArray? {
        return nil
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = SwitchGLU.sortThreshold > 0 && indices.size >= SwitchGLU.sortThreshold

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // Single gatherQuantizedMM for both gate and up projections
        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)

        let activated: MLXArray
        // Use fused kernel for decode (small N), fall back to native ops for prefill.
        // During decode N ≈ topK (2-8), so totalElements ≈ topK * H (small).
        // During prefill N ≈ topK * T, so totalElements can be millions (native ops better).
        let totalElements = gateUp.shape.dropLast().reduce(1, *) * hiddenDims
        if let fusedType = fusedActivationType, totalElements <= 32768 {
            // Fused Metal kernel: split + activation + multiply in one dispatch
            activated = FusedGateActivationKernel.shared(
                gateUp, hiddenDims: hiddenDims, activationType: fusedType)
        } else if let twoArgActivation {
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            activated = twoArgActivation(parts[1], parts[0])
        } else {
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            activated = activation(parts[0]) * parts[1]
        }

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
