import Foundation
import MLX
import MLXNN

// MARK: - FusedGateUpMLP

/// Dense MLP that fuses `gate_proj` + `up_proj` into a single `gate_up_proj`
/// Linear with output dim `2 * hiddenDimensions`. Saves one Metal dispatch per
/// layer vs. separate gate/up matmuls — dispatch overhead dominates at T=1
/// decode, so the single kernel is a measurable win on dense models.
///
/// Weight layout: `gate_up_proj.weight` shape `[2 * hiddenDims, inputDims]`,
/// rows `[0 ..< hiddenDims)` → gate, rows `[hiddenDims ..< 2 * hiddenDims)` → up.
/// Concatenation happens in `sanitize()` via `fuseGateUpWeights` on axis 0.
///
/// Two activation modes mirror `FusedGateUpSwitchGLU`:
/// - Single-argument (default): `activation(gate) * up` — silu / gelu.
/// - Two-argument: `twoArgActivation(up, gate)` — asymmetric activations.
public final class FusedGateUpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") public var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public let hiddenDims: Int
    public let activation: (MLXArray) -> MLXArray
    public let twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)?

    /// If set, enables the inline fused-activation Metal kernel at decode
    /// (T=1) when `MLX_INLINE_ACTIVATION=1`. The kernel replaces the
    /// split + activation + multiply chain with a single dispatch.
    /// Spec 002.
    public let activationKind: DenseActivationKind?

    public init(
        dimensions: Int,
        hiddenDimensions: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)? = nil,
        activationKind: DenseActivationKind? = nil,
        bias: Bool = false
    ) {
        self.hiddenDims = hiddenDimensions
        self.activation = activation
        self.twoArgActivation = twoArgActivation
        self.activationKind = activationKind
        self._gateUpProj.wrappedValue = Linear(
            dimensions, 2 * hiddenDimensions, bias: bias)
        self._downProj.wrappedValue = Linear(
            hiddenDimensions, dimensions, bias: bias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        let activated: MLXArray
        if twoArgActivation == nil,
            let kind = activationKind,
            isInlineDenseActivationEnabled(),
            canUseInlineDenseActivation(gateUp: gateUp, hiddenDims: hiddenDims)
        {
            activated = fusedDenseGateActivation(
                gateUp, hiddenDims: hiddenDims, kind: kind)
        } else {
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            if let twoArgActivation {
                activated = twoArgActivation(parts[1], parts[0])
            } else {
                activated = activation(parts[0]) * parts[1]
            }
        }
        return downProj(activated)
    }
}

// MARK: - Weight fusion helper

/// Concatenate `gate_proj.*` and `up_proj.*` tensors into a single
/// `gate_up_proj.*` entry along the output axis. Call from a model's
/// `sanitize()` once the other weight-map transformations have run.
///
/// Covers `.weight`, `.scales`, and `.biases` uniformly — the substring
/// `"gate_proj"` matches every suffix produced by the quantization pipeline.
///
/// - Parameters:
///   - weights: mutable weight map.
///   - keyFilter: substring that narrows which `gate_proj` entries to fuse.
///     Use a tight filter like `".mlp.gate_proj."` to avoid catching sibling
///     paths such as `.mlp.switch_mlp.gate_proj.` on MoE models.
///   - outputAxis: axis to concatenate on (0 for `Linear`, 1 for
///     `SwitchLinear`'s `[E, outDim, inDim]` layout).
public func fuseGateUpWeights(
    _ weights: inout [String: MLXArray],
    keyFilter: String,
    outputAxis: Int
) {
    let gateKeys = weights.keys.filter {
        $0.contains(keyFilter) && $0.contains("gate_proj")
    }
    for gateKey in gateKeys {
        let upKey = gateKey.replacingOccurrences(
            of: "gate_proj", with: "up_proj")
        guard let gateVal = weights[gateKey],
              let upVal = weights[upKey]
        else { continue }
        let fusedKey = gateKey.replacingOccurrences(
            of: "gate_proj", with: "gate_up_proj")
        weights[fusedKey] = concatenated([gateVal, upVal], axis: outputAxis)
        weights.removeValue(forKey: gateKey)
        weights.removeValue(forKey: upKey)
    }
}
