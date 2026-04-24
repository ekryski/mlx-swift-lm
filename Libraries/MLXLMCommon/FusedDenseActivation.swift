import Foundation
import MLX

// MARK: - FusedDenseGateActivation
//
// Thin wrapper around the precompiled `MLXFast.fusedGateActivation` kernel
// (added to mlx / mlx-c / mlx-swift under branch `feat/fused-dense-gate-activation`).
// Takes a fused `gateUp` tensor of shape `[..., 2*hiddenDims]` and produces
// `activation(gate) * up` of shape `[..., hiddenDims]` in a single Metal
// dispatch. Replaces the Split + activation + Multiply chain (4 dispatches)
// in the dense MLP forward pass — at decode (T=1), dispatch overhead dominates.
//
// Spec 002.

/// Activation variants supported by the inline fused-gate kernel.
public enum DenseActivationKind: Hashable, Sendable {
    case silu
    case geluApprox

    fileprivate var fastVariant: MLXFast.DenseGateActivation {
        switch self {
        case .silu: return .silu
        case .geluApprox: return .geluApprox
        }
    }
}

// MARK: - Feature gate

/// Reads `MLX_INLINE_ACTIVATION` at call time. Defaults to `false` (the
/// kernel path is opt-in until correctness and perf are validated per
/// model).
@inline(__always)
public func isInlineDenseActivationEnabled() -> Bool {
    switch ProcessInfo.processInfo.environment["MLX_INLINE_ACTIVATION"] {
    case "1", "true", "TRUE", "on", "ON": return true
    default: return false
    }
}

// MARK: - Fast-path preconditions

/// True when the inline kernel's fast-path preconditions are met.
@inline(__always)
public func canUseInlineDenseActivation(
    gateUp: MLXArray,
    hiddenDims: Int
) -> Bool {
    // Decode-only — prefill's GEMM path is already dispatch-efficient.
    guard gateUp.dim(-2) == 1 else { return false }
    // Fast path is fp16 / bf16; the precompiled kernel also instantiates
    // fp32 but real weights ship in fp16/bf16 at decode, so restrict here
    // to keep behavior consistent with the spec.
    switch gateUp.dtype {
    case .bfloat16, .float16: break
    default: return false
    }
    // Last-axis layout sanity — fusion requires the 2*hidden layout.
    guard gateUp.dim(-1) == 2 * hiddenDims else { return false }
    return true
}

// MARK: - Public helper

/// Run the inline fused `activation(gate) * up` kernel.
///
/// Caller must check `canUseInlineDenseActivation(...)` first. Returns a
/// tensor with the same leading dims as `gateUp` and last-axis size
/// `hiddenDims`.
public func fusedDenseGateActivation(
    _ gateUp: MLXArray,
    hiddenDims: Int,
    kind: DenseActivationKind
) -> MLXArray {
    MLXFast.fusedGateActivation(
        gateUp, hiddenDims: hiddenDims, activation: kind.fastVariant)
}
