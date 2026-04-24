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
    /// GPT-OSS clipped swiglu. Fused two-arg op — replaces the
    /// `twoArgActivation` closure on `FusedGateUpSwitchGLU`.
    case clippedSwiglu

    fileprivate var fastVariant: MLXFast.DenseGateActivation {
        switch self {
        case .silu: return .silu
        case .geluApprox: return .geluApprox
        case .clippedSwiglu: return .clippedSwiglu
        }
    }
}

// MARK: - Feature gate

/// Reads `MLX_INLINE_ACTIVATION` at call time. Defaults to `false` — the
/// current kernel does not beat MLX's native split+activation+multiply
/// path on the measured hidden sizes, so the path is opt-in pending
/// further kernel tuning.
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
    switch gateUp.dtype {
    case .bfloat16, .float16: break
    default: return false
    }
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
