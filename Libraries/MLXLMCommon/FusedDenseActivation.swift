import Foundation
import MLX

// MARK: - FusedDenseGateActivation
//
// Thin wrapper around the precompiled `MLXFast.fusedGateActivation`
// Metal kernel (added upstream in mlx / mlx-c / mlx-swift on
// `feat/fused-gate-activation-gpt-oss`). Takes a fused `gateUp` tensor of
// shape `[..., 2*hiddenDims]` and produces `activation(gate) * up` of
// shape `[..., hiddenDims]` in a single Metal dispatch.
//
// The only production caller today is `FusedGateUpSwitchGLU` for GPT-OSS's
// clipped-swiglu activation, where the C kernel replaces a compiled-closure
// two-arg fallback and yields +37–45% decode across contexts. Dense-MLP
// callers are deliberately not wired — the benchmark data showed no
// consistent win on Generation tok/s for silu / gelu_approx variants.
// See `benchmarks/notes/fused-kernel-findings-2026-04-24.md`.

/// Activation variants supported by the inline fused-gate kernel.
public enum DenseActivationKind: Hashable, Sendable {
    case silu
    case geluApprox
    /// GPT-OSS clipped swiglu: `clamp(-7,7)`, `sigmoid(1.702·g)`, `·(u+1)`.
    case clippedSwiglu

    fileprivate var fastVariant: MLXFast.DenseGateActivation {
        switch self {
        case .silu: return .silu
        case .geluApprox: return .geluApprox
        case .clippedSwiglu: return .clippedSwiglu
        }
    }
}

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

/// Run the inline fused `activation(gate) * up` kernel.
public func fusedDenseGateActivation(
    _ gateUp: MLXArray,
    hiddenDims: Int,
    kind: DenseActivationKind
) -> MLXArray {
    MLXFast.fusedGateActivation(
        gateUp, hiddenDims: hiddenDims, activation: kind.fastVariant)
}
