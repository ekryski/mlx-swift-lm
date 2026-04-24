import Foundation
import MLX
import MLXNN

// MARK: - FusedNormProjection
//
// Apply RMSNorm and a quantized Linear projection as a single fused Metal
// kernel via `MLXFast.rmsNormQuantizedGEMV`. Decode-only fast path — falls
// back to separate `rmsNorm` + `proj` dispatches when preconditions don't
// hold.
//
// Spec 004.

/// Reads `MLX_FUSED_NORM_MLP` at call time. Defaults to false until
/// benchmarks confirm the fused path is a net win per model.
@inline(__always)
public func isFusedNormMLPEnabled() -> Bool {
    switch ProcessInfo.processInfo.environment["MLX_FUSED_NORM_MLP"] {
    case "1", "true", "TRUE", "on", "ON": return true
    default: return false
    }
}

/// True when the fused `rmsNorm + quantized GEMV` primitive can be used.
@inline(__always)
public func canUseFusedNormProj(x: MLXArray, proj: Linear) -> Bool {
    guard let qProj = proj as? QuantizedLinear,
        qProj.bits == 4,
        qProj.biases != nil,
        proj.bias == nil
    else { return false }
    guard x.dim(-2) == 1 else { return false }
    switch x.dtype {
    case .bfloat16, .float16: return true
    default: return false
    }
}

/// Apply an RMSNorm followed by a (possibly quantized) Linear projection.
/// When `proj` is a 4-bit `QuantizedLinear`, input is a decode-shape GEMV,
/// and dtype is fp16/bf16, the two ops collapse into one
/// `MLXFast.rmsNormQuantizedGEMV` dispatch. Otherwise falls back.
///
/// Callers must also gate on `isFusedNormMLPEnabled()` if they want the
/// feature off by default.
public func applyNormLinear(
    _ x: MLXArray, normWeight: MLXArray, eps: Float, proj: Linear
) -> MLXArray {
    if let qProj = proj as? QuantizedLinear,
        qProj.bits == 4,
        let biases = qProj.biases,
        proj.bias == nil,
        x.dim(-2) == 1,
        (x.dtype == .bfloat16 || x.dtype == .float16)
    {
        return MLXFast.rmsNormQuantizedGEMV(
            x, normWeight: normWeight,
            w: qProj.weight, scales: qProj.scales, biases: biases,
            eps: eps, groupSize: qProj.groupSize)
    }
    let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
    return proj(normed)
}

// MARK: - PreNormHolder
//
// Opaque reference type that holds a borrowed RMSNorm weight + eps for an
// MLP. Wrapping in a non-MLXArray / non-Module class hides the MLXArray
// from Module's Mirror-based parameter discovery, so the MLP does not
// advertise `preNormWeight` as a loadable parameter in its state dict.

public final class PreNormHolder: @unchecked Sendable {
    public var weight: MLXArray?
    public var eps: Float = 1e-6
    public init() {}
}
