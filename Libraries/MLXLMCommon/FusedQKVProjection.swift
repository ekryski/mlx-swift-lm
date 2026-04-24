import Foundation
import MLX
import MLXNN

// MARK: - FusedQKVProjection
//
// Applies `qProj`, `kProj`, `vProj` to the same input as a single
// `MLXFast.batchedQKVQuantizedGEMV` Metal dispatch when guards are met.
// Falls back to three separate matmuls when not.
//
// Spec 003.

/// True when `batchedQKVQuantizedGEMV` preconditions are satisfied for the
/// given attention projections.
@inline(__always)
public func canUseFusedQKVProjection(
    x: MLXArray, qProj: Linear, kProj: Linear, vProj: Linear
) -> Bool {
    // Decode-only (GEMV, not GEMM).
    guard x.dim(-2) == 1 else { return false }
    // Quantized primitive — needs all three to be QuantizedLinear with matching params.
    guard let qQ = qProj as? QuantizedLinear,
        let kQ = kProj as? QuantizedLinear,
        let vQ = vProj as? QuantizedLinear
    else { return false }
    guard qQ.bits == kQ.bits, kQ.bits == vQ.bits,
        qQ.groupSize == kQ.groupSize, kQ.groupSize == vQ.groupSize,
        qQ.bits == 4
    else { return false }
    // Biases on the quantized representation must exist.
    guard qQ.biases != nil, kQ.biases != nil, vQ.biases != nil else { return false }
    // Bias on the Linear (not quantization bias) breaks the fused path — the
    // primitive doesn't fold a per-output-element bias into its GEMV. Three
    // separate Linears with biases would each add their own; not worth
    // supporting at the helper layer.
    guard qProj.bias == nil, kProj.bias == nil, vProj.bias == nil else { return false }
    // dtype: fp16 / bf16 only.
    switch x.dtype {
    case .bfloat16, .float16: break
    default: return false
    }
    return true
}

/// Reads `MLX_FUSED_QKV` at call time. Defaults to false — the batched QKV
/// kernel stays opt-in until per-model benchmarks validate it.
@inline(__always)
public func isFusedQKVProjectionEnabled() -> Bool {
    switch ProcessInfo.processInfo.environment["MLX_FUSED_QKV"] {
    case "1", "true", "TRUE", "on", "ON": return true
    default: return false
    }
}

/// Apply qProj, kProj, vProj as a single fused Metal dispatch when guards
/// are met; fall back to three separate Linear calls otherwise. Returns the
/// three projection outputs in their caller-expected shapes.
///
/// Caller is responsible for any post-projection reshape / norm / RoPE —
/// this helper only collapses the three matmuls.
public func fusedQKVProjection(
    _ x: MLXArray,
    qProj: Linear, kProj: Linear, vProj: Linear
) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
    if isFusedQKVProjectionEnabled(),
        canUseFusedQKVProjection(x: x, qProj: qProj, kProj: kProj, vProj: vProj),
        let qQ = qProj as? QuantizedLinear,
        let kQ = kProj as? QuantizedLinear,
        let vQ = vProj as? QuantizedLinear
    {
        let q_dim = qQ.weight.dim(0)
        let k_dim = kQ.weight.dim(0)
        let v_dim = vQ.weight.dim(0)
        let fused = MLXFast.batchedQKVQuantizedGEMV(
            x,
            wQ: qQ.weight, scalesQ: qQ.scales, biasesQ: qQ.biases!,
            wK: kQ.weight, scalesK: kQ.scales, biasesK: kQ.biases!,
            wV: vQ.weight, scalesV: vQ.scales, biasesV: vQ.biases!,
            groupSize: qQ.groupSize)
        // Split last axis into [q_dim, k_dim, v_dim].
        let parts = MLX.split(fused, indices: [q_dim, q_dim + k_dim], axis: -1)
        return (q: parts[0], k: parts[1], v: parts[2])
    }
    return (q: qProj(x), k: kProj(x), v: vProj(x))
}
