import Foundation
import MLX

// MARK: - FusedDenseGateActivation
//
// Inline Metal kernel that takes a fused `gateUp` tensor of shape
// `[..., 2*hiddenDims]` and produces `activation(gate) * up` of shape
// `[..., hiddenDims]` in a single dispatch. Replaces the
// Slice + Slice + activation + Multiply chain (4 dispatches) in the dense
// MLP forward pass — at decode (T=1), dispatch overhead dominates.
//
// Spec 002.

/// Activation variants supported by the inline fused-gate kernel.
public enum DenseActivationKind: Hashable, Sendable {
    case silu
    case geluApprox
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

// MARK: - Kernel sources

private func siluKernelSource() -> String {
    """
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint hidden = threads_per_grid.x;
        uint rows = threads_per_grid.y;
        if (col >= hidden || row >= rows) return;
        uint base = row * (2 * hidden);
        float g = static_cast<float>(gateUp[base + col]);
        float u = static_cast<float>(gateUp[base + hidden + col]);
        float s = g / (1.0f + fast::exp(-g));
        out[row * hidden + col] = static_cast<T>(s * u);
    """
}

private func geluApproxKernelSource() -> String {
    """
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint hidden = threads_per_grid.x;
        uint rows = threads_per_grid.y;
        if (col >= hidden || row >= rows) return;
        uint base = row * (2 * hidden);
        float g = static_cast<float>(gateUp[base + col]);
        float u = static_cast<float>(gateUp[base + hidden + col]);
        // gelu tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c = 0.7978845608028654f;
        float x3 = g * g * g;
        float t = fast::tanh(c * (g + 0.044715f * x3));
        float s = 0.5f * g * (1.0f + t);
        out[row * hidden + col] = static_cast<T>(s * u);
    """
}

private func makeKernel(for kind: DenseActivationKind) -> MLXFast.MLXFastKernel {
    let source: String
    let name: String
    switch kind {
    case .silu:
        source = siluKernelSource()
        name = "fused_dense_gate_silu"
    case .geluApprox:
        source = geluApproxKernelSource()
        name = "fused_dense_gate_gelu_approx"
    }
    return MLXFast.metalKernel(
        name: name,
        inputNames: ["gateUp"],
        outputNames: ["out"],
        source: source
    )
}

// MARK: - Kernel manager

private final class DenseActivationKernelManager: @unchecked Sendable {
    static let shared = DenseActivationKernelManager()

    private let lock = NSLock()
    private var kernels: [DenseActivationKind: MLXFast.MLXFastKernel] = [:]

    func kernel(for kind: DenseActivationKind) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = kernels[kind] { return k }
        let k = makeKernel(for: kind)
        kernels[kind] = k
        return k
    }
}

// MARK: - Public helper

/// Thread-group width along the `hiddenDims` axis. 64 is a safe power-of-2
/// that divides every dense `hiddenDims` we currently ship (Qwen3.5: 3072,
/// 6912, 9728, 11008, 13824; Gemma 4: 8192, 16384, 22528). Rows get 1
/// thread each at decode so the Y group stays at 1.
private let denseActivationThreadGroupWidth = 64

/// True when the inline kernel's fast-path preconditions are met.
@inline(__always)
public func canUseInlineDenseActivation(
    gateUp: MLXArray,
    hiddenDims: Int
) -> Bool {
    // Decode-only — prefill's GEMM path is already dispatch-efficient.
    guard gateUp.dim(-2) == 1 else { return false }
    // Fast path is fp16 / bf16; no fp32 specialization (not worth the build cost).
    switch gateUp.dtype {
    case .bfloat16, .float16: break
    default: return false
    }
    // Threadgroup divisibility guard.
    guard hiddenDims % denseActivationThreadGroupWidth == 0 else { return false }
    // Last-axis layout sanity — fusion requires the 2*hidden layout.
    guard gateUp.dim(-1) == 2 * hiddenDims else { return false }
    return true
}

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
    // Collapse leading dims into a single `rows` axis for the kernel.
    let dtype = gateUp.dtype
    let leading = Array(gateUp.shape.dropLast())
    let rows = leading.reduce(1, *)
    let flat = gateUp.reshaped([rows, 2 * hiddenDims])

    let kernel = DenseActivationKernelManager.shared.kernel(for: kind)

    let outputs = kernel(
        [flat],
        template: [("T", dtype)],
        grid: (hiddenDims, rows, 1),
        threadGroup: (denseActivationThreadGroupWidth, 1, 1),
        outputShapes: [[rows, hiddenDims]],
        outputDTypes: [dtype]
    )

    return outputs[0].reshaped(leading + [hiddenDims])
}
