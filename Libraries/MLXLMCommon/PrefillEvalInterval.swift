import Foundation

/// Number of transformer layers to batch into one lazy-graph window before
/// issuing an `asyncEval` during multi-token prefill on eligible Qwen3.5
/// dense hybrids (see `Qwen35TextModelInner.batchedPrefillEvalEligible`).
/// Larger = fewer graph splits (faster prefill), smaller = tighter
/// peak-memory bound on activations within the window.
///
/// Override at runtime with `MLX_PREFILL_EVAL_INTERVAL=<N>`. `N=1`
/// force-disables the optimization globally and restores the pre-spec
/// per-layer `asyncEval` behavior, even on eligible models. Non-positive /
/// unparseable values are clamped to 1.
///
/// Default `N=8` — picked from the A/B matrix on M1 Max across Qwen3.5
/// 0.8B / 2B / 4B / 9B; see `benchmarks/notes/prefill-eval-every-n-layers-2026-04-24.md`.
public enum PrefillEvalInterval {
    /// Read once on first access. Constant for the process lifetime — we
    /// don't re-read the env in the per-layer hot loop.
    public static let value: Int = {
        let raw = ProcessInfo.processInfo.environment["MLX_PREFILL_EVAL_INTERVAL"] ?? ""
        return max(1, Int(raw) ?? 8)
    }()
}
