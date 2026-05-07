// Copyright © 2026 Apple Inc.
//
// Shared Gemma 4 building blocks. Like Qwen 3 (PR #177) and GLM 4
// (PR #178), the Gemma 4 family has substantive architectural
// divergence between LLM and VLM that prevents a full layer-stack
// consolidation in this PR. The LLM has alpha-branch perf work — most
// importantly compiled QKV projection, the `MLXFast.rmsNormRoPE` fused
// norm+rope kernel path (gated by `_fusedInvFreqs`), the conditional
// `Gemma4SharedMLP` gate+down compile (tuned ON for dense E2B/E4B/31B,
// OFF for MoE 26B-A4B per a measured ~10% regression), the fused
// logit softcap and the fused gelu+mul PLE projection. Forcing a
// unified `Attention` / `MLP` / `DecoderLayer` would either silently
// regress these or impose an indirection on the LLM hot path.
//
// What lives here is bit-identical and has no perf-sensitive
// instantiation pattern: the two specialised RMSNorm variants the
// VLM defines as private final classes. The LLM currently uses
// `MLXNN.RMSNorm` directly (which initialises weight=ones and calls
// the same `MLXFast.rmsNorm` kernel — functionally equivalent to
// `RMSNormZeroShift`) and an inline `MLXFast.rmsNorm(weight: .mlxNone)`
// for v_norm (functionally equivalent to `RMSNormNoScale`); both
// equivalences are documented in the LLM file so the deliberate
// non-share is auditable rather than accidental.
//
// Future-work alignment with `IMPLEMENTATION-PLAN.md`:
//
// - **Spec 024 — KV cache write fusion** (Tier 4 follow-up). Gemma 4
//   E2B was the trigger model for the 60-copy decode-path regression.
//   The shared K/V update pattern lifted by that spec is what would
//   eventually justify a consolidated `Gemma4DecoderLayer` here.
// - **Spec 029 — ANE LM head + Gemma 4 PLE projection**. PLE
//   projection is the natural ANE-offload boundary; a consolidated
//   `Gemma4Backbone` would expose it cleanly to the dispatch shim.
// - **Issue #115 — `MLXFast.batchedQKVQuantizedGEMV`** and
//   **issue #117 — RMSNorm + GEMV fusion**. Today the LLM uses
//   `MLXFast.rmsNormRoPE` (a similar fused kernel) only for Gemma 4;
//   when those two issues land, both attention paths converge on the
//   same primitive and an `Attention` consolidation becomes safe.
// - **Cross-family M-RoPE helper** (deferred from PR #177 / #178).
//   Gemma 4 doesn't need it (no multimodal positionIds stack), but
//   the helper landing for Qwen3VL / GlmOcr unblocks the same kind of
//   second-pass consolidation here.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Gemma 4 text decoder. Layer-stack classes
/// (Attention / MLP / DecoderLayer / Backbone) are intentionally NOT
/// included here; see file-level note.
public enum Gemma4 {

    // MARK: - RMSNorm variants

    /// RMSNorm with no learnable scale — `weight` is `.mlxNone` and the
    /// kernel applies pure `x / rms(x)`. Used for the value projection
    /// (v_norm) and for the MoE router's input norm in the Gemma 4 text
    /// decoder. Bit-identical between the LLM (which calls
    /// `MLXFast.rmsNorm(weight: .mlxNone)` inline) and the VLM (which
    /// wraps the same call in a `private final class`).
    public final class RMSNormNoScale: Module, UnaryLayer {
        public let eps: Float

        public init(eps: Float = 1e-6) {
            self.eps = eps
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
        }
    }

    /// RMSNorm with a learnable weight initialised to ones — `out =
    /// x / rms(x) * weight`, *no* `(1 + weight)` shift (as in the
    /// original Gemma 1/2 family). Functionally equivalent to
    /// `MLXNN.RMSNorm`; the explicit class exists so the VLM's
    /// `Gemma4Text*` decoder can reuse the same name shape as the
    /// upstream Python port and so any future Gemma 4-specific norm
    /// kernel (e.g. issue #117 RMSNorm+GEMV fusion) has one type to
    /// dispatch through.
    public final class RMSNormZeroShift: Module, UnaryLayer {
        public let eps: Float
        @ModuleInfo public var weight: MLXArray

        public init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            self._weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }
    }
}
