// Copyright © 2026 Apple Inc.
//
// Shared GLM 4 building blocks. Like Qwen 3 (PR #177), the GLM 4
// family has substantive architectural divergence between LLM and VLM
// that prevents a full layer-stack consolidation in this PR:
//
// - LLM `GLM4Attention` uses standard `applyRotaryPosition(rope, ...)`
//   with `partialRotaryFactor` and an internally-managed RoPE.
// - VLM `GlmOcr` Attention takes a precomputed `(cos, sin)` tuple from
//   outside via `positionEmbeddings` parameter, with a custom
//   `GlmOcrRotaryEmbedding` class generating the (cos, sin) pair using
//   multimodal positionIds. Same M-RoPE family of design as Qwen3VL.
//
// Forcing a unified shared Attention would either degrade the LLM's
// inline-RoPE path or impose an indirection on the LLM hot path.
// Instead, this namespace lifts what's bit-identical (the fused
// gate-up-proj SwiGLU MLP) and leaves Attention / DecoderLayer /
// ModelInner per-target.
//
// Future-work alignment with `IMPLEMENTATION-PLAN.md`:
//
// - **M-RoPE shared helper** — same blocking factor as PR #177 (Qwen
//   3). Once GlmOcr and Qwen3VL both consume a shared M-RoPE helper,
//   their respective Attention classes shrink enough for fuller
//   consolidation. Track via the consolidation umbrella issue.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the GLM 4 text decoder. The layer-stack
/// classes are intentionally NOT included here; see file-level note.
public enum GLM4 {

    // MARK: - LayerArgs (adapter)

    public struct LayerArgs: Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float

        public init(hiddenSize: Int, intermediateSize: Int, rmsNormEps: Float) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.rmsNormEps = rmsNormEps
        }
    }

    // MARK: - MLP

    /// Fused gate-up SwiGLU MLP — single `gate_up_proj` matmul producing
    /// `2 * intermediate` outputs, then split + silu(gate) * up + down.
    /// Bit-identical between the LLM `GLM4MLP` and VLM `GlmOcrLanguage.MLP`.
    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gateUp.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let xx = gateUp(x)
            let parts = split(xx, parts: 2, axis: -1)
            return down(silu(parts[0]) * parts[1])
        }
    }
}
