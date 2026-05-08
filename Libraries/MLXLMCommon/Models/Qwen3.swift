// Copyright © 2026 Apple Inc.
//
// Shared Qwen 3 building blocks. The Qwen 3 family has substantive
// architectural divergence between LLM and VLM that prevents a full
// layer-stack consolidation in this PR:
//
// - LLM `Qwen3Attention` uses standard `applyRotaryPosition(rope, ...)`
//   and ships alpha-side perf work (`batchedForward`,
//   `fullyBatchedForward` for batched inference).
// - VLM `Qwen3VL` Attention uses M-RoPE (multimodal rotary): takes a
//   `positionIds` array, computes `(cos, sin)` via a custom
//   `RotaryEmbedding` class, and applies via
//   `Qwen3VLLanguage.applyMultimodalRotary(q:k:cos:sin:)`.
//
// Forcing a single shared Attention would either gut the LLM's batched
// forward optimisations or require a closure/protocol-based dispatch
// that costs an indirection on the LLM hot path. Instead, this
// namespace lifts the small set of pieces that ARE genuinely shareable
// (configuration adapter + the SwiGLU MLP) and leaves Attention /
// DecoderLayer / ModelInner per-target.
//
// Future work (aligned with `IMPLEMENTATION-PLAN.md`):
//
// - **M-RoPE shared helper** — once a second VLM family adopts the
//   same M-RoPE pattern (Qwen3VL is currently the only one in tree
//   that needs it; Qwen 2.5 VL still uses standard rope), lift the
//   `applyMultimodalRotary` + position-id construction to a shared
//   helper and have both VLMs consume it.
// - **Issue #115 (QKV batched fusion)** — when this lands, the
//   LLM's `batchedForward` paths simplify to a single fused matmul,
//   reducing the per-target divergence enough to enable a fuller
//   Attention consolidation.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Qwen 3 text decoder. The layer-stack
/// classes are intentionally NOT included here; see file-level note.
public enum Qwen3 {

    // MARK: - LayerArgs (adapter)

    /// Minimum field set the shared MLP needs from any of the consuming
    /// configurations. Each consumer's config provides a `var layerArgs:
    /// Qwen3.LayerArgs` accessor.
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

    /// SwiGLU MLP — gate_proj / up_proj / down_proj with silu(gate) * up.
    /// Bit-identical between the LLM and VLM Qwen 3 implementations.
    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }
}
