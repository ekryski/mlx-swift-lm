// Copyright © 2026 Apple Inc.
//
// Shared Qwen 3.5 building blocks. The Qwen 3.5 family is the
// trickiest of the WS-C consolidation sprint because it is a hybrid
// model — `GatedDeltaNet` (GDN) SSM blocks alternate with standard
// attention blocks via a `layer_types` config — and because the LLM
// has substantial alpha-branch perf work that we explicitly preserve:
//
// - **Fused `gate_up_proj` MLP** (`Qwen3NextMLP` in
//   `Libraries/MLXLLM/Models/Qwen3Next.swift`): the LLM merges the
//   per-expert `gate_proj` and `up_proj` linears into a single
//   `gate_up_proj` matmul at sanitize time (see `fuseGateUpWeights`
//   in `Libraries/MLXLLM/Models/Qwen35.swift`). Forcing the VLM to
//   use this would change its weight-loading shape; forcing the LLM
//   onto the separate-projection class would silently regress decode.
// - **Fused GDN decode dispatch**
//   (`Qwen35GatedDeltaNet.callAsFunction` decode S==1 path): dispatches
//   `fusedGatedDeltaUpdate` (single Metal kernel) instead of the
//   ops-based `gatedDeltaUpdate`, saving 4–6 dispatches/layer/token.
//   Plus `.contiguous()` safety on the conv-state slice that prevents
//   the 9 GB lazy-chain blow-up at long contexts. The VLM's GDN runs
//   the slower ops path everywhere and has no such safety calls; the
//   two GDN classes are *near*-identical but not interchangeable.
// - **Batched speculative-decoding paths** (`fullyBatchedForward` on
//   both `Qwen35Attention` and `Qwen35GatedDeltaNet`): LLM-only.
// - **Turbo KV boundary-layer skip** in
//   `Qwen35TextModel.newCache(parameters:)`: preserves first-N /
//   last-N attention layers uncompressed. Not in the VLM (which has
//   no turbo path).
//
// What lives here is the **standard separate-projection SwiGLU MLP**
// the VLM uses for both its dense decoder layers and the
// `SparseMoeBlock.sharedExpert`. It is bit-identical to other
// Qwen-family MLPs (Qwen 2 / Qwen 3 — both already lifted via PR
// #175 / #177), shipped here with the `Qwen35.` namespace prefix so
// the consolidation pattern stays per-family rather than relying on
// cross-family typealias chains.
//
// Future-work alignment with `IMPLEMENTATION-PLAN.md`:
//
// - **Spec 020 phase 2 — `SSMStateCache: StateReplayCache` + Metal
//   kernel** (Tier 1). Both LLM
//   (`Libraries/MLXLLM/Models/Qwen35.swift` `newCache(parameters:)`)
//   and VLM (`Libraries/MLXVLM/Models/Qwen35.swift`
//   `LanguageModel.makeCache`) instantiate `SSMStateCache()` inline
//   for hybrid linear-attention layers. Phase 2 will need to update
//   both factories. The right time to lift them into a single
//   `Qwen35.makeHybridCache(...)` helper here is when phase 2 lands —
//   the helper signature is shaped by the new `StateReplayCache`
//   protocol, which doesn't exist yet.
// - **Spec 028 — chunkwise WY GatedDeltaNet prefill** (Tier 4). The
//   LLM's fused decode path is the natural target for the chunkwise
//   variant; once spec 028 lands, the GDN class becomes a candidate
//   for cross-target lift.
// - **Issue #115 — `MLXFast.batchedQKVQuantizedGEMV`**, **issue #117
//   — RMSNorm + GEMV fusion**. Same pattern as Qwen 3 (PR #177): once
//   the LLM's `fullyBatchedForward` paths simplify to a single fused
//   matmul, an `Attention` consolidation becomes safe.
// - **Cross-family M-RoPE helper** (deferred from PR #177 / #178).
//   The VLM's `Qwen35Language.RotaryEmbedding` is the same M-RoPE
//   shape as Qwen3VL / GlmOcr; the helper landing for those unblocks
//   the same kind of second-pass consolidation here.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Qwen 3.5 text decoder. Only the standard
/// separate-projection SwiGLU MLP is included; see file-level note for
/// why everything else is per-target.
public enum Qwen35 {

    // MARK: - MLP

    /// Standard SwiGLU MLP — `gate_proj` / `up_proj` / `down_proj` with
    /// `silu(gate) * up`. Bit-identical to the Qwen 3 / Qwen 2 standard
    /// SwiGLU; shipped under the `Qwen35.` prefix to keep the
    /// consolidation-namespace boundary per-family. Used by the VLM's
    /// dense `DecoderLayer.mlp` and by its `SparseMoeBlock.sharedExpert`.
    /// The LLM uses its alpha-branch fused-`gate_up_proj` variant
    /// (`Qwen3NextMLP`) instead — see file-level note.
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
