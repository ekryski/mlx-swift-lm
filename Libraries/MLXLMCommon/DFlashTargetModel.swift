// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Target-side protocol surface for DFlash speculative decoding
//
// DFlash speculative decoding pairs a small block-diffusion draft model with
// the target model (the model the user actually wants outputs from). The
// draft conditions on **target-side hidden states** captured at a small
// configured set of layers, and the target verifies the K=16 candidate
// tokens produced by the draft in one batched forward pass.
//
// This file defines the **target-side** protocol — what a model adopting
// DFlash must expose to the iterator. The complementary **draft-side**
// protocol is in `DFlashDraftBackend.swift`.
//
// The shape mirrors SwiftLM's `Sources/DFlash/DFlashTargetModel.swift`
// surface (which in turn ports the `bstnxbt/dflash-mlx` Python reference).
// We keep it small + Swift-idiomatic — no Core ML, no Python-shape glue
// here. Per-target conformance lives next to each target model in
// `Libraries/MLXLLM/Models/*+DFlash.swift`.

/// A target model that can drive DFlash speculative decoding.
///
/// "Target" = the big model whose outputs we want. The DFlash draft
/// reads target hidden states from selected layers and emits 16 candidate
/// tokens in one parallel-decoder forward pass; this protocol exposes the
/// hidden-state capture hook the draft needs plus the standard embed /
/// LM-head pieces the iterator uses to glue the cycle together.
///
/// Per-target conformance is a small extension that taps into the existing
/// per-layer activations on the model's forward path — see spec 015 phase
/// 1 for the bootstrapping pattern.
public protocol DFlashTargetModel: AnyObject {
    /// Token-embedding lookup. Maps `[B, T]` token IDs to `[B, T, H]`
    /// hidden states. Used by the iterator's prefill path.
    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray

    /// LM-head projection from `[B, T, H]` hidden states to `[B, T, V]`
    /// logits. Used by the iterator's verify path to sample target argmax
    /// at each verify-position.
    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray

    /// Forward pass over `inputIDs` of shape `[B, T]` with hidden-state
    /// capture at the configured `captureLayerIDs`. Returns:
    ///   - `logits`: `[B, T, V]` — the standard forward output.
    ///   - `captured`: `[Int: MLXArray]` — `layerID -> [B, T, H]` post-block
    ///     hidden state at each captured layer.
    ///
    /// The captured hidden states feed the draft's cross-attention input
    /// on the next speculative cycle.
    ///
    /// Implementations on hybrid GDN/Mamba models must respect any active
    /// tape-replay-rollback session on `cache` — see spec 020 for the
    /// recording protocol.
    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, captured: [Int: MLXArray])

    /// Whether this target uses GatedDeltaNet / Mamba / SSM layers anywhere
    /// in its stack. When `true`, the iterator routes through the
    /// tape-replay-rollback path (spec 020) on partial accept; when
    /// `false`, plain positional cache trim is sufficient.
    ///
    /// Pure-attention targets (Llama, Qwen 3 dense, Gemma 4) return
    /// `false`. Hybrid targets (Qwen 3.5 / 3.6, Nemotron H, Jamba, etc.)
    /// return `true`.
    var dflashIsHybridGDN: Bool { get }
}

// MARK: - Default selection of capture layers

/// Pick a small spread of target layers for the draft to condition on.
/// dflash-mlx's reference convention: roughly evenly spaced layers from
/// just past the embedding to a few layers shy of the LM head.
///
/// For a target with `numLayers` total decoder layers and a draft model
/// trained to consume `numDraftLayers` captures, returns the layer IDs
/// the target should expose to the draft.
///
/// The choice is identical to dflash-mlx's `buildTargetLayerIDs` helper
/// (cited inline). Single-layer drafts grab the middle of the stack;
/// multi-layer drafts evenly span layers `[1, numLayers - 3]`.
public func dflashDefaultCaptureLayerIDs(
    numTargetLayers: Int,
    numDraftLayers: Int
) -> [Int] {
    if numDraftLayers <= 1 {
        return [numTargetLayers / 2]
    }
    let start = 1
    let end = numTargetLayers - 3
    let span = max(1, end - start)
    return (0 ..< numDraftLayers).map { i in
        Int((Double(start) + Double(i) * Double(span) / Double(numDraftLayers - 1))
            .rounded())
    }
}
