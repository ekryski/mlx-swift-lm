// Copyright © 2026 ekryski.
//
// Multi-token prediction (MTP) — protocol surface for self-speculative decoding.
//
// Spec: specs/030-multi-token-prediction.md (variant A — in-trunk MTP heads).
//
// Background. Several model families (DeepSeek-V3/V4, Qwen3.5/3.6, Qwen3-Next,
// MiMo, GLM-4-MoE) ship checkpoints with multi-token prediction heads stacked
// on top of the same trunk. At decode time these heads predict the next k
// tokens (k typically 1–4) in a single forward pass; the trunk then verifies
// in batch. Same model is target and draft — no separate draft load.
//
// This file defines the loader-side and forward-side protocols. The actual
// iterator is in `MTPSelfSpeculativeDecoding.swift`.

import Foundation
import MLX

// MARK: - MTPContract

/// Per-family knobs for the MTP forward pass. The two axes vary across
/// the in-trunk MTP families and have to be modeled explicitly:
///
/// - `hiddenVariant`: where the trunk-norm is applied relative to the
///   embedding-norm + concat. DeepSeek-V3/V4 use `preNorm` (the trunk
///   hidden is normalized before concat). Qwen3-Next uses `postNorm`.
/// - `concatOrder`: which of `(embedding, hidden)` comes first in the
///   `eh_proj` input. DeepSeek-V3/V4 use `embeddingHidden`. Qwen3-Next
///   uses `hiddenEmbedding`.
///
/// MTPLX (`mtplx/runtime.py` in github.com/youssofal/MTPLX) ships the
/// canonical per-family contract values for the Python reference; the
/// Swift port mirrors those values via per-target `MTPInjector`
/// implementations.
public struct MTPContract: Sendable {

    /// Where the hidden-state norm sits in the eh_proj input pipeline.
    public enum HiddenVariant: Sendable {
        /// Norm applied to the trunk hidden state BEFORE the concat /
        /// eh_proj. DeepSeek-V3/V4 convention.
        case preNorm
        /// Norm applied AFTER the eh_proj projection but before the
        /// transformer block. Qwen3-Next convention.
        case postNorm
    }

    /// Concat order on the eh_proj input.
    public enum ConcatOrder: Sendable {
        /// `[norm(embedding), norm(hidden)]` — DeepSeek-V3/V4.
        case embeddingHidden
        /// `[norm(hidden), norm(embedding)]` — Qwen3-Next.
        case hiddenEmbedding
    }

    public let hiddenVariant: HiddenVariant
    public let concatOrder: ConcatOrder
    /// Number of nextn heads available on the bundle. The iterator may
    /// use fewer per cycle (driven by `GenerateParameters.mtpDraftCount`)
    /// but cannot exceed this.
    public let maxHeads: Int
    /// Family identifier — used in logs and the diagnostic banner the
    /// iterator emits at construction time. Examples: "deepseek-v4",
    /// "qwen3-next", "mimo".
    public let family: String

    public init(
        hiddenVariant: HiddenVariant,
        concatOrder: ConcatOrder,
        maxHeads: Int,
        family: String
    ) {
        self.hiddenVariant = hiddenVariant
        self.concatOrder = concatOrder
        self.maxHeads = maxHeads
        self.family = family
    }
}

// MARK: - MTPHeadOutput

/// Output of one MTP-head forward pass. The head consumes `(prevHidden,
/// prevToken)` and emits `(nextHidden, nextLogits)` — the next-hidden
/// becomes the input to the subsequent head; the logits feed the
/// iterator's accept/reject loop.
public struct MTPHeadOutput {
    /// Hidden state for the *next* MTP step. Shape `[B, 1, hidden]`.
    public let nextHidden: MLXArray
    /// Logits for the predicted token at the current MTP step.
    /// Shape `[B, 1, vocab]`.
    public let logits: MLXArray

    public init(nextHidden: MLXArray, logits: MLXArray) {
        self.nextHidden = nextHidden
        self.logits = logits
    }
}

// MARK: - MTPInjector

/// Protocol implemented by target models that carry in-trunk MTP heads.
///
/// **Two responsibilities**:
///   1. *Capture forward* — the iterator needs the same forward pass the
///      base `LanguageModel` runs, plus the trunk hidden state (pre-LM-head)
///      so it can be threaded into the MTP heads. `runCaptureForward` is
///      the dual of the model's `callAsFunction` returning both logits
///      and the hidden state at the configured capture position
///      (typically the final post-norm output, just before lm_head).
///   2. *MTP-head forward* — `runMTPHead(headIndex:)` runs head `headIndex`
///      on the given `(prevHidden, prevToken)` and returns the next-hidden
///      + logits.
///
/// Implementers also expose:
///   - `mtpContract` — per-family knobs.
///   - `mtpHeadCaches(parameters:)` — per-head KV caches. Each MTP head is
///     itself a transformer block with its own attention KV. Returned
///     length must equal `mtpContract.maxHeads`.
///   - `isMTPLoaded` — true after `MTPLoader.attach(...)` has installed
///     the head weights. Iterator construction throws if false.
///
/// The protocol is intentionally narrow: it does NOT require the
/// implementer to expose its internal layer modules. The MTP head's
/// forward stays inside the model, which can use private state freely.
public protocol MTPInjector: AnyObject {

    /// True once MTP-head weights have been loaded into the model.
    var isMTPLoaded: Bool { get }

    /// Per-family contract values. Stable across decode cycles.
    var mtpContract: MTPContract { get }

    /// Build per-head KV caches for the MTP transformer blocks. One
    /// entry per head; entries are passed back in order to
    /// `runMTPHead(headIndex:)` calls.
    func mtpHeadCaches(parameters: GenerateParameters?) -> [KVCache]

    /// Run a "capturing" forward pass on the trunk. Equivalent to the
    /// model's standard `callAsFunction` but additionally returns the
    /// post-norm hidden state at the LM-head input.
    ///
    /// - Parameters:
    ///   - inputs: shape `[B, L]` token ids (typically `[1, 1]` during
    ///     decode, `[1, k+1]` during MTP verify).
    ///   - cache: trunk KV cache (the same one the iterator uses for
    ///     plain AR decode).
    /// - Returns:
    ///   - `logits`: shape `[B, L, vocab]`.
    ///   - `hidden`: shape `[B, L, hidden]` — the post-trunk-norm hidden
    ///     state, *before* lm_head projection. Used as the input to the
    ///     first MTP head.
    func runCaptureForward(
        inputs: MLXArray, cache: [KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray)

    /// Run MTP head `headIndex` on `(prevHidden, prevToken)`.
    ///
    /// - Parameters:
    ///   - headIndex: 0-based head index. Must be in
    ///     `[0, mtpContract.maxHeads)`.
    ///   - prevHidden: shape `[B, 1, hidden]` — output of the previous
    ///     head (or the trunk for `headIndex == 0`).
    ///   - prevToken: shape `[B, 1]` — the previously committed (or
    ///     previously-proposed) token id.
    ///   - headCache: the per-head KV cache returned by
    ///     `mtpHeadCaches(...)` at index `headIndex`.
    /// - Returns: `MTPHeadOutput` — next-hidden + logits at this step.
    func runMTPHead(
        headIndex: Int,
        prevHidden: MLXArray,
        prevToken: MLXArray,
        headCache: KVCache
    ) -> MTPHeadOutput
}

// MARK: - MTPLoader

/// Probes a target's weight bag for in-trunk MTP keys and reports
/// whether the bundle ships the heads. Used by sanitize routines to
/// decide whether to keep or drop `mtp.*` / `model.layers.{N}.*` keys.
///
/// Conventions across families:
///   - DeepSeek-V3/V4: heads packed at `model.layers.{numHiddenLayers}.*`
///     plus `eh_proj`, `enorm`, `hnorm` siblings. Layer index is the
///     *next* layer beyond the declared count.
///   - Qwen3.5 / Qwen3-Next / MiMo / GLM-4-MoE: heads under `mtp.*`
///     prefix.
///   - Some bundles ship MTP as a sidecar file (`mtp.safetensors`,
///     `mtp/weights.safetensors`, `model-mtp.safetensors`). This loader
///     does not handle sidecars — that's a follow-up to
///     `MTPLoader.discoverSidecar(...)` once we ship the custom
///     converter.
public enum MTPLoader {

    /// Standard key prefixes to probe for MTP heads. Conservative — does
    /// not catch the DSV4 layer-N convention (use
    /// `hasInTrunkMTPHead(...)` for that).
    public static let standardPrefixes: [String] = [
        "mtp.",
        "model.mtp.",
        "language_model.mtp.",
    ]

    /// True iff any of the supplied weight keys match an MTP-head shape
    /// — either a standard prefix or the DeepSeek-V3/V4 layer-N
    /// convention `model.layers.{layerN}.*`.
    ///
    /// Keys-only variant; sanitize routines call the
    /// `weights:` overload below which forwards here.
    public static func hasInTrunkMTPHead<S: Sequence>(
        keys: S,
        deepseekStyleLayerN: Int? = nil
    ) -> Bool where S.Element == String {
        for k in keys {
            for prefix in standardPrefixes {
                if k.hasPrefix(prefix) { return true }
            }
            if let layerN = deepseekStyleLayerN,
                k.hasPrefix("model.layers.\(layerN).")
            {
                return true
            }
        }
        return false
    }

    /// True iff the weight bag carries any MTP-head-shaped keys.
    /// Convenience overload over the keys-only variant.
    public static func hasInTrunkMTPHead(
        weights: [String: MLXArray],
        deepseekStyleLayerN: Int? = nil
    ) -> Bool {
        hasInTrunkMTPHead(
            keys: weights.keys, deepseekStyleLayerN: deepseekStyleLayerN)
    }

    /// Feature-flag gate. Returns true when MTP loading should be active
    /// for the current process. Driven by either
    /// `parameters.mtpEnabled == true` or `MLX_MTP_ENABLED=1`.
    ///
    /// Sanitize routines call this from inside `sanitize(weights:)` to
    /// decide whether to keep or drop MTP-named keys. Without an
    /// active flag, the sanitize routine *must* drop the MTP keys to
    /// preserve loader-time invariants (no orphan weights, no extra
    /// memory).
    public static func mtpLoadEnabled(parameters: GenerateParameters?) -> Bool {
        if parameters?.mtpEnabled == true { return true }
        if ProcessInfo.processInfo.environment["MLX_MTP_ENABLED"] == "1" {
            return true
        }
        return false
    }

    /// Static fallback for sanitize routines that don't have access to
    /// `GenerateParameters` at load time. Sanitize is called early in
    /// the model load path, before the iterator/parameters are
    /// constructed, so the env var is the only signal available there.
    public static var mtpLoadEnabledFromEnv: Bool {
        ProcessInfo.processInfo.environment["MLX_MTP_ENABLED"] == "1"
    }
}
