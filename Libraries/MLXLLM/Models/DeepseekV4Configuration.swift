// Copyright © 2026 TheTom.
// Adapted from osaurus-ai/vmlx-swift-lm (MIT) — see
// Libraries/MLXLLM/Models/DEEPSEEK_V4_ATTRIBUTION.md for upstream license.
//
// Original: Copyright © 2026 Osaurus AI. SPDX-License-Identifier: MIT.
// Upstream: https://github.com/osaurus-ai/vmlx-swift-lm
//
// DeepSeek-V4 (DSV4-Flash / DSV4-Pro) configuration. Phase 1 port —
// the model loads and produces coherent output, but several numerical
// details are not yet bug-for-bug equivalent to the Python reference;
// see the attribution doc for the Phase 2 follow-up list.
//
// DSV4 is architecturally distinct from DSV3: mHC (manifold hyper-
// connections), MLA with head_dim=512 (no split nope/pe), grouped
// low-rank O projection, learned attention sinks, sqrtsoftplus routing,
// hash routing for the first `numHashLayers` layers, per-layer
// compress_ratio (0/4/128), YaRN RoPE only on compress_ratio>0 layers,
// and swiglu_limit=10.

import Foundation
import MLXLMCommon

/// DeepSeek-V4 architecture + tokenizer + quant configuration.
/// Every field is decoded from `config.json` (via `CodingKeys`) and
/// has a sensible default matching DSV4-Flash (284B / 21B active).
public struct DeepseekV4Configuration: Codable, Sendable {
    // MARK: - Core transformer

    public var vocabSize: Int = 129_280
    public var hiddenSize: Int = 4096
    public var numHiddenLayers: Int = 43
    public var numAttentionHeads: Int = 64
    /// DSV4 uses a SINGLE latent KV head broadcast to all Q heads.
    public var numKeyValueHeads: Int = 1
    public var headDim: Int = 512
    /// Rotary applied only to last `qkRopeHeadDim` dims of the
    /// head-dim=512 vector; the first (headDim - qkRopeHeadDim) = 448
    /// dims are "no-position".
    public var qkRopeHeadDim: Int = 64
    public var qLoraRank: Int = 1024
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 1_048_576

    // MARK: - MLA — grouped low-rank O

    /// `wo_a` splits head output into `oGroups` × `oLoraRank` via an
    /// einsum `bsgd,grd→bsgr`, then concatenates groups before `wo_b`.
    public var oGroups: Int = 8
    public var oLoraRank: Int = 1024

    // MARK: - MoE (Mixture of Experts)

    public var nRoutedExperts: Int = 256
    public var nSharedExperts: Int = 1
    public var numExpertsPerTok: Int = 6
    public var moeIntermediateSize: Int = 2048
    /// Hash routing bypasses topk for the first `numHashLayers` layers —
    /// a learned `tid2eid` table maps token id → expert id directly.
    public var numHashLayers: Int = 3
    public var scoringFunc: String = "sqrtsoftplus"
    public var normTopkProb: Bool = true
    public var routedScalingFactor: Float = 1.5
    /// Clamp for DSV4 SwiGLU: `silu(min(gate, lim)) * clip(up, ±lim)`.
    /// Set to 10.0 in DSV4-Flash; essential to prevent activation blow-up.
    public var swigluLimit: Float = 10.0

    // MARK: - mHC (Manifold Hyper-Connections)

    /// Number of parallel residual-stream copies threaded through each
    /// decoder block (collapse → process → expand). Sinkhorn
    /// doubly-stochastic mixing matrix preserves residual norm.
    public var hcMult: Int = 4
    /// Sinkhorn iterations for `comb` row/col normalization.
    public var hcSinkhornIters: Int = 20
    public var hcEps: Float = 1e-6

    // MARK: - RoPE

    /// Rope theta for layers with `compress_ratio == 0` (no YaRN).
    public var ropeTheta: Float = 10000.0
    /// Rope theta for layers with `compress_ratio > 0` (with YaRN).
    public var compressRopeTheta: Float = 160000.0
    public var ropeScaling: [String: StringOrNumber]? = nil

    // MARK: - Sliding window + compressor

    public var slidingWindow: Int = 128
    /// Per-layer compress ratio ∈ {0, 4, 128}. Layers with >0 use the
    /// Compressor + (for ratio=4) Indexer path for global context.
    public var compressRatios: [Int] = []

    // MARK: - Indexer (sparse attention, only layers with ratio=4)

    public var indexNHeads: Int = 64
    public var indexHeadDim: Int = 128
    public var indexTopk: Int = 512

    // MARK: - Attention sink (learned per-head logit prepended pre-softmax)

    /// Whether the model ships a learned per-head `attn_sink` bias that
    /// is appended as a logit column before softmax (then dropped). DSV4
    /// ships it per layer; setting to false disables the contribution.
    public var useAttnSink: Bool = true

    // MARK: - CodingKeys

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHashLayers = "num_hash_layers"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case compressRopeTheta = "compress_rope_theta"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case useAttnSink = "use_attn_sink"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        func req<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }

        self.vocabSize = req(.vocabSize, 129_280)
        self.hiddenSize = req(.hiddenSize, 4096)
        self.numHiddenLayers = req(.numHiddenLayers, 43)
        self.numAttentionHeads = req(.numAttentionHeads, 64)
        self.numKeyValueHeads = req(.numKeyValueHeads, 1)
        self.headDim = req(.headDim, 512)
        self.qkRopeHeadDim = req(.qkRopeHeadDim, 64)
        self.qLoraRank = req(.qLoraRank, 1024)
        self.rmsNormEps = req(.rmsNormEps, 1e-6)
        self.maxPositionEmbeddings = req(.maxPositionEmbeddings, 1_048_576)
        self.oGroups = req(.oGroups, 8)
        self.oLoraRank = req(.oLoraRank, 1024)
        self.nRoutedExperts = req(.nRoutedExperts, 256)
        self.nSharedExperts = req(.nSharedExperts, 1)
        self.numExpertsPerTok = req(.numExpertsPerTok, 6)
        self.moeIntermediateSize = req(.moeIntermediateSize, 2048)
        self.numHashLayers = req(.numHashLayers, 3)
        self.scoringFunc = req(.scoringFunc, "sqrtsoftplus")
        self.normTopkProb = req(.normTopkProb, true)
        self.routedScalingFactor = req(.routedScalingFactor, 1.5)
        self.swigluLimit = req(.swigluLimit, 10.0)
        self.hcMult = req(.hcMult, 4)
        self.hcSinkhornIters = req(.hcSinkhornIters, 20)
        self.hcEps = req(.hcEps, 1e-6)
        self.ropeTheta = req(.ropeTheta, 10000.0)
        self.compressRopeTheta = req(.compressRopeTheta, 160_000.0)
        self.ropeScaling = try? c.decode([String: StringOrNumber].self, forKey: .ropeScaling)
        self.slidingWindow = req(.slidingWindow, 128)
        self.compressRatios = req(.compressRatios, [])
        self.indexNHeads = req(.indexNHeads, 64)
        self.indexHeadDim = req(.indexHeadDim, 128)
        self.indexTopk = req(.indexTopk, 512)
        self.useAttnSink = req(.useAttnSink, true)
    }

    public init() {}
}

extension DeepseekV4Configuration {
    /// True for layers that carry the compressor (and, at ratio=4, the
    /// indexer). The `Compressor` + `Indexer` modules attach only to
    /// layers with `compress_ratio > 0`.
    public func hasCompressor(layer: Int) -> Bool {
        guard layer < compressRatios.count else { return false }
        return compressRatios[layer] > 0
    }

    /// True for the first `numHashLayers` — these bypass softmax topk
    /// and route tokens to experts via the `tid2eid` hash table.
    public func isHashLayer(_ layer: Int) -> Bool {
        layer < numHashLayers
    }

    /// Per-layer rope theta. DSV4 uses a higher theta on compressor
    /// layers (with YaRN scaling), lower theta on plain attention.
    public func ropeTheta(forLayer layer: Int) -> Float {
        hasCompressor(layer: layer) ? compressRopeTheta : ropeTheta
    }
}
