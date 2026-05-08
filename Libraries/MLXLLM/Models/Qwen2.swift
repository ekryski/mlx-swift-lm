//
//  Qwen2.swift
//  LLM
//
//  Created by John Mai on 2024/3/3.
//
//  Layer stack lifted into MLXLMCommon.Qwen2 namespace during the issue
//  #168 consolidation pass (2026-05-06). This file owns only the LLM-side
//  outer model + Configuration.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false

    /// Adapter producing the shared layer-args struct consumed by
    /// `Qwen2.{Attention, MLP, DecoderLayer, ModelInner}`.
    public var layerArgs: Qwen2.LayerArgs {
        Qwen2.LayerArgs(
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeTraditional: ropeTraditional,
            ropeScaling: ropeScaling)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Public LLM-side Qwen 2 model. Wraps `Qwen2.ModelInner` with an optional
/// Linear lm_head (or tied embedding).
public class Qwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen2.ModelInner
    let configuration: Qwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2.ModelInner(args.layerArgs, vocabularySize: args.vocabularySize)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

extension Qwen2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
