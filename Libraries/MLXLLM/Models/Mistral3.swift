// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ministral3.py
//
// Renamed from Mistral3Text.swift on 2026-05-06 (issue #168 consolidation).
// The text-decoder layer stack now lives in
// `Libraries/MLXLMCommon/Models/Mistral3.swift` as the public `Mistral3`
// namespace. This file owns only the LLM-side outer model + Configuration.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Mistral3TextConfiguration: Codable, Sendable {
    public var modelType: String = "ministral3"
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var headDimensions: Int?
    public var maxPositionEmbeddings: Int?
    public var kvHeads: Int
    public var ropeTheta: Float = 10_000
    public var ropeParameters: [String: StringOrNumber]?
    public var tieWordEmbeddings: Bool = false
    public var layerTypes: [String]
    public var slidingWindow: Int?

    public var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    /// Adapter producing the shared layer-args struct consumed by
    /// `Mistral3.Attention` / `MLP` / `TransformerBlock` / `ModelInner`.
    public var layerArgs: Mistral3.LayerArgs {
        Mistral3.LayerArgs(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: resolvedHeadDimensions,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeParameters: ropeParameters,
            maxPositionEmbeddings: maxPositionEmbeddings,
            layerTypes: layerTypes,
            slidingWindow: slidingWindow)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let topLevelContainer = try decoder.container(keyedBy: CodingKeys.self)
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // For VLM-converted repos the text fields nest under `text_config`.
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "ministral3"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? (try topLevelContainer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
            ?? false

        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
    }

    public init(
        modelType: String = "ministral3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        layerTypes: [String]? = nil,
        slidingWindow: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = slidingWindow
    }
}

// MARK: - Outer LLM model

/// Mistral 3 / Ministral 3 LLM. Wraps `Mistral3.ModelInner` (in MLXLMCommon)
/// with an optional Linear lm_head (or tied embedding) and
/// `LLMModel`-protocol-required hooks (sanitize, newCache).
public class Mistral3TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Mistral3.ModelInner
    fileprivate let args: Mistral3TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Mistral3TextConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Mistral3.ModelInner(args.layerArgs, vocabularySize: args.vocabularySize)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, inputEmbeddings: nil)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray?
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Strip the VLM `language_model.` prefix when present.
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Drop precomputed rotary frequency artifacts.
        var sanitizedWeights = processedWeights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        if args.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Apply `weight_scale_inv` (legacy quantized format) and drop
        // `activation_scale` artifacts.
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    newWeights[weightKey] = weight * value
                }
            } else if key.contains("activation_scale") {
                continue
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }
        return newWeights.isEmpty ? sanitizedWeights : newWeights
    }

    /// Per-layer cache: sliding-attention layers get a windowed
    /// `StandardKVCache`; full-attention layers get unbounded.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let affineStep = defaultPrefillStepSize
        return model.layers.map { layer in
            let isSliding = layer.useSliding
            let maxSize: Int? = isSliding ? args.slidingWindow : nil
            return makeAttentionCache(
                parameters: parameters, maxSize: maxSize, affineStep: affineStep,
                architecturalSlidingWindow: isSliding)
        }
    }
}

extension Mistral3TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
