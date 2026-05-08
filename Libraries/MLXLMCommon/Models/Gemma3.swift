// Copyright © 2026 Apple Inc.
//
// Shared Gemma 3 text-decoder building blocks. Both `MLXLLM/Models/Gemma3.swift`
// (the text-only Gemma3TextModel) and `MLXVLM/Models/Gemma3.swift` (the
// vision-language wrapper) consume this namespace to avoid duplication.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Gemma 3 text decoder. The outer LLM and VLM model
/// classes live in their respective targets and share this layer stack.
public enum Gemma3 {

    // MARK: - Configuration

    /// JSON config parser shared across LLM and VLM Gemma 3 variants.
    ///
    /// Handles both flat config (LLM-side `gemma3_text` repos) and the
    /// nested `text_config` wrapper used by VLM repos when a converter has
    /// merged text + vision config into a single file.
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeLocalBaseFreq: Float
        public let ropeTraditional: Bool
        public let queryPreAttnScalar: Float
        public let slidingWindow: Int
        public let slidingWindowPattern: Int
        public let maxPositionEmbeddings: Int
        public let ropeScaling: [String: StringOrNumber]?
        public let finalLogitSoftcapping: Float?

        public init(
            modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
            attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int,
            kvHeads: Int, ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
            queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int,
            maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil,
            finalLogitSoftcapping: Float? = nil
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.vocabularySize = vocabularySize
            self.kvHeads = kvHeads
            self.ropeTheta = ropeTheta
            self.ropeLocalBaseFreq = ropeLocalBaseFreq
            self.ropeTraditional = ropeTraditional
            self.queryPreAttnScalar = queryPreAttnScalar
            self.slidingWindow = slidingWindow
            self.slidingWindowPattern = slidingWindowPattern
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.ropeScaling = ropeScaling
            self.finalLogitSoftcapping = finalLogitSoftcapping
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeLocalBaseFreq = "rope_local_base_freq"
            case ropeTraditional = "rope_traditional"
            case queryPreAttnScalar = "query_pre_attn_scalar"
            case slidingWindow = "sliding_window"
            case slidingWindowPattern = "sliding_window_pattern"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeScaling = "rope_scaling"
            case finalLogitSoftcapping = "final_logit_softcapping"
        }

        enum VLMCodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }

        public init(from decoder: Decoder) throws {
            // VLM repos converted via mlx_lm.convert nest the text fields under
            // `text_config`; LLM repos have the fields at the top level.
            let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
            let container =
                if nestedContainer.contains(.textConfig) {
                    try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
                } else {
                    try decoder.container(keyedBy: CodingKeys.self)
                }

            modelType = try container.decode(String.self, forKey: .modelType)
            hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
            intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
            // 4B+ variants set these explicitly; 1B uses the defaults below.
            attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
            headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
            rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
            // 1B = 262144, 4B+ = 262208 — always JSON-driven.
            vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
            kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
            ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
            ropeLocalBaseFreq =
                try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
            ropeTraditional =
                try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            queryPreAttnScalar =
                try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
            slidingWindow =
                try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
            slidingWindowPattern =
                try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
            maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            finalLogitSoftcapping =
                try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        }
    }

    // MARK: - Attention

    public class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        let scale: Float
        let isSliding: Bool

        @ModuleInfo(key: "q_proj") var queryProj: Linear
        @ModuleInfo(key: "k_proj") var keyProj: Linear
        @ModuleInfo(key: "v_proj") var valueProj: Linear
        @ModuleInfo(key: "o_proj") var outputProj: Linear

        @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
        @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

        @ModuleInfo var rope: RoPELayer

        public init(_ config: TextConfiguration, layerIdx: Int) {
            let dim = config.hiddenSize
            self.nHeads = config.attentionHeads
            self.nKVHeads = config.kvHeads
            self.headDim = config.headDim
            self.scale = pow(config.queryPreAttnScalar, -0.5)
            self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

            self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            self._queryNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            self._keyNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)

            // Sliding-window layers use the local rope frequency; global layers
            // use the configured theta with optional scaling.
            if isSliding {
                self.rope = initializeRope(
                    dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                    scalingConfig: nil, maxPositionEmbeddings: nil)
            } else {
                self.rope = initializeRope(
                    dims: headDim, base: config.ropeTheta, traditional: false,
                    scalingConfig: config.ropeScaling,
                    maxPositionEmbeddings: config.maxPositionEmbeddings)
            }
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache? = nil
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = queryProj(x)
            var keys = keyProj(x)
            var values = valueProj(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            queries = queryNorm(queries)
            keys = keyNorm(keys)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return outputProj(output)
        }
    }

    // MARK: - MLP

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(geluApproximate(gateProj(x)) * upProj(x))
        }
    }

    // MARK: - TransformerBlock

    public class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

        public init(_ config: TextConfiguration, layerIdx: Int) {
            self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

            self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache? = nil
        ) -> MLXArray {
            let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
            let r2 = mlp(preFeedforwardLayerNorm(h))
            let out = Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
            return out
        }
    }

    // MARK: - Backbone

    /// Shared transformer backbone (embed → N transformer blocks → norm).
    /// The LLM target wraps this with a `Linear` lm_head; the VLM target wraps
    /// it with an lm_head that may be `Linear` or `QuantizedLinear` and adds
    /// optional final-logit softcapping. Both pass `inputs`; the VLM path also
    /// passes `inputEmbedding` (a vision-fused embed) when prefilling with an
    /// image attachment.
    public class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        @ModuleInfo public var layers: [TransformerBlock]
        @ModuleInfo public var norm: Gemma.RMSNorm

        public let config: TextConfiguration

        public init(_ config: TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize)
            self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
                TransformerBlock(config, layerIdx: layerIdx)
            }
            self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray? = nil,
            inputEmbedding: MLXArray? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache?]? = nil
        ) -> MLXArray {
            let h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("Backbone requires either `inputs` or `inputEmbedding`")
            }

            // sqrt(hiddenSize) scale, computed in bf16 then cast to runtime dtype.
            let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
                .asType(h.dtype)
            var stream = h * scale

            var layerCache = cache
            if layerCache == nil {
                layerCache = Array(repeating: nil as KVCache?, count: layers.count)
            }

            // Build masks once. Sliding-window layers and global layers use
            // different masks; cache[0] is sliding (if pattern>1) and
            // cache[pattern-1] is global.
            let globalMask = createAttentionMask(
                h: stream, cache: cache?[config.slidingWindowPattern - 1])
            let slidingWindowMask =
                if config.slidingWindowPattern > 1 {
                    createAttentionMask(
                        h: stream, cache: cache?[0], windowSize: config.slidingWindow)
                } else {
                    MLXFast.ScaledDotProductAttentionMaskMode.none
                }

            for (i, layer) in layers.enumerated() {
                let isGlobal =
                    (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
                let m = isGlobal ? globalMask : slidingWindowMask
                stream = layer(stream, mask: m, cache: layerCache?[i])
            }
            return norm(stream)
        }
    }
}
