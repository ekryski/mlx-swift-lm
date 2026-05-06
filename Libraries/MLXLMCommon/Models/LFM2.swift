// Copyright © 2026 Apple Inc.
//
// Shared LFM 2 text-decoder building blocks. Both `MLXLLM/Models/LFM2.swift`
// and `MLXVLM/Models/LFM2VL.swift` consume this namespace.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the LFM 2 text decoder. The outer LLM and VLM model
/// classes live in their respective targets and share this layer stack.
public enum LFM2 {

    // MARK: - Configuration

    /// JSON config parser shared across LLM and VLM LFM 2 variants. Decodes
    /// every numeric/bool field with sensible defaults so both LFM2-text
    /// repos and LFM2-VL nested `text_config` repos load cleanly.
    public struct Configuration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let maxPositionEmbeddings: Int?
        public let normEps: Float
        public let convBias: Bool
        public let convLCache: Int

        // Optional with hiddenSize fallback (per-model overrides).
        private let _blockDim: Int?
        public var blockDim: Int { _blockDim ?? hiddenSize }
        private let _blockFFDim: Int?
        public var blockFFDim: Int { _blockFFDim ?? hiddenSize }

        public let blockMultipleOf: Int
        public let blockFFNDimMultiplier: Float
        public let blockAutoAdjustFFDim: Bool

        // Either explicit `full_attn_idxs` or derived from `layer_types`.
        private let _fullAttnIdxs: [Int]?
        private let layerTypes: [String]?
        public var fullAttnIdxs: [Int] {
            if let _fullAttnIdxs { return _fullAttnIdxs }
            if let layerTypes {
                return layerTypes.enumerated().compactMap { idx, type in
                    type == "full_attention" ? idx : nil
                }
            }
            return Array(0 ..< hiddenLayers)
        }

        public let ropeTheta: Float
        // Stored so the auto-synthesized `encode(to:)` is happy; the value is
        // also folded into `ropeTheta` at decode time when present.
        public let ropeParameters: [String: StringOrNumber]?

        public var headDimensions: Int { hiddenSize / attentionHeads }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case normEps = "norm_eps"
            case convBias = "conv_bias"
            case convLCache = "conv_L_cache"
            case _blockDim = "block_dim"
            case _blockFFDim = "block_ff_dim"
            case blockMultipleOf = "block_multiple_of"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case _fullAttnIdxs = "full_attn_idxs"
            case layerTypes = "layer_types"
            case ropeTheta = "rope_theta"
            case ropeParameters = "rope_parameters"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm2"
            self.vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65536
            self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            self.maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            self.normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1.0e-5
            self.convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
            self.convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
            self._blockDim = try c.decodeIfPresent(Int.self, forKey: ._blockDim)
            self._blockFFDim = try c.decodeIfPresent(Int.self, forKey: ._blockFFDim)
            self.blockMultipleOf = try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
            self.blockFFNDimMultiplier = try c.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1.0
            self.blockAutoAdjustFFDim = try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
            self._fullAttnIdxs = try c.decodeIfPresent([Int].self, forKey: ._fullAttnIdxs)
            self.layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            // `rope_theta` may be top-level OR nested in `rope_parameters`.
            let topRopeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
            self.ropeParameters = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)
            self.ropeTheta = self.ropeParameters?["rope_theta"]?.asFloat() ?? topRopeTheta
        }
    }

    // MARK: - Attention

    public class Attention: Module {
        let scale: Float
        let headDim: Int
        let heads: Int
        let kvHeads: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        let rope: RoPE

        public init(_ config: Configuration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = config.headDimensions
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return outProj(output)
        }
    }

    // MARK: - Short Conv (LFM2's hybrid SSM-style layer)

    public class ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        public init(_ config: Configuration, layerIdx: Int) {
            self.lCache = config.convLCache
            self.hiddenSize = config.hiddenSize
            let bias = config.convBias

            self._conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
                kernelSize: lCache, groups: config.hiddenSize, bias: bias)
            self._inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: bias)
            self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: bias)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray, cache: SSMStateCache?) -> MLXArray {
            let BCx = inProj(x)
            let parts = BCx.split(parts: 3, axis: -1)
            let B = parts[0]
            let C = parts[1]
            let xPart = parts[2]
            var Bx = B * xPart

            var state: MLXArray? = nil
            if let cache { state = cache[0] }
            if state == nil {
                state = MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            }

            Bx = concatenated([state!, Bx], axis: -2)
            if let cache {
                cache[0] = Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...].contiguous()
            }

            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    // MARK: - MLP

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        public init(
            dim: Int, ffDim: Int, multipleOf: Int, autoAdjustFFDim: Bool, ffnDimMultiplier: Float?
        ) {
            var adjustedFFDim = ffDim
            if autoAdjustFFDim {
                adjustedFFDim = Int(Float(2 * ffDim) / 3.0)
                if let multiplier = ffnDimMultiplier {
                    adjustedFFDim = Int(multiplier * Float(adjustedFFDim))
                }
                adjustedFFDim = multipleOf * ((adjustedFFDim + multipleOf - 1) / multipleOf)
            }
            self._w1.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, dim, bias: false)
            self._w3.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    // MARK: - DecoderLayer

    public class DecoderLayer: Module {
        public let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        public init(_ config: Configuration, layerIdx: Int) {
            self.isAttentionLayer = config.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._attention.wrappedValue = Attention(config)
            } else {
                self._conv.wrappedValue = ShortConv(config, layerIdx: layerIdx)
            }

            self._feedForward.wrappedValue = MLP(
                dim: config.blockDim,
                ffDim: config.blockFFDim,
                multipleOf: config.blockMultipleOf,
                autoAdjustFFDim: config.blockAutoAdjustFFDim,
                ffnDimMultiplier: config.blockFFNDimMultiplier)
            self._operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = attention!(operatorNorm(x), mask: mask, cache: cache)
            } else {
                r = conv!(operatorNorm(x), cache: cache as? SSMStateCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    // MARK: - ModelInner

    /// Shared transformer backbone (embed → N hybrid attention/short-conv
    /// blocks → norm). Both LLM and VLM outer model classes wrap this.
    public class ModelInner: Module {
        public let config: Configuration
        public let vocabularySize: Int
        public let numHiddenLayers: Int

        public let layers: [DecoderLayer]

        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") public var embeddingNorm: RMSNorm

        public init(_ config: Configuration) {
            self.config = config
            self.vocabularySize = config.vocabularySize
            self.numHiddenLayers = config.hiddenLayers
            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: config.hiddenSize)
            self.layers = (0 ..< numHiddenLayers).map { i in
                DecoderLayer(config, layerIdx: i)
            }
            self._embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil,
            inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens(inputs)

            // The default mask draws from the first attention layer's cache —
            // short-conv layers use SSMStateCache and don't contribute a key
            // length to the attention mask.
            let resolvedMask =
                mask
                ?? {
                    let firstAttnIdx = config.fullAttnIdxs.first ?? 0
                    let c = cache != nil && firstAttnIdx < cache!.count
                        ? cache![firstAttnIdx] : nil
                    return createAttentionMask(h: h, cache: c)
                }()

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: resolvedMask, cache: cache?[i])
            }
            return embeddingNorm(h)
        }
    }
}
