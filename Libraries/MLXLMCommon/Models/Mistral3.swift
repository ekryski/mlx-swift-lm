// Copyright © 2026 Apple Inc.
//
// Shared Mistral 3 / Ministral 3 text-decoder building blocks. Both
// `MLXLLM/Models/Mistral3.swift` (renamed from Mistral3Text.swift) and
// `MLXVLM/Models/Mistral3.swift` consume this namespace.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Mistral 3 / Ministral 3 text decoder. The LLM
/// target uses `attentionHeads` / `vocabularySize` / etc. naming on its
/// config; the VLM target uses `numAttentionHeads` / `vocabSize`. Rather
/// than force-unify the two configs, this namespace's layer classes
/// take a thin `LayerArgs` adapter that each target's config produces.
public enum Mistral3 {

    // MARK: - LayerArgs (adapter)

    /// Minimum field set the layer stack needs from either the LLM or VLM
    /// configuration. Both targets compute one of these from their own
    /// config struct (see the typealiases at the bottom of each target file).
    public struct LayerArgs: Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeParameters: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int?
        public let layerTypes: [String]
        public let slidingWindow: Int?

        public init(
            hiddenSize: Int, intermediateSize: Int, attentionHeads: Int, kvHeads: Int,
            headDim: Int, rmsNormEps: Float, ropeTheta: Float,
            ropeParameters: [String: StringOrNumber]?, maxPositionEmbeddings: Int?,
            layerTypes: [String], slidingWindow: Int?
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.ropeParameters = ropeParameters
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.layerTypes = layerTypes
            self.slidingWindow = slidingWindow
        }
    }

    // MARK: - Llama4 attention scaling helper

    /// Llama 4 style position-based attention scaling. Used by Mistral 3 /
    /// Ministral 3 when `rope_parameters.llama_4_scaling_beta` is set.
    public static func llama4AttentionScale(
        start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
    ) -> MLXArray {
        let positions = MLXArray(Int32(start) ..< Int32(stop))
        let scaling =
            1
            + beta
            * MLX.log(
                1 + MLX.floor(positions.asType(.float32) / Float(maxPositionEmbeddings)))
        return scaling[0..., .newAxis]
    }

    // MARK: - Attention

    public class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPELayer

        public init(_ args: LayerArgs) {
            let dim = args.hiddenSize
            self.nHeads = args.attentionHeads
            self.nKVHeads = args.kvHeads
            self.headDim = args.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            // Prefer rope_parameters.rope_theta over the top-level value.
            let ropeTheta = args.ropeParameters?["rope_theta"]?.asFloat() ?? args.ropeTheta
            self.rope = initializeRope(
                dims: headDim, base: ropeTheta, traditional: false,
                scalingConfig: args.ropeParameters,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, attnScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
            queries = queries * attnScale

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return wo(output)
        }
    }

    // MARK: - MLP

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(_ args: LayerArgs) {
            let dim = args.hiddenSize
            let hiddenDim = args.intermediateSize
            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            return down(silu(gate(x)) * up(x))
        }
    }

    // MARK: - TransformerBlock

    public class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public let useSliding: Bool

        public init(_ args: LayerArgs, useSliding: Bool = false) {
            self.useSliding = useSliding
            self._attention.wrappedValue = Attention(args)
            self._mlp.wrappedValue = MLP(args)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, attnScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    // MARK: - ModelInner

    /// Shared transformer backbone. Handles full-attention vs sliding-attention
    /// dispatch via `args.layerTypes`, applies Llama-4 attention scaling when
    /// `rope_parameters.llama_4_scaling_beta` is set.
    public class ModelInner: Module {
        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        public let layers: [TransformerBlock]
        public let norm: RMSNorm

        public let args: LayerArgs
        public let vocabularySize: Int
        public let faIndex: Int
        public let swaIndex: Int?

        public init(_ args: LayerArgs, vocabularySize: Int) {
            self.args = args
            self.vocabularySize = vocabularySize
            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: args.hiddenSize)
            self.layers = args.layerTypes.map { layerType in
                TransformerBlock(args, useSliding: layerType == "sliding_attention")
            }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.faIndex = args.layerTypes.firstIndex(of: "full_attention") ?? 0
            self.swaIndex = args.layerTypes.firstIndex(of: "sliding_attention")
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbeddings {
                h = inputEmbeddings
            } else {
                h = embedTokens(inputs)
            }

            let offset = cache?.first?.offset ?? 0

            let faMask = createAttentionMask(h: h, cache: cache?[faIndex])
            let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let swaIndex = swaIndex {
                swaMask = createAttentionMask(
                    h: h, cache: cache?[swaIndex], windowSize: args.slidingWindow)
            } else {
                swaMask = .none
            }

            // Llama-4 attention scaling (or constant 1.0 when not configured).
            let attnScale: MLXArray
            if let ropeParams = args.ropeParameters,
                let beta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
                let originalMaxPos = ropeParams["original_max_position_embeddings"]?.asInt()
            {
                attnScale = Self.scale(
                    h: h, offset: offset, length: h.dim(1),
                    beta: beta, maxPositionEmbeddings: originalMaxPos)
            } else {
                attnScale = MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
            }

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }

        /// Static helper so callers can compute the attention scale outside
        /// the model if needed.
        public static func scale(
            h: MLXArray, offset: Int, length: Int, beta: Float, maxPositionEmbeddings: Int
        ) -> MLXArray {
            llama4AttentionScale(
                start: offset, stop: offset + length, beta: beta,
                maxPositionEmbeddings: maxPositionEmbeddings
            ).asType(h.dtype)
        }
    }
}
