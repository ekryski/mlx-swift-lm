// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma 4 Multi-Token Prediction (MTP) drafter — Swift port of
// Blaizzy/mlx-vlm:mlx_vlm/speculative/drafters/gemma4_assistant/.
//
// Architecture notes:
//   - 4-layer "assistant" model trained to draft K candidate tokens per
//     round; full Gemma 4 target verifies in one parallel forward pass.
//   - Tightly coupled to target: no own KV cache, reads K/V directly
//     from target's last full-attention + last sliding-attention layers.
//   - Each draft step input = concat([target_embed(last_token),
//     last_hidden_state]) → pre_projection → drafter hidden_size.
//   - Position is held constant across all draft steps within a block
//     (RoPE rotates queries at the bonus token's absolute position).
//
// Implementation scope: 26B-A4B variant (no masked embedder, tied dense
// LM head, use_ordered_embeddings=false). MaskedEmbedder for E2B/E4B
// can be added later — same interface, different LM head.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public let modelType: String
    public let backboneHiddenSize: Int
    public let useOrderedEmbeddings: Bool
    public let numCentroids: Int
    public let centroidIntermediateTopK: Int
    public let tieWordEmbeddings: Bool
    public let textConfig: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case tieWordEmbeddings = "tie_word_embeddings"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decode(String.self, forKey: .modelType)
        self.backboneHiddenSize = try c.decode(Int.self, forKey: .backboneHiddenSize)
        self.useOrderedEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        self.numCentroids =
            try c.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        self.centroidIntermediateTopK =
            try c.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.textConfig = try c.decode(
            Gemma4TextConfiguration.self, forKey: .textConfig)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(backboneHiddenSize, forKey: .backboneHiddenSize)
        try c.encode(useOrderedEmbeddings, forKey: .useOrderedEmbeddings)
        try c.encode(numCentroids, forKey: .numCentroids)
        try c.encode(centroidIntermediateTopK, forKey: .centroidIntermediateTopK)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(textConfig, forKey: .textConfig)
    }
}

// MARK: - Parent KV Extraction

/// Pull the parent's last full-attention and last sliding-attention
/// K/V from its cache list, accounting for KV-sharing donors. The
/// drafter's 4 layers all run kv_shared_only against these two
/// donor tensors.
public func extractDrafterSharedKV(
    layerTypes: [String],
    previousKVs: [Int],
    caches: [KVCache]
) -> (full: (MLXArray, MLXArray), sliding: (MLXArray, MLXArray))? {
    var lastFull: Int? = nil
    var lastSliding: Int? = nil
    for (i, t) in layerTypes.enumerated() {
        if t == "full_attention" { lastFull = i }
        else if t == "sliding_attention" { lastSliding = i }
    }
    guard let lf = lastFull, let ls = lastSliding else { return nil }
    let fullDonor = previousKVs[lf]
    let slidingDonor = previousKVs[ls]
    guard fullDonor < caches.count, slidingDonor < caches.count else {
        return nil
    }
    guard let fullKV = caches[fullDonor].peek(),
          let slidingKV = caches[slidingDonor].peek()
    else { return nil }
    return (full: fullKV, sliding: slidingKV)
}

// MARK: - Bidirectional Drafter Masks

/// Build a bidirectional sliding-window mask for the drafter's queries
/// against the target's cached K/V. For each query at absolute position
/// `q ∈ [queryOffset, queryOffset+queryLen)`, allow attention to keys in
/// `(q - window, q + window)`. Returns nil when no masking is needed.
private func bidirectionalSWAMask(
    queryLen: Int, queryOffset: Int, kvLen: Int,
    window: Int, dtype: DType
) -> MLXArray? {
    if kvLen <= window && queryOffset + queryLen <= kvLen + window {
        return nil
    }
    let qIdx = MLXArray(
        Array(stride(from: Int32(queryOffset),
                     to: Int32(queryOffset + queryLen),
                     by: 1)))
        .reshaped([queryLen, 1])
    let kIdx = MLXArray(
        Array(stride(from: Int32(0), to: Int32(kvLen), by: 1)))
        .reshaped([1, kvLen])
    let dist = qIdx - kIdx
    let lo = MLX.greater(dist, MLXArray(-Int32(window)))
    let hi = MLX.less(dist, MLXArray(Int32(window)))
    let inside = MLX.logicalAnd(lo, hi)
    let bias = MLX.where(
        inside,
        MLXArray(0.0).asType(dtype),
        MLXArray(-Float.infinity).asType(dtype))
    return bias.reshaped([1, 1, queryLen, kvLen])
}

// MARK: - Drafter Attention (Q-only — K/V come from parent's shared KV)

private class Gemma4DrafterAttention: Module {
    let isSliding: Bool
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm

    let rmsNormEps: Float
    let rope: any OffsetLayer
    let _fusedInvFreqs: MLXArray?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self.nHeads = config.attentionHeads
        if isSliding {
            self.headDim = config.headDim
            self.nKVHeads = config.kvHeads
        } else {
            self.headDim = config.globalHeadDim
            self.nKVHeads = config.globalKvHeads
        }
        self.scale = 1.0
        let dim = config.hiddenSize
        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self.rmsNormEps = config.rmsNormEps

        if isSliding {
            self.rope = RoPE(
                dimensions: headDim, traditional: false,
                base: config.ropeTheta)
            let exponents = MLXArray(
                stride(from: Float(0), to: Float(headDim), by: 2)
            ) / Float(headDim)
            let freqs = pow(MLXArray(config.ropeTheta), exponents)
            self._fusedInvFreqs = 1.0 / freqs
        } else {
            let ropeDim = Int(Float(headDim) * config.partialRotaryFactor)
            self.rope = ProportionalRoPE(
                dims: headDim, rotatedDims: ropeDim,
                traditional: false, base: config.globalRopeTheta)
            let propExponents = MLXArray(
                stride(from: Float(0), to: Float(ropeDim), by: 2)
            ) / Float(headDim)
            let realFreqs = pow(MLXArray(config.globalRopeTheta), propExponents)
            let paddingCount = (headDim - ropeDim) / 2
            let infPadding = MLXArray(
                Array(repeating: Float.infinity, count: paddingCount))
            let allFreqs = concatenated([realFreqs, infPadding], axis: 0)
            self._fusedInvFreqs = 1.0 / allFreqs
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sharedKV: (MLXArray, MLXArray),
        donorOffset: Int
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        var queries = qProj(x).reshaped(B, L, nHeads, -1)

        if let invFreqs = _fusedInvFreqs {
            queries = MLXFast.rmsNormRoPE(
                queries, weight: qNorm.weight, invFreqs: invFreqs,
                eps: rmsNormEps, offset: donorOffset,
                nHeads: nHeads, seqLen: L)
            queries = queries.transposed(0, 2, 1, 3)
        } else {
            queries = qNorm(queries)
            queries = queries.transposed(0, 2, 1, 3)
            queries = rope(queries, offset: donorOffset)
        }

        let (cachedK, cachedV) = sharedKV
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedK, values: cachedV,
            scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }
}

// MARK: - Drafter Decoder Layer

private class Gemma4DrafterDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Gemma4DrafterAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: DrafterMLP

    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self._attention.wrappedValue = Gemma4DrafterAttention(
            config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = DrafterMLP(config)
        // layer_scalar is loaded from checkpoint
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sharedKV: (MLXArray, MLXArray),
        donorOffset: Int
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let attnOut = attention(
            inputNorm, mask: mask, sharedKV: sharedKV,
            donorOffset: donorOffset)
        var h = MLXFast.rmsNormResidual(
            attnOut, residual: x,
            weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps)
        let preFFNNorm = preFeedforwardLayerNorm(h)
        let ffnOut = mlp(preFFNNorm)
        h = MLXFast.rmsNormResidual(
            ffnOut, residual: h,
            weight: postFeedforwardLayerNorm.weight,
            eps: postFeedforwardLayerNorm.eps)
        h = h * layerScalar
        return h
    }
}

private class DrafterMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration) {
        let hidden = config.hiddenSize
        let inter = config.intermediateSize
        self._gateProj.wrappedValue = Linear(hidden, inter, bias: false)
        self._upProj.wrappedValue = Linear(hidden, inter, bias: false)
        self._downProj.wrappedValue = Linear(inter, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Drafter Inner Model

private class Gemma4AssistantInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DrafterDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let config: Gemma4TextConfiguration

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Gemma4DrafterDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

// MARK: - Drafter Model

public class Gemma4AssistantModel: Module, LanguageModel {
    public var vocabularySize: Int { config.textConfig.vocabularySize }
    public let kvHeads: [Int] = []  // No own cache
    @ModuleInfo private var model: Gemma4AssistantInner
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4AssistantConfiguration

    /// Bound parent embedding (set via `bind(parent:)`).
    private var parentEmbed: Embedding?
    private var parentEmbedScale: Float = 1.0

    /// Shared K/V from parent (set via `setSharedKV(...)` per draft block).
    /// Keys are layer types: "full_attention" and "sliding_attention".
    private var sharedKV: [String: (MLXArray, MLXArray)] = [:]
    private var sharedKVOffset: Int = 0
    private var draftPosition: Int = 0

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        let textCfg = config.textConfig
        self._model.wrappedValue = Gemma4AssistantInner(textCfg)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabularySize, bias: false)
        }
        super.init()
    }

    /// Bind to the parent target model's input embedding. Required before
    /// `draftBlock` can run (drafter uses parent's embedding to embed
    /// the last accepted token).
    public func bind(parentEmbedding: Embedding, embedScale: Float) {
        self.parentEmbed = parentEmbedding
        self.parentEmbedScale = embedScale
    }

    /// Provide shared K/V from parent's caches. Call this BEFORE every
    /// `draftBlock` so the drafter sees the up-to-date target activations.
    /// - Parameters:
    ///   - fullAttn: parent's last full-attention layer's (K, V)
    ///   - slidingAttn: parent's last sliding-attention layer's (K, V)
    ///   - kvOffset: parent cache.offset (number of cells filled)
    ///   - position: absolute position of bonus query (typically kvOffset)
    public func setSharedKV(
        fullAttn: (MLXArray, MLXArray),
        slidingAttn: (MLXArray, MLXArray),
        kvOffset: Int,
        position: Int? = nil
    ) {
        self.sharedKV = [
            "full_attention": fullAttn,
            "sliding_attention": slidingAttn,
        ]
        self.sharedKVOffset = kvOffset
        self.draftPosition = position ?? kvOffset
    }

    /// Pre-computed masks for one draft block (constant across all
    /// blockSize-1 steps within a block — position is fixed and KV is
    /// not appended to during drafting).
    private var cachedMasks: [String: MLXFast.ScaledDotProductAttentionMaskMode] = [:]

    /// One forward step of the drafter.
    /// - Parameter inputsEmbeds: shape `[B, L, 2*backboneHiddenSize]`
    /// - Returns: (last_hidden in backbone size, logits)
    public func forwardStep(
        _ inputsEmbeds: MLXArray
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig
        var h = preProjection(inputsEmbeds)
        let queryOffset = draftPosition

        for (layerIdx, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[layerIdx]
            guard let kv = sharedKV[layerType] else {
                fatalError("setSharedKV not called for \(layerType)")
            }
            let mask = cachedMasks[layerType] ?? .none
            h = layer(
                h, mask: mask, sharedKV: kv,
                donorOffset: queryOffset)
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)

        let logits: MLXArray
        if let lm = lmHead {
            logits = lm(h)
        } else {
            logits = model.embedTokens.asLinear(h)
        }
        return (lastHidden, logits)
    }

    /// Build per-layer-type masks once at the start of each draft block.
    private func buildBlockMasks(queryLen: Int, dtype: DType) {
        let textCfg = config.textConfig
        cachedMasks.removeAll(keepingCapacity: true)
        for (layerType, kv) in sharedKV {
            if layerType == "sliding_attention" {
                if let m = bidirectionalSWAMask(
                    queryLen: queryLen, queryOffset: draftPosition,
                    kvLen: kv.0.dim(2), window: textCfg.slidingWindow,
                    dtype: dtype)
                {
                    cachedMasks[layerType] = .array(m)
                } else {
                    cachedMasks[layerType] = .none
                }
            } else {
                cachedMasks[layerType] = .none
            }
        }
    }

    /// Autoregressive K-step drafting (lazy). Returns an `MLXArray` of
    /// shape `[1, blockSize-1]` (Int32) — the drafted candidate tokens
    /// — WITHOUT forcing GPU↔CPU sync. The caller is expected to thread
    /// this directly into the parent's verify forward and sync once at
    /// the end of the round, so the full drafter chain fuses into one
    /// lazy MLX graph evaluation.
    ///
    /// `lastBonus` is the most recently accepted token (B=1: scalar Int);
    /// `hidden` is the parent's last hidden state at that position.
    public func draftBlock(
        lastBonus: Int, hidden: MLXArray, blockSize: Int
    ) -> MLXArray {
        precondition(parentEmbed != nil,
                     "bind(parentEmbedding:) must be called before draftBlock")
        precondition(!sharedKV.isEmpty,
                     "setSharedKV must be called before draftBlock")
        precondition(blockSize > 1, "blockSize must be >= 2")

        // Build masks once per draft block (queryLen=1 fixed across steps,
        // queryOffset and KV layout are also constant per block).
        buildBlockMasks(queryLen: 1, dtype: hidden.dtype)

        // Token state stays as MLXArray throughout — never pulls to CPU
        // until the caller decides to sync.
        var tokArr = MLXArray([Int32(lastBonus)]).reshaped([1, 1])
        var hPrev = hidden
        var draftedArrs: [MLXArray] = []
        for _ in 0 ..< (blockSize - 1) {
            let tokEmbed = parentEmbed!(tokArr) * parentEmbedScale
            let inputsEmbeds = concatenated([tokEmbed, hPrev], axis: -1)
            let (newHidden, logits) = forwardStep(inputsEmbeds)
            // Greedy argmax → next token (shape [1, 1] int32). Stays
            // lazy; no .item() call.
            let nextTokArr = argMax(logits[0, 0], axis: -1)
                .asType(.int32).reshaped([1, 1])
            draftedArrs.append(nextTokArr)
            hPrev = newHidden
            tokArr = nextTokArr
        }
        // shape [1, blockSize-1]
        return concatenated(draftedArrs, axis: 1)
    }

    /// Stub for LanguageModel protocol conformance — Gemma4Assistant is
    /// driven via `runMTPSpeculative`, not the standard generate path.
    /// Calling `prepare` or `callAsFunction` on this model directly is a
    /// programming error.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        fatalError(
            "Gemma4AssistantModel must be driven via runMTPSpeculative, " +
            "not the standard generate() path")
    }

    /// Drafter has no own KV cache — reads K/V from the parent's caches
    /// via setSharedKV during draft_block.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    /// Strip checkpoint keys that aren't loaded (e.g. tied LM head, ordering
    /// indices not used in 26B-A4B path). Mirror of mlx-vlm sanitize().
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights

        // Strip language_model. prefix if present (some checkpoints wrap it).
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            out = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Drop tied lm_head.weight (we use embed_tokens.asLinear).
        if config.tieWordEmbeddings {
            out.keys.filter { $0.hasPrefix("lm_head.") }
                .forEach { out.removeValue(forKey: $0) }
        }

        // Drop masked embedder weights (E2B/E4B-only — not used in 26B-A4B).
        if !config.useOrderedEmbeddings {
            out.keys.filter { $0.hasPrefix("masked_embedding.") }
                .forEach { out.removeValue(forKey: $0) }
        }

        return out
    }
}
