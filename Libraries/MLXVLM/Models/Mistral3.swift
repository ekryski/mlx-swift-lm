import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mistral3
// Note: Mistral3 reuses the vision model from Pixtral

// MARK: - Configuration

// Re-export PixtralVisionConfiguration for Mistral3 use
public typealias Mistral3VisionConfiguration = PixtralVisionConfiguration

// MARK: - Text Configuration

public struct Mistral3VLMTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int

    public var headDim: Int? { _headDim }
    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
    public var ropeTheta: Float { _ropeTheta ?? 1_000_000_000 }
    public var ropeParameters: [String: StringOrNumber]? { _ropeParameters }
    public var ropeTraditional: Bool { _ropeTraditional ?? false }
    public var ropeScaling: [String: StringOrNumber]? { _ropeScaling }
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }
    public var layerTypes: [String]? { _layerTypes }
    public var slidingWindow: Int? { _slidingWindow }
    public var useQkNorm: Bool { _useQkNorm ?? false }

    /// Adapter producing the shared layer-args struct consumed by
    /// `MLXLMCommon.Mistral3.{Attention, MLP, TransformerBlock, ModelInner}`.
    public var layerArgs: Mistral3.LayerArgs {
        Mistral3.LayerArgs(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            attentionHeads: numAttentionHeads,
            kvHeads: numKeyValueHeads,
            headDim: headDim ?? (hiddenSize / numAttentionHeads),
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeParameters: ropeParameters,
            maxPositionEmbeddings: maxPositionEmbeddings,
            layerTypes: layerTypes
                ?? Array(repeating: "full_attention", count: numHiddenLayers),
            slidingWindow: slidingWindow)
    }

    private let _headDim: Int?
    private let _maxPositionEmbeddings: Int?
    private let _numKeyValueHeads: Int?
    private let _ropeTheta: Float?
    private let _ropeParameters: [String: StringOrNumber]?
    private let _ropeTraditional: Bool?
    private let _ropeScaling: [String: StringOrNumber]?
    private let _tieWordEmbeddings: Bool?
    private let _layerTypes: [String]?
    private let _slidingWindow: Int?
    private let _useQkNorm: Bool?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case _headDim = "head_dim"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _numKeyValueHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeParameters = "rope_parameters"
        case _ropeTraditional = "rope_traditional"
        case _ropeScaling = "rope_scaling"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _layerTypes = "layer_types"
        case _slidingWindow = "sliding_window"
        case _useQkNorm = "use_qk_norm"
    }
}

// MARK: - Model Configuration

public struct Mistral3VLMConfiguration: Codable, Sendable {
    public let textConfig: Mistral3VLMTextConfiguration
    public let visionConfig: Mistral3VisionConfiguration
    public let modelType: String

    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 32000 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }
    public var eosTokenId: [Int]? { _eosTokenId }

    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureSelectStrategy: String?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?
    private let _spatialMergeSize: Int?
    private let _multimodalProjectorBias: Bool?
    private let _eosTokenId: [Int]?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _imageTokenId = "image_token_id"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Unfold (im2col)

/// Extract sliding local blocks from a batched input tensor.
/// Equivalent to PyTorch's nn.functional.unfold / im2col operation.
private func unfold(
    _ input: MLXArray,
    kernelSize: Int,
    dilation: Int = 1,
    padding: Int = 0,
    stride: Int = 1
) -> MLXArray {
    var x = input
    let (batchSize, channels, height, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

    // Add padding if needed
    if padding > 0 {
        x = MLX.padded(
            x,
            widths: [
                0,  // batch
                0,  // channels
                .init((padding, padding)),  // height
                .init((padding, padding)),  // width
            ])
    }

    let paddedH = height + 2 * padding
    let paddedW = width + 2 * padding

    // Calculate output dimensions
    let heightOut = (paddedH - dilation * (kernelSize - 1) - 1) / stride + 1
    let widthOut = (paddedW - dilation * (kernelSize - 1) - 1) / stride + 1

    // Extract blocks using array indexing
    var blocks: [MLXArray] = []

    for i in Swift.stride(from: 0, to: paddedH - kernelSize * dilation + 1, by: stride) {
        for j in Swift.stride(from: 0, to: paddedW - kernelSize * dilation + 1, by: stride) {
            var block: [MLXArray] = []
            for di in 0 ..< kernelSize {
                for dj in 0 ..< kernelSize {
                    let hIdx = i + di * dilation
                    let wIdx = j + dj * dilation
                    block.append(x[0..., 0..., hIdx, wIdx])
                }
            }
            // Stack the channel-blocks: (B, C, k*k)
            let stackedBlock = MLX.stacked(block, axis: 1).transposed(0, 2, 1)
            blocks.append(stackedBlock)
        }
    }

    // Stack all blocks: (B, C, k*k, L)
    let result = MLX.stacked(blocks, axis: -1)

    // Reshape to (B, C*k*k, L)
    return result.reshaped(batchSize, channels * kernelSize * kernelSize, heightOut * widthOut)
}

// MARK: - Mistral3 Patch Merger

private class Mistral3PatchMerger: Module {
    let spatialMergeSize: Int
    let patchSize: Int

    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.patchSize = config.visionConfig.patchSize

        let hiddenSize = config.visionConfig.hiddenSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        // Convert image sizes to patch sizes
        let patchSizes = imageSizes.map { (h, w) in
            (h / patchSize, w / patchSize)
        }

        let tokensPerImage = patchSizes.map { $0.0 * $0.1 }
        let d = imageFeatures.dim(-1)
        var features = imageFeatures.asType(.bfloat16)

        // Split the image features into chunks based on tokens per image
        var splitIndices: [Int] = []
        var currentIndex = 0
        for tokens in tokensPerImage.dropLast() {
            currentIndex += tokens
            splitIndices.append(currentIndex)
        }

        let chunks: [MLXArray]
        if splitIndices.isEmpty {
            chunks = [features[0, 0..., 0...]]
        } else {
            chunks = MLX.split(features[0], indices: splitIndices, axis: 0)
        }

        var permutedTensors: [MLXArray] = []

        for (imageIndex, imageTokens) in chunks.enumerated() {
            if imageTokens.dim(0) > 0 {
                let (h, w) = patchSizes[imageIndex]

                // Reshape to grid: (h, w, d) -> (1, d, h, w)
                let imageGrid = imageTokens.reshaped(h, w, d).transposed(2, 0, 1)[
                    .newAxis, 0..., 0..., 0...]

                // Apply unfold
                var grid = unfold(imageGrid, kernelSize: spatialMergeSize, stride: spatialMergeSize)

                // Reshape: (d * spatial_merge_size^2, -1).T
                grid = grid.reshaped(d * spatialMergeSize * spatialMergeSize, -1).transposed()
                permutedTensors.append(grid)
            }
        }

        features = MLX.concatenated(permutedTensors, axis: 0)
        features = mergingLayer(features)

        return features[.newAxis, 0..., 0...]
    }
}

// MARK: - Mistral3 MultiModal Projector

private class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)
        self._patchMerger.wrappedValue = Mistral3PatchMerger(config)
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
    }

    func callAsFunction(_ x: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        var result = norm(x)
        result = patchMerger(result, imageSizes: imageSizes)
        result = linear1(result)
        result = gelu(result)
        result = linear2(result)
        return result
    }
}

// MARK: - Language Model Components

private enum Language {

    // MARK: - Layer stack lifted to MLXLMCommon
    //
    // Mistral 3 / Ministral 3 layer classes (Attention, MLP, TransformerBlock,
    // Ministral3ModelInner) and the Llama-4 attention-scaling helper were
    // lifted into `MLXLMCommon.Mistral3` during the issue #168 consolidation
    // pass. The VLM target now consumes them via the shared namespace; only
    // `Language.LanguageModel` remains in this file.

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: Mistral3VLMTextConfiguration
        let modelType: String

        @ModuleInfo(key: "model") private var model: Mistral3.ModelInner
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            return layerTypes.map { _ in config.numKeyValueHeads }
        }

        /// Access to embed_tokens
        var embedTokens: Embedding {
            model.embedTokens
        }

        /// Access to layers for LoRA
        var layers: [Mistral3.TransformerBlock] {
            model.layers
        }

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            // `Mistral3.ModelInner` handles both model types:
            //  - ministral3: sliding attention + llama4 scaling from rope_parameters
            //  - mistral:    full attention with attnScale=1.0 (beta=0 / missing)
            self._model.wrappedValue = Mistral3.ModelInner(
                config.layerArgs, vocabularySize: config.vocabSize)

            if !config.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var out = model(inputs, cache: cache, inputEmbeddings: inputsEmbeds)

            if config.tieWordEmbeddings {
                out = embedTokens.asLinear(out)
            } else if let lmHead {
                out = lmHead(out)
            }
            return out
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            // LLM Mistral3 leaves `defaultPrefillStepSize` at the protocol
            // default (1024); align the affine cache step here.
            let affineStep = 1024

            return layerTypes.map { layerType in
                if layerType == "sliding_attention", let slidingWindow = config.slidingWindow {
                    return makeAttentionCache(
                        parameters: parameters,
                        slidingWindow: slidingWindow,
                        affineStep: affineStep)
                } else {
                    // Full-attention layer: function reads
                    // `parameters?.maxKVSize` internally as the budget cap.
                    return makeAttentionCache(
                        parameters: parameters,
                        keep: 4,
                        affineStep: affineStep)
                }
            }
        }
    }
}

// MARK: - Mistral3 VLM Model

public class Mistral3VLM: Module, VLMModel, KVCacheDimensionProvider {
    // Use PixtralVision.VisionModel from Pixtral.swift
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Mistral3MultiModalProjector

    public let config: Mistral3VLMConfiguration
    let visionFeatureLayer: Int

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Mistral3VLMConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer

        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray?,
        pixelValues: MLXArray?,
        imageSizes: [(Int, Int)]?
    ) -> MLXArray {
        guard var pixelValues, let imageSizes else {
            guard let inputIds else {
                fatalError("Either inputIds or pixelValues must be provided")
            }
            return languageModel.embedTokens(inputIds)
        }

        guard let inputIds else {
            fatalError("inputIds required when pixelValues provided")
        }

        let inputsEmbeds = languageModel.embedTokens(inputIds)

        // Handle 3D pixel values (missing batch dimension)
        if pixelValues.ndim == 3 {
            pixelValues = pixelValues.expandedDimensions(axis: 0)
        }

        // Process through vision tower (reuses Pixtral vision model)
        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true
        )

        // Select features from specified layer
        guard let hiddenStates else {
            fatalError("Vision model must return hidden states")
        }

        let layerIndex =
            visionFeatureLayer < 0
            ? hiddenStates.count + visionFeatureLayer
            : visionFeatureLayer
        let selectedFeatures = hiddenStates[layerIndex]

        // Project to text space using Mistral3's patch merger projector
        let imageFeatures = multiModalProjector(selectedFeatures, imageSizes: imageSizes)

        // Merge embeddings
        return mergeInputIdsWithImageFeatures(
            imageTokenIndex: config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds
        )
    }

    private func mergeInputIdsWithImageFeatures(
        imageTokenIndex: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let (_, numImagePatches, _) = (
            imageFeatures.dim(0),
            imageFeatures.dim(1),
            imageFeatures.dim(2)
        )

        // Find image token positions (assuming batch size is 1)
        let inputIdArray: [Int32] = inputIds[0].asArray(Int32.self)
        let imagePositions = inputIdArray.enumerated().compactMap {
            $1 == Int32(imageTokenIndex) ? $0 : nil
        }

        // Validate that the number of image tokens matches the number of image patches
        guard imagePositions.count == numImagePatches else {
            fatalError(
                "Image token count (\(imagePositions.count)) does not match image patches (\(numImagePatches)). Ensure the processor adds exactly numImagePatches image tokens."
            )
        }

        // Build text segments - text before each image token
        var textSegments: [MLXArray] = []
        var startIdx = 0

        for position in imagePositions {
            textSegments.append(inputsEmbeds[0..., startIdx ..< position, 0...])
            startIdx = position + 1
        }

        // Split image features into separate embeddings for each image
        // imageFeatures shape: (numImages, numImagePatches, embedDim)
        // Split along axis 1 into numImagePatches parts (one per patch)
        let splitIndices = Array(1 ..< numImagePatches)
        let imageEmbeddings = MLX.split(imageFeatures, indices: splitIndices, axis: 1)

        // Interleave text and image embeddings
        // [text0, img0, text1, img1, ...]
        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text)
            finalEmbeddings.append(image)
        }

        // Add remaining text after the last image token
        finalEmbeddings.append(inputsEmbeds[0..., startIdx..., 0...])

        // Create a final embedding of shape
        // (1, num_image_patches*num_images + sequence_len, embed_dim)
        return MLX.concatenated(finalEmbeddings, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        // Extract image sizes from frames or fall back to config defaults
        let imageSizes: [(Int, Int)]?
        if let frames = input.image?.frames {
            imageSizes = frames.map { ($0.h, $0.w) }
        } else if pixelValues != nil {
            imageSizes = [(config.visionConfig.imageSize, config.visionConfig.imageSize)]
        } else {
            imageSizes = nil
        }

        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )

        let logits = languageModel(inputIds, cache: cache, inputsEmbeds: embeddings)

        // Mirror the Gemma 3 / Gemma 4 VLM prefill-sync barrier (issue #169 /
        // PR #172): hard `eval()` on cache + logits before returning
        // `.logits(...)` so the iterator's first decode forward doesn't read
        // pending K/V writes. Preempting the same class of bug here.
        var cacheArrays: [MLXArray] = []
        for c in cache {
            cacheArrays.append(contentsOf: c.innerState())
        }
        eval(cacheArrays + [logits])

        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Transform keys to match model structure
            // Vision tower keys: vision_tower.X -> vision_tower.vision_model.X (for pixtral structure)
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                // Alternative key format: model.vision_encoder.X -> vision_tower.vision_model.X
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "model.vision_encoder", with: "vision_tower.vision_model")
                }
            } else if key.contains("model.language_model") && !key.contains("language_model.model")
            {
                newKey = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.contains("lm_head") && !key.contains("language_model") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.contains("model.vision_projection") {
                newKey = key.replacingOccurrences(
                    of: "model.vision_projection", with: "multi_modal_projector")
            }

            // Skip rotary embeddings
            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }

            // Handle weight scale patterns
            if newKey.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = newKey.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[key.replacingOccurrences(of: "_scale_inv", with: "")] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if newKey.contains("activation_scale") {
                continue
            } else if newWeights[newKey] == nil {
                newWeights[newKey] = value
            }
        }

        return newWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

// MARK: - LoRA Support

extension Mistral3VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}

// MARK: - Processor Configuration

public struct Mistral3VLMProcessorConfiguration: Codable, Sendable {
    public let imageProcessor: ImageProcessorConfig
    public let imageToken: String
    public let imageBreakToken: String?
    public let imageEndToken: String?
    public let patchSize: Int
    public let spatialMergeSize: Int?

    public struct ImageProcessorConfig: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let size: ProcessorSize
        public let patchSize: Int
        public let doNormalize: Bool?
        public let doRescale: Bool?
        public let doResize: Bool?
        public let rescaleFactor: Float?

        public struct ProcessorSize: Codable, Sendable {
            public let width: Int?
            public let height: Int?
            public let longestEdge: Int?

            enum CodingKeys: String, CodingKey {
                case width
                case height
                case longestEdge = "longest_edge"
            }
        }

        public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }

        public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size
            case patchSize = "patch_size"
            case doNormalize = "do_normalize"
            case doRescale = "do_rescale"
            case doResize = "do_resize"
            case rescaleFactor = "rescale_factor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
        case imageBreakToken = "image_break_token"
        case imageEndToken = "image_end_token"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
    }

    /// Mlx-community's Pixtral / Mistral 3.1 VL repos ship a flat
    /// `preprocessor_config.json` (image_mean / image_std / size at top
    /// level) PLUS a separate `processor_config.json` (image tokens +
    /// patch_size). Hugging Face's processor_config nests image-processor
    /// fields under `image_processor`. Decode either shape: prefer the
    /// nested `image_processor` key, else read fields top-level.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // imageToken is required in either shape.
        self.imageToken = (try? c.decode(String.self, forKey: .imageToken)) ?? "[IMG]"
        self.imageBreakToken = try c.decodeIfPresent(String.self, forKey: .imageBreakToken)
        self.imageEndToken = try c.decodeIfPresent(String.self, forKey: .imageEndToken)
        self.patchSize = (try? c.decode(Int.self, forKey: .patchSize)) ?? 14
        self.spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize)

        if let nested = try? c.decode(ImageProcessorConfig.self, forKey: .imageProcessor) {
            self.imageProcessor = nested
        } else {
            // Decode the flat preprocessor_config.json shape (Pixtral
            // convention): image_mean / image_std / size / patch_size /
            // do_* / rescale_factor are top-level siblings of image_token.
            self.imageProcessor = try ImageProcessorConfig(from: decoder)
        }
    }
}

// MARK: - Message Generator for Mistral3 VLM

/// Message generator for Mistral3 VLM that creates structured messages with image placeholders
public struct Mistral3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        // For Mistral3 VLM, images come before text in the content
        [
            "role": message.role.rawValue,
            "content": message.images.map { _ in
                ["type": "image"]
            } + [["type": "text", "text": message.content]],
        ]
    }
}

// MARK: - Processor

public struct Mistral3VLMProcessor: UserInputProcessor {
    private let config: Mistral3VLMProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageToken: String
    private let imageTokenId: Int

    private struct PreprocessResult {
        let pixels: MLXArray  // BCHW
        let frames: [THW]
        let numImageTokens: Int
    }

    public init(_ config: Mistral3VLMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageToken = config.imageToken
        // Get image token ID from tokenizer, fallback to 10 (default for Mistral3)
        if let vocabTokenId = tokenizer.convertTokenToId(config.imageToken) {
            self.imageTokenId = vocabTokenId
        } else {
            self.imageTokenId = 10
        }
    }

    private func preprocessImage(
        _ image: CIImage,
        processing: UserInput.Processing?,
        patchSize: Int,
        spatialMergeSize: Int,
        longestEdge: Int?
    ) throws -> PreprocessResult {
        var image = MediaProcessing.inSRGBToneCurveSpace(image)
        image = MediaProcessing.apply(image, processing: processing)

        let maxVisionEdge = patchSize * 24  // Pixtral vision expects 24x24 patches (336px for patchSize=14)
        let targetEdge = min(longestEdge ?? maxVisionEdge, maxVisionEdge)

        let originalSize = image.extent.size
        let scale = min(CGFloat(targetEdge) / max(originalSize.width, originalSize.height), 1.0)
        let newWidth = max(1, Int((originalSize.width * scale).rounded()))
        let newHeight = max(1, Int((originalSize.height * scale).rounded()))

        // Round to patch size multiples for padding
        let paddedWidth = ((newWidth + patchSize - 1) / patchSize) * patchSize
        let paddedHeight = ((newHeight + patchSize - 1) / patchSize) * patchSize

        // Resize
        image = MediaProcessing.resampleBicubic(
            image,
            to: CGSize(width: newWidth, height: newHeight)
        )

        // Pad to patch boundaries (bottom-right padding)
        if newWidth != paddedWidth || newHeight != paddedHeight {
            let background = CIImage(color: .black).cropped(
                to: CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight))
            let tx = 0.0
            let ty = CGFloat(paddedHeight - newHeight)
            let transformed = image.transformed(by: CGAffineTransform(translationX: tx, y: ty))
            image = transformed.composited(over: background)
        }

        image = MediaProcessing.normalize(
            image,
            mean: config.imageProcessor.imageMeanTuple,
            std: config.imageProcessor.imageStdTuple
        )

        var pixels = MediaProcessing.asMLXArray(image)

        if pixels.ndim == 2 {
            pixels = pixels.expandedDimensions(axis: -1)
        }
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }
        // Convert to BCHW format for vision model
        if pixels.dim(-1) == 3 {
            pixels = pixels.transposed(0, 3, 1, 2)
        }

        // Calculate number of image tokens needed after spatial merging
        let numPatchesH = paddedHeight / patchSize
        let numPatchesW = paddedWidth / patchSize
        let mergedPatchesH = numPatchesH / spatialMergeSize
        let mergedPatchesW = numPatchesW / spatialMergeSize
        let numImageTokens = mergedPatchesH * mergedPatchesW

        return PreprocessResult(
            pixels: pixels,
            frames: [THW(1, paddedHeight, paddedWidth)],
            numImageTokens: numImageTokens
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Generate structured messages using the message generator
        let messages = Mistral3MessageGenerator().generate(from: input)

        if input.images.isEmpty {
            // No image - just apply chat template
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }
        let spatialMergeSize = config.spatialMergeSize ?? 2
        let patchSize = config.imageProcessor.patchSize

        // Apply chat template to get tokenized prompt with image placeholder
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Decode to find and replace image placeholder token
        let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)

        // Process image to get dimensions
        let preprocessResult = try preprocessImage(
            input.images[0].asCIImage(),
            processing: input.processing,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            longestEdge: config.imageProcessor.size.longestEdge
        )

        // Replace the image placeholder token with the correct number of image tokens
        // The chat template should have inserted the imageToken (e.g., "[IMG]") which we need to expand
        if decoded.contains(imageToken) {
            // Split by image token and re-encode with expanded image tokens
            let pieces = decoded.components(separatedBy: imageToken)
            var expandedTokens: [Int] = []

            for (index, piece) in pieces.enumerated() {
                if !piece.isEmpty {
                    let pieceTokens = tokenizer.encode(text: piece)
                    expandedTokens.append(contentsOf: pieceTokens)
                }
                // Add image tokens between pieces (not after the last one)
                if index < pieces.count - 1 {
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                }
            }
            promptTokens = expandedTokens
        } else {
            // Fallback: If no image token placeholder found, try to find and replace the single image token ID
            // or insert at the beginning after BOS
            var foundImageToken = false
            var expandedTokens: [Int] = []

            for token in promptTokens {
                if token == imageTokenId && !foundImageToken {
                    // Replace single image token with expanded tokens
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                    foundImageToken = true
                } else {
                    expandedTokens.append(token)
                }
            }

            if foundImageToken {
                promptTokens = expandedTokens
            } else {
                // Last resort: insert image tokens after BOS (if present) or at start
                var insertIndex = 0
                if !promptTokens.isEmpty && promptTokens[0] == 1 {
                    insertIndex = 1  // After BOS token
                }
                promptTokens.insert(
                    contentsOf: Array(
                        repeating: imageTokenId, count: preprocessResult.numImageTokens),
                    at: insertIndex
                )
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: preprocessResult.pixels, frames: preprocessResult.frames)
        )
    }
}
