//
//  LFM2.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/7/12.
//
//  Layer stack lifted into MLXLMCommon.LFM2 namespace during issue #168
//  consolidation pass (2026-05-06). This file owns only the LLM-side
//  outer model wrapper.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Back-compat alias. The configuration moved into `LFM2.Configuration`
/// in MLXLMCommon during the issue #168 consolidation pass; existing
/// callers that referenced `LFM2Configuration` continue to compile.
public typealias LFM2Configuration = LFM2.Configuration

public class LFM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM2.ModelInner
    let configuration: LFM2.Configuration

    public init(_ args: LFM2.Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            args.fullAttnIdxs.contains(layerIdx) ? args.kvHeads : 0
        }
        self.model = LFM2.ModelInner(args)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (name, param) in weights {
            var p = param
            // Conv1d weight layout adjustment: HF-format is [out, kernel, in],
            // MLX expects [out, in, kernel] for grouped conv.
            if name.contains("conv.weight"),
                p.shape[p.shape.count - 1] > p.dim(1)
            {
                p = p.transposed(0, 2, 1)
            }
            sanitized[name] = p
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layerIdx in
            if configuration.fullAttnIdxs.contains(layerIdx) {
                makeAttentionCache(parameters: parameters)
            } else {
                SSMStateCache()
            }
        }
    }
}

extension LFM2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
