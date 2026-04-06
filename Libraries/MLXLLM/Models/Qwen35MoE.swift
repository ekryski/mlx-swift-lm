//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen35Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Qwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen35TextConfiguration(from: decoder)
        }
    }
}

public class Qwen35MoEModel: Qwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            newWeights[key] = value
        }

        for l in 0 ..< languageModel.configuration.hiddenLayers {
            let prefix = "language_model.model.layers.\(l).mlp"

            // Handle TWO cases for expert weights:
            //
            // Case A: Non-quantized model — safetensor has fused experts.gate_up_proj
            //   Remap and split into switch_mlp.gate_up_proj.weight
            //
            // Case B: Pre-quantized model (e.g., mlx-community 4-bit) — safetensor already
            //   has split switch_mlp.gate_proj.weight + switch_mlp.up_proj.weight.
            //   Concatenate them into switch_mlp.gate_up_proj.weight for FusedGateUpSwitchGLU.

            let gateUpKey = "\(prefix).experts.gate_up_proj"
            if let gateUp = newWeights[gateUpKey] {
                // Case A: fused experts.gate_up_proj from non-quantized model
                newWeights[gateUpKey] = nil
                newWeights["\(prefix).switch_mlp.gate_up_proj.weight"] = gateUp
            }

            // Handle experts.down_proj → switch_mlp.down_proj.weight (non-quantized)
            if let downProj = newWeights["\(prefix).experts.down_proj"] {
                newWeights["\(prefix).experts.down_proj"] = nil
                newWeights["\(prefix).switch_mlp.down_proj.weight"] = downProj
            }

            // Case B: Pre-quantized model with already-split gate_proj + up_proj.
            // Concatenate back into fused gate_up_proj for FusedGateUpSwitchGLU.
            for suffix in [".weight", ".scales", ".biases"] {
                let gateKey = "\(prefix).switch_mlp.gate_proj\(suffix)"
                let upKey = "\(prefix).switch_mlp.up_proj\(suffix)"
                if let gateVal = newWeights[gateKey], let upVal = newWeights[upKey] {
                    newWeights[gateKey] = nil
                    newWeights[upKey] = nil
                    // Concatenate along outputDims axis (dim -2 for weights, dim -2 for scales)
                    newWeights["\(prefix).switch_mlp.gate_up_proj\(suffix)"] =
                        MLX.concatenated([gateVal, upVal], axis: -2)
                }
            }
        }

        // Morton order experiment: reorder expert weights for cache locality.
        // MOE_EXPERT_ORDER=shuffle → random permutation (A/B test baseline)
        // MOE_EXPERT_ORDER=morton → Z-order interleaving (future)
        // Default: no reordering
        if let orderMode = ProcessInfo.processInfo.environment["MOE_EXPERT_ORDER"],
           orderMode == "shuffle"
        {
            let numExperts = languageModel.configuration.numExperts
            guard numExperts > 0 else { return languageModel.sanitize(weights: newWeights) }

            // Generate a random permutation of expert indices
            let perm = (0..<numExperts).shuffled()
            let permArray = MLXArray(perm.map { Int32($0) })
            print("[MOE-REORDER] Shuffling \(numExperts) experts for cache locality A/B test")

            for l in 0 ..< languageModel.configuration.hiddenLayers {
                let prefix = "language_model.model.layers.\(l).mlp"

                // Reorder gate weight columns: gate.weight is [hiddenSize, numExperts]
                // or [numExperts, hiddenSize] depending on quantization
                for suffix in [".weight", ".scales", ".biases"] {
                    let gateKey = "\(prefix).gate\(suffix)"
                    if let gateW = newWeights[gateKey] {
                        // Gate weight: reorder along the expert axis (last axis for weight,
                        // which corresponds to output dimension = numExperts)
                        // For quantized: weight is [numExperts, packedHidden], reorder axis 0
                        newWeights[gateKey] = gateW[permArray]
                    }
                }

                // Reorder expert weights: shape [numExperts, ...]
                for proj in ["gate_up_proj", "down_proj"] {
                    for suffix in [".weight", ".scales", ".biases"] {
                        let key = "\(prefix).switch_mlp.\(proj)\(suffix)"
                        if let w = newWeights[key] {
                            newWeights[key] = w[permArray]
                        }
                    }
                }
            }
        }

        return languageModel.sanitize(weights: newWeights)
    }
}
