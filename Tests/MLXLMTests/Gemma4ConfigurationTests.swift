// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLLM
import XCTest

/// Tests for `Gemma4TextConfiguration` decoding. The Gemma 4 family ships
/// HF configs in two shapes:
///
///   1. **VLM-style nested**: top-level config has a `text_config` object
///      that owns the language-model fields (used when the model has
///      vision/audio towers).
///   2. **Text-only flat**: top-level config IS the text config (used by
///      E2B / E4B / 31B text-only checkpoints).
///
/// The configuration's custom `init(from:)` handles both. This test locks
/// that dual-path behaviour so a future refactor (e.g. one that replaces
/// the standalone Gemma4TextModel with a wrapper-based pattern) doesn't
/// silently regress text-only loads.
public class Gemma4ConfigurationTests: XCTestCase {

    func testFlatConfigDecoding() throws {
        // E2B / E4B / 31B style: text fields at top level, no `text_config` wrapper.
        let json =
            """
            {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "num_key_value_heads": 8,
                "sliding_window": 1024,
                "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 256
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.modelType, "gemma4_text")
        XCTAssertEqual(config.hiddenSize, 2816)
        XCTAssertEqual(config.hiddenLayers, 30)
        XCTAssertEqual(config.attentionHeads, 16)
        XCTAssertEqual(config.kvHeads, 8)
        // globalKvHeads defaults to kvHeads when not present.
        XCTAssertEqual(config.globalKvHeads, 8)
        XCTAssertEqual(config.layerTypes.count, 3)
        XCTAssertTrue(config.tieWordEmbeddings)
    }

    func testNestedTextConfigDecoding() throws {
        // VLM style: text fields wrapped in `text_config`. The configuration's
        // custom decoder must reach into `text_config` rather than the top
        // level. (Vision-tower fields above are ignored — they belong to a
        // different config type in the VLM tree.)
        let json =
            """
            {
                "model_type": "gemma4",
                "vision_config": { "model_type": "siglip" },
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 5376,
                    "num_hidden_layers": 62,
                    "intermediate_size": 21504,
                    "moe_intermediate_size": 5376,
                    "num_attention_heads": 32,
                    "head_dim": 128,
                    "global_head_dim": 256,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262144,
                    "num_key_value_heads": 16,
                    "num_global_key_value_heads": 8,
                    "sliding_window": 4096,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": false,
                    "enable_moe_block": true,
                    "num_experts": 8,
                    "top_k_experts": 2,
                    "hidden_size_per_layer_input": 256
                }
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.modelType, "gemma4_text")
        XCTAssertEqual(config.hiddenSize, 5376)
        XCTAssertEqual(config.hiddenLayers, 62)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 16)
        // Distinct globalKvHeads when explicitly set in the nested block.
        XCTAssertEqual(config.globalKvHeads, 8)
        XCTAssertTrue(config.enableMoeBlock)
        XCTAssertEqual(config.numExperts, 8)
        XCTAssertEqual(config.topKExperts, 2)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testRopeParametersNested() throws {
        // Real Gemma 4 configs ship RoPE thetas inside a `rope_parameters`
        // object keyed by attention-mode (`sliding_attention`,
        // `full_attention`). Verify the decoder pulls them out correctly.
        let json =
            """
            {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 1,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "num_key_value_heads": 8,
                "sliding_window": 1024,
                "layer_types": ["sliding_attention"],
                "rope_parameters": {
                    "sliding_attention": { "rope_theta": 10000.0 },
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.5
                    }
                },
                "hidden_size_per_layer_input": 256
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertEqual(config.globalRopeTheta, 1_000_000.0)
        XCTAssertEqual(config.partialRotaryFactor, 0.5)
    }
}
