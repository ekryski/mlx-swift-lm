//
//  QwenPrefillBridge.swift
//  mlx-swift-lm
//
//  Qwen-family dedicated native prefill bridge.
//  Wraps the C API from prefill_bridge_qwen.h for use by Qwen2, Qwen3, and Qwen3MoE.
//  Separated from GenericPrefillBridge to allow Qwen-specific tuning
//  (eval barriers, MoE dispatch) without risking regressions on other models.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NativePrefillBridge

final class QwenPrefillBridge {
    static let shared = QwenPrefillBridge()

    private var initializedModelType: String? = nil

    func ensureInitialized(modelType: String, model: Module, config configJSON: String) -> Bool {
        if initializedModelType == modelType { return true }

        // If switching model types, cleanup first
        if initializedModelType != nil {
            qwen_cleanup()
            initializedModelType = nil
        }

        let rc = configJSON.withCString { qwen_init($0) }
        if rc != 0 { print("[QwenPrefill] init failed"); return false }

        let params = model.parameters().flattened()
        var weightCount = 0
        for (key, arr) in params {
            let bridgeKey = "model." + key
            let rawPtr = arr.ctx.ctx
            let status = bridgeKey.withCString { cKey in
                qwen_set_weight(cKey, rawPtr!)
            }
            if status == 0 { weightCount += 1 }
        }
        print("[QwenPrefill] Passed \(weightCount) weights")

        let finRC = qwen_finalize()
        if finRC != 0 { print("[QwenPrefill] finalize failed"); return false }

        initializedModelType = modelType
        print("[QwenPrefill] Initialized for \(modelType)")

        // Warmup: run a dummy forward pass to materialize all weight GPU buffers
        var warmMs: Double = 0
        let warmTokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let _ = qwen_run(warmTokens.ctx.ctx!, &warmMs)
        print(String(format: "[QwenPrefill] Pre-warmed in %.0fms", warmMs))

        return true
    }

    func runAndInjectKV(tokenArray: MLXArray, cache: [KVCache], numLayers: Int) -> (Double, Bool) {
        guard initializedModelType != nil else { return (0, false) }

        let tokens2d = tokenArray.dim(0) == 1 ? tokenArray : tokenArray.reshaped(1, tokenArray.size)
        var ms: Double = 0
        let rc = qwen_run(tokens2d.ctx.ctx!, &ms)
        if rc != 0 { return (0, false) }

        for i in 0..<min(numLayers, cache.count) {
            guard let kPtr = qwen_get_k_ptr(Int32(i)),
                  let vPtr = qwen_get_v_ptr(Int32(i)) else {
                return (ms, false)
            }
            let kArr = MLXArray.fromCppArray(kPtr).contiguous()
            let vArr = MLXArray.fromCppArray(vPtr).contiguous()

            let _ = cache[i].update(keys: kArr, values: vArr)
        }

        return (ms, true)
    }
}
