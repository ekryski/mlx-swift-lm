//
//  GemmaPrefillBridge.swift
//  mlx-swift-lm
//
//  Gemma 4 native prefill bridge using dlopen/dlsym.
//  Loads libprefill_bridge_gemma.dylib at runtime to avoid SPM LTO
//  miscompilation. The dylib shares the host MLX allocator via
//  -undefined dynamic_lookup.
//
//  Opt-in: set NATIVE_PREFILL=1 environment variable.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - C function pointer types

// int gemma_init(int, int, int, int, int, int)
private typealias GemmaInitFn = @convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int32

// int gemma_set_weight(const char*, void*)
private typealias GemmaSetWeightFn = @convention(c) (UnsafePointer<CChar>, UnsafeMutableRawPointer) -> Int32

// int gemma_finalize(void)
private typealias GemmaFinalizeFn = @convention(c) () -> Int32

// int gemma_run_array(void*, double*, float*)
private typealias GemmaRunArrayFn = @convention(c) (
    UnsafeMutableRawPointer, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Float>
) -> Int32

// void gemma_get_kv_handles(int, void**, void**)
private typealias GemmaGetKVHandlesFn = @convention(c) (
    Int32, UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> Void

// void gemma_cleanup(void)
private typealias GemmaCleanupFn = @convention(c) () -> Void

// MARK: - Bridge

final class GemmaPrefillBridge: @unchecked Sendable {
    nonisolated(unsafe) static let shared = GemmaPrefillBridge()

    private var handle: UnsafeMutableRawPointer?
    private var initialized = false

    // Resolved function pointers
    private var fnInit: GemmaInitFn?
    private var fnSetWeight: GemmaSetWeightFn?
    private var fnFinalize: GemmaFinalizeFn?
    private var fnRunArray: GemmaRunArrayFn?
    private var fnGetKVHandles: GemmaGetKVHandlesFn?
    private var fnCleanup: GemmaCleanupFn?

    /// Whether the native prefill bridge is available and opted-in
    var isEnabled: Bool {
        ProcessInfo.processInfo.environment["NATIVE_PREFILL"] == "1"
    }

    /// Load the dylib and resolve all symbols. Returns false on failure.
    private func loadLibrary() -> Bool {
        if handle != nil { return true }

        // Search paths: next to the binary, .build release dir, Sources dir
        let candidates = [
            "libprefill_bridge_gemma.dylib",
            Bundle.main.bundlePath + "/libprefill_bridge_gemma.dylib",
            Bundle.main.bundlePath + "/../libprefill_bridge_gemma.dylib",
        ]

        for path in candidates {
            if let h = dlopen(path, RTLD_NOW | RTLD_LOCAL) {
                handle = h
                break
            }
        }

        guard let h = handle else {
            print("[pb2] Failed to load libprefill_bridge_gemma.dylib")
            return false
        }

        // Resolve symbols
        fnInit = unsafeBitCast(dlsym(h, "gemma_init"), to: GemmaInitFn.self)
        fnSetWeight = unsafeBitCast(dlsym(h, "gemma_set_weight"), to: GemmaSetWeightFn.self)
        fnFinalize = unsafeBitCast(dlsym(h, "gemma_finalize"), to: GemmaFinalizeFn.self)
        fnRunArray = unsafeBitCast(dlsym(h, "gemma_run_array"), to: GemmaRunArrayFn.self)
        fnGetKVHandles = unsafeBitCast(
            dlsym(h, "gemma_get_kv_handles"), to: GemmaGetKVHandlesFn.self)
        fnCleanup = unsafeBitCast(dlsym(h, "gemma_cleanup"), to: GemmaCleanupFn.self)

        // Verify all symbols resolved
        guard fnInit != nil, fnSetWeight != nil, fnFinalize != nil,
            fnRunArray != nil, fnGetKVHandles != nil, fnCleanup != nil
        else {
            print("[pb2] Failed to resolve all symbols from bridge dylib")
            dlclose(h)
            handle = nil
            return false
        }

        print("[pb2] Loaded libprefill_bridge_gemma.dylib")
        return true
    }

    /// Initialize the bridge with model config and weights.
    func ensureInitialized(config: Gemma4TextConfiguration, model: Module) -> Bool {
        if initialized { return true }
        guard loadLibrary() else { return false }

        // TODO: derive sliding_window_pattern from config.layerTypes instead of hardcoding
        let rc = fnInit!(
            Int32(config.hiddenLayers),
            Int32(config.hiddenSize),
            Int32(config.attentionHeads),
            Int32(config.kvHeads),
            Int32(config.slidingWindow),
            3  // sliding_window_pattern: every 5th layer is full attention
        )
        if rc != 0 {
            print("[pb2] init failed with code \(rc)")
            return false
        }

        // Pass all model weights to the C++ bridge
        let params = model.parameters().flattened()
        var weightCount = 0
        for (key, arr) in params {
            let bridgeKey = "model." + key
            guard let rawPtr = arr.ctx.ctx else { continue }
            let status = bridgeKey.withCString { cKey in
                fnSetWeight!(cKey, rawPtr)
            }
            if status == 0 { weightCount += 1 }
        }
        print("[pb2] Passed \(weightCount) weights to bridge")

        let finRC = fnFinalize!()
        if finRC != 0 {
            print("[pb2] finalize failed with code \(finRC)")
            return false
        }

        initialized = true
        print("[pb2] Bridge initialized for gemma4_text")
        return true
    }

    /// Run native prefill and inject K/V into Swift caches.
    /// Returns (elapsed_ms, success).
    func runAndInjectKV(tokenArray: MLXArray, cache: [KVCache], numLayers: Int) -> (Double, Bool) {
        guard initialized else { return (0, false) }

        let tokens2d = tokenArray.dim(0) == 1 ? tokenArray : tokenArray.reshaped(1, tokenArray.size)
        var ms: Double = 0
        var checksum: Float = 0

        guard let rawPtr = tokens2d.ctx.ctx else { return (0, false) }
        let rc = fnRunArray!(rawPtr, &ms, &checksum)
        if rc != 0 { return (0, false) }

        // Inject K/V from bridge into Swift caches
        for i in 0..<min(numLayers, cache.count) {
            var kPtr: UnsafeMutableRawPointer?
            var vPtr: UnsafeMutableRawPointer?
            fnGetKVHandles!(Int32(i), &kPtr, &vPtr)

            guard let k = kPtr, let v = vPtr else {
                return (ms, false)
            }

            let kArr = MLXArray.fromCppArray(k).contiguous()
            let vArr = MLXArray.fromCppArray(v).contiguous()
            let _ = cache[i].update(keys: kArr, values: vArr)
        }

        return (ms, true)
    }

    /// Cleanup bridge state
    func cleanup() {
        if initialized {
            fnCleanup?()
            initialized = false
        }
    }

    deinit {
        cleanup()
        if let h = handle {
            dlclose(h)
        }
    }
}
