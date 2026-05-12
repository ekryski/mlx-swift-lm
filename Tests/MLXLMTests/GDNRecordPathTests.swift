// Copyright © 2026 Apple Inc.
//
// Direct test of `gatedDeltaUpdateRecord` end-to-end with the GDN
// recording session active. Reproduces what
// `Qwen35GatedDeltaNet.update` does on a speculative-decoder verify
// forward, but standalone — no model load, no iterator, no async stream.
//
// Goal: isolate the "SmallVector out of range" crash that ngram-spot
// hits on Qwen 3.5 9B. Either the layer's record path itself crashes
// (and we get a real Swift stack trace), or it succeeds (proving the
// bug is iterator-side, in the prime-the-pump prefill or some other
// path that runs before the layer is called).

import Foundation
import MLX
@testable import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite("GDN record-path direct test (Qwen 3.5 9B shapes)")
struct GDNRecordPathTests {

    // Realistic Qwen 3.5 9B GDN dimensions, per the model's config.json:
    //   linear_key_head_dim   = 128 → Dk
    //   linear_value_head_dim = 128 → Dv
    //   linear_num_key_heads  = 16  → Hk
    //   linear_num_value_heads = 32 → Hv
    static let B = 1
    static let T = 3            // verify-window size for n=3 D=2 (y + 2 drafts)
    static let Hk = 16
    static let Hv = 32
    static let Dk = 128
    static let Dv = 128

    /// Make a deterministic small-int input array of the given shape and dtype.
    static func makeInput(shape: [Int], seed: Float, dtype: DType = .bfloat16) -> MLXArray {
        let total = shape.reduce(1, *)
        let values = (0..<total).map { Float($0 + 1) * seed * 0.001 }
        return MLXArray(values).reshaped(shape).asType(dtype)
    }

    @Test("gatedDeltaUpdateRecord runs without crash on Qwen 3.5 9B shapes")
    func recordPathSurvives() {
        // Inputs match the call-site contract in `Qwen35.swift:317`:
        //   q, k:   [B, T, Hk, Dk]   bf16 (post-rmsNorm)
        //   v:      [B, T, Hv, Dv]   bf16
        //   a, b:   [B, T, Hv]       bf16
        //   aLog:   [Hv]             bf16
        //   dtBias: [Hv]             bf16
        //   state:  [B, Hv, Dv, Dk]  fp32 (forced by gatedDeltaUpdateRecord)
        let q = Self.makeInput(shape: [Self.B, Self.T, Self.Hk, Self.Dk], seed: 1.0)
        let k = Self.makeInput(shape: [Self.B, Self.T, Self.Hk, Self.Dk], seed: 2.0)
        let v = Self.makeInput(shape: [Self.B, Self.T, Self.Hv, Self.Dv], seed: 3.0)
        let a = Self.makeInput(shape: [Self.B, Self.T, Self.Hv], seed: 0.5)
        let b = Self.makeInput(shape: [Self.B, Self.T, Self.Hv], seed: 0.5)
        let aLog = Self.makeInput(shape: [Self.Hv], seed: -1.0)
        let dtBias = Self.makeInput(shape: [Self.Hv], seed: 0.0)

        // Cache: SSMStateCache with `beginRecord()` called so isRecording == true.
        let cache = SSMStateCache()
        // Seed slot 0 (conv state) and slot 1 (recurrent state) — both as
        // they'd look after a real prefill. Slot 1 is fp32 [B,Hv,Dv,Dk]
        // (the dtype `gatedDeltaUpdate` forces).
        cache[0] = MLXArray.zeros([Self.B, 3, Self.Hk * Self.Dk + 2 * Self.Hv * Self.Dv])
        cache[1] = MLXArray.zeros([Self.B, Self.Hv, Self.Dv, Self.Dk], dtype: .float32)
        cache.beginRecord()
        #expect(cache.isRecording == true)

        // Run the record-path dispatcher — this is the exact call the
        // GDN layer makes on a verify forward when the iterator has
        // called `beginCacheRecord`.
        let (out, newState) = gatedDeltaUpdateRecord(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: cache[1], mask: nil, cache: cache)

        // Force eval — any deferred kernel error surfaces here.
        eval(out)
        eval(newState)

        #expect(out.shape == [Self.B, Self.T, Self.Hv, Self.Dv])
        #expect(newState.shape == [Self.B, Self.Hv, Self.Dv, Self.Dk])
    }

    @Test("gatedDeltaUpdateRecord with state == nil (cold start)")
    func recordPathColdStart() {
        // Same shapes but no pre-existing state in the cache —
        // exercises the `state ?? MLXArray.zeros(...)` fallback.
        let q = Self.makeInput(shape: [Self.B, Self.T, Self.Hk, Self.Dk], seed: 1.0)
        let k = Self.makeInput(shape: [Self.B, Self.T, Self.Hk, Self.Dk], seed: 2.0)
        let v = Self.makeInput(shape: [Self.B, Self.T, Self.Hv, Self.Dv], seed: 3.0)
        let a = Self.makeInput(shape: [Self.B, Self.T, Self.Hv], seed: 0.5)
        let b = Self.makeInput(shape: [Self.B, Self.T, Self.Hv], seed: 0.5)
        let aLog = Self.makeInput(shape: [Self.Hv], seed: -1.0)
        let dtBias = Self.makeInput(shape: [Self.Hv], seed: 0.0)

        let cache = SSMStateCache()
        cache.beginRecord()

        let (out, newState) = gatedDeltaUpdateRecord(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: nil, mask: nil, cache: cache)

        eval(out)
        eval(newState)

        #expect(out.shape == [Self.B, Self.T, Self.Hv, Self.Dv])
        #expect(newState.shape == [Self.B, Self.Hv, Self.Dv, Self.Dk])
    }
}
