// Copyright © 2026 Apple Inc.

import Foundation
import os

/// `os_signpost`-based lifecycle tracing for the inference benchmark.
///
/// When enabled, emits begin/end signpost intervals at every notable
/// phase — model load, prompt prep, prefill, each decode step, each
/// sampler step — into the `.pointsOfInterest` category of a dedicated
/// subsystem (`ai.mlx.bench`). These show up as a labelled track in
/// Instruments' "Points of Interest" instrument, so you can overlay
/// them on CPU samples (Time Profiler) and GPU traces (Metal System
/// Trace) to see where CPU/GPU time is spent at each stage.
///
/// ## Runtime cost
///
/// Each `signpostID` + begin/end pair is ~40ns on M-series silicon.
/// When `signpostsEnabled` is false (no attached Instruments session)
/// the kernel path short-circuits after a single flag check — no
/// buffer writes, no string formatting. Cost in the "profile
/// attached" case is dominated by Instruments' own kernel-to-user
/// buffer copies, not by the app. For 200-token decode loops the
/// total overhead is well under 50 µs — imperceptible against the
/// ~4 s total.
///
/// ## How to capture
///
/// ### Option A — Attach Instruments interactively
///
/// 1. Run the bench with the profile level set:
///    ```
///    MLX_BENCH_PROFILE=2 MLX_BENCH_MODEL=gpt-oss-20b \
///      MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
///      swift test --skip-build -c release --filter benchmark
///    ```
/// 2. In Instruments, choose the "Points of Interest" template (or
///    any template — "Time Profiler" + "Metal System Trace" + "Points
///    of Interest" is the ideal combo).
/// 3. Select the `mlx-swift-lmPackageTests` process as the target and
///    start recording before the bench enters its hot loop.
///
/// ### Option B — xctrace (headless, saved to .trace)
///
/// ```
/// xctrace record --template 'Time Profiler' \
///    --output profile.trace \
///    --attach mlx-swift-lmPackageTests
/// ```
///
/// Then open `profile.trace` in Instruments to inspect. Filter by
/// subsystem `ai.mlx.bench` to isolate the benchmark signposts from
/// system noise.
///
/// ## What gets traced
///
/// See `PhaseLabel` for the full list. The per-decode-step signpost
/// fires once per generated token with metadata `(token_idx, token_id,
/// tokens_generated_so_far)` — enough to correlate individual tokens
/// with the CPU samples and GPU command buffers that produced them.
public enum BenchmarkSignpost {

    /// Fixed subsystem identifier. Filter on this in Instruments to
    /// isolate benchmark-specific signposts.
    public static let subsystem = "ai.mlx.bench"

    /// Gated activation — only true when `MLX_BENCH_PROFILE >= 2`.
    /// Checking this once at init and storing into an `OSLog` that is
    /// either `.pointsOfInterest` (active) or `.disabled` (no-op) lets
    /// the signpost API's own fast path handle zero-cost gating.
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_BENCH_PROFILE"],
              let level = Int(raw) else {
            return false
        }
        return level >= 2
    }()

    /// Shared log handle. Resolves to `.disabled` when the env gate
    /// is off — all signpost calls against a disabled log are no-ops
    /// inside the OS and never cross into the kernel.
    public static let log: OSLog = {
        if enabled {
            return OSLog(subsystem: subsystem, category: .pointsOfInterest)
        } else {
            return .disabled
        }
    }()

    /// Stable category names for each phase. Instruments groups
    /// signpost intervals by name, so phases with the same label
    /// stack vertically in the timeline. Typed as `StaticString`
    /// because `os_signpost` requires a compile-time name.
    public enum PhaseLabel {
        // ── Lifecycle phases (emitted from `Tests/Benchmarks/InferenceBenchmark.swift`)
        public static let modelLoad:       StaticString = "model_load"
        public static let promptPrep:      StaticString = "prompt_prep"
        public static let prefill:         StaticString = "prefill"
        public static let decodeStep:      StaticString = "decode_step"
        public static let sampler:         StaticString = "sampler"
        public static let kldBaseline:     StaticString = "kld_baseline"
        public static let kldForcedDecode: StaticString = "kld_forced_decode"

        // ── Attention sub-phases (emitted from `attentionWithCacheUpdate`).
        // Apply to all KV cache variants (KVCacheSimple, RotatingKVCache,
        // QuantizedKVCacheProtocol, TurboQuant A path). Lets us compare
        // phase breakdown across cache types on the same axes.
        public static let kvUpdate:  StaticString = "kv_update"   // cache.update / updateAndDequant / updateQuantized
        public static let sdpa:      StaticString = "sdpa"        // MLXFast.scaledDotProductAttention
        public static let qsdpa:     StaticString = "qsdpa"       // quantizedScaledDotProductAttention (affine path)

        // ── TurboQuant B path (`compressedAttention`) phase intervals.
        // Nest inside `decode_step` (alongside `kv_update`/`sdpa` for
        // other cache types). Correlate with MLX's per-kernel-dispatch
        // signposts (subsystem `ai.mlx.metal`) in Metal System Trace to
        // attribute GPU time per phase.
        public static let tqEncode:  StaticString = "tq_encode"   // encodeNewToken
        public static let tqScore:   StaticString = "tq_score"    // Q*K (matmul or compressed mseScore)
        public static let tqSoftmax: StaticString = "tq_softmax"  // softmax over scores (separated path)
        public static let tqValue:   StaticString = "tq_value"    // Attn*V (TurboFlash or mseWeightedSum)
        public static let tqRotate:  StaticString = "tq_rotate"   // Q rotation + inverse value rotation
    }

    /// Interval handle — returned by `begin`, consumed by `end`.
    /// Wraps an `OSSignpostID` so the call sites read cleanly and
    /// nothing leaks when signposts are disabled (the ID is just a
    /// `UInt64` either way).
    public struct IntervalHandle {
        public let id: OSSignpostID
        public let name: StaticString
    }

    /// Begin a labelled interval. `label` must be a `StaticString`
    /// because `os_signpost` takes one at compile time — rejecting
    /// dynamic format strings keeps the kernel path branch-free.
    /// Use the constants in `PhaseLabel` to get compile-time strings.
    @inline(__always)
    public static func begin(
        _ label: StaticString,
        metadata: String = ""
    ) -> IntervalHandle {
        let id = OSSignpostID(log: log)
        if metadata.isEmpty {
            os_signpost(.begin, log: log, name: label, signpostID: id)
        } else {
            os_signpost(.begin, log: log, name: label, signpostID: id, "%{public}s", metadata)
        }
        return IntervalHandle(id: id, name: label)
    }

    /// End a previously-begun interval. No-ops safely when
    /// signposts are disabled — `handle.id` is still valid, and the
    /// disabled log's `.end` emission short-circuits.
    @inline(__always)
    public static func end(
        _ handle: IntervalHandle,
        metadata: String = ""
    ) {
        if metadata.isEmpty {
            os_signpost(.end, log: log, name: handle.name, signpostID: handle.id)
        } else {
            os_signpost(.end, log: log, name: handle.name, signpostID: handle.id, "%{public}s", metadata)
        }
    }

    /// Scoped convenience wrapper — `begin` + `end` around a closure
    /// so the call site reads as a block.
    ///
    /// ```swift
    /// try BenchmarkSignpost.interval(PhaseLabel.prefill) {
    ///     try await model.prefill(...)
    /// }
    /// ```
    @inline(__always)
    @discardableResult
    public static func interval<T>(
        _ label: StaticString,
        metadata: String = "",
        _ body: () throws -> T
    ) rethrows -> T {
        let h = begin(label, metadata: metadata)
        defer { end(h) }
        return try body()
    }

    /// Async variant.
    @inline(__always)
    @discardableResult
    public static func intervalAsync<T>(
        _ label: StaticString,
        metadata: String = "",
        _ body: () async throws -> T
    ) async rethrows -> T {
        let h = begin(label, metadata: metadata)
        defer { end(h) }
        return try await body()
    }

    /// Emit a single-point event (no interval) — useful for
    /// boundaries that don't naturally span a closure (e.g. first
    /// token arrival).
    @inline(__always)
    public static func event(
        _ label: StaticString,
        metadata: String = ""
    ) {
        if metadata.isEmpty {
            os_signpost(.event, log: log, name: label)
        } else {
            os_signpost(.event, log: log, name: label, "%{public}s", metadata)
        }
    }
}
