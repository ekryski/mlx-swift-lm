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
    /// `UInt64` either way). Also captures the wall-clock start time
    /// for the in-process CPU aggregator (active under
    /// `MLX_BENCH_PROFILE >= 2`).
    public struct IntervalHandle {
        public let id: OSSignpostID
        public let name: StaticString
        public let startTime: CFAbsoluteTime
    }

    // ── In-process CPU wall-clock aggregator
    //
    // Why this exists: gives a per-phase breakdown printable to stdout
    // (`[MLX-PROFILE]` table at end of each bench cell) without
    // requiring Instruments / xctrace setup. Set `MLX_BENCH_PROFILE=2`
    // and run the bench — the table appears below the `[PROFILE]`
    // lifecycle block. Useful for quick A/B comparisons in the
    // terminal where attaching Instruments would be overkill.
    //
    // What it captures: a per-label (count, totalNs) accumulator over
    // every `begin`/`end` pair fired during the run, populated only
    // when `enabled` (`MLX_BENCH_PROFILE >= 2`). The handle returned
    // from `begin()` carries the wall-clock start time; `end()`
    // computes elapsed and updates the counter under a lock.
    //
    // What it does NOT capture: GPU execution time. CPU `begin`/`end`
    // brackets the dispatch + synchronous prep, not the async GPU
    // work the kernel does after the dispatch returns. For decode
    // phases that mostly queue Metal kernels, the CPU-side numbers
    // are a *lower bound* on actual phase work — useful for ranking
    // phases by activity and comparing dispatch overhead between
    // configurations, not for absolute GPU attribution. For accurate
    // GPU per-phase timing, capture the trace under Metal System
    // Trace + Points of Interest in Instruments and correlate
    // signpost intervals with the GPU command buffer execution
    // windows. See `benchmarks/README.md` for the xctrace recipe.

    nonisolated(unsafe) private static var aggregator: [String: (count: Int, totalNs: UInt64)] = [:]
    private static let aggregatorLock = NSLock()

    private enum PadAlign { case left, right }
    private static func pad(_ s: String, _ w: Int, _ align: PadAlign = .left) -> String {
        if s.count >= w { return s }
        let p = String(repeating: " ", count: w - s.count)
        return align == .left ? s + p : p + s
    }

    /// Print + reset the per-label aggregator to stdout. No-op when
    /// the aggregator is empty.
    public static func dumpAggregator() {
        aggregatorLock.lock()
        defer { aggregatorLock.unlock() }
        guard !aggregator.isEmpty else { return }
        let totalNs: UInt64 = aggregator.values.reduce(0) { $0 + $1.totalNs }
        let totalMs = Double(totalNs) / 1_000_000.0
        print("[MLX-PROFILE] CPU wall-clock aggregator (in-process; not GPU time):")
        print("[MLX-PROFILE]   \(pad("label", 14)) \(pad("count", 10, .right)) \(pad("total(ms)", 12, .right)) \(pad("%", 7, .right)) \(pad("avg(µs)", 10, .right))")
        for (label, agg) in aggregator.sorted(by: { $0.value.totalNs > $1.value.totalNs }) {
            let ms = Double(agg.totalNs) / 1_000_000.0
            let pct = totalNs > 0 ? Double(agg.totalNs) / Double(totalNs) * 100 : 0
            let avgUs = agg.count > 0 ? Double(agg.totalNs) / Double(agg.count) / 1_000.0 : 0
            let lbl = pad(label, 14)
            let c = pad("\(agg.count)", 10, .right)
            let t = pad(String(format: "%.2f", ms), 12, .right)
            let p = pad(String(format: "%.1f%%", pct), 7, .right)
            let a = pad(String(format: "%.2f", avgUs), 10, .right)
            print("[MLX-PROFILE]   \(lbl) \(c) \(t) \(p) \(a)")
        }
        let totalCount = aggregator.values.reduce(0) { $0 + $1.count }
        print(String(format: "[MLX-PROFILE]   total %.2f ms across %d intervals", totalMs, totalCount))
        aggregator.removeAll()
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
        return IntervalHandle(id: id, name: label, startTime: enabled ? CFAbsoluteTimeGetCurrent() : 0)
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
        if enabled {
            let elapsedNs = UInt64(max(0, (CFAbsoluteTimeGetCurrent() - handle.startTime) * 1e9))
            let key = String(describing: handle.name)
            aggregatorLock.lock()
            let prev = aggregator[key] ?? (count: 0, totalNs: 0)
            aggregator[key] = (count: prev.count + 1, totalNs: prev.totalNs + elapsedNs)
            aggregatorLock.unlock()
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
