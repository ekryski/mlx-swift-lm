// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Prefix-KV-cache routing helpers (spec 017 phase 1B)
//
// `prefixCacheRoute(...)` is the entry-point shim that:
//   1. computes a ``PrefixKey`` for the request,
//   2. attempts a longest-prefix lookup in ``PrefixKVCache/shared``
//      (and ``PrefixKVCacheDisk/shared`` when phase-4 disk is on),
//   3. on hit, hydrates a fresh `[KVCache]` from the snapshot and
//      rewrites the request `LMInput` to the suffix tokens,
//   4. exposes ``PrefixCacheRouteState/wrapStreamForSnapshot(_:cache:)``
//      so the caller can wrap their result stream — on stream
//      completion we capture a snapshot of the cache at the stable-
//      prefix boundary and insert it.
//
// Failure modes are silently absorbed: hydrate/serialise throws drop
// us back onto the plain uncached generate path. The user-visible
// behaviour is "prefix cache helped or did nothing", never "prefix
// cache broke generation".

/// Routing state returned from ``prefixCacheRoute(input:cache:parameters:model:)``.
public struct PrefixCacheRouteState: @unchecked Sendable {
    /// The rewritten LMInput to feed the iterator. When the prefix
    /// cache was hit, this is the **suffix** input (matched tokens
    /// stripped). When not hit, this is the original `input` verbatim.
    public let input: LMInput

    /// The hydrated cache, or nil if the prefix cache was not engaged
    /// (or was engaged but missed). Caller falls back to its existing
    /// cache argument.
    public let cache: [KVCache]?

    /// True when prefix-cache routing is active. False on opt-out paths
    /// (env not set, parameters disabled, no model id).
    public let shouldHydrate: Bool

    /// Optional snapshotter closure. Captured at route-construction
    /// time so the caller can fire it once the iterator's cache has
    /// completed prefill (passing the live iterator cache in).
    fileprivate let snapshotter: ((_ liveCache: [KVCache]) -> Void)?

    /// Wrap an AsyncStream<Generation> so that on stream completion we
    /// snapshot the live cache. When this state has no snapshotter
    /// (e.g. cache disabled), the stream is returned verbatim — no
    /// allocation, no Task.
    public func wrapStreamForSnapshot(
        _ stream: AsyncStream<Generation>, cache: [KVCache]
    ) -> AsyncStream<Generation> {
        guard let snapshotter else { return stream }
        // Box non-Sendable references so the forwarder Task can capture
        // them without tripping the strict-concurrency sender checks.
        // `[KVCache]` is intentionally non-Sendable (MLXArray graph
        // mutation), and the SendableBox + sole-consumer pattern is the
        // codebase's established escape hatch (see `generateLoopTask`).
        let cacheBox = SendableBox(cache)
        let snapBox = SendableBox(snapshotter)
        return AsyncStream<Generation> { continuation in
            let forwarder = Task {
                var sawInfo = false
                for await event in stream {
                    if case .info = event { sawInfo = true }
                    continuation.yield(event)
                }
                // Snapshot only if the generation actually reached the
                // `.info` finish event — protects against early break /
                // cancellation paths leaving the cache mid-write.
                if sawInfo {
                    let liveCache = cacheBox.consume()
                    let snap = snapBox.consume()
                    snap(liveCache)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                forwarder.cancel()
            }
        }
    }
}

/// Compute prefix-cache routing for one generate(...) call.
///
/// - Parameters:
///   - input: the request's input as supplied to generate(...).
///   - cache: the caller-supplied cache (nil = model.newCache(...)).
///   - parameters: the generate parameters. `prefixCacheEnabled`
///     gates everything; when false, this returns a pass-through
///     state.
///   - model: the language model — used to allocate a fresh cache
///     when hydrating from a snapshot.
public func prefixCacheRoute(
    input: LMInput, cache: [KVCache]?,
    parameters: GenerateParameters, model: any LanguageModel
) -> PrefixCacheRouteState {
    // Env override: `MLX_PREFIX_CACHE=1` forces on; `MLX_PREFIX_CACHE=0`
    // forces off; unset honors the parameter field.
    let env = ProcessInfo.processInfo.environment
    let enabled: Bool = {
        if let v = env["MLX_PREFIX_CACHE"] {
            return v == "1"
        }
        return parameters.prefixCacheEnabled
    }()
    if !enabled {
        return PrefixCacheRouteState(
            input: input, cache: nil, shouldHydrate: false, snapshotter: nil)
    }

    let diskEnabled: Bool = {
        if let v = env["MLX_PREFIX_CACHE_DISK"] {
            return v == "1"
        }
        return parameters.prefixCacheDiskEnabled
    }()

    // Resolve identity key. modelID can come from parameters (preferred)
    // or fall back to a placeholder — single-model setups can leave it
    // unset, multi-model setups must populate it.
    let modelID = parameters.prefixCacheModelID ?? "unspecified"
    let promptTokens = tokensArrayFromLMInput(input)
    guard !promptTokens.isEmpty else {
        return PrefixCacheRouteState(
            input: input, cache: nil, shouldHydrate: false, snapshotter: nil)
    }

    // Configuration-key inference. We need a concrete cache to read
    // `storageKind` from; if the caller supplied one use it, otherwise
    // allocate a probe.
    let probeForKey = cache ?? model.newCache(parameters: parameters)
    let key = prefixKey(forCache: probeForKey, modelID: modelID)

    // L1 lookup. If we miss and L2 is enabled, fall through to disk;
    // promote disk hits back into L1 so the next request short-circuits.
    var lookupResult: PrefixCacheLookupResult? = nil
    do {
        lookupResult = try PrefixKVCache.shared.lookup(prefix: promptTokens, key: key)
    } catch {
        // Closed cache or schema mismatch — surface as miss.
        lookupResult = nil
    }
    if lookupResult == nil, diskEnabled {
        if let snap = try? PrefixKVCacheDisk.shared.lookup(prefix: promptTokens, key: key) {
            // Promote to L1. Failure to promote is silent — the snapshot
            // is still usable on this request.
            try? PrefixKVCache.shared.insert(snap)
            lookupResult = PrefixCacheLookupResult(
                matchedLength: snap.tokens.count, snapshot: snap)
        }
    }

    // Build the snapshotter closure. We capture key + policy + tokens
    // so the post-stream hook has everything it needs without re-reading
    // mutable state.
    let policy: any StablePrefixPolicy =
        parameters.prefixCachePolicy ?? IdentityPolicy()
    let debug = env["MLX_PREFIX_CACHE_DEBUG"] == "1"
    let snapshotter: (([KVCache]) -> Void) = { liveCache in
        // Determine stable-prefix length. Snapshot only what the policy
        // says is stable — for chat templates this excludes the trailing
        // assistant-opener tokens that regenerate next turn.
        let stableLen = policy.stablePrefixLen(promptTokens)
        guard stableLen > 0, stableLen <= promptTokens.count else {
            if debug { print("[PREFIX-CACHE-DEBUG] skip: stableLen out of range (\(stableLen)/\(promptTokens.count))") }
            return
        }
        // Trim the cache back to the stable-prefix boundary. For chat
        // workloads the cache holds (prompt + generated response); we
        // want to snapshot at the stable prefix point of the prompt.
        // For pure-attention (trimmable) caches we trim; for hybrid we
        // best-effort trim the trimmable layers only.
        //
        // For hybrid models (Qwen 3.5 / 3.6 GDN+attention) the first
        // layer can be an SSM whose `offset` stays at 0 — read the
        // offset from any trimmable layer instead. Falls back to the
        // first layer's offset for pure-attention stacks.
        let currentOffset =
            liveCache.first(where: { $0.isTrimmable })?.offset
            ?? liveCache.first?.offset
            ?? 0
        if currentOffset > stableLen {
            if canTrimPromptCache(liveCache) {
                trimPromptCache(liveCache, numTokens: currentOffset - stableLen)
            } else if canRollbackPromptCache(liveCache) {
                // Mixed stack — best-effort trim of the trimmable
                // layers. Non-trimmable layers retain their (post-
                // generation) state; the per-layer SSM snapshot is
                // semantically stale but the attention layers still
                // accelerate the next turn's prefill, and the insert
                // invariant explicitly skips SSM-kind layers.
                for c in liveCache where c.isTrimmable {
                    c.trim(currentOffset - stableLen)
                }
            } else {
                // The cache has wrapped (sliding-window layers past
                // their window size) or contains a non-trim, non-
                // state-replay cache class. Either way the prefix
                // tokens are no longer recoverable from current state.
                // Common case: GPT-OSS-20B's interleaved sliding-window
                // layers at window=128 — once generation pushes offset
                // past 128 the prefix is gone. This is an honest
                // "no cache benefit" outcome for that model family.
                if debug {
                    let summary = liveCache.enumerated().map { (i, c) in
                        "L\(i)=\(type(of: c))(off=\(c.offset),trim=\(c.isTrimmable))"
                    }.joined(separator: " ")
                    print("[PREFIX-CACHE-DEBUG] skip wrapped/non-trimmable: \(summary)")
                }
                return
            }
        }
        // Use the policy-trimmed token list for the snapshot.
        let snapshotTokens = Array(promptTokens.prefix(stableLen))
        do {
            let snap = try serialisePrefixSnapshot(
                cache: liveCache, tokens: snapshotTokens, key: key)
            try PrefixKVCache.shared.insert(snap)
            if debug {
                print("[PREFIX-CACHE-DEBUG] inserted tokens=\(snap.tokens.count) bytes=\(snap.byteSize)")
            }
            if diskEnabled {
                try PrefixKVCacheDisk.shared.write(snap)
            }
        } catch {
            if debug { print("[PREFIX-CACHE-DEBUG] insert/serialise failed: \(error)") }
            // Silent in non-debug — a snapshot failure doesn't affect
            // the user's generation result.
        }
    }

    guard let hit = lookupResult, hit.matchedLength > 0,
          hit.matchedLength <= promptTokens.count else {
        return PrefixCacheRouteState(
            input: input, cache: nil, shouldHydrate: true, snapshotter: snapshotter)
    }

    // Hit — hydrate a fresh cache from the snapshot. The caller's cache
    // (if any) is discarded; this mirrors how `cache ?? model.newCache(...)`
    // works in the uncached path.
    let freshCache = model.newCache(parameters: parameters)
    do {
        try hydratePrefixSnapshot(hit.snapshot, into: freshCache)
    } catch {
        // Hydrate failure → fall back to uncached path. Snapshotter is
        // still attached so we still capture a snapshot post-generate.
        return PrefixCacheRouteState(
            input: input, cache: nil, shouldHydrate: true, snapshotter: snapshotter)
    }

    // Rewrite input to the suffix tokens. If the hit covers the entire
    // prompt we still need at least one token to drive the next forward,
    // so we leave the last matched token in place (the iterator's
    // prefill will compute one token, which becomes the first decode
    // step's prior).
    let suffixInput: LMInput
    if hit.matchedLength == promptTokens.count {
        // Exact hit. Drop matched tokens down to the last one so the
        // iterator has something to forward.
        let lastToken = promptTokens.last ?? 0
        suffixInput = makeLMInput(input: input, tokens: [lastToken])
        // Note: we also need to step the cache back by 1, because the
        // hydrated cache has matched_length tokens in it but the
        // iterator will write at offset = matched_length when it
        // forwards the last token. Trim the trailing token off.
        for c in freshCache where c.isTrimmable {
            c.trim(1)
        }
    } else {
        let suffix = Array(promptTokens[hit.matchedLength...])
        suffixInput = makeLMInput(input: input, tokens: suffix)
    }

    return PrefixCacheRouteState(
        input: suffixInput, cache: freshCache,
        shouldHydrate: true, snapshotter: snapshotter)
}

// MARK: - LMInput helpers

/// Read prompt tokens out of an ``LMInput`` as a plain Swift `[Int]`.
func tokensArrayFromLMInput(_ input: LMInput) -> [Int] {
    let arr = input.text.tokens
    if arr.ndim == 0 { return [] }
    let int32s = arr.reshaped(arr.size).asArray(Int32.self)
    return int32s.map { Int($0) }
}

/// Construct a new ``LMInput`` from an existing one but with a rewritten
/// token list. Preserves vision payloads (image / video) so VLM call
/// sites continue to work when the prefix cache strips text tokens.
func makeLMInput(input: LMInput, tokens: [Int]) -> LMInput {
    let arr = MLXArray(tokens.map { Int32($0) })
    let text = LMInput.Text(tokens: arr, mask: nil)
    return LMInput(text: text, image: input.image, video: input.video)
}
