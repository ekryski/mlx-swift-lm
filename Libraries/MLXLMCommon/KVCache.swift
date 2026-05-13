// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = StandardKVCache()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = AffineQuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
public protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Read the current cached keys and values without modifying the cache.
    /// Used by KV sharing (Gemma 4) where shared layers read a donor's cached K/V.
    /// Returns nil if the cache is empty.
    func peek() -> (MLXArray, MLXArray)?

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Actual memory footprint in bytes of all arrays held by this cache.
    /// Computed directly from stored array shapes and dtypes — not from MLX memory pool.
    var memoryBytes: Int { get }

    /// Create an attention mask for this cache
    ///
    /// This method encapsulates cache-specific mask creation logic. Implementations should handle offset capping, window size logic,
    /// and optimization decisions (symbolic vs array masks).
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    /// Create an independent deep copy of this cache.
    func copy() -> any KVCache

    /// Whether this cache is a KV-sharing donor (other layers read its K/V).
    /// Donor caches must NOT be converted to compressed formats that return
    /// rotated/transformed K/V, because shared layers expect raw fp16 data.
    var isDonor: Bool { get set }

    /// Reflects what the cache currently holds. Self-transitioning caches
    /// (`AffineQuantizedKVCache`, `TurboQuantizedKVCache`) report their
    /// post-transition state, which may differ from how they were constructed.
    /// Used by `AttentionUtils.attentionWithCacheUpdate` to dispatch without
    /// `as?` downcasts. See `KVStorageKind` in `KVCacheTypes.swift`.
    var storageKind: KVStorageKind { get }
}

// QuantizedKVCacheProtocol was removed in spec 006 PR 2. The only quantized
// cache type today is `AffineQuantizedKVCache`; external dispatch is via
// `cache.storageKind == .affineQuantized(...)` (see `KVStorageKind`) or a
// direct `as? AffineQuantizedKVCache` downcast when the concrete-class
// methods (groupSize / bits / mode / updateQuantized / getQuantizedState)
// are needed.

/// Compute the byte size of an MLXArray from its shape and dtype.
func arrayBytes(_ array: MLXArray?) -> Int {
    guard let array else { return 0 }
    let elements = array.shape.reduce(1, *)
    return elements * array.dtype.bytesPerElement
}

extension DType {
    var bytesPerElement: Int {
        switch self {
        case .bool: return 1
        case .uint8, .int8: return 1
        case .uint16, .int16, .float16, .bfloat16: return 2
        case .uint32, .int32, .float32: return 4
        case .uint64, .int64: return 8
        case .float64: return 8
        case .complex64: return 8
        // `DType` is a public non-frozen enum in mlx-swift; future cases
        // (e.g. fp8 / int4 ML quant types) should not silently break this
        // call site. Default to 4 bytes as a reasonable conservative
        // fallback — `@unknown default` is the Swift idiom for keeping the
        // switch exhaustive at compile time while remaining resilient if
        // a newer mlx-swift binary introduces additional cases at link
        // time. Update this site explicitly when adding support for any
        // new dtype.
        @unknown default: return 4
        }
    }
}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }
    public var isDonor: Bool = false

    public func innerState() -> [MLXArray] { [] }

    /// Default: sum bytes of all state arrays. Subclasses should override for accuracy.
    open var memoryBytes: Int {
        state.reduce(0) { $0 + arrayBytes($1) }
    }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func peek() -> (MLXArray, MLXArray)? {
        return nil  // Subclasses override to return their stored K/V
    }

    open func copy() -> any KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    /// Default implementation for caches without special mask requirements
    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For single token, no mask needed
        if n == 1 {
            return .none
        }

        // For multi-token sequences
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }

    /// Default storage kind. Subclasses override to report their actual
    /// (post-transition) storage state.
    open var storageKind: KVStorageKind { .raw }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    return mask
}

/// Create an attention mask matching mlx-lm's create_attention_mask helper.
///
/// This returns `.causal` when a symbolic mask is sufficient, avoiding
/// materializing a full mask array.
public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also `MultiHeadAttention.createAdditiveCausalMask(_:dtype:)` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

@available(
    *, deprecated,
    message: "Use createAttentionMask(h:cache:windowSize:returnArray:) with a single cache instead"
)
public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize - 1, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

/// Create an attention mask with explicit window size parameter.
///
/// - Parameters:
///   - h: The input array (used to determine sequence length)
///   - cache: Optional single KV cache
///   - windowSize: Optional sliding window size (if provided, creates windowed attention)
///   - returnArray: Force return of array mask instead of symbolic "causal"
/// - Returns: Attention mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's makeMask if available
    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    // Fallback for no cache
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: SSMStateCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

/// Standard raw-FP16/BF16 KV cache with two eviction strategies:
/// `.unbounded` grows linearly (the legacy `StandardKVCache` shape), `.window` rotates
/// in-place with optional sink tokens (the legacy `StandardKVCache` shape).
///
/// Renamed + consolidated in spec 006 (2026-05-04). Typealiases keep the old
/// names alive for one release: `StandardKVCache = StandardKVCache` (default-init
/// produces an unbounded cache) and `StandardKVCache = StandardKVCache` (the
/// `init(maxSize:keep:step:)` convenience init produces a windowed cache).
public class StandardKVCache: BaseKVCache, CustomDebugStringConvertible {
    /// Eviction strategy. `private(set)` because `metaState` may need to update
    /// the window size / keep on persistence load.
    public private(set) var eviction: KVEviction

    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step: Int

    /// Last K/V returned from `update()` — used by Gemma 4 KV sharing to avoid
    /// redundant `peek()` slice ops for donor layers.
    public var lastReturnedKeys: MLXArray?
    public var lastReturnedValues: MLXArray?

    /// Window-only state. Dormant when `eviction == .unbounded`.
    private var idx: Int = 0
    /// True once a write has overwritten a slot that previously held a
    /// different sequence position — i.e. the rotating ring buffer has
    /// completed a full cycle. From that point on, buffer position N no
    /// longer maps onto sequence position N, so a `state` capture is
    /// not a faithful prefix snapshot.
    ///
    /// The prefix-cache snapshotter consults this to refuse a snapshot
    /// of a rotated buffer (`serialiseStandard`). Surfaced as an issue
    /// on GPT-OSS-20B (`sliding_window = 128`) and Gemma 4 windowed
    /// layers when the prompt length matches the window exactly: the
    /// first decode step rotates position 0 onto sequence token
    /// `maxSize`, corrupting the prefix's K/V. `trim(...)` only
    /// decrements `offset`; it does not undo rotation, so an
    /// `offset <= maxSize` check alone is insufficient.
    ///
    /// Always `false` for `eviction == .unbounded`.
    public private(set) var hasWrappedRotatingBuffer: Bool = false
    /// Optional first-allocation size hint set via ``reserve(_:)``. Window-only.
    /// When set, the first write allocates `[B, kvHeads, hint, headDim]` upfront
    /// instead of growing in `step`-sized chunks. Eliminates the per-chunk
    /// `concatenated` transient surge for callers that know their workload
    /// size up front (most generation: `prompt + maxTokens`). Capped to the
    /// window size; `nil` falls back to step-based growth.
    private var initialAllocSize: Int?

    /// Window-only convenience accessor. Returns 0 for unbounded eviction.
    private var keep: Int {
        if case .window(_, let k) = eviction { return k } else { return 0 }
    }

    /// Window-only convenience accessor. Returns `Int.max` for unbounded eviction.
    /// Internal callers that need the raw window size should pattern-match `eviction`.
    private var maxCacheSize: Int {
        if case .window(let size, _) = eviction { return size } else { return Int.max }
    }

    public override var maxSize: Int? {
        if case .window(let size, _) = eviction { return size } else { return nil }
    }

    /// Default unbounded init — equivalent to legacy `StandardKVCache()`.
    public override init() {
        self.eviction = .unbounded
        self.step = 256
        super.init()
    }

    /// Eviction-typed init. Use this from the `makeKVCache` factory.
    public init(eviction: KVEviction, step: Int = 256) {
        self.eviction = eviction
        self.step = step
        super.init()
    }

    /// Window convenience init — equivalent to legacy `StandardKVCache(maxSize:keep:step:)`.
    public convenience init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.init(eviction: .window(size: maxSize, keep: keep), step: step)
    }

    /// **Opt-in** workload-size hint. Triggers a single up-front allocation on
    /// the first write instead of step-incremental growth, eliminating the
    /// per-chunk `concatenated` transient that the lazy path produces during
    /// multi-token prefill.
    ///
    /// No-op on unbounded caches (the unbounded growth path already allocates
    /// in `step`-multiples without a per-chunk transient).
    ///
    /// - Idempotent: only takes effect before the cache has been written
    ///   to. After first write, this is a no-op.
    /// - Cap: clamped to the window size. The cache can still grow past the
    ///   hint up to the window size if the workload exceeds it.
    /// - Floor: never allocates less than `step` (small hints are rounded up).
    public func reserve(_ size: Int) {
        guard self.keys == nil else { return }
        guard size > 0 else { return }
        guard case .window(let maxSize, _) = eviction else { return }
        initialAllocSize = max(step, min(size, maxSize))
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    // MARK: - Update dispatch

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        switch eviction {
        case .unbounded:
            return updateUnbounded(keys: keys, values: values)
        case .window:
            // When `reserve` was called, route multi-token writes through
            // `updateInPlace` too. Its first allocation is already sized for the
            // workload, so writes go straight into the pre-allocated buffer
            // without the `updateConcat` per-chunk `concatenated` surge that
            // compounds at B>1 long-context prefill. Without `reserve`, keep the
            // legacy split: single-token decode is in-place, multi-token prefill
            // builds the buffer via `updateConcat`.
            let useInPlace = keys.dim(2) == 1 || initialAllocSize != nil
            let result =
                if useInPlace {
                    updateInPlace(keys: keys, values: values)
                } else {
                    updateConcat(keys: keys, values: values)
                }
            self.lastReturnedKeys = result.0
            self.lastReturnedValues = result.1
            return result
        }
    }

    private func updateUnbounded(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
                // Materialize to break the lazy graph chain — without this,
                // each resize keeps all prior arrays alive in the compute graph,
                // causing GPU memory to grow monotonically with context length.
                eval(self.keys!, self.values!)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        self.lastReturnedKeys = returnedKeys
        self.lastReturnedValues = returnedValues

        return (returnedKeys, returnedValues)
    }

    // MARK: - Window-mode helpers (active when eviction == .window)

    private func windowTrim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let maxCacheSize = self.maxCacheSize
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            // Allow temporary cache growth during multi-token processing (e.g., prompt prefill).
            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context.
            //
            // Note: this means the actual memory ceiling during a multi-token
            // (prefill) write is `maxCacheSize + S - 1`, not `maxCacheSize`. With
            // a typical prefill chunk size of 1024 the overshoot is 1023 tokens
            // (per-layer); negligible at sane chunk sizes but worth keeping in
            // mind when sizing a wired-memory ticket. The single-token decode
            // path (`updateInPlace`) enforces `maxCacheSize` strictly via its
            // rotation logic.
            let trimSize = idx - maxCacheSize + 1
            // `windowTrim(trimSize > 0, ...)` drops sequence positions from
            // the front of the buffer to keep the window bound — those
            // tokens are gone, so the snapshot's "buffer position N ==
            // sequence position N" invariant no longer holds. Mark the
            // buffer wrapped so the prefix-cache snapshotter refuses it.
            if trimSize > 0 {
                hasWrappedRotatingBuffer = true
            }
            self.keys = windowTrim(trimSize: trimSize, self.keys!, append: keys)
            self.values = windowTrim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let maxCacheSize = self.maxCacheSize
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // Allocate or grow when:
        //   * The buffer hasn't been created yet, OR
        //   * The incoming write doesn't fit in the current buffer and we
        //     haven't reached `maxCacheSize`.
        // The growth condition uses `prev + S` rather than just `prev` so
        // multi-token writes (S > 1, e.g. prefill chunks) trigger growth
        // when needed — the legacy `prev >= buffer.dim(2)` check only
        // worked for single-token decode writes.
        if self.keys == nil
            || (prev + S > self.keys!.dim(2) && self.keys!.dim(2) < maxCacheSize)
        {
            // First-time allocation respects the `reserve` hint when set, but
            // never falls below `S` (the incoming write must fit). Subsequent
            // growth uses `step` and slots between buffer-end and maxCacheSize.
            let newSize: Int
            if self.keys == nil {
                let target = initialAllocSize.map { max($0, S) } ?? max(step, S)
                newSize = min(target, maxCacheSize - prev)
            } else {
                newSize = min(max(step, S), maxCacheSize - self.keys!.dim(2))
            }

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = windowTrim(trimSize: trimSize, self.keys!)
            self.values = windowTrim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
            idx = keep
            hasWrappedRotatingBuffer = true
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice
        if offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    // MARK: - peek

    public override func peek() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        switch eviction {
        case .unbounded:
            return (keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...])
        case .window:
            // Used by KV sharing (e.g., Gemma 4) where shared layers read a donor's cached K/V.
            let orderedKeys = temporalOrder(keys)
            let orderedValues = temporalOrder(values)
            let len = min(offset, orderedKeys.dim(2))
            return (
                orderedKeys[.ellipsis, ..<len, 0...],
                orderedValues[.ellipsis, ..<len, 0...]
            )
        }
    }

    // MARK: - Persistence

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            switch eviction {
            case .unbounded:
                if offset == keys.dim(2) {
                    return [keys, values]
                } else {
                    return [
                        keys[.ellipsis, ..<offset, 0...],
                        values[.ellipsis, ..<offset, 0...],
                    ]
                }
            case .window:
                if offset < keys.dim(2) {
                    return [
                        keys[.ellipsis, ..<offset, 0...],
                        values[.ellipsis, ..<offset, 0...],
                    ]
                } else {
                    return [keys, values]
                }
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("StandardKVCache state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Legacy parity: StandardKVCache's state setter sets offset from keys.dim(2);
            // StandardKVCache's setter does NOT (offset is restored via metaState).
            if case .unbounded = eviction {
                self.offset = self.keys!.dim(2)
            }
        }
    }

    public override var metaState: [String] {
        get {
            switch eviction {
            case .unbounded:
                // StandardKVCache inherited BaseKVCache's [""] default — preserve.
                return [""]
            case .window(let size, let keep):
                return [String(keep), String(size), String(step), String(offset), String(idx)]
            }
        }
        set {
            switch newValue.count {
            case 0:
                // No-op
                break
            case 1:
                // Unbounded — nothing to restore (BaseKVCache's [""] default).
                break
            case 5:
                // Windowed (legacy StandardKVCache shape).
                guard let keepVal = Int(newValue[0]),
                    let stepVal = Int(newValue[2]),
                    let offsetVal = Int(newValue[3]),
                    let idxVal = Int(newValue[4])
                else {
                    fatalError("Failed to convert metaState values to integers")
                }
                if newValue[1] == "None" {
                    fatalError(
                        "StandardKVCache window mode requires a non-nil maxSize. Cannot load cache with maxSize=None.")
                }
                guard let maxSizeVal = Int(newValue[1]) else {
                    fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
                }
                self.eviction = .window(size: maxSizeVal, keep: keepVal)
                self.step = stepVal
                self.offset = offsetVal
                self.idx = idxVal
            default:
                fatalError(
                    "StandardKVCache metaState must have 1 or 5 values, got \(newValue.count)")
            }
        }
    }

    // MARK: - Trim + mask

    public override var isTrimmable: Bool {
        switch eviction {
        case .unbounded: return true
        case .window(let size, _): return offset < size
        }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        if case .window = eviction {
            idx -= trimmed
        }
        return trimmed
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard case .window(let maxCacheSize, _) = eviction else {
            // Unbounded falls back to BaseKVCache's default makeMask.
            return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }

        // Window-mode optimized mask creation with offset capping.
        if n > 1 {
            let actualWindowSize = windowSize ?? maxCacheSize
            let cappedOffset = min(maxCacheSize - 1, offset)

            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single-token decode.
            guard let windowSize = windowSize else { return .none }

            // May need a mask when window_size < max_size and cache has wrapped.
            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }
                let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
                let rolledMask = roll(mask, shift: currentIdx + 1)
                return .array(rolledMask)
            }
            return .none
        }
    }

    // MARK: - Memory + copy + debug

    override public var memoryBytes: Int {
        arrayBytes(keys) + arrayBytes(values)
    }

    public override func copy() -> any KVCache {
        let new = StandardKVCache(eviction: eviction, step: step)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        if case .window = eviction {
            // Window mode: also restore idx + offset from metaState.
            new.metaState = self.metaState
        }
        return new
    }

    public var debugDescription: String {
        switch eviction {
        case .unbounded:
            return
                "\(String(describing: Self.self))(unbounded) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
        case .window(let size, let keep):
            return
                "\(String(describing: Self.self))(window: \(size), keep: \(keep)) offset: \(offset), step: \(step), idx: \(idx), keys: \(keys?.shape.description ?? "-")"
        }
    }

    // MARK: - Conversion to quantized variants

    /// Convert to TurboQuant compressed cache.
    ///
    /// For window-mode caches the K/V are reordered into temporal sequence first
    /// (TurboQuant's WHT rotation operates on the actual token order, not the
    /// circular buffer order).
    public func toTurboQuantized(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil
    ) -> TurboQuantizedKVCache {
        let turboCache = TurboQuantizedKVCache(bits: bits, keyBits: keyBits, valueBits: valueBits)
        guard let keys = self.keys, let values = self.values, offset > 0 else {
            return turboCache
        }

        let orderedKeys: MLXArray
        let orderedValues: MLXArray
        if case .window = eviction {
            orderedKeys = temporalOrder(keys)
            orderedValues = temporalOrder(values)
        } else {
            orderedKeys = keys
            orderedValues = values
        }
        let len = min(offset, orderedKeys.dim(2))

        turboCache.loadRawKV(
            keys: orderedKeys[.ellipsis, ..<len, 0...],
            values: orderedValues[.ellipsis, ..<len, 0...]
        )
        return turboCache
    }

    /// Convert to affine-quantized cache. For window-mode caches, K/V are
    /// reordered into temporal sequence first so quantization group boundaries
    /// align with the actual token order.
    ///
    /// The returned ``AffineQuantizedKVCache`` does NOT rotate — it grows
    /// linearly. At 4-bit quantization the per-token footprint is ~4× smaller,
    /// so the effective memory bound is comparable to the original window.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> AffineQuantizedKVCache {
        let quantizedCache = AffineQuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset

        guard let keys = self.keys, let values = self.values else {
            return quantizedCache
        }

        let orderedKeys: MLXArray
        let orderedValues: MLXArray
        if case .window = eviction {
            orderedKeys = temporalOrder(keys)
            orderedValues = temporalOrder(values)
        } else {
            orderedKeys = keys
            orderedValues = values
        }
        let len = min(offset, orderedKeys.dim(2))
        let currentKeys = orderedKeys[.ellipsis, ..<len, 0...]
        let currentValues = orderedValues[.ellipsis, ..<len, 0...]

        let quantizedKeys = quantized(currentKeys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(currentValues, groupSize: groupSize, bits: bits)

        quantizedCache.state = [
            quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
            quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
        ].compactMap { $0 }
        return quantizedCache
    }
}

/// Affine-quantized KV cache (group quantization via MLX) for memory efficiency.
///
/// Renamed from `AffineQuantizedKVCache` in spec 006 (2026-05-04) for symmetry with
/// `TurboQuantizedKVCache`. The typealias `AffineQuantizedKVCache = AffineQuantizedKVCache`
/// is kept for one release.
public class AffineQuantizedKVCache: BaseKVCache {
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    /// SDPA strategy for this cache (spec 041 phases 1+2+4).
    ///
    /// - `.auto` (default): route `L > 1` prefill through the flash path
    ///   (dequant K/V to FP16 → `MLXFast.scaledDotProductAttention`) and
    ///   `L = 1` decode through the discrete path
    ///   (`quantizedMM → softmax → quantizedMM`). Best of both: flash
    ///   avoids the `[B, H, L, T]` score-tensor materialisation at long
    ///   prefill; discrete avoids per-step K/V dequant at decode.
    /// - `.flash`: force the flash path everywhere. Wins on small-model
    ///   long-context shapes where score materialisation dominates and
    ///   the dequant transient is acceptable.
    /// - `.discrete`: force the discrete path everywhere. Useful for
    ///   A/B regression checks against the legacy materialise-scores
    ///   shape.
    ///
    /// Env var `MLX_AFFINE_SDPA={auto,flash,discrete}` overrides this
    /// per-cache value globally when set — for diagnostics across a
    /// whole run without rebuilding cache instances.
    public let sdpaStrategy: AffineSDPAStrategy

    /// - Parameters:
    ///   - groupSize: Affine quantization group size (default 64).
    ///   - bits: Quantization bit-width (default 8).
    ///   - mode: Affine vs. MXFP4 quantization mode.
    ///   - step: Minimum allocation increment for buffer growth, in tokens.
    ///     `updateQuantized` grows the storage tuple by `step`-rounded chunks
    ///     when it needs more room. Each growth triggers an `expandQuant`
    ///     concat + `eval()` barrier; a step matched to the model's
    ///     `defaultPrefillStepSize` collapses N prefill chunks down to N
    ///     growth events instead of `N × ceil(chunk/step)`. Callers building
    ///     caches via `makeAttentionCache(...)` thread the per-model prefill
    ///     chunk through automatically; direct constructions default to 256
    ///     for backward compatibility.
    ///   - sdpaStrategy: SDPA path selector (default `.auto`). See the
    ///     `sdpaStrategy` property docs for the trade-offs. Env var
    ///     `MLX_AFFINE_SDPA` overrides this when set.
    public init(
        groupSize: Int = 64,
        bits: Int = 8,
        mode: QuantizationMode = .affine,
        step: Int = 256,
        sdpaStrategy: AffineSDPAStrategy = .auto
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = step
        self.mode = mode
        self.sdpaStrategy = sdpaStrategy
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
                // Materialise after each growth to break the lazy graph
                // chain. Each `expandQuant` concatenates the previous
                // packed/scale/bias tensor with a fresh `zeros` block,
                // and without an eval the compute graph retains every
                // prior generation. With `step=256` and prefill chunks
                // of 1024+ tokens, a 32k-context request grows the
                // buffer ≥128 times — without this barrier the held
                // intermediates dominate peak GPU (a 0.8B model jumps
                // from ~2 GB to ~7 GB at ctx=32k). Mirrors
                // `StandardKVCache.updateUnbounded`'s eval after the
                // `concatenated([currentKeys, newK], axis: 2)` resize.
                //
                // Single batched eval — feeding all 4-6 arrays to one
                // call lets MLX flush the command buffer once per
                // growth instead of once per array, which the 4-call
                // form was costing ~10-20% in prefill / decode tok/s.
                var pendingEval: [MLXArray] = [
                    self.keys!.0, self.keys!.1, self.values!.0, self.values!.1,
                ]
                if let kBiases = self.keys!.2 { pendingEval.append(kBiases) }
                if let vBiases = self.values!.2 { pendingEval.append(vBiases) }
                eval(pendingEval)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// Memory for quantized K/V: wq (packed) + scales + biases per K and V.
    override public var memoryBytes: Int {
        var total = 0
        if let k = keys { total += arrayBytes(k.0) + arrayBytes(k.1) + arrayBytes(k.2) }
        if let v = values { total += arrayBytes(v.0) + arrayBytes(v.1) + arrayBytes(v.2) }
        return total
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with AffineQuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `AffineQuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    /// Array of keys and values -- this will have either 6 elements or 4 elements (if biases are nil).
    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 4:
                // nil biases case
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "AffineQuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("AffineQuantizedKVCache metaState must have exactly 4 values")
            }

            self.offset = Int(newValue[1]) ?? 0
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = AffineQuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> StandardKVCache {
        let simpleCache = StandardKVCache()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }

    public override var storageKind: KVStorageKind {
        .affineQuantized(bits: bits, groupSize: groupSize)
    }
}

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
    private var cache: [MLXArray?]
    internal var leftPadding: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    public override func copy() -> any KVCache {
        let new = ArraysCache(size: cache.count)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    /// In-place filter to keep just the given indices in the cache
    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = nil
    }

    /// In-place extend this cache with the other cache
    public func extend(other: ArraysCache) {
        cache = zip(cache, other.cache).map { (c, o) in
            if let c = c, let o = o {
                return MLX.concatenated([c, o])
            }
            return c ?? o
        }
        leftPadding = nil
    }

    /// Create attention mask based on left padding
    public func makeMask(N: Int) -> MLXArray? {
        if cache[0] == nil, let leftPadding = leftPadding {
            return MLXArray(0 ..< N) .>= leftPadding[0..., .newAxis]
        } else {
            return nil
        }
    }

    // MARK: - Serialization

    /// metaState format: [slotCount, presentSlots (comma-separated), leftPadding (comma-separated, optional)]
    /// Legacy format (BaseKVCache default): [""]
    public override var metaState: [String] {
        get {
            var result = [
                "\(cache.count)",
                presentSlotIndices.map(String.init).joined(separator: ","),
            ]
            if let lp = leftPaddingValues {
                result.append(lp.map(String.init).joined(separator: ","))
            }
            return result
        }
        set {
            assertionFailure(
                "ArraysCache.metaState should not be set directly. Use restoreFromMetaState() instead"
            )
        }
    }

    /// Restore from saved metaState + state arrays. Handles both new (slot-aware) and legacy formats.
    internal func restoreFromMetaState(state: [MLXArray], savedMetaState: [String]) {
        // Detect new format: first element parses as int (slotCount), second element is present slots
        if savedMetaState.count >= 2, let slotCount = Int(savedMetaState[0]) {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }
            let lp: [Int]? =
                savedMetaState.count >= 3
                ? savedMetaState[2].split(separator: ",").compactMap({ Int($0) }) : nil

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = lp.map { MLXArray($0) }
        } else {
            // Legacy: best-effort, state is compacted
            self.cache = state.map { $0 as MLXArray? }
        }
    }

    /// Total number of slots (including nil)
    internal var slotCount: Int { cache.count }

    /// Indices of non-nil slots
    internal var presentSlotIndices: [Int] {
        cache.enumerated().compactMap { (i, v) in v != nil ? i : nil }
    }

    /// Left padding values as Int array, or nil
    internal var leftPaddingValues: [Int]? {
        guard let lp = leftPadding else { return nil }
        return lp.asArray(Int.self)
    }
}

/// Simple cache for Mamba-style state space models
/// Cache for SSM (state space model) state — used by GatedDeltaNet (Qwen 3.5 / 3.6),
/// Mamba (NemotronH), Jamba, FalconH1, and other hybrid attention/SSM architectures.
///
/// The cache holds cumulative recurrent state, not K/V tensors. It is **not
/// trimmable** because SSM state has no positional rollback in the general case;
/// speculative decoders that need rollback use `snapshot()` / `restore()` (legacy
/// hooks that spec 020 phase 2 will replace with `StateReplayCache` conformance).
///
/// Renamed from `SSMStateCache` in spec 006 (2026-05-04). The cache is misleadingly
/// named after Mamba but is generic SSM state — Qwen 3.5 / 3.6 use GatedDeltaNet,
/// not Mamba. The typealias `SSMStateCache = SSMStateCache` is kept for one release.
public class SSMStateCache: ArraysCache {

    /// Saved state for snapshot/restore during speculative decoding.
    /// `SSMStateCache` is not trimmable (SSM state is cumulative), so we
    /// snapshot before speculation and restore on rejection.
    private var snapshotState: [MLXArray]?
    private var snapshotOffset: Int?

    // MARK: - Tape-replay state (spec 020 phase 2)
    //
    // The delta log stores per-step innovation tuples handed in by the GDN
    // layer during a recording session. Each tuple is `[delta_t, k_t,
    // g_t]` where:
    //   - `delta_t` is the post-norm pre-state-update GDN innovation
    //     (`(v_t - kv_mem_t) * beta_t`),
    //   - `k_t` is the key projection at time t (already GQA-expanded
    //     to `Hv` heads),
    //   - `g_t` is the per-Hv-head decay scalar.
    //
    // On `rollback(acceptedPrefix: k)`, we restore the recurrent state
    // from the pre-record snapshot and re-fold the first k entries via
    // the standard GDN recurrence (`state = state * g + k * delta`).
    // The ops-based path defined here is the correctness reference; the
    // Metal kernel `gated_delta_replay` (in the `mlx-swift` sibling
    // repo) is the speed path with the same numerical contract.
    fileprivate var deltaLog: [[MLXArray]]?

    /// Whether a state-replay recording session is currently active.
    /// GDN layers read this to decide whether to route through the
    /// forward-with-record kernel (which captures per-step `delta_t` for
    /// future rollback) vs. the standard fast forward kernel.
    public var isRecording: Bool { deltaLog != nil }

    /// Whether this cache supports state-replay rollback.
    ///
    /// **Default `true`** — `SSMStateCache` was added for GatedDeltaNet
    /// (Qwen 3.5 / 3.6) where state replay is implemented and validated.
    ///
    /// **Mamba / Mamba 2** models (Nemotron Cascade 2, Jamba, Granite-MoE-
    /// Hybrid, FalconH1) also use `SSMStateCache` for their selective-SSM
    /// layers, but the Mamba recurrence
    /// (`s_{t+1} = exp(dt·A)·s_t + dt·B·x`) is numerically distinct from
    /// the GatedDeltaNet recurrence (`s_{t+1} = g·s_t + k·δ`), and the
    /// production S>1 forward uses a chunked parallel scan (`ssmAttn`)
    /// that doesn't expose per-step deltas. Mamba state replay needs its
    /// own native kernel pair — see spec 020 §"Mamba / Mamba 2
    /// follow-up". Until that ships, Mamba-using model factories should
    /// set this property to `false` on the caches they emit; the
    /// speculative iterators see `canRollbackPromptCache == false` and
    /// gracefully fall back to vanilla `TokenIterator`.
    public var canStateReplay: Bool = true

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    public override func copy() -> any KVCache {
        let new = SSMStateCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    public override var storageKind: KVStorageKind { .ssm }

    // MARK: - Speculative Decoding Support

    /// Save the current state so it can be restored if speculation is rejected.
    public func snapshot() {
        let s = self.state
        snapshotState = s.isEmpty ? nil : s.map { $0[.ellipsis] }
        snapshotOffset = self.offset
    }

    /// Restore state from a previous snapshot (used when draft tokens are rejected).
    public func restore() {
        guard let savedState = snapshotState, let savedOffset = snapshotOffset else { return }
        self.state = savedState
        self.offset = savedOffset
        discardSnapshot()
    }

    /// Discard the snapshot without restoring (all draft tokens accepted).
    public func discardSnapshot() {
        snapshotState = nil
        snapshotOffset = nil
    }
}

// MARK: - SSMStateCache: StateReplayCache (spec 020 phase 2)

extension SSMStateCache: StateReplayCache {

    // `canStateReplay` is a stored property on the class itself (defined
    // above) so model factories can opt out per-cache. GDN models
    // (Qwen 3.5 / 3.6) leave it at the default `true`. Mamba / Mamba 2
    // models (Nemotron Cascade 2, Jamba, …) set it to `false` until
    // their own state-replay kernel pair lands — see spec 020 §"Mamba /
    // Mamba 2 follow-up".

    /// Linear in accepted-prefix length — each step is a constant-time
    /// elementwise op on the recurrent state.
    public var replayCost: StateReplayCost { .ok }

    public func beginRecord() {
        precondition(deltaLog == nil, "SSMStateCache.beginRecord called twice without commit/cancel")
        snapshot()
        deltaLog = []
    }

    /// Append the per-step recurrence tensors. The GDN layer hands in
    /// `[delta_t, k_t, g_t]`; the cache stores them verbatim for possible
    /// re-fold during `rollback(acceptedPrefix:)`.
    ///
    /// `commitFull()` discards the delta log (the verify forward already
    /// advanced the state through `update(...)`); `cancel()` discards
    /// the delta log and restores the pre-record snapshot.
    ///
    /// **Per-round, T-axis recording** (preferred — used by the GDN layer
    /// dispatcher `gatedDeltaUpdateRecord`). Each call appends a single
    /// entry with the whole verify forward's tensors:
    ///   - delta:   `[B, T, Hv, Dv]`
    ///   - k:       `[B, T, Hv, Dk]`  (GQA-expanded)
    ///   - g:       `[B, T, Hv]`
    /// This is ~5× cheaper than per-step recording (one call per layer
    /// per round instead of T calls × per-step slice creation).
    ///
    /// **Per-step recording** (legacy — used by some unit tests). Each
    /// call appends one step's tensors:
    ///   - delta:   `[B, Hv, Dv]`
    ///   - k:       `[B, Hv, Dk]`
    ///   - g:       `[B, Hv]`
    /// The replay dispatcher (`stateReplayUpdate` in
    /// `StateReplayKernels.swift`) detects which format the entries are
    /// in (via the delta tensor's `ndim`) and stacks accordingly.
    public func recordStep(_ tensors: [MLXArray]) {
        precondition(
            deltaLog != nil,
            "SSMStateCache.recordStep called outside an active recording session")
        // Two record shapes are accepted:
        //   - GDN: 3 tensors per entry: [delta, k, g]      (spec 020)
        //   - Mamba: 2 tensors per entry: [dA_log, dBx_log] (spec 040)
        // The rollback dispatcher detects which shape by entry length.
        precondition(
            tensors.count == 3 || tensors.count == 2,
            "SSMStateCache expects 3 tensors (GDN: delta, k, g) or 2 tensors (Mamba: dA_log, dBx_log); got \(tensors.count)")
        deltaLog?.append(tensors)
    }

    public func commitFull() {
        precondition(deltaLog != nil, "SSMStateCache.commitFull called without an active recording session")
        // State already advanced through update(); we just clear the delta log
        // and discard the snapshot.
        deltaLog = nil
        discardSnapshot()
    }

    public func rollback(acceptedPrefix k: Int) {
        precondition(deltaLog != nil, "SSMStateCache.rollback called without an active recording session")
        let recordedLog = deltaLog ?? []
        precondition(k >= 0, "acceptedPrefix (\(k)) must be non-negative")
        // The k <= T_log range check happens inside `stateReplayUpdate` —
        // T_log depends on the record format (per-round T-axis vs per-step
        // entries) and the dispatcher resolves it before invoking the
        // kernel.

        // 1) Restore state from the pre-record snapshot.
        restore()

        // 2) Re-fold the first k delta log entries through the `state_replay`
        //    Metal kernel. The kernel adopts the masked-timestep
        //    correctness fix (`3217e15`) and branchless `metal::select`
        //    pattern (`c9f992e`) from day 1.
        //
        //    Slot assignment matches the production GDN layer
        //    (`Qwen35GatedDeltaNet.callAsFunction` / `Qwen3NextGatedDeltaNet`):
        //      - `state[0]` is the **conv state** (3-dim `[B, kernel-1, conv_dim]`).
        //      - `state[1]` is the **recurrent SSM state** (4-dim
        //        `[B, Hv, Dv, Dk]`).
        //    State replay only updates the recurrent slot; the conv state
        //    is re-initialised by the layer on the next forward.
        if self.state.count >= 2 && k > 0 {
            let firstEntry = recordedLog.first ?? []
            if firstEntry.count == 2 {
                // Spec 040: Mamba flavour. Per-round entry is [dA_log, dBx_log]
                // with dA_log shape [B, T, H, ds] and dBx_log [B, T, H, dh, ds].
                // Each entry covers the full T-axis of one verify forward, so
                // we use the most recent entry's `[..<k]` slice.
                let dALogFull = firstEntry[0]
                let dBxLogFull = firstEntry[1]
                let dALogK = dALogFull[0..., ..<k, .ellipsis]
                let dBxLogK = dBxLogFull[0..., ..<k, .ellipsis]
                let s = MLXFast.ssmReplay(
                    stateSnapshot: self.state[1],
                    dALog: dALogK,
                    dBxLog: dBxLogK,
                    acceptedPrefix: k,
                    mask: nil)
                var newState = self.state
                newState[1] = s
                self.state = newState
            } else {
                // GDN flavour (spec 020): [delta, k, g] triples.
                let s = stateReplayUpdate(
                    state: self.state[1],
                    deltaLog: recordedLog,
                    acceptedPrefix: k)
                var newState = self.state
                newState[1] = s
                self.state = newState
            }
        }

        // 3) Update offset to reflect k accepted tokens past the snapshot.
        self.offset = (self.offset) + k

        // 4) Clear the delta log; recording session is over.
        deltaLog = nil
    }

    public func cancel() {
        precondition(deltaLog != nil, "SSMStateCache.cancel called without an active recording session")
        restore()
        deltaLog = nil
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    /// Internal initializer for reconstruction from deserialized children
    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var storageKind: KVStorageKind { .composite }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    /// Internal accessor for child caches (used by serialization)
    internal var children: [KVCache] { caches }

    // MARK: - Serialization

    /// metaState format: [childCount, (className, stateCount, metaStateCount, ...metaState)*]
    ///
    /// Like Python's CacheList.meta_state which returns [child_class_names, child_meta_states],
    /// but flattened for Swift's [String] format.
    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className = cacheClassName(cache)
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList.metaState should not be set directly. Use CacheList.fromState() instead")
        }
    }

    /// Reconstruct a CacheList from flattened state + metaState, like Python's from_state()
    internal static func fromState(state: [MLXArray], metaState: [String]) throws -> CacheList {
        guard let childCount = metaState.first.flatMap({ Int($0) }) else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }

        var children: [KVCache] = []
        var metaIdx = 1  // skip childCount
        var stateIdx = 0

        for _ in 0 ..< childCount {
            guard metaIdx + 2 < metaState.count else {
                throw KVCacheError(message: "CacheList metaState truncated")
            }
            let className = metaState[metaIdx]
            guard let stateCount = Int(metaState[metaIdx + 1]) else {
                throw KVCacheError(message: "CacheList: invalid stateCount for child")
            }
            guard let metaCount = Int(metaState[metaIdx + 2]) else {
                throw KVCacheError(message: "CacheList: invalid metaStateCount for child")
            }
            metaIdx += 3

            let childMeta = Array(metaState[metaIdx ..< min(metaIdx + metaCount, metaState.count)])
            metaIdx += metaCount

            let childState = Array(state[stateIdx ..< min(stateIdx + stateCount, state.count)])
            stateIdx += stateCount

            let child = try restoreCacheFromMetaState(
                className: className, state: childState, metaState: childMeta)
            children.append(child)
        }

        return CacheList(caches: children)
    }
}

// MARK: - Error Types

struct KVCacheError: Error {
    let message: String
}

// MARK: - Utility Functions

/// Map a cache instance to its Python-compatible class name for serialization.
private func cacheClassName(_ cache: KVCache) -> String {
    // Python-compatible class-name strings for cross-platform persistence.
    // The spec-006 cleanup PR removed the legacy typealiases (`KVCacheSimple`
    // / `RotatingKVCache` / `QuantizedKVCache` / `MambaCache` / `ChunkedKVCache`),
    // so this helper discriminates on the post-rename Swift types but emits
    // the legacy names so checkpoints stay loadable by mlx-lm Python.
    // Discriminate `StandardKVCache` on `eviction` to pick "KVCache"
    // (unbounded) vs "RotatingKVCache" (windowed); the loader accepts both
    // names and routes to `StandardKVCache` with the appropriate eviction.
    switch cache {
    case is SSMStateCache: return "MambaCache"   // must precede ArraysCache (subclass)
    case is ArraysCache: return "ArraysCache"
    case let standard as StandardKVCache:
        if case .window = standard.eviction { return "RotatingKVCache" }
        return "KVCache"
    case is AffineQuantizedKVCache: return "QuantizedKVCache"
    case is CacheList: return "CacheList"
    default: return "KVCache"
    }
}

/// Save a pre-computed prompt cache to a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    let cacheClasses = cache.map { cacheClassName($0) }

    // Flatten cache data using tree_flatten compatible structure: "i.j" format
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // Create cache_metadata structure compatible with Python: [cache_info, metadata, cache_classes]
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata)
    for (key, value) in metadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    try save(arrays: flattenedData, metadata: flattenedMetadata, url: url)
}

/// Load a prompt cache from a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)

    // Unflatten arrays using tree_unflatten compatible logic
    let cacheData = unflattenArrays(arrays)

    // Unflatten metadata using tree_unflatten compatible logic
    let unflattenedMetadata = unflattenMetadata(metadata)

    // Extract cache_info, user_metadata, and cache_classes from unflattened structure
    // Structure: [cache_info, user_metadata, cache_classes]
    guard unflattenedMetadata.count >= 3 else {
        throw KVCacheError(message: "Invalid cache metadata format")
    }

    let cacheInfo = unflattenedMetadata[0] as? [[String]] ?? []
    let userMetadata = unflattenedMetadata[1] as? [String: String] ?? [:]
    let cacheClasses = unflattenedMetadata[2] as? [String] ?? []

    guard cacheData.count == cacheInfo.count && cacheData.count == cacheClasses.count else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    // Reconstruct cache instances
    var caches: [KVCache] = []
    for i in 0 ..< cacheData.count {
        let className = cacheClasses[i]
        let info = i < cacheInfo.count ? cacheInfo[i] : []

        let cache = try restoreCacheFromMetaState(
            className: className, state: cacheData[i], metaState: info)
        caches.append(cache)
    }

    return (caches, userMetadata)
}

/// Reconstruct a single cache from its class name, state arrays, and metaState.
///
/// Like Python's `globals()[className].from_state(state, meta_state)`, each cache type
/// encodes enough info in `metaState` to reconstruct itself.
private func restoreCacheFromMetaState(
    className: String,
    state: [MLXArray],
    metaState: [String]
) throws -> KVCache {
    // Class-name strings on disk stay as the legacy Python-compatible names
    // ("KVCache", "RotatingKVCache", "QuantizedKVCache", "MambaCache", …) for
    // cross-platform interop with mlx-lm Python checkpoints. The spec-006
    // cleanup PR removed the legacy Swift typealiases, so this helper maps
    // each on-disk name to the post-rename concrete type.
    switch className {
    case "KVCache", "KVCacheSimple":
        let cache = StandardKVCache()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "RotatingKVCache":
        guard metaState.count >= 5 else {
            throw KVCacheError(
                message: "Invalid RotatingKVCache metaState - expected 5 values")
        }
        if metaState[1] == "None" {
            throw KVCacheError(
                message: "RotatingKVCache with maxSize=None is not supported.")
        }
        guard let maxSize = Int(metaState[1]) else {
            throw KVCacheError(
                message: "Failed to parse RotatingKVCache maxSize from: \(metaState[1])")
        }
        let cache = StandardKVCache(maxSize: maxSize)  // window-eviction
        cache.state = state
        cache.metaState = metaState
        return cache

    case "QuantizedKVCache":
        let cache = AffineQuantizedKVCache()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "MambaCache":
        let cache = SSMStateCache()
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "ArraysCache":
        let cache = ArraysCache(size: 0)
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "CacheList":
        return try CacheList.fromState(state: state, metaState: metaState)

    case "ChunkedKVCache":
        // ChunkedKVCache was deleted in the spec-006 cleanup PR (audit
        // confirmed zero in-tree usage). Old checkpoints serialised under
        // this name surface here so we can fail loud rather than silently
        // returning the wrong cache shape.
        throw KVCacheError(
            message: "ChunkedKVCache was removed in spec 006; cannot restore.")

    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

/// Unflatten arrays from tree_flatten format (e.g., "0.1", "1.0") to nested structure
private func unflattenArrays(_ flatArrays: [String: MLXArray]) -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        if components.count >= 2,
            let i = Int(components[0]),
            let j = Int(components[1])
        {
            if arrayMap[i] == nil {
                arrayMap[i] = [:]
            }
            arrayMap[i]![j] = array
        }
    }

    // Convert to ordered array structure
    var result: [[MLXArray]] = []
    let maxI = arrayMap.keys.max() ?? -1

    for i in 0 ... maxI {
        if let innerMap = arrayMap[i] {
            let maxJ = innerMap.keys.max() ?? -1
            var innerArray: [MLXArray] = []
            for j in 0 ... maxJ {
                if let array = innerMap[j] {
                    innerArray.append(array)
                }
            }
            result.append(innerArray)
        } else {
            result.append([])
        }
    }

    return result
}

/// Unflatten metadata from tree_flatten format to nested structure
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) -> [KVCache] {
    if let maxKVSize = maxKVSize {
        return (0 ..< numLayers).map { _ in
            StandardKVCache(maxSize: maxKVSize, keep: 4)
        }
    } else {
        return (0 ..< numLayers).map { _ in StandardKVCache() }
    }
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Quantized Attention Operations

/// Spec 041 phase 1: pick between the discrete-pass quantized SDPA and the
/// flash quantized SDPA (currently implemented as dequant-then-MLXFastSDPA).
///
/// The discrete path computes scores via `quantizedMM(transpose=true)`,
/// materialises `[B, H, L, T]` softmax weights, then `quantizedMM(transpose=false)`
/// for the output. At ctx=32k that score matrix is ~1 GiB per layer for
/// Qwen 3.5-0.8B and the dominant peak-GPU contributor on `--kv affine*`.
///
/// The flash path dequantises K and V to FP16 once and routes through
/// `MLXFast.scaledDotProductAttention`, which tiles K/V internally and never
/// materialises the score matrix. Peak GPU drops to within a few hundred MiB
/// of the `--kv none` baseline. Sinks, sliding-window, and GQA all pass
/// through to MLX's native SDPA support unchanged.
///
/// The ideal Phase 1 fused-kernel implementation would dequantise per-tile
/// inside the SDPA loop (no full K_fp / V_fp materialisation between layer
/// calls). That further optimisation is scoped for follow-up; the dequant-then-
/// SDPA stop-gap already achieves the spec's acceptance criterion of "peak
/// GPU within 200 MiB of `--kv none` baseline at ctx=32k" on the smoke matrix.
///
/// L>1 vs L=1 trade-off:
///   - **L > 1 (prefill)**: flash wins big. Score matrix is [B, H, L, T]
///     — at L=2048, T=32k that's ~1 GiB per layer on a small model. The
///     flash path drops it to ~K_fp + V_fp = a few hundred MiB total at
///     long context. Empirically 30-40% peak GPU reduction.
///   - **L = 1 (decode)**: discrete wins. Score matrix is [B, H, 1, T]
///     — small. The dequant materialises K_fp/V_fp = ~32 MB per layer
///     PER STEP, which costs ~15-20% decode tok/s at long context.
///
/// Per-call gate auto-selects flash vs discrete based on L. Can be pinned
/// per-cache via `AffineQuantizedKVCache.sdpaStrategy` (constructor arg) or
/// globally via the `MLX_AFFINE_SDPA` env var:
///   - `MLX_AFFINE_SDPA=auto` — flash for L>1, discrete for L=1 (default)
///   - `MLX_AFFINE_SDPA=kernel` — spec 041 phase 1.1 fused Metal kernel
///     (`fusedFlashQuantizedSDPA`). Correct but unoptimised; tiled MMA +
///     threadgroup codebook + INT8 score accumulator follow-ups in spec
///     042 Phase 1b will close the perf gap. Opt-in for kernel validation
///     and future-perf-work development.
///   - `MLX_AFFINE_SDPA=flash` — dequant-then-MLXFastSDPA stop-gap.
///     Materialises K_fp/V_fp transient before MLX SDPA's flash kernel.
///     Best perf today; what `.auto` picks for L>1 prefill.
///   - `MLX_AFFINE_SDPA=discrete` — legacy `quantizedMM → softmax →
///     quantizedMM`. Materialises full `[B, H, L, T]` score matrix. What
///     `.auto` picks for L=1 decode (score matrix is small and the
///     dequant transient cost dominates).
///
/// Precedence: env var when set ▶ per-cache `sdpaStrategy` ▶ `.auto`.
public enum AffineSDPAStrategy: Sendable {
    case auto, kernel, flash, discrete

    /// Env-var override read once at process startup. When set, takes
    /// precedence over per-cache `sdpaStrategy`. `nil` when unset (honour
    /// the cache's own strategy).
    static let envOverride: AffineSDPAStrategy? = {
        switch ProcessInfo.processInfo.environment["MLX_AFFINE_SDPA"] {
        case "discrete": return .discrete
        case "flash": return .flash
        case "kernel": return .kernel
        case "auto": return .auto
        default: return nil
        }
    }()

    /// Resolved dispatch path. `.auto` never appears here — it's an input
    /// directive that resolves to a concrete one of `.kernel`/`.flash`/
    /// `.discrete` based on `L`.
    public enum Path { case kernel, flash, discrete }

    /// Pick path given the call's query length, with the caller's per-cache
    /// strategy as the fallback when no env override is set.
    static func choose(L: Int, strategy: AffineSDPAStrategy = .auto) -> Path {
        let effective = envOverride ?? strategy
        switch effective {
        case .kernel: return .kernel
        case .flash: return .flash
        case .discrete: return .discrete
        case .auto: return L > 1 ? .flash : .discrete
        }
    }
}

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine,
    strategy: AffineSDPAStrategy = .auto
) -> MLXArray {
    let queryL = queries.dim(2)
    switch AffineSDPAStrategy.choose(L: queryL, strategy: strategy) {
    case .kernel:
        return fusedFlashQuantizedSDPA(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale, mask: mask, sinks: sinks,
            groupSize: groupSize, bits: bits, mode: mode
        )
    case .flash:
        return flashQuantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale, mask: mask, sinks: sinks,
            groupSize: groupSize, bits: bits, mode: mode
        )
    case .discrete:
        break
    }

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Apply mask
    switch mask {
    case .causal, .slidingWindow:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let qExpanded = expandedDimensions(qIndices, axis: -1)
        let kExpanded = expandedDimensions(kIndices, axis: -2)
        var causalMask = greaterEqual(qExpanded, kExpanded)
        if case .slidingWindow(let window) = mask {
            let lowerBound = qExpanded - MLXArray(Int32(window))
            causalMask = logicalAnd(causalMask, greater(kExpanded, lowerBound))
        }
        // Cast the negative-infinity sentinel to `scores.dtype` so
        // `MLX.where(bool, bf16, fp32_scalar)` doesn't promote the
        // entire scores tensor to fp32. For prefill chunks with
        // L=1024 and T=32k that promotion doubled the materialised
        // score matrix from ~1 GB to ~2 GB per layer — visible as
        // peak GPU 3-4× larger than the no-quant baseline at long
        // context. `TurboQuantizedKVCache.compressedAttention` and
        // `Gemma3nText` already had the dtype cast; this is the
        // affine path's matching fix.
        scores = MLX.where(
            causalMask, scores,
            MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(
                maskArray, scores,
                MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(
                    maskArray, scores,
                    MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    let attentionWeights: MLXArray
    if let sinks {
        // Attention-sink fold (GPT-OSS family) via softmax-of-augmented-scores.
        //
        // Standard softmax-with-sinks adds one extra implicit logit per Q head
        // to the denominator; the sink contributes nothing to the output
        // accumulator (its V is implicit zero). Math:
        //
        //   M = max(max_t s_t, sink_h)
        //   p_t = exp(s_t - M) / (Σ_t exp(s_t - M) + exp(sink_h - M))
        //   output_h = Σ_t p_t · V_t
        //
        // Implementation: concat the per-head sink as a (T+1)th score column,
        // run MLX's fused `softmax` over `T+1`, then drop the sink column on
        // the way to the second matmul (V_sink is implicit zero so its
        // contribution to the output is zero — equivalent to multiplying that
        // column out by V_sink=0).
        //
        // Why fused-softmax-of-(T+1) instead of an explicit manual softmax
        // (max → subtract → exp → sum-with-sink → divide): MLX's softmax is a
        // single fused Metal op that doesn't materialize the per-token exp /
        // running-sum intermediates. The naive manual version peaks at ~3
        // copies of the score tensor, which on GPT-OSS-20B summarization at
        // ctx=8192 grew peak GPU from 11.9 GB (no-quant) to 20.1 GB. The
        // augmented-column path only adds T+1 elements per row and keeps the
        // softmax kernel fused. Sinks input is cast to `scores.dtype` to keep
        // the concat in bf16/fp16 (avoiding the fp32 upcast that also blew
        // peak GPU).
        //
        // Sink broadcasting:
        //   - GQA: scores [B, nKVH, nRep, L, T], sinks [nQH=nKVH·nRep]
        //          → reshape sinks to [1, nKVH, nRep, 1, 1]; broadcast to
        //            [B, nKVH, nRep, L, 1] for concat along T.
        //   - MHA: scores [B, nQH, L, T], sinks [nQH]
        //          → reshape sinks to [1, nQH, 1, 1]; broadcast to
        //            [B, nQH, L, 1].
        let scoresDtype = scores.dtype
        let sinkColumnShape: [Int]
        let sinkReshape: [Int]
        if nRepeats > 1 {
            sinkReshape = [1, nKVHeads, nRepeats, 1, 1]
            sinkColumnShape = [
                scores.dim(0), nKVHeads, nRepeats, scores.dim(-2), 1,
            ]
        } else {
            sinkReshape = [1, nQHeads, 1, 1]
            sinkColumnShape = [
                scores.dim(0), nQHeads, scores.dim(-2), 1,
            ]
        }
        let sinkColumn = MLX.broadcast(
            sinks.reshaped(sinkReshape).asType(scoresDtype),
            to: sinkColumnShape)
        let scoresAugmented = concatenated([scores, sinkColumn], axis: -1)
        let weightsAugmented = softmax(scoresAugmented, axis: -1)
        // Slice off the (T+1)th column. It carries the sink's softmax mass,
        // but that mass weighs against V_sink≡0 — equivalent to zero
        // contribution to the second matmul.
        let T = scores.dim(-1)
        attentionWeights = weightsAugmented[.ellipsis, ..<T]
    } else {
        attentionWeights = softmax(scores, axis: -1)
    }

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output
}

/// Spec 041 phase 1.1: fused flash-quantized SDPA via `MLXFast.flashQuantizedSDPA`.
///
/// Calls the new Metal kernel that dequantises K/V per tile inside the
/// online-softmax loop. Unlike the dequant-then-SDPA stop-gap below, this
/// path never materialises the full FP K_fp / V_fp transient — the only
/// long-lived per-call buffer is the [B, n_q, T_q, V] output.
///
/// The Metal kernel supports `bits ∈ {2,3,4,6,8}`, `groupSize = 64`, and head
/// dimensions in `{64, 96, 128, 256, 512}`. Shapes outside those instantiations
/// fall through to the dequant-then-SDPA path (`flashQuantizedScaledDotProductAttention`)
/// which handles arbitrary shapes via MLX's generic `dequantized()` op.
///
/// Mask modes:
///   - `.causal`   → kernel-side causal mask (no array materialised)
///   - `.array(m)` → bool / float mask passed through
///   - `.none`     → unmasked
///   - `.slidingWindow(_)` → caller must supply the mask as `.array(...)`;
///     the kernel doesn't yet have a fused sliding-window path. Falls back
///     to the discrete + manual softmax fold.
private func fusedFlashQuantizedSDPA(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray {
    // Only affine + groupSize=64 + supported bits are instantiated in the
    // kernel. Anything else: fall back to dequant-then-SDPA.
    let supportedBits: Set<Int> = [2, 3, 4, 6, 8]
    let headDim = queries.dim(-1)
    let supportedDim: Set<Int> = [64, 96, 128, 256, 512]
    if mode != .affine || groupSize != 64 || !supportedBits.contains(bits)
        || !supportedDim.contains(headDim)
    {
        return flashQuantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale, mask: mask, sinks: sinks,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    // Map mask mode → (causal flag, array mask).
    let causal: Bool
    let maskArr: MLXArray?
    switch mask {
    case .causal:
        causal = true
        maskArr = nil
    case .array(let m):
        causal = false
        maskArr = m
    case .arrays(let arrs):
        causal = false
        maskArr = arrs.first
    case .slidingWindow:
        // Sliding-window: kernel doesn't have a native windowed path yet;
        // route through the dequant-then-SDPA stop-gap which calls
        // `MLXFast.scaledDotProductAttention(... mask: .slidingWindow(...))`.
        return flashQuantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale, mask: mask, sinks: sinks,
            groupSize: groupSize, bits: bits, mode: mode)
    case .none:
        causal = false
        maskArr = nil
    }

    // The kernel's bias arg is non-optional; affine quantize always emits
    // a bias array. Defensive guard for mxfp4 / unbiased schemes (which
    // route through dequant-then-SDPA path above before reaching here).
    guard let kBiases = quantizedKeys.2, let vBiases = quantizedValues.2 else {
        return flashQuantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale, mask: mask, sinks: sinks,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    return MLXFast.flashQuantizedSDPA(
        queries: queries,
        kPacked: quantizedKeys.0,
        kScales: quantizedKeys.1,
        kBiases: kBiases,
        vPacked: quantizedValues.0,
        vScales: quantizedValues.1,
        vBiases: vBiases,
        scale: scale,
        bits: bits,
        groupSize: groupSize,
        causal: causal,
        mask: maskArr,
        sinks: sinks)
}

/// Spec 041 phase 1: flash-quantized SDPA via dequant-then-MLXFastSDPA.
///
/// This is the "stop-gap" implementation of phase 1: rather than write a
/// fused Metal kernel that dequants K/V per tile inside the flash loop, we
/// dequantise K and V to FP16 once per call and route through
/// `MLXFast.scaledDotProductAttention`. That kernel tiles K/V internally and
/// runs an online softmax — never materialising the `[B, H, L, T]` score
/// matrix that dominated peak GPU on the discrete `quantizedMM → softmax →
/// quantizedMM` path.
///
/// The dequant produces a transient FP16 K/V tensor of shape
/// `[B, nKVH, T, D]` per call, which the SDPA call consumes and then frees.
/// At ctx=32k for Qwen 3.5-0.8B this is ~32 MB per layer per K (and per V),
/// concurrently live for one layer at a time. Compared with the discrete
/// path's ~1 GiB score matrix, that's a ~30× reduction in peak per-layer
/// workspace.
///
/// All MLXFast SDPA features pass through unchanged:
/// - **GQA / MQA**: don't pre-tile keys/values; MLX SDPA handles the
///   broadcast internally (`nQHeads > nKVHeads`).
/// - **Sliding window**: `mask: .slidingWindow(size:)` routes to the
///   sliding-causal kernel.
/// - **Sinks**: `sinks:` folds the per-head logit into the online softmax
///   (native MLX SDPA support).
/// - **Bit-widths**: any `bits` supported by MLX's `dequantized()` works —
///   spec 041 Phase 2's broader bit-widths come for free.
///
/// Phase 1 fused-kernel optimisation (per-tile dequant, no full K_fp / V_fp
/// materialisation) is a known follow-up. Profitable mainly at very long
/// context where the K_fp / V_fp transient becomes appreciable compared
/// with the K/V cache itself.
private func flashQuantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray {
    // Dequantise K. Output dtype follows queries (typically bf16/fp16).
    let keysFP = dequantized(
        quantizedKeys.0,
        scales: quantizedKeys.1, biases: quantizedKeys.2,
        groupSize: groupSize, bits: bits, mode: mode,
        dtype: queries.dtype)
    // Dequantise V — same shape and dtype rules.
    let valuesFP = dequantized(
        quantizedValues.0,
        scales: quantizedValues.1, biases: quantizedValues.2,
        groupSize: groupSize, bits: bits, mode: mode,
        dtype: queries.dtype)

    // MLXFast SDPA handles GQA, masks, sinks natively.
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keysFP,
        values: valuesFP,
        scale: scale,
        mask: mask,
        sinks: sinks
    )
}

/// Parse a turbo scheme string like "turbo4", "turbo4v2", "turbo0v4" into bit-widths.
/// Kept for legacy callsites (e.g., model factories' direct turbo construction).
/// Prefer `KVCache.CompressionAlgorithm.init?(_:)` for new code.
public func parseTurboScheme(_ scheme: String) -> (bits: Int, keyBits: Int?, valueBits: Int?) {
    // "turbo4v2" → keyBits=4, valueBits=2
    // "turbo4"   → bits=4 (symmetric)
    // "turbo0v4" → keyBits=0 (fp16), valueBits=4
    let digits = scheme.dropFirst(5) // drop "turbo"
    if let vIdx = digits.firstIndex(of: "v") {
        let kb = Int(digits[digits.startIndex..<vIdx]) ?? 4
        let vb = Int(digits[digits.index(after: vIdx)...]) ?? 4
        return (bits: max(kb, vb), keyBits: kb, valueBits: vb)
    } else {
        let b = Int(digits) ?? 4
        return (bits: b, keyBits: nil, valueBits: nil)
    }
}
