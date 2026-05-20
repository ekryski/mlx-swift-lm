// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines scalar or per-sequence offset), or `nil`
///     for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    // TriAttention V3-aware: when the cache is a TriAttentionKVCache, the
    // post-compaction `offset` is the storage length (smaller than the
    // logical position range). RoPE must use the LOGICAL position so the
    // model sees positional encodings tied to the original token stream,
    // not the compacted slab. Falls back through the BatchPositionedKVCache
    // path and finally to `cache?.offset` for non-V3 caches — fully
    // backwards compatible.
    if let triCache = cache as? TriAttentionKVCache {
        return rope(x, offset: triCache.logicalOffset)
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return rope(x, offset: batchCache.batchOffset)
    }
    return rope(x, offset: cache?.offset ?? 0)
}
