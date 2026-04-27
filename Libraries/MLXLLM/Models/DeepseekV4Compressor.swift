// Copyright © 2026 TheTom.
// Adapted from osaurus-ai/vmlx-swift-lm (MIT) — see
// Libraries/MLXLLM/Models/DEEPSEEK_V4_ATTRIBUTION.md for upstream license.
//
// Original: Copyright © 2026 Osaurus AI. SPDX-License-Identifier: MIT.
// Upstream: https://github.com/osaurus-ai/vmlx-swift-lm
//
// DSV4 Compressor + Indexer + per-layer DeepseekV4Cache.
//
// These power the "compressed global context" path that augments the
// 128-token local sliding window with pooled summaries of older tokens.
// Every decoder layer with `compress_ratio > 0` (~41 of 43 layers in
// DSV4-Flash) carries a Compressor; layers with `compress_ratio == 4`
// ALSO carry an Indexer that picks the top-k most relevant pooled
// entries per query position.
//
// Phase 1 caveat: long-context paths are wired but not yet validated
// against the Python reference. Short prompts take a bypass fast-path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DeepseekV4Cache
//
// Per-layer composite cache. Wraps a `RotatingKVCache` for the local
// sliding window plus persistent buffer state for the compressor and
// indexer. Multi-call stateful: on each prefill step it accumulates
// raw-token windows until a full `compress_ratio`-sized chunk is ready,
// then pools and stores. The pooled sequence grows across calls so
// turn 2 and beyond see the full history summary.
//
// For short prompts (L < compress_ratio) and no V4Cache provided, the
// attention forward takes a fast-path that skips the compressor
// entirely (Python mirror: `if v4_cache is None and L < compress_ratio
// → skip`).
public final class DeepseekV4Cache: KVCache {
    /// Expose the inner rotating cache so `TQDiskSerializer` and
    /// `restoreRotatingLayer` can round-trip the sliding-window state.
    /// Compressor/Indexer pool buffers are NOT serialized — they get
    /// recomputed from prompt tokens on the next prefill.
    public var rotating: RotatingKVCache { local }
    /// Local sliding-window cache (compress_ratio-agnostic).
    public let local: RotatingKVCache
    let slidingWindow: Int
    /// Compressor buffer state (raw kv/gate not yet ready to pool)
    /// and pooled summary so far.
    fileprivate var compBufferKV: MLXArray?
    fileprivate var compBufferGate: MLXArray?
    fileprivate var compPooled: MLXArray?
    /// Indexer's own buffer state (separate branch — compressor inside
    /// the indexer uses its own buffers).
    fileprivate var idxBufferKV: MLXArray?
    fileprivate var idxBufferGate: MLXArray?
    fileprivate var idxPooled: MLXArray?

    public init(slidingWindow: Int) {
        self.slidingWindow = slidingWindow
        self.local = RotatingKVCache(maxSize: slidingWindow, keep: 0)
    }

    // KVCache protocol implementation — delegate everything to `local`.
    public var offset: Int { local.offset }
    public var maxSize: Int? { local.maxSize }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        local.update(keys: keys, values: values)
    }

    public func peek() -> (MLXArray, MLXArray)? { local.peek() }
    public var memoryBytes: Int { local.memoryBytes }
    public var isDonor: Bool {
        get { local.isDonor }
        set { local.isDonor = newValue }
    }

    public var state: [MLXArray] {
        get { local.state }
        set { local.state = newValue }
    }

    public var metaState: [String] {
        get { local.metaState }
        set { local.metaState = newValue }
    }

    public var isTrimmable: Bool { local.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int { local.trim(n) }

    public func innerState() -> [MLXArray] { local.innerState() }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        local.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        // Compressor pool state is NOT deep-copied for snapshot/restore;
        // the caller should treat compressor state as ephemeral (it's
        // recomputable from prompt tokens via re-prefill).
        let dup = DeepseekV4Cache(slidingWindow: slidingWindow)
        dup.state = local.state
        return dup
    }

    // State accessors for Compressor/Indexer. Public so disk round-trip
    // tests and any future cache-inspection code can verify the
    // ephemeral buffers are cleared on restore (they recompute from
    // prompt tokens on the next prefill).
    public func getBuffers(_ key: BranchKey) -> (kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor: return (compBufferKV, compBufferGate)
        case .indexer: return (idxBufferKV, idxBufferGate)
        }
    }

    public func setBuffers(_ key: BranchKey, kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor:
            compBufferKV = kv
            compBufferGate = gate
        case .indexer:
            idxBufferKV = kv
            idxBufferGate = gate
        }
    }

    public func getPooled(_ key: BranchKey) -> MLXArray? {
        key == .compressor ? compPooled : idxPooled
    }

    public func setPooled(_ key: BranchKey, value: MLXArray) {
        if key == .compressor {
            compPooled = value
        } else {
            idxPooled = value
        }
    }

    public enum BranchKey { case compressor, indexer }
}

// MARK: - Compressor
//
// Projects input x through `wkv` + `wgate`, accumulates raw windows
// until a full `compress_ratio`-chunk is ready, then pools the chunk
// via softmax(gate)-weighted sum. For compress_ratio=4 the output is
// 2× widened (overlap mode) so each pool spans TWO adjacent chunks —
// strictly increases context coverage. After pooling, applies the
// absolute position embedding (APE) and partial RoPE at the chunk
// positions. Updates the per-layer V4Cache pool if provided.
public final class DeepseekV4Compressor: Module {
    let compressRatio: Int
    let headDim: Int
    let outDim: Int
    let overlap: Bool
    let rmsNormEps: Float

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    /// APE: (compress_ratio, out_dim) learned positional bias inside
    /// each pool window.
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration, compressRatio: Int, headDim: Int) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.rmsNormEps = config.rmsNormEps
        self.overlap = compressRatio == 4
        self.outDim = headDim * (overlap ? 2 : 1)
        self._wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._ape.wrappedValue = zeros([compressRatio, outDim])
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    /// Forward. Returns pooled summary of shape (B, pooled_count, headDim).
    /// When `v4Cache` is provided, pooled is appended to the cache pool
    /// and the full cached pool is returned.
    func callAsFunction(
        _ x: MLXArray,
        rope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int,
        branch: DeepseekV4Cache.BranchKey = .compressor
    ) -> MLXArray {
        let B = x.dim(0)
        var kv = wkv(x)
        var gate = wgate(x)

        // Accumulate windows. When cache present, prepend unused-tail
        // buffers from prior calls.
        var poolBase = startPos
        if let cache = v4Cache {
            let (bufKV, bufGate) = cache.getBuffers(branch)
            if let bKV = bufKV, bKV.dim(1) > 0, let bG = bufGate {
                kv = concatenated([bKV, kv], axis: 1)
                gate = concatenated([bG, gate], axis: 1)
                poolBase -= bKV.dim(1)
            }
            let total = kv.dim(1)
            let usable = (total / compressRatio) * compressRatio
            // Stash the tail for the next call.
            let tailKV = usable < total ? kv[0..., usable..., 0...] : nil
            let tailGate = usable < total ? gate[0..., usable..., 0...] : nil
            cache.setBuffers(branch, kv: tailKV, gate: tailGate)
            kv = kv[0..., 0..<usable, 0...]
            gate = gate[0..., 0..<usable, 0...]
        } else {
            let total = kv.dim(1)
            let usable = (total / compressRatio) * compressRatio
            kv = kv[0..., 0..<usable, 0...]
            gate = gate[0..., 0..<usable, 0...]
        }

        let Lready = kv.dim(1)
        if Lready == 0 {
            let empty = MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
            if let cache = v4Cache {
                return cache.getPooled(branch) ?? empty
            }
            return empty
        }

        let W = Lready / compressRatio
        var kvWin = kv.reshaped(B, W, compressRatio, outDim)
        var gateWin =
            gate.reshaped(B, W, compressRatio, outDim) + ape.asType(gate.dtype)

        if overlap {
            kvWin = overlapTransform(kvWin, fillValue: 0.0)
            // For gate, the pre-allocated fill is -inf so softmax assigns
            // zero mass to the padding half.
            gateWin = overlapTransform(
                gateWin, fillValue: -Float.infinity)
        }

        let weights =
            softmax(gateWin.asType(.float32), axis: 2, precise: true).asType(
                kvWin.dtype)
        var pooled = (kvWin * weights).sum(axis: 2)
        pooled = norm(pooled.asType(x.dtype))

        // Apply RoPE at the chunk centers (position = chunk_idx * ratio
        // + pool_base).
        let positions =
            MLXArray(
                Int32(0)..<Int32(pooled.dim(1))
            ).asType(.float32) * Float(compressRatio) + Float(poolBase)
        // Build cos/sin at those positions.
        let angles =
            positions.expandedDimensions(axis: -1)
            * rope.invFreq.expandedDimensions(axis: 0)
        let cosP = cos(angles).expandedDimensions(axes: [0])
        let sinP = sin(angles).expandedDimensions(axes: [0])
        pooled = DeepseekV4Math.applyPartialRoPE(
            pooled, cos: cosP, sin: sinP, ropeDim: rope.dim)

        if let cache = v4Cache {
            if let existing = cache.getPooled(branch) {
                let merged = concatenated([existing, pooled], axis: 1)
                cache.setPooled(branch, value: merged)
                return merged
            } else {
                cache.setPooled(branch, value: pooled)
                return pooled
            }
        }
        return pooled
    }

    /// Overlap transform for compress_ratio=4. Expands (B, W, R, D) to
    /// (B, W, 2R, D) where the first R columns are the first half of
    /// the previous window's output and the last R columns are the
    /// current window's second half — gives each pool access to both
    /// chunks.
    private func overlapTransform(_ x: MLXArray, fillValue: Float) -> MLXArray {
        let B = x.dim(0)
        let W = x.dim(1)
        let R = x.dim(2)
        // Build output in two halves via concatenate; avoids
        // mutable-assignment patterns that mlx-swift Module doesn't
        // provide a clean API for.
        //
        // Layout:
        //   out[:, 0, :R]   = fill
        //   out[:, 1:, :R]  = x[:, :-1, :, :headDim]
        //   out[:,  :, R:]  = x[:, :, :, headDim:]
        let firstHalfAll = x[0..., 0..., 0..., 0..<headDim]  // (B, W, R, hd)
        // Shift: prepend a fill-window at position 0.
        let fillWindow = MLXArray.full(
            [B, 1, R, headDim], values: MLXArray(fillValue).asType(x.dtype))
        let shifted = concatenated(
            [fillWindow, firstHalfAll[0..., 0..<(W - 1), 0..., 0...]],
            axis: 1)  // (B, W, R, hd)
        let secondHalfAll = x[0..., 0..., 0..., headDim...]  // (B, W, R, hd)
        // Concat along the R axis: (B, W, 2R, hd).
        return concatenated([shifted, secondHalfAll], axis: 2)
    }
}

// MARK: - Indexer

/// Per-query top-k selector over the Compressor's pooled output.
/// Only present on compress_ratio=4 layers. Given `x` and `q_residual`
/// (the post-`q_norm` low-rank Q), projects Q into `n_heads`×`head_dim`,
/// scores against the Compressor's pooled keys, weights by a per-head
/// coefficient from `weights_proj`, and returns the top-`index_topk`
/// indices per query position.
public final class DeepseekV4Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let topK: Int
    let scale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(config: DeepseekV4Configuration, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.scale = 1.0 / sqrt(Float(headDim))
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, nHeads * headDim, bias: false)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, nHeads, bias: false)
        self._compressor.wrappedValue = DeepseekV4Compressor(
            config: config, compressRatio: compressRatio, headDim: headDim)
    }

    /// Returns top-k indices shape (B, L, k) into the pooled sequence
    /// of the attention's Compressor, or nil when there's nothing to
    /// select (empty pool).
    func callAsFunction(
        _ x: MLXArray,
        qResidual: MLXArray,
        rope: DeepseekV4RoPE,
        positionRope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int
    ) -> MLXArray? {
        let pooled = compressor(
            x, rope: rope, v4Cache: v4Cache,
            startPos: startPos, branch: .indexer)
        if pooled.dim(1) == 0 { return nil }

        let B = x.dim(0)
        let L = x.dim(1)
        var q = wqB(qResidual)
            .reshaped(B, L, nHeads, headDim)
            .transposed(0, 2, 1, 3)
        // Partial RoPE on Q using the plain (non-compressor) RoPE.
        let (cosT, sinT) = positionRope.cosSin(offset: startPos, length: L)
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(
            q, cos: cosQ, sin: sinQ, ropeDim: rope.dim)

        // scores: (B, nHeads, L, pooledLen). Match Python shape.
        // q is (B, nHeads, L, headDim); pooled is (B, pooledLen, headDim).
        // Expand pooled to (B, 1, pooledLen, headDim) for broadcast.
        let pooledBroad = pooled.expandedDimensions(axis: 1)
        var scores = q.asType(.float32).matmul(
            pooledBroad.asType(.float32).swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(0.0)) * MLXArray(scale)

        // weights: (B, L, nHeads) * n_heads^-0.5. Broadcast over the
        // pooled axis and sum over heads.
        let wRaw = weightsProj(x).asType(.float32)
            * MLXArray(1.0 / sqrt(Float(nHeads)))
        // Reshape scores sum axis: (B, 1, L, nHeads) multiply → sum.
        let wExpanded = wRaw.swappedAxes(-1, -2).expandedDimensions(axis: -1)
        scores = (scores * wExpanded).sum(axis: 1)  // (B, L, pooledLen)

        let k = min(topK, pooled.dim(1))
        let topIdx = argPartition(-scores, kth: k - 1, axis: -1)[
            .ellipsis, 0..<k]
        return topIdx
    }
}
