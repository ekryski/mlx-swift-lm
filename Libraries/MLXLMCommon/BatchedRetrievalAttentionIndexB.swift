// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedRetrievalAttentionIndexB — block-feature selector index with an
// explicit per-request batch dimension. Each (B, KV-head) gets its own
// running mean of block features in the JL-projected content space and
// produces its own top-K block selection per decode step.
//
// Shape conventions:
//   fineBlockFeatures:    [B, nKVHeads, nFineBlocks, contentDim]
//   coarseBlockFeatures:  [B, nKVHeads, nCoarseBlocks, contentDim]
//   q (input to project): [B, nKVHeads, dHead]
//   projectedQ:           [B, nKVHeads, contentDim]
//   topK output:          [B, nKVHeads, K_fine]  / [B, nKVHeads, K_coarse]
//
// Storage decision: drops per-token feature storage; maintains block
// features INCREMENTALLY via running mean. This is the lower-memory
// path — saves on the order of tens of GB at B=8 T=128K for a 14B model.
//
// Running-mean math (per block b, per content dim c):
//   mean_new = mean_old + (added - mean_old) / new_count
// Cheap to update as each new K row arrives — no scan over T.

import Foundation
import MLX

public final class BatchedRetrievalAttentionIndexB {

    public let config: RetrievalAttentionConfig
    public let B: Int
    public let dHead: Int
    public let nKVHeads: Int
    public let ropeBase: Float
    public let layerIdx: Int

    /// JL projection matrix `[dHead, contentDim]`. Shared across all
    /// (B, KV-head) — JL is a model-architecture property, not per-slot.
    private var jlMatrixT: MLXArray?
    public var jlW: MLXArray? { jlMatrixT }

    /// Block-pooled features, `[B, nKVH, blockCap, contentDim]` fp32.
    /// Grown geometrically. Valid prefix is `[0..<currentFineBlocks]`.
    /// Only contentDim — trig features are skipped (λ=0 ship config).
    private var fineBlockBuffer: MLXArray?
    private var coarseBlockBuffer: MLXArray?
    private var fineBlockCapacity: Int = 0
    private var coarseBlockCapacity: Int = 0

    /// Per (B, KV-head) running-sum tracking for the partial last block.
    /// Currently uses in-place rolling mean — no separate sum tensors
    /// needed since `mean_new = mean_old + (added - mean_old) / count`.
    private var fineTokensInCurrentBlock: [[Int]]   // [B][nKVH]
    private var coarseTokensInCurrentBlock: [[Int]]

    /// Per-slot populated count. Decode / prefill can advance independently
    /// (e.g. continuous batching) so each slot tracks its own T.
    private var populatedPerSlot: [Int]
    /// Convenience: max-populated across slots (drives block-buffer capacity).
    public var seqLenMax: Int { populatedPerSlot.max() ?? 0 }
    /// Convenience: min populated (for rectangular T assumption check).
    public var seqLenMin: Int { populatedPerSlot.min() ?? 0 }
    /// Per-slot getter.
    public func seqLen(slot: Int) -> Int { populatedPerSlot[slot] }

    /// Block counts driven by `seqLenMax` — buffer holds enough for the
    /// "tallest" slot; shorter slots have zeros at the tail.
    public var currentFineBlocks: Int {
        let m = seqLenMax
        guard m > 0 else { return 0 }
        return (m + config.fineBlockSize - 1) / config.fineBlockSize
    }
    public var currentCoarseBlocks: Int {
        let m = seqLenMax
        guard m > 0 else { return 0 }
        return (m + config.coarseBlockSize - 1) / config.coarseBlockSize
    }

    /// Public views (valid prefix only).
    public var fineBlockFeatures: MLXArray? {
        guard let buf = fineBlockBuffer, currentFineBlocks > 0 else { return nil }
        return buf[0..., 0..., ..<currentFineBlocks, 0...]
    }
    public var coarseBlockFeatures: MLXArray? {
        guard let buf = coarseBlockBuffer, currentCoarseBlocks > 0 else { return nil }
        return buf[0..., 0..., ..<currentCoarseBlocks, 0...]
    }

    public init(
        config: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        B: Int,
        dHead: Int,
        nKVHeads: Int,
        ropeBase: Float = 10_000.0,
        layerIdx: Int
    ) {
        precondition(B > 0, "B must be > 0")
        self.config = config
        self.B = B
        self.dHead = dHead
        self.nKVHeads = nKVHeads
        self.ropeBase = ropeBase
        self.layerIdx = layerIdx
        self.populatedPerSlot = Array(repeating: 0, count: B)
        self.fineTokensInCurrentBlock =
            Array(repeating: Array(repeating: 0, count: nKVHeads), count: B)
        self.coarseTokensInCurrentBlock =
            Array(repeating: Array(repeating: 0, count: nKVHeads), count: B)
        // `usesTrigFeatures` path not supported here; selector ships
        // λ=0 (pure content). Revisit if a customer flips lambdaPos > 0.
        precondition(!config.usesTrigFeatures,
            "BatchedRetrievalAttentionIndexB supports λ=0 (no trig) only")
    }

    private func ensureJL() {
        guard jlMatrixT == nil else { return }
        let W = retrievalAttentionJLProjection(dHead: dHead, config: config)
        jlMatrixT = W.transposed(1, 0)
        eval(jlMatrixT!)
    }

    /// Update with a chunk of new K rows for ALL slots and ALL heads.
    ///
    /// - Parameter newKeys: `[B, nKVHeads, L, dHead]` post-RoPE keys.
    ///   All slots get the same L — assumes rectangular prefill / decode.
    ///   For continuous batching where slots have different L per step,
    ///   call this once per (slot, L) — ragged path is a future concern.
    public func update(newKeys: MLXArray) {
        precondition(newKeys.shape.count == 4,
            "expected [B, nKVHeads, L, dHead], got \(newKeys.shape)")
        precondition(newKeys.dim(0) == B, "B mismatch (got \(newKeys.dim(0)) expected \(B))")
        precondition(newKeys.dim(1) == nKVHeads, "nKVHeads mismatch")
        precondition(newKeys.dim(3) == dHead, "dHead mismatch")
        ensureJL()
        let L = newKeys.dim(2)
        let WT = jlMatrixT!

        // Project all (B, KV-head) tokens at once.
        //   [B, nKVH, L, dHead] @ [dHead, contentDim] = [B, nKVH, L, contentDim]
        let content = matmul(newKeys.asType(.float32), WT).asType(.float32)
        // Force materialization here — downstream blockwise pool reads
        // contiguous slabs, and lazy graph through index slicing showed up
        // as a perf cliff in the single-batch index path.
        eval(content)

        for slot in 0..<B {
            let oldT = populatedPerSlot[slot]
            let newT = oldT + L
            populatedPerSlot[slot] = newT
        }

        // Grow buffers BEFORE writing.
        let contentDim = content.dim(3)
        ensureBlockBuffer(isFine: true, contentDim: contentDim)
        if config.coarseRescueEnabled {
            ensureBlockBuffer(isFine: false, contentDim: contentDim)
        }

        // For prefill (L large) recompute affected blocks via slab mean
        // — cheaper than running-sum because we're rewriting the whole
        // tail. For decode (L=1) running-mean wins: just bump 1 partial-
        // block update.
        if L == 1 {
            updateDecodeStep(content: content, contentDim: contentDim)
        } else {
            updatePrefillChunk(content: content, L: L, contentDim: contentDim)
        }
    }

    /// L=1 decode update — running-mean of the partial last block per slot.
    private func updateDecodeStep(content: MLXArray, contentDim: Int) {
        // content: [B, nKVH, 1, contentDim] → squeeze L → [B, nKVH, contentDim]
        let added = content[0..., 0..., 0, 0...]
        // For each slot, fold into fine + coarse partial blocks.
        // mean_new = mean_old + (added - mean_old) / new_count
        updateRunningMean(isFine: true, added: added)
        if config.coarseRescueEnabled {
            updateRunningMean(isFine: false, added: added)
        }
    }

    /// Closed-form rolling-mean update of the per-slot partial last block.
    private func updateRunningMean(isFine: Bool, added: MLXArray) {
        let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
        let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
        for slot in 0..<B {
            let popPost = populatedPerSlot[slot]
            // Block index of the new token = (popPost - 1) / bs.
            let blockIdx = (popPost - 1) / bs
            // Count of tokens now in this block (1..bs).
            let countInBlock = popPost - blockIdx * bs
            let addedRow = added[slot, 0..., 0...]  // [nKVH, contentDim]
            let oldMean = buf[slot, 0..., blockIdx, 0...]
            let countF = Float(countInBlock)
            let newMean: MLXArray
            if countInBlock == 1 {
                // First token in this block = mean = added.
                newMean = addedRow
            } else {
                newMean = oldMean + (addedRow - oldMean) / countF
            }
            buf[slot, 0..., blockIdx, 0...] = newMean
        }
    }

    /// Multi-token prefill update — recompute every affected block via
    /// slab-mean. Cheaper than per-token running-mean when L is large.
    private func updatePrefillChunk(content: MLXArray, L: Int, contentDim: Int) {
        let rectangular: Bool = {
            guard B > 0 else { return false }
            let firstPop = populatedPerSlot[0]
            return populatedPerSlot.allSatisfy { $0 == firstPop }
        }()

        if rectangular {
            updatePrefillRectangular(content: content, L: L, contentDim: contentDim)
        } else {
            // Fallback per-slot path (ragged-T continuous batching path).
            for slot in 0..<B {
                let slotContent = content[slot ..< (slot + 1), 0..., 0..., 0...]
                updateSlotPrefillSlice(
                    slot: slot, slotContent: slotContent,
                    L: L, contentDim: contentDim)
            }
        }
    }

    /// Rectangular prefill — all slots at the same oldT. Vectorize across B.
    private func updatePrefillRectangular(content: MLXArray, L: Int, contentDim: Int) {
        let newT = populatedPerSlot[0]
        let oldT = newT - L
        updateBlockRectangularInPlace(
            content: content, oldT: oldT, newT: newT, L: L,
            isFine: true)
        if config.coarseRescueEnabled {
            updateBlockRectangularInPlace(
                content: content, oldT: oldT, newT: newT, L: L,
                isFine: false)
        }
    }

    /// Per-block mean recompute for the rectangular case.
    /// content: [B, nKVH, L, contentDim]
    private func updateBlockRectangularInPlace(
        content: MLXArray, oldT: Int, newT: Int, L: Int, isFine: Bool
    ) {
        let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
        let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
        let firstBlock = oldT / bs
        let lastBlock = (newT - 1) / bs  // inclusive
        for blk in firstBlock ... lastBlock {
            let blockStartAbs = blk * bs
            let blockEndAbs = min(blockStartAbs + bs, newT)
            let leftStart = max(0, blockStartAbs)
            let leftEnd = min(oldT, blockEndAbs)
            let leftCount = max(0, leftEnd - leftStart)
            let rightStart = max(oldT, blockStartAbs)
            let rightEnd = blockEndAbs
            let rightCount = max(0, rightEnd - rightStart)
            let cStart = rightStart - oldT
            let cEnd = rightEnd - oldT
            let totalCount = leftCount + rightCount
            precondition(totalCount > 0, "block must have >=1 token")
            let rightSlice = content[0..., 0..., cStart ..< cEnd, 0...]
            let rightSum = rightSlice.sum(axis: 2)   // [B, nKVH, contentDim]
            if leftCount == 0 {
                let mean = rightSum / Float(totalCount)
                buf[0..., 0..., blk, 0...] = mean
            } else {
                let oldMean = buf[0..., 0..., blk, 0...]
                let leftSum = oldMean * Float(leftCount)
                let mean = (leftSum + rightSum) / Float(totalCount)
                buf[0..., 0..., blk, 0...] = mean
            }
        }
    }

    /// Per-slot prefill slice update. Slow path for non-rectangular.
    private func updateSlotPrefillSlice(
        slot: Int, slotContent: MLXArray, L: Int, contentDim: Int
    ) {
        let newT = populatedPerSlot[slot]
        let oldT = newT - L
        for isFine in [true, false] {
            if !isFine && !config.coarseRescueEnabled { continue }
            let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
            let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
            let firstBlock = oldT / bs
            let lastBlock = (newT - 1) / bs
            for blk in firstBlock ... lastBlock {
                let blockStartAbs = blk * bs
                let blockEndAbs = min(blockStartAbs + bs, newT)
                let leftStart = blockStartAbs
                let leftEnd = min(oldT, blockEndAbs)
                let leftCount = max(0, leftEnd - leftStart)
                let rightStart = max(oldT, blockStartAbs)
                let rightEnd = blockEndAbs
                let rightCount = max(0, rightEnd - rightStart)
                let cStart = rightStart - oldT
                let cEnd = rightEnd - oldT
                let totalCount = leftCount + rightCount
                let rightSlice = slotContent[0..., 0..., cStart ..< cEnd, 0...]
                let rightSum = rightSlice.sum(axis: 2)   // [1, nKVH, contentDim]
                if leftCount == 0 {
                    let mean = rightSum / Float(totalCount)
                    buf[slot ..< (slot + 1), 0..., blk, 0...] = mean
                } else {
                    let oldMean = buf[slot ..< (slot + 1), 0..., blk, 0...]
                    let leftSum = oldMean * Float(leftCount)
                    let mean = (leftSum + rightSum) / Float(totalCount)
                    buf[slot ..< (slot + 1), 0..., blk, 0...] = mean
                }
            }
        }
    }

    private func ensureBlockBuffer(isFine: Bool, contentDim: Int) {
        let needed = isFine ? currentFineBlocks : currentCoarseBlocks
        let cap = isFine ? fineBlockCapacity : coarseBlockCapacity
        let existing = isFine ? fineBlockBuffer : coarseBlockBuffer
        if cap >= needed && existing != nil {
            return
        }
        // Geometric growth — contentDim is small (16) and block counts at
        // T=128K with fineBS=64 = 2048 blocks. At B=8, nKVH=8, contentDim=16
        // that's ~8 MB fp32 — cheap.
        let chunkSize = 256
        var newCap = max(cap, chunkSize)
        while newCap < needed { newCap *= 2 }
        let dtype: DType = .float32
        let newBuf = MLXArray.zeros([B, nKVHeads, newCap, contentDim], dtype: dtype)
        if cap > 0, let old = existing {
            newBuf[0..., 0..., ..<cap, 0...] = old[0..., 0..., ..<cap, 0...]
        }
        if isFine {
            fineBlockBuffer = newBuf
            fineBlockCapacity = newCap
        } else {
            coarseBlockBuffer = newBuf
            coarseBlockCapacity = newCap
        }
    }

    // MARK: - scoring + top-K

    /// Project decode queries for all (B, KV-heads) in one batched matmul.
    /// - Parameter q: `[B, nKVHeads, dHead]` decode-step query (rep-per-group).
    /// - Returns: `[B, nKVHeads, contentDim]`.
    public func projectQueriesBatched(_ q: MLXArray) -> MLXArray {
        precondition(q.shape == [B, nKVHeads, dHead],
            "expected [\(B), \(nKVHeads), \(dHead)], got \(q.shape)")
        ensureJL()
        let WT = jlMatrixT!
        // [B, nKVH, dHead] @ [dHead, contentDim] = [B, nKVH, contentDim]
        return matmul(q.asType(.float32), WT)
    }

    /// Per-(B, KV-head) top-K fine block starts.
    /// - Parameter projectedQ: `[B, nKVHeads, contentDim]`
    /// - Returns: `[B, nKVHeads, K_fine]` int32 token-position block starts.
    public func topKFineBlockStarts(projectedQ: MLXArray) -> MLXArray {
        guard let features = fineBlockFeatures else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        let k = config.effectiveFineTopK(seqLen: seqLenMax)
        return computeTopK(
            features: features, projectedQ: projectedQ,
            k: k, blockSize: config.fineBlockSize)
    }

    public func topKCoarseBlockStarts(projectedQ: MLXArray) -> MLXArray {
        guard config.coarseRescueEnabled,
              let features = coarseBlockFeatures
        else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        return computeTopK(
            features: features, projectedQ: projectedQ,
            k: config.coarseTopK, blockSize: config.coarseBlockSize)
    }

    /// Score + argPartition on a [B, nKVH, nBlocks, contentDim] buffer.
    /// Returns top-K block starts `[B, nKVH, k]` in token coords.
    private func computeTopK(
        features: MLXArray, projectedQ: MLXArray, k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = features.dim(2)
        let take = min(k, nBlocks)
        guard take > 0 else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        // Score: per (B, nKVH, block, content) inner-product with projectedQ.
        //   features: [B, nKVH, nBlocks, contentDim]
        //   projQ:    [B, nKVH, contentDim] → reshape [B, nKVH, 1, contentDim]
        let qExp = projectedQ.reshaped(B, nKVHeads, 1, projectedQ.dim(2))
        let scores = (features * qExp).sum(axis: -1)  // [B, nKVH, nBlocks]
        // argPartition on axis -1 picks top-`take` at positions [N-take..N).
        let pivot = nBlocks - take
        let partitioned: MLXArray
        if pivot <= 0 {
            let rangeArr = MLXArray(0..<Int32(nBlocks))
                .reshaped(1, 1, nBlocks)
            partitioned = broadcast(rangeArr, to: [B, nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(scores, kth: pivot, axis: -1)
        }
        let topKIdx = partitioned[0..., 0..., (nBlocks - take)...]
        return topKIdx * Int32(blockSize)
    }

    /// Per-(B, KV-head) sorted gather list of ALL positions to attend
    /// (static prefix + sliding window + top-K fine + optional top-K
    /// coarse). Padded to fixed K_padded for the kernel grid; positions
    /// are clipped to `[0, T-1]` so OOB protection in the kernel doesn't
    /// fire.
    ///
    /// All slots share the same static / sliding range when rectangular.
    ///
    // MARK: - F-83 sparse-prefill API (L > 1)

    /// Project a chunk of queries (L > 1) for the selector.
    ///
    /// Mirrors `projectQueriesBatched` but with an L dimension on the
    /// query side. Used by the sparse-prefill path to compute per-chunk
    /// union top-K block selections.
    ///
    /// - Parameter q: `[B, nKVHeads, L, dHead]` post-RoPE queries
    ///   (rep-per-group across the GQA groups).
    /// - Returns: `[B, nKVHeads, L, contentDim]`.
    public func projectQueriesBatchedL(_ q: MLXArray) -> MLXArray {
        precondition(q.shape.count == 4,
            "expected [B, nKVHeads, L, dHead], got \(q.shape)")
        precondition(q.dim(0) == B && q.dim(1) == nKVHeads && q.dim(3) == dHead,
            "expected [\(B), \(nKVHeads), L, \(dHead)], got \(q.shape)")
        ensureJL()
        let WT = jlMatrixT!
        // [B, nKVH, L, dHead] @ [dHead, contentDim] = [B, nKVH, L, contentDim]
        return matmul(q.asType(.float32), WT)
    }

    /// Per-(B, KV-head) top-K fine block starts, union-pooled across L
    /// chunk queries. For each KV head, a block's score is the *max*
    /// over the L queries of `feature . projectedQ_l`. Top-K is taken
    /// on the max-pooled score. NSA GQA pattern: queries within a chunk
    /// in the same KV head all select the same blocks.
    ///
    /// - Parameter projectedQL: `[B, nKVHeads, L, contentDim]` —
    ///   typically the output of `projectQueriesBatchedL`.
    /// - Returns: `[B, nKVHeads, K_fine]` int32 token-position block
    ///   starts. Shape contract matches `topKFineBlockStarts` so the
    ///   downstream mask/gather builder can ingest either.
    public func topKFineBlockStartsUnionL(projectedQL: MLXArray) -> MLXArray {
        guard let features = fineBlockFeatures else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        let k = config.effectiveFineTopK(seqLen: seqLenMax)
        return computeTopKUnionL(
            features: features, projectedQL: projectedQL,
            k: k, blockSize: config.fineBlockSize)
    }

    /// Per-(B, KV-head) top-K coarse block starts, union-pooled across L.
    /// Mirrors `topKCoarseBlockStarts` for the prefill path.
    public func topKCoarseBlockStartsUnionL(projectedQL: MLXArray) -> MLXArray {
        guard config.coarseRescueEnabled,
              let features = coarseBlockFeatures
        else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        return computeTopKUnionL(
            features: features, projectedQL: projectedQL,
            k: config.coarseTopK, blockSize: config.coarseBlockSize)
    }

    /// Union-pool over L → argPartition top-K. Same shape contract as
    /// `computeTopK` so callers can swap between L=1 and L>1 paths.
    private func computeTopKUnionL(
        features: MLXArray,    // [B, nKVH, nBlocks, contentDim]
        projectedQL: MLXArray, // [B, nKVH, L, contentDim]
        k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = features.dim(2)
        let take = min(k, nBlocks)
        guard take > 0 else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        // perQ[b, h, blk, l] = features[b, h, blk, :] . projectedQL[b, h, l, :]
        // Computed via batched matmul on the last dim. Materializing the
        // elementwise [B, nKVH, nBlocks, L, contentDim] tensor would be
        // pathological at long context — matmul keeps peak memory bounded.
        //
        //   features:    [B, nKVH, nBlocks, contentDim]
        //   projectedQL: [B, nKVH, L, contentDim] -> [B, nKVH, contentDim, L]
        // result: [B, nKVH, nBlocks, L]
        let perQ = matmul(features, projectedQL.transposed(0, 1, 3, 2))
        // Max-pool across L → [B, nKVH, nBlocks]. NSA union semantics.
        let scores = perQ.max(axis: -1)
        let pivot = nBlocks - take
        let partitioned: MLXArray
        if pivot <= 0 {
            let rangeArr = MLXArray(0..<Int32(nBlocks))
                .reshaped(1, 1, nBlocks)
            partitioned = broadcast(rangeArr, to: [B, nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(scores, kth: pivot, axis: -1)
        }
        let topKIdx = partitioned[0..., 0..., (nBlocks - take)...]
        return topKIdx * Int32(blockSize)
    }

    /// - Parameters:
    ///   - projectedQ: `[B, nKVHeads, contentDim]`
    ///   - T: rectangular cache length (caller passes K.dim(2)).
    /// - Returns: (gather [B, nKVHeads, K_padded] int32, K_padded)
    public func perKVHeadGatherBatched(
        projectedQ: MLXArray, seqLen T: Int
    ) -> (MLXArray, Int) {
        let staticInit = config.staticInit
        let slidingWindow = config.slidingWindow
        let fineBS = config.fineBlockSize
        let coarseBS = config.coarseBlockSize
        let kFineEff = min(config.effectiveFineTopK(seqLen: T),
                           fineBlockFeatures?.dim(2) ?? 0)
        let kCoarseEff = config.coarseRescueEnabled
            ? min(config.coarseTopK, coarseBlockFeatures?.dim(2) ?? 0)
            : 0
        let staticEnd = min(staticInit, T)
        let slidingStart = max(0, T - slidingWindow)
        let staticCount = staticEnd
        let slidingCount = max(0, T - slidingStart)
        let kPadded = staticCount + slidingCount
            + kFineEff * fineBS + kCoarseEff * coarseBS

        // Static + sliding ranges — same across (B, nKVH).
        let staticPositions = staticCount > 0
            ? MLXArray(Int32(0)..<Int32(staticEnd))
            : MLXArray.zeros([0], dtype: .int32)
        let slidingPositions = slidingCount > 0
            ? MLXArray(Int32(slidingStart)..<Int32(T))
            : MLXArray.zeros([0], dtype: .int32)
        let baseSS = concatenated([staticPositions, slidingPositions], axis: 0)
            .reshaped(1, 1, staticCount + slidingCount)
        let staticBroadcast = broadcast(
            baseSS, to: [B, nKVHeads, staticCount + slidingCount])

        var pieces: [MLXArray] = [staticBroadcast]
        if kFineEff > 0 {
            let fineStarts = topKFineBlockStarts(projectedQ: projectedQ)
            // expand to per-token block positions:
            //   [B, nKVH, kFineEff] + [1, 1, 1, fineBS] = [B, nKVH, kFineEff, fineBS]
            let offsets = MLXArray(0..<Int32(fineBS)).reshaped(1, 1, 1, fineBS)
            let expanded = fineStarts.expandedDimensions(axis: 3) + offsets
            pieces.append(expanded.reshaped(B, nKVHeads, kFineEff * fineBS))
        }
        if kCoarseEff > 0 {
            let coarseStarts = topKCoarseBlockStarts(projectedQ: projectedQ)
            let offsets = MLXArray(0..<Int32(coarseBS)).reshaped(1, 1, 1, coarseBS)
            let expanded = coarseStarts.expandedDimensions(axis: 3) + offsets
            pieces.append(expanded.reshaped(B, nKVHeads, kCoarseEff * coarseBS))
        }
        let unsorted = concatenated(pieces, axis: 2)  // [B, nKVH, kPadded]
        let clipped = clip(unsorted, min: Int32(0), max: Int32(T - 1))
        // `sorted` on int32 returns int64; group SDPA kernel takes int32
        // gather. Force back to int32 before handing to the kernel.
        let sortedGather = sorted(clipped, axis: -1).asType(.int32)
        return (sortedGather, kPadded)
    }
}
