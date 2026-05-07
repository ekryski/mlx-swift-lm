// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// TriAttention V3 — KV cache eviction policy for long-context inference.
//
// Direct Swift port of the Python engine + scoring + policy in
// vllm-turboquant/vllm/v1/attention/triattention/. Same math, same
// semantics, no behavior changes — re-implemented in MLX so the M5 Max /
// Apple Silicon stack can run V3 + Tier 2 (eviction → longctx-svc) +
// Tier 3 (next-turn rehydrate) end-to-end alongside the AMD path.
//
// What V3 is, in 30 seconds:
//   - On prefill, K is accumulated layer-by-layer and scored against
//     calibration centers. The trig formula scores HIGH for tokens to
//     evict (it's an orthogonality measure, not alignment).
//   - When cache pressure crosses budget + divide_length, the policy
//     picks N cells to evict respecting a protected prefix window and
//     a recent-window slice, distributing eviction quota across
//     `n_segments` buckets to avoid clustering.
//   - Eviction is recorded in a per-sequence valid mask. Storage path
//     (raw fp16 / TurboQuant / etc) is orthogonal — V3 is just a
//     policy on top.
//
// This Swift port is paper-V3 only — Tier 1 (lambda_query_attn blend)
// is NOT included. Three independent AMD experiments today
// (NIAH-32K cliff, multi-question MRCR, low-budget heavy-eviction) all
// found Tier 1 produces zero measurable signal vs paper-V3. Until a
// workload shows benefit we skip the extra complexity here.
import Foundation
import MLX
import MLXNN

/// Configuration for the V3 engine. Mirrors Python `TriAttentionV3Config`.
public struct TriAttentionV3Config: Sendable {
    public var budget: Int
    public var divideLength: Int
    public var windowSize: Int
    public var prefixProtect: Int
    public var nSegments: Int
    public var warmupTokens: Int
    public var emaAlpha: Float
    public var adaptiveCalibration: Bool
    /// 0 = V1 global sort, 1 = V2 quota-only, 2 = V3 prefix + quota.
    public var hybridMode: Int
    public var boundarySkip: Int

    public init(
        budget: Int = 2048,
        divideLength: Int = 128,
        windowSize: Int = 128,
        prefixProtect: Int = 128,
        nSegments: Int = 8,
        warmupTokens: Int = 1024,
        emaAlpha: Float = 0.1,
        adaptiveCalibration: Bool = false,
        hybridMode: Int = 2,
        boundarySkip: Int = 0
    ) {
        self.budget = budget
        self.divideLength = divideLength
        self.windowSize = windowSize
        self.prefixProtect = prefixProtect
        self.nSegments = nSegments
        self.warmupTokens = warmupTokens
        self.emaAlpha = emaAlpha
        self.adaptiveCalibration = adaptiveCalibration
        self.hybridMode = hybridMode
        self.boundarySkip = boundarySkip
    }

    /// Read every knob from `VLLM_TRIATT_*` env vars (matches the
    /// Python config's `from_env`). Lets callers wire one config source
    /// across both AMD-Python and Apple-Swift halves of the stack.
    public static func fromEnv() -> TriAttentionV3Config {
        var cfg = TriAttentionV3Config()
        let env = ProcessInfo.processInfo.environment
        if let v = env["VLLM_TRIATT_BUDGET"], let n = Int(v) { cfg.budget = n }
        if let v = env["VLLM_TRIATT_HYBRID"], let n = Int(v) { cfg.hybridMode = n }
        if let v = env["VLLM_TRIATT_PREFIX"], let n = Int(v) { cfg.prefixProtect = n }
        if let v = env["VLLM_TRIATT_WINDOW"], let n = Int(v) { cfg.windowSize = n }
        if let v = env["VLLM_TRIATT_SEGMENTS"], let n = Int(v) { cfg.nSegments = n }
        if let v = env["VLLM_TRIATT_WARMUP"], let n = Int(v) { cfg.warmupTokens = n }
        if let v = env["VLLM_TRIATT_ADAPTIVE"], let n = Int(v) {
            cfg.adaptiveCalibration = (n != 0)
        }
        return cfg
    }
}

/// Per-sequence runtime state. Held inside the engine, keyed by seq id.
fileprivate final class SeqState {
    var validMask: MLXArray?
    var nEvicted: Int = 0
    var maxPos: Int = -1
    var evictRounds: Int = 0
    var pendingScores: MLXArray?  // [seq_len] fp32
    var pendingNBlocks: Int = 0
    var pendingSeqLen: Int = 0
    var pendingLayers: Set<Int> = []
}

/// V3 engine — calibration + scoring + per-sequence eviction state.
/// One instance per process. Calibration is global across sequences;
/// per-sequence state lives in `seqStates`.
public final class TriAttentionV3Engine: @unchecked Sendable {
    public let cfg: TriAttentionV3Config
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let nRot: Int
    public let freqCount: Int
    public let ropeTheta: Float

    /// RoPE angular frequencies omega[i] = 1 / theta^(2i/n_rot). [freqCount]
    public let omega: MLXArray
    /// Geometric offsets [1, 2, 4, ..., 65536]. [nOff]
    public let offsets: MLXArray

    // Per-(layer, kv_head) accumulators + centers, shape [nLayers*nKVHeads, freqCount].
    private var qSumReal: MLXArray
    private var qSumImag: MLXArray
    private var qSumAbs: MLXArray
    private var qPrevSumReal: MLXArray
    private var qPrevSumImag: MLXArray
    private var qPrevSumAbs: MLXArray
    public private(set) var centerReal: MLXArray?
    public private(set) var centerImag: MLXArray?
    public private(set) var centerAbs: MLXArray?

    public private(set) var qSamples: Int = 0
    public private(set) var qSamplesAtLastUpdate: Int = 0
    public private(set) var calibrated: Bool = false
    public private(set) var firstAttnLayer: Int = -1
    public private(set) var totalEvictRounds: Int = 0

    private var seqStates: [Int: SeqState] = [:]
    private let lock = NSLock()

    /// Tier 2 hook: callback fired AFTER each successful eviction round.
    /// Backend / longctx integration registers this to capture evicted-span
    /// text. Engine itself stays I/O-free; the callback owns tokenizer
    /// access + HTTP. Signature: (seqId, evictedPositions, nEvicted) -> Void
    public var evictionCallback: ((Int, [Int], Int) -> Void)?

    /// Per-layer cache registry — one entry per attention layer's
    /// TriAttentionKVCache. Populated by `registerCache` from cache
    /// init, drained by `unregisterCache` from cache deinit. Engine
    /// uses this to apply physical eviction compaction to every layer
    /// at finalize time. NSMapTable with weak values so dropped caches
    /// drop out automatically without us tracking lifecycle by hand.
    internal let cacheRegistry =
        NSMapTable<NSNumber, TriAttentionKVCache>(
            keyOptions: .strongMemory, valueOptions: .weakMemory
        )

    public init(
        cfg: TriAttentionV3Config,
        nLayers: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        ropeTheta: Float,
        nRot: Int? = nil
    ) {
        self.cfg = cfg
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        let resolvedNRot = nRot ?? headDim
        precondition(
            resolvedNRot % 2 == 0, "n_rot must be even (real/imag halves)"
        )
        self.nRot = resolvedNRot
        self.freqCount = resolvedNRot / 2
        self.ropeTheta = ropeTheta

        // omega[i] = 1 / theta^(2i/n_rot)
        let i = MLXArray(0..<self.freqCount).asType(.float32)
        let exponent = 2.0 * i / Float(self.nRot)
        self.omega = MLX.pow(MLXArray(ropeTheta), exponent).reciprocal()

        // Geometric offsets [1, 2, 4, ..., 65536]
        var offs: [Int32] = []
        var v: Int32 = 1
        while v <= 65536 {
            offs.append(v)
            v *= 2
        }
        self.offsets = MLXArray(offs).asType(.float32)

        let nTotal = nLayers * nKVHeads
        let zero = MLXArray.zeros([nTotal, self.freqCount], dtype: .float32)
        self.qSumReal = zero
        self.qSumImag = zero
        self.qSumAbs = zero
        self.qPrevSumReal = zero
        self.qPrevSumImag = zero
        self.qPrevSumAbs = zero
    }

    // MARK: - Calibration

    /// Accumulate Q stats for one attention layer. `q` is pre-RoPE,
    /// shape `[nTokens, nHeads, headDim]` float.
    public func accumulateQ(_ q: MLXArray, layerIdx: Int) {
        if calibrated && !cfg.adaptiveCalibration { return }
        guard layerIdx >= 0 && layerIdx < nLayers else { return }

        let fc = freqCount
        // Slice the rotated half and split real/imag (RoPE half-layout).
        let qRot = q[.ellipsis, ..<self.nRot].asType(.float32)
        let qReal = qRot[.ellipsis, ..<fc]                // [T, H, fc]
        let qImag = qRot[.ellipsis, fc..<self.nRot]       // [T, H, fc]
        let qAbs = MLX.sqrt(qReal * qReal + qImag * qImag + 1e-8)

        // Group H queries into nKVHeads groups of (heads_per_kv) and average.
        let headsPerKV = nHeads / nKVHeads
        let T = q.dim(0)
        let qrG = qReal.reshaped([T, nKVHeads, headsPerKV, fc]).mean(axis: 2)
        let qiG = qImag.reshaped([T, nKVHeads, headsPerKV, fc]).mean(axis: 2)
        let qaG = qAbs.reshaped([T, nKVHeads, headsPerKV, fc]).mean(axis: 2)

        let sumReal = qrG.sum(axis: 0)  // [nKVHeads, fc]
        let sumImag = qiG.sum(axis: 0)
        let sumAbs = qaG.sum(axis: 0)

        lock.lock()
        defer { lock.unlock() }

        let base = layerIdx * nKVHeads
        let lo = base
        let hi = base + nKVHeads
        qSumReal[lo..<hi] = qSumReal[lo..<hi] + sumReal
        qSumImag[lo..<hi] = qSumImag[lo..<hi] + sumImag
        qSumAbs[lo..<hi] = qSumAbs[lo..<hi] + sumAbs

        // Token counter — count once per pass on first attention layer seen.
        if firstAttnLayer < 0 || layerIdx < firstAttnLayer {
            firstAttnLayer = layerIdx
        }
        if layerIdx == firstAttnLayer {
            qSamples += T
            if !calibrated && qSamples >= cfg.warmupTokens {
                updateCalibrationLocked()
            }
        }
    }

    private func updateCalibrationLocked() {
        guard qSamples > 0 else { return }
        if !calibrated {
            let invN = Float(1.0) / Float(qSamples)
            centerReal = qSumReal * invN
            centerImag = qSumImag * invN
            centerAbs = qSumAbs * invN
            calibrated = true
        } else {
            let alpha = cfg.emaAlpha
            let newSamples = qSamples - qSamplesAtLastUpdate
            guard newSamples > 0 else { return }
            let invN = Float(1.0) / Float(newSamples)
            let newR = (qSumReal - qPrevSumReal) * invN
            let newI = (qSumImag - qPrevSumImag) * invN
            let newA = (qSumAbs - qPrevSumAbs) * invN
            centerReal = (1 - alpha) * centerReal! + alpha * newR
            centerImag = (1 - alpha) * centerImag! + alpha * newI
            centerAbs = (1 - alpha) * centerAbs! + alpha * newA
        }
        qPrevSumReal = qSumReal
        qPrevSumImag = qSumImag
        qPrevSumAbs = qSumAbs
        qSamplesAtLastUpdate = qSamples
    }

    // MARK: - Per-sequence valid-mask state

    /// Get the [seqLen] bool valid mask for `seqId`, growing if needed.
    public func getValidMask(seqId: Int, seqLen: Int) -> MLXArray {
        let st = seqStates[seqId] ?? {
            let s = SeqState()
            seqStates[seqId] = s
            return s
        }()
        if let m = st.validMask, m.dim(0) >= seqLen {
            return m[..<seqLen]
        }
        let new = MLXArray.ones([seqLen], dtype: .bool)
        if let m = st.validMask {
            // Carry forward the prior mask for already-seen positions.
            let oldLen = m.dim(0)
            new[..<oldLen] = m
        }
        st.validMask = new
        return new
    }

    public func nUsed(seqId: Int, seqLen: Int) -> Int {
        guard let st = seqStates[seqId], let m = st.validMask else {
            return seqLen
        }
        return m[..<seqLen].sum().item(Int.self)
    }

    public func shouldEvict(seqId: Int, seqLen: Int) -> Bool {
        guard calibrated else { return false }
        let used = nUsed(seqId: seqId, seqLen: seqLen)
        let effBudget = max(cfg.budget, cfg.windowSize + 1)
        return used > effBudget + cfg.divideLength
    }

    // MARK: - Per-layer score accumulation

    public func beginScoreRound(seqId: Int, seqLen: Int) {
        let st = seqStates[seqId] ?? {
            let s = SeqState()
            seqStates[seqId] = s
            return s
        }()
        st.pendingScores = MLXArray.zeros([seqLen], dtype: .float32)
        st.pendingNBlocks = 0
        st.pendingSeqLen = seqLen
        st.pendingLayers = []
    }

    /// Add one layer's score contribution to the pending buffer.
    /// `K` shape `[seqLen, nKVHeads, headDim]`.
    public func accumulateLayerScore(
        seqId: Int, layerIL: Int, K: MLXArray, maxPos: Int, windowThr: Int
    ) {
        guard calibrated, layerIL >= cfg.boundarySkip else { return }
        guard let st = seqStates[seqId], let pending = st.pendingScores else {
            return
        }
        // Shape drift guard — the AMD path occasionally fires layers with
        // mismatched K shape during compile/capture; skip those quietly.
        guard pending.dim(0) == K.dim(0) else { return }
        let valid = getValidMask(seqId: seqId, seqLen: pending.dim(0))
        let cb = layerIL * nKVHeads
        let cR = centerReal![cb..<(cb + nKVHeads)]
        let cI = centerImag![cb..<(cb + nKVHeads)]
        let cA = centerAbs![cb..<(cb + nKVHeads)]
        let layerScore = TriAttentionScoring.scoreCells(
            K: K.asType(.float32),
            centerReal: cR.asType(.float32),
            centerImag: cI.asType(.float32),
            centerAbs: cA.asType(.float32),
            omega: omega,
            offsets: offsets,
            maxPos: maxPos,
            validMask: valid,
            windowThr: windowThr,
            nRot: nRot
        )
        st.pendingScores = pending + layerScore
        st.pendingNBlocks += nKVHeads
        st.pendingLayers.insert(layerIL)
    }

    /// Run V3 policy on accumulated scores; return # positions evicted.
    @discardableResult
    public func finalizeEvictRound(seqId: Int) -> Int {
        guard let st = seqStates[seqId], let scoresBuf = st.pendingScores else {
            return 0
        }
        let nBlocks = st.pendingNBlocks
        let seqLen = st.pendingSeqLen
        st.pendingScores = nil
        st.pendingNBlocks = 0
        st.pendingSeqLen = 0
        st.pendingLayers = []

        let scores: MLXArray = nBlocks > 0
            ? scoresBuf / Float(nBlocks) : scoresBuf
        let valid = getValidMask(seqId: seqId, seqLen: seqLen)
        let used = valid.sum().item(Int.self)
        let nToEvict = used - cfg.budget
        guard nToEvict > 0 else { return 0 }

        // Live-position scan to find max + window threshold.
        let positions = MLXArray(0..<Int32(seqLen))
        let livePos = positions[valid]
        guard livePos.size > 0 else { return 0 }
        let maxPos = livePos.max().item(Int.self)
        let windowThr = maxPos - cfg.windowSize + 1
        let prefixLo = cfg.hybridMode == 2 ? cfg.prefixProtect : 0

        let evictPos = TriAttentionPolicy.selectEvictions(
            scores: scores,
            valid: valid,
            nToEvict: nToEvict,
            windowThr: windowThr,
            prefixLo: prefixLo,
            nSegments: cfg.nSegments,
            mode: cfg.hybridMode
        )
        guard !evictPos.isEmpty else { return 0 }

        // Apply the eviction: mark valid=false at each evicted position.
        // Cheap CPU loop — eviction is off the hot decode path and
        // typically picks O(100s-1000s) cells.
        var newValid = valid
        for p in evictPos {
            newValid[p] = MLXArray(false)
        }
        st.validMask = newValid
        st.nEvicted += evictPos.count
        st.maxPos = maxPos
        st.evictRounds += 1
        totalEvictRounds += 1

        // Tier 2 callback. Mirrors Python — exceptions are suppressed so
        // a misbehaving rescue path can't crash decoding.
        if let cb = evictionCallback, !evictPos.isEmpty {
            cb(seqId, evictPos, evictPos.count)
        }
        return evictPos.count
    }

    public func setEvictionCallback(_ cb: ((Int, [Int], Int) -> Void)?) {
        self.evictionCallback = cb
    }
}
