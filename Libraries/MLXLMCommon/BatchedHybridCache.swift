// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

// MARK: - BatchedMambaCache

/// Batched cache for GatedDeltaNet (and Mamba-style) layers.
///
/// Unlike attention KV cache where state grows with sequence length,
/// GDN state is fixed-size per slot:
///   - `convState`: rolling Conv1d window, `[B, kernel-1, convDim]`
///   - `recState`:  recurrent SSM state, `[B, Hv, Dv, Dk]` in fp32
///
/// Pre-allocates one tensor per state shape, indexed by slot. Slot lifetime is
/// managed by `addSlot` (returns a slot index, zero-inits state) and
/// `removeSlot(_:)` (swap-from-end with the last active slot).
///
/// IMPORTANT: zero-init happens in `addSlot`, NOT in `init`. After
/// `removeSlot` swaps a freed slot's state into the last position, the next
/// `addSlot` must wipe stale state in that newly-allocated slot or the GDN
/// recurrence will leak the previous request's history.
public class BatchedMambaCache {
    public let maxBatch: Int
    public let kernelMinusOne: Int
    public let convDim: Int
    public let Hv: Int
    public let Dv: Int
    public let Dk: Int

    /// Conv state: [maxBatch, kernel-1, convDim]
    public var convState: MLXArray

    /// Recurrent state: [maxBatch, Hv, Dv, Dk] fp32
    public var recState: MLXArray

    /// Active slot count (first `active` slots are in use)
    public var active: Int = 0

    public init(
        maxBatch: Int,
        kernelMinusOne: Int,
        convDim: Int,
        Hv: Int,
        Dv: Int,
        Dk: Int,
        dtype: DType = .bfloat16
    ) {
        self.maxBatch = maxBatch
        self.kernelMinusOne = kernelMinusOne
        self.convDim = convDim
        self.Hv = Hv
        self.Dv = Dv
        self.Dk = Dk

        // Don't bother zeroing here — slots get zeroed lazily in addSlot.
        // (We still allocate so the buffer exists with the right shape.)
        self.convState = MLXArray.zeros(
            [maxBatch, kernelMinusOne, convDim], dtype: dtype)
        self.recState = MLXArray.zeros(
            [maxBatch, Hv, Dv, Dk], dtype: .float32)
        eval(self.convState, self.recState)
    }

    /// Add a new slot. Returns the slot index. **Zero-inits the new slot's
    /// conv + rec state** so reused slot positions don't leak prior state.
    @discardableResult
    public func addSlot() -> Int {
        precondition(active < maxBatch, "BatchedMambaCache: out of slots")
        let slot = active
        active += 1
        zeroSlot(slot)
        return slot
    }

    /// Remove a slot by swapping the last active slot into its position.
    /// After the swap, `active` is decremented — the freed tail position is
    /// effectively garbage and will be wiped on its next `addSlot`.
    public func removeSlot(_ slot: Int) {
        precondition(slot >= 0 && slot < active, "BatchedMambaCache: bad slot \(slot)")
        let last = active - 1
        if slot != last {
            // Copy last → slot for both states (pure tensor assign, no concat)
            convState[slot, 0..., 0...] = convState[last, 0..., 0...]
            recState[slot, 0..., 0..., 0...] = recState[last, 0..., 0..., 0...]
        }
        active -= 1
    }

    /// Slice for the active prefix. Caller mutates the returned tensors only
    /// via `writeback(...)`.
    public func slice(active: Int) -> (conv: MLXArray, rec: MLXArray) {
        precondition(active <= self.active, "BatchedMambaCache: slice exceeds active")
        let conv = convState[..<active, 0..., 0...]
        let rec = recState[..<active, 0..., 0..., 0...]
        return (conv, rec)
    }

    /// Writeback updated conv + rec state for the first `active` slots.
    /// `conv` shape: [active, kernel-1, convDim].
    /// `rec` shape:  [active, Hv, Dv, Dk] (fp32).
    public func writeback(conv: MLXArray, rec: MLXArray) {
        let B = conv.dim(0)
        precondition(B <= active, "writeback B (\(B)) exceeds active (\(active))")
        convState[..<B, 0..., 0...] = conv
        // Force fp32 to match storage; defensive — caller should already be fp32.
        recState[..<B, 0..., 0..., 0...] =
            (rec.dtype == .float32) ? rec : rec.asType(.float32)
    }

    /// Reset all slots.
    public func reset() {
        active = 0
        // No need to wipe storage — addSlot wipes per-slot on reuse.
    }

    // MARK: - Private

    /// Zero a single slot's conv + rec state in place.
    private func zeroSlot(_ slot: Int) {
        let convZ = MLXArray.zeros(
            [kernelMinusOne, convDim], dtype: convState.dtype)
        let recZ = MLXArray.zeros([Hv, Dv, Dk], dtype: .float32)
        convState[slot, 0..., 0...] = convZ
        recState[slot, 0..., 0..., 0...] = recZ
    }
}

// MARK: - BatchedHybridCache

/// Polymorphic per-layer batched cache for hybrid models that interleave
/// attention layers and linear (GDN/Mamba-style) layers.
///
/// Layers are kept in lockstep: every `addSlot`/`removeSlot` propagates to
/// every layer cache, so `active` is consistent across the model.
public class BatchedHybridCache {
    public enum BatchedLayerCache {
        case attention(BatchedKVCache)
        case gdn(BatchedMambaCache)
    }

    public let layers: [BatchedLayerCache]

    /// All layers stay in lockstep — `active` is read from the first layer.
    public var active: Int {
        guard let first = layers.first else { return 0 }
        switch first {
        case .attention(let c): return c.active
        case .gdn(let c): return c.active
        }
    }

    public init(layers: [BatchedLayerCache]) {
        precondition(!layers.isEmpty, "BatchedHybridCache: must have at least one layer")
        self.layers = layers
    }

    /// Add a new request slot to every layer. Returns the slot index
    /// (consistent across all layers).
    @discardableResult
    public func addSlot() -> Int {
        var assigned = -1
        for layer in layers {
            let slot: Int
            switch layer {
            case .attention(let c): slot = c.addRequest()
            case .gdn(let c): slot = c.addSlot()
            }
            if assigned < 0 { assigned = slot }
            assert(slot == assigned,
                   "BatchedHybridCache: layer slot \(slot) drifted from \(assigned)")
        }
        return assigned
    }

    /// Remove a slot from every layer. Swap-from-end semantics; caller is
    /// responsible for re-mapping any external slot→request bookkeeping.
    public func removeSlot(_ slot: Int) {
        for layer in layers {
            switch layer {
            case .attention(let c):
                // BatchedKVCache uses swap-from-end too; mirror it.
                let last = c.active - 1
                if slot != last {
                    c.keys[slot, 0..., 0..., 0...] = c.keys[last, 0..., 0..., 0...]
                    c.values[slot, 0..., 0..., 0...] = c.values[last, 0..., 0..., 0...]
                    c.offsets[slot] = c.offsets[last]
                }
                c.offsets[last] = 0
                c.active -= 1
            case .gdn(let c):
                c.removeSlot(slot)
            }
        }
    }

    /// Reset all slots across every layer.
    public func reset() {
        for layer in layers {
            switch layer {
            case .attention(let c): c.reset()
            case .gdn(let c): c.reset()
            }
        }
    }
}

// MARK: - BatchedHybridLLM

/// Models that can do fully-batched decode through a hybrid (attention + GDN)
/// stack should conform to this protocol. The bridge can `as?`-cast and
/// dispatch fully batched if available; otherwise it falls back to the
/// sequential per-request decode path.
public protocol BatchedHybridLLM: AnyObject {
    /// Single-step batched decode. `inputs` shape: `[B, 1]`. Returns
    /// `[B, 1, vocab]` logits.
    func fullyBatchedDecode(_ inputs: MLXArray, caches: BatchedHybridCache) -> MLXArray

    /// Build a fresh `BatchedHybridCache` sized for `maxBatch` requests.
    func newBatchedHybridCache(
        maxBatch: Int, parameters: GenerateParameters?
    ) -> BatchedHybridCache
}
