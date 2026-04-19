import Foundation

// MARK: - Parallel Inference Slots

enum SlotState: String {
    case idle
    case prefilling
    case generating
}

/// A single inference slot that tracks state for one concurrent request.
final class ServerSlot: @unchecked Sendable {
    let id: Int
    var state: SlotState = .idle
    var requestId: String = ""
    var startTime: CFAbsoluteTime = 0
    var promptTokenCount: Int = 0
    var generationTokenCount: Int = 0

    init(id: Int) { self.id = id }

    func reset() {
        state = .idle
        requestId = ""
        startTime = 0
        promptTokenCount = 0
        generationTokenCount = 0
    }
}

/// Manages the pool of inference slots with FIFO queuing.
/// Requests acquire a slot before starting generation and release it when done.
/// Prefill is serialized: only one slot prefills at a time to avoid GPU contention.
actor SlotManager {
    let slots: [ServerSlot]
    let slotCount: Int
    private var slotWaiters: [CheckedContinuation<ServerSlot, Never>] = []
    private var prefillBusy = false
    private var prefillWaiters: [CheckedContinuation<Void, Never>] = []

    init(slotCount: Int) {
        self.slotCount = slotCount
        var s: [ServerSlot] = []
        for i in 0..<slotCount { s.append(ServerSlot(id: i)) }
        self.slots = s
    }

    func acquirePrefill() async {
        if !prefillBusy { prefillBusy = true; return }
        await withCheckedContinuation { cont in prefillWaiters.append(cont) }
    }

    func releasePrefill() {
        if let next = prefillWaiters.first {
            prefillWaiters.removeFirst()
            next.resume()
        } else {
            prefillBusy = false
        }
    }

    func acquireSlot() async -> ServerSlot {
        if let slot = slots.first(where: { $0.state == .idle }) {
            slot.state = .prefilling
            return slot
        }
        return await withCheckedContinuation { cont in slotWaiters.append(cont) }
    }

    func tryAcquireSlot() -> ServerSlot? {
        if let slot = slots.first(where: { $0.state == .idle }) {
            slot.state = .prefilling
            return slot
        }
        return nil
    }

    func releaseSlot(_ slot: ServerSlot) {
        slot.reset()
        if let waiter = slotWaiters.first {
            slotWaiters.removeFirst()
            slot.state = .prefilling
            waiter.resume(returning: slot)
        }
    }

    func slotStatus() -> [(id: Int, state: String, requestId: String, promptTokens: Int, genTokens: Int, elapsed: Double)] {
        slots.map { slot in
            let elapsed = slot.state == .idle ? 0 : CFAbsoluteTimeGetCurrent() - slot.startTime
            return (id: slot.id, state: slot.state.rawValue, requestId: slot.requestId,
                    promptTokens: slot.promptTokenCount, genTokens: slot.generationTokenCount,
                    elapsed: elapsed)
        }
    }

    func activeSlotCount() -> Int { slots.filter { $0.state != .idle }.count }
    func queueDepth() -> Int { slotWaiters.count }
}
