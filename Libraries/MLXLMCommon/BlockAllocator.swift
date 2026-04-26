// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Shared physical block pool for `PagedKVCache`. Refcounted free list —
/// `retain` lets multiple sequences share blocks (prefix caching).
///
/// **Thread safety:** all public methods serialize on an internal lock.
public final class BlockAllocator: @unchecked Sendable {

    public let numBlocks: Int

    /// Refcount per block. 0 = free; >0 = referenced by N sequences.
    private var refcounts: [Int]

    /// LIFO free list — recently freed blocks reuse first for cache locality.
    private var freeStack: [Int]

    private let lock = NSLock()

    public init(numBlocks: Int) {
        precondition(numBlocks > 0, "numBlocks must be > 0")
        self.numBlocks = numBlocks
        self.refcounts = Array(repeating: 0, count: numBlocks)
        // Reverse so id 0 pops first — predictable in tests.
        self.freeStack = (0 ..< numBlocks).reversed()
    }

    /// Allocate `count` fresh blocks. Throws if pool exhausted.
    public func allocate(_ count: Int) throws -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        guard count <= freeStack.count else {
            throw AllocatorError.exhausted(requested: count, available: freeStack.count)
        }
        var allocated: [Int] = []
        allocated.reserveCapacity(count)
        for _ in 0 ..< count {
            let id = freeStack.removeLast()
            refcounts[id] += 1
            allocated.append(id)
        }
        return allocated
    }

    /// Decrement refcount; return to free list when it hits 0.
    public func free(_ id: Int) {
        lock.lock()
        defer { lock.unlock() }
        precondition(id >= 0 && id < numBlocks, "block id out of range")
        precondition(refcounts[id] > 0, "double-free of block \(id)")
        refcounts[id] -= 1
        if refcounts[id] == 0 {
            freeStack.append(id)
        }
    }

    public func free(_ ids: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        for id in ids {
            precondition(id >= 0 && id < numBlocks, "block id out of range")
            precondition(refcounts[id] > 0, "double-free of block \(id)")
            refcounts[id] -= 1
            if refcounts[id] == 0 {
                freeStack.append(id)
            }
        }
    }

    /// Increment refcount on already-allocated blocks (prefix sharing).
    public func retain(_ ids: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        for id in ids {
            precondition(id >= 0 && id < numBlocks, "block id out of range")
            precondition(refcounts[id] >= 1, "retain on free block \(id)")
            refcounts[id] += 1
        }
    }

    public var freeCount: Int {
        lock.lock(); defer { lock.unlock() }
        return freeStack.count
    }

    public var allocatedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return numBlocks - freeStack.count
    }

    public func refcount(of id: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        return refcounts[id]
    }
}

public enum AllocatorError: Error, CustomStringConvertible {
    case exhausted(requested: Int, available: Int)

    public var description: String {
        switch self {
        case let .exhausted(req, avail):
            return "BlockAllocator exhausted: requested \(req), \(avail) free"
        }
    }
}
