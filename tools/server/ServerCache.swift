import Foundation
import MLXLMCommon

// MARK: - Server Prompt Cache (Multi-Session)

struct CachedSession {
    let id: UUID = UUID()
    var tokenIds: [Int]
    var kvCache: [KVCache]
    var lastUsed: Date
}

struct CacheMetrics {
    var totalRequests: Int = 0
    var cacheHits: Int = 0
    var cacheMisses: Int = 0
    var trimFailures: Int = 0
    var totalPrefillTokens: Int = 0
    var totalReusedTokens: Int = 0
    var totalPrefillMs: Double = 0
    var totalDecodeMs: Double = 0
    var totalDecodeTokens: Int = 0
    var evictions: Int = 0

    var hitRate: Double { totalRequests > 0 ? Double(cacheHits) / Double(totalRequests) : 0 }
    var avgPrefillTokens: Double { totalRequests > 0 ? Double(totalPrefillTokens) / Double(totalRequests) : 0 }
    var avgPrefillMs: Double { totalRequests > 0 ? totalPrefillMs / Double(totalRequests) : 0 }
    var avgDecodeTokensPerSec: Double { totalDecodeMs > 0 ? Double(totalDecodeTokens) / (totalDecodeMs / 1000) : 0 }
}

enum CacheStatus {
    case hit(prefixReused: Int, totalTokens: Int, newTokens: Int)
    case miss(totalTokens: Int, sessionsCount: Int)
    case trimFailed

    var logString: String {
        switch self {
        case .hit(let prefix, let total, let new):
            return "cache=hit prefix=\(prefix)/\(total) new=\(new)"
        case .miss(let total, let sessions):
            return "cache=miss tokens=\(total) sessions=\(sessions)"
        case .trimFailed:
            return "cache=trim_failed"
        }
    }
}

/// Multi-session prompt cache with LCP (longest common prefix) matching.
/// Keeps up to `maxSessions` cached KV states. When a new request arrives,
/// finds the session with the longest matching token prefix, trims KV to
/// that prefix, and returns only the new tokens to prefill.
actor ServerPromptCache {
    var sessions: [CachedSession] = []
    let maxSessions: Int
    let kvScheme: String?
    var metrics = CacheMetrics()

    init(maxSessions: Int = 3, kvScheme: String? = nil) {
        self.maxSessions = maxSessions
        self.kvScheme = kvScheme
    }

    func recordRequest(hit: Bool, prefillTokens: Int, reusedTokens: Int) {
        metrics.totalRequests += 1
        if hit { metrics.cacheHits += 1 } else { metrics.cacheMisses += 1 }
        metrics.totalPrefillTokens += prefillTokens
        metrics.totalReusedTokens += reusedTokens
    }

    func recordTiming(prefillMs: Double, decodeMs: Double, decodeTokens: Int) {
        metrics.totalPrefillMs += prefillMs
        metrics.totalDecodeMs += decodeMs
        metrics.totalDecodeTokens += decodeTokens
    }

    func recordEviction() { metrics.evictions += 1 }
    func recordTrimFailure() { metrics.trimFailures += 1 }
    func getMetrics() -> CacheMetrics { metrics }
    func getSessionCount() -> Int { sessions.count }

    /// Find the session with the longest common prefix match.
    func fetch(tokens newTokens: [Int], model: any LanguageModel) -> ([KVCache], [Int], CacheStatus, UUID) {
        var bestIdx = -1
        var bestPrefix = 0

        for (i, session) in sessions.enumerated() {
            let prefix = commonPrefix(session.tokenIds, newTokens)
            if prefix > bestPrefix {
                bestPrefix = prefix
                bestIdx = i
            }
        }

        if bestIdx >= 0 && bestPrefix > 0 {
            let session = sessions[bestIdx]
            let actualCacheSize = session.kvCache.first?.offset ?? session.tokenIds.count
            let trimAmount = actualCacheSize - bestPrefix
            let isExtension = (bestPrefix == session.tokenIds.count) || (trimAmount == 0)

            if isExtension {
                if trimAmount > 0 {
                    for c in session.kvCache {
                        if c.trim(trimAmount) == 0 {
                            metrics.trimFailures += 1
                            return freshCache(tokens: newTokens, model: model)
                        }
                    }
                }
                sessions[bestIdx].lastUsed = Date()
                sessions[bestIdx].tokenIds = Array(newTokens[0..<bestPrefix])
                let remaining = Array(newTokens[bestPrefix...])
                let status = CacheStatus.hit(prefixReused: bestPrefix, totalTokens: newTokens.count, newTokens: remaining.count)
                recordRequest(hit: true, prefillTokens: remaining.count, reusedTokens: bestPrefix)
                return (session.kvCache, remaining, status, session.id)
            } else {
                let copiedCache = session.kvCache.map { $0.copy() }
                if trimAmount > 0 {
                    for c in copiedCache {
                        if c.trim(trimAmount) == 0 {
                            metrics.trimFailures += 1
                            return freshCache(tokens: newTokens, model: model)
                        }
                    }
                }

                evictIfNeeded()
                let newSession = CachedSession(tokenIds: Array(newTokens[0..<bestPrefix]),
                                               kvCache: copiedCache, lastUsed: Date())
                sessions.append(newSession)
                sessions[bestIdx].lastUsed = Date()

                let remaining = Array(newTokens[bestPrefix...])
                let status = CacheStatus.hit(prefixReused: bestPrefix, totalTokens: newTokens.count, newTokens: remaining.count)
                recordRequest(hit: true, prefillTokens: remaining.count, reusedTokens: bestPrefix)
                return (copiedCache, remaining, status, newSession.id)
            }
        }

        return freshCache(tokens: newTokens, model: model)
    }

    private func freshCache(tokens: [Int], model: any LanguageModel) -> ([KVCache], [Int], CacheStatus, UUID) {
        evictIfNeeded()
        var cacheParams: GenerateParameters? = nil
        if let scheme = kvScheme {
            var p = GenerateParameters()
            p.kvScheme = scheme
            cacheParams = p
        }
        let cache = model.newCache(parameters: cacheParams)
        let session = CachedSession(tokenIds: [], kvCache: cache, lastUsed: Date())
        sessions.append(session)
        let status = CacheStatus.miss(totalTokens: tokens.count, sessionsCount: sessions.count)
        recordRequest(hit: false, prefillTokens: tokens.count, reusedTokens: 0)
        return (cache, tokens, status, session.id)
    }

    func save(sessionId: UUID, tokens: [Int]) {
        if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[idx].tokenIds = tokens
            sessions[idx].lastUsed = Date()
        }
    }

    private func evictIfNeeded() {
        while sessions.count >= maxSessions {
            if let oldest = sessions.enumerated().min(by: { $0.element.lastUsed < $1.element.lastUsed }) {
                log("Evicting session \(oldest.offset) (\(oldest.element.tokenIds.count) tokens, idle \(Int(-oldest.element.lastUsed.timeIntervalSinceNow))s)")
                sessions.remove(at: oldest.offset)
            }
        }
    }

    func evictIdle(keep: Int) {
        while sessions.count > keep {
            if let oldest = sessions.enumerated().min(by: { $0.element.lastUsed < $1.element.lastUsed }) {
                log("Memory pressure eviction: session \(oldest.offset)")
                sessions.remove(at: oldest.offset)
                metrics.evictions += 1
            }
        }
    }

    func flush() {
        let count = sessions.count
        sessions.removeAll()
        metrics.evictions += count
        log("Flushed all \(count) cached sessions")
    }

    private func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let maxLen = min(a.count, b.count)
        for i in 0..<maxLen {
            if a[i] != b[i] { return i }
        }
        return maxLen
    }
}
