import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func log(_ msg: String) {
    FileHandle.standardError.write(Data("[MLXServer] \(msg)\n".utf8))
}

// MARK: - OpenAI Types

struct ChatMessage: Codable {
    let role: String
    let content: String?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        // Content can be string, null, or array — just grab string or nil
        if let str = try? container.decode(String.self, forKey: .content) {
            content = str
        } else {
            content = nil
        }
    }

    enum CodingKeys: String, CodingKey { case role, content }
}

struct ChatRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let max_tokens: Int?
    let temperature: Float?
    let stream: Bool?
    // Accept but ignore extra fields
    let tools: AnyCodable?
    let tool_choice: AnyCodable?
    let top_p: Float?
    let frequency_penalty: Float?
    let presence_penalty: Float?
    let stop: AnyCodable?
    let n: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, max_tokens, temperature, stream
        case tools, tool_choice, top_p, frequency_penalty, presence_penalty, stop, n
    }
}

// Wraps any JSON value
struct AnyCodable: Codable {
    let value: Any?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([JSONValue].self) {
            value = arr.map { $0.toAny() }
        } else if let dict = try? container.decode([String: JSONValue].self) {
            value = dict.mapValues { $0.toAny() }
        } else {
            value = nil
        }
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encodeNil()
    }
}

// Recursive JSON value for preserving tools structure
enum JSONValue: Codable {
    case string(String), int(Int), double(Double), bool(Bool), null
    case array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int.self) { self = .int(v) }
        else if let v = try? c.decode(Double.self) { self = .double(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else if let v = try? c.decode([String: JSONValue].self) { self = .object(v) }
        else if c.decodeNil() { self = .null }
        else { self = .null }
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
    func toAny() -> Any {
        switch self {
        case .string(let v): return v
        case .int(let v): return v
        case .double(let v): return v
        case .bool(let v): return v
        case .null: return NSNull()
        case .array(let v): return v.map { $0.toAny() }
        case .object(let v): return v.mapValues { $0.toAny() } as [String: Any]
        }
    }
}

struct ChatResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Codable {
        let index: Int
        let message: Message?
        let delta: Message?
        let finish_reason: String?
    }

    struct Message: Codable {
        let role: String?
        let content: String?
    }

    struct Usage: Codable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
    }
}

// MARK: - Minimal HTTP Server using URLSession's HTTPServer
// Using a raw socket server via Foundation for zero dependencies

// MARK: - Server Prompt Cache (Multi-Session)

struct CachedSession {
    let id: UUID = UUID()
    var tokenIds: [Int]
    var kvCache: [KVCache]
    var lastUsed: Date
}

/// Multi-session prompt cache with LCP (longest common prefix) matching.
/// Keeps up to `maxSessions` cached KV states. When a new request arrives,
/// finds the session with the longest matching token prefix, trims KV to
/// that prefix, and returns only the new tokens to prefill.
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

actor ServerPromptCache {
    var sessions: [CachedSession] = []
    let maxSessions: Int
    var metrics = CacheMetrics()

    init(maxSessions: Int = 3) {
        self.maxSessions = maxSessions
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
    /// Returns (kvCache, newTokensToProcess, cacheStatus, sessionId).
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
            let trimAmount = session.tokenIds.count - bestPrefix

            // If the new request extends the cached session (same prefix, more tokens),
            // we can trim and use in-place. If it diverges (different suffix), we need
            // to copy so the original stays intact for future reuse.
            let isExtension = (bestPrefix == session.tokenIds.count) || (trimAmount == 0)

            if isExtension {
                // Same conversation continuing — use in-place, no copy needed
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
                // Different conversation forking from this prefix — deep copy
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
        let cache = model.newCache(parameters: nil)
        let session = CachedSession(tokenIds: [], kvCache: cache, lastUsed: Date())
        sessions.append(session)
        let status = CacheStatus.miss(totalTokens: tokens.count, sessionsCount: sessions.count)
        recordRequest(hit: false, prefillTokens: tokens.count, reusedTokens: 0)
        return (cache, tokens, status, session.id)
    }

    /// Save token state after generation completes.
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

    /// Evict idle sessions, keeping at most `keep` sessions.
    func evictIdle(keep: Int) {
        while sessions.count > keep {
            if let oldest = sessions.enumerated().min(by: { $0.element.lastUsed < $1.element.lastUsed }) {
                log("Memory pressure eviction: session \(oldest.offset) (\(oldest.element.tokenIds.count) tokens)")
                sessions.remove(at: oldest.offset)
                metrics.evictions += 1
            }
        }
    }

    /// Flush all cached sessions (e.g., on model change or critical memory pressure)
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

final class SimpleHTTPServer {
    let port: UInt16
    let container: ModelContainer
    let modelId: String
    let promptCache: ServerPromptCache
    private var serverSocket: Int32 = -1
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    static let maxBodySize = 10 * 1024 * 1024

    /// Compute max sessions based on available physical memory.
    /// Conservative: assume ~1.5GB per cached session (14K tokens FP16 KV for 30B MoE).
    static func autoMaxSessions() -> Int {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        // Reserve 20GB for model + OS, then ~1.5GB per session
        let available = max(totalGB - 20, 2)
        let sessions = Int(available / 1.5)
        return max(2, min(sessions, 10))  // clamp 2-10
    }

    init(port: UInt16, container: ModelContainer, modelId: String) {
        self.port = port
        self.container = container
        self.modelId = modelId
        let maxSess = SimpleHTTPServer.autoMaxSessions()
        self.promptCache = ServerPromptCache(maxSessions: maxSess)
        log("Auto-configured: \(maxSess) max cached sessions (\(ProcessInfo.processInfo.physicalMemory / (1024*1024*1024))GB RAM)")
    }

    func start() throws {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0)
        guard serverSocket >= 0 else { throw ServerError.socketCreation }

        var opt: Int32 = 1
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(serverSocket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bindResult == 0 else { throw ServerError.bind(port) }

        guard listen(serverSocket, 64) == 0 else { throw ServerError.listen }

        log("Listening on http://127.0.0.1:\(port)")
        log("Endpoints: GET /v1/models, POST /v1/chat/completions, GET /tokenizer_info, POST /tokenize, GET /metrics")

        // Monitor macOS memory pressure — evict idle sessions under pressure
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global())
        source.setEventHandler { [promptCache] in
            Task {
                let event = source.data
                if event.contains(.critical) {
                    log("MEMORY PRESSURE: critical — flushing all cached sessions")
                    await promptCache.flush()
                } else if event.contains(.warning) {
                    log("MEMORY PRESSURE: warning — evicting idle sessions")
                    await promptCache.evictIdle(keep: 1)
                }
            }
        }
        source.resume()
        memoryPressureSource = source

        while true {
            let client = accept(serverSocket, nil, nil)
            guard client >= 0 else { continue }
            Task { await handleClient(client) }
        }
    }

    func handleClient(_ fd: Int32) async {
        defer { close(fd) }

        let requestStart = CFAbsoluteTimeGetCurrent()
        var method = "?"
        var path = "?"

        do {
            // Read headers first
            var headerData = Data()
            var byte: UInt8 = 0
            while headerData.count < 65536 {
                let n = read(fd, &byte, 1)
                guard n == 1 else { return }
                headerData.append(byte)
                // Detect end of headers: \r\n\r\n
                if headerData.count >= 4 &&
                   headerData[headerData.count-4] == 0x0D && headerData[headerData.count-3] == 0x0A &&
                   headerData[headerData.count-2] == 0x0D && headerData[headerData.count-1] == 0x0A {
                    break
                }
            }

            let headerStr = String(data: headerData, encoding: .utf8) ?? ""
            let lines = headerStr.split(separator: "\r\n", omittingEmptySubsequences: false)
            guard let firstLine = lines.first else { return }

            let parts = firstLine.split(separator: " ")
            guard parts.count >= 2 else { return }
            method = String(parts[0])
            path = String(parts[1]).split(separator: "?").first.map(String.init) ?? String(parts[1])

            // Parse Content-Length and read body
            var contentLength = 0
            for line in lines {
                let lower = line.lowercased()
                if lower.hasPrefix("content-length:") {
                    contentLength = Int(lower.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
                }
            }

            // Enforce max body size (10MB)
            if contentLength > SimpleHTTPServer.maxBodySize {
                log("\(method) \(path) — body too large (\(contentLength) bytes)")
                sendResponse(fd: fd, status: 413,
                           body: "{\"error\":\"request body too large (max \(SimpleHTTPServer.maxBodySize / 1024 / 1024)MB)\"}",
                           contentType: "application/json")
                return
            }

            var bodyData = Data()
            if contentLength > 0 {
                bodyData.reserveCapacity(contentLength)
                var remaining = contentLength
                var buf = [UInt8](repeating: 0, count: min(remaining, 65536))
                while remaining > 0 {
                    let toRead = min(remaining, buf.count)
                    let n = read(fd, &buf, toRead)
                    guard n > 0 else { break }
                    bodyData.append(contentsOf: buf[0..<n])
                    remaining -= n
                }
            }
            let bodyStr = String(data: bodyData, encoding: .utf8) ?? ""

            log("\(method) \(path) (\(bodyStr.count) bytes)")

            switch (method, path) {
            case ("GET", "/v1/models"):
                let models = "{\"object\":\"list\",\"data\":[{\"id\":\"\(modelId)\",\"object\":\"model\",\"created\":\(Int(Date().timeIntervalSince1970))}]}"
                sendResponse(fd: fd, status: 200, body: models, contentType: "application/json")

            case ("POST", "/v1/chat/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let request = try? JSONDecoder().decode(ChatRequest.self, from: data) else {
                    sendResponse(fd: fd, status: 400,
                               body: "{\"error\":\"invalid request\"}", contentType: "application/json")
                    return
                }
                await handleChat(fd: fd, request: request)

            case ("GET", "/tokenizer_info"), ("GET", "/v1/tokenizer_info"):
                let ctx = await container.perform { ctx in ctx }
                let eos = ctx.tokenizer.eosToken ?? ""
                let bos = ctx.tokenizer.bosToken ?? ""
                let eosId = ctx.tokenizer.eosTokenId ?? -1
                let bosId = ctx.tokenizer.bosTokenId ?? -1
                let info = "{\"eos_token\":\"\(eos)\",\"bos_token\":\"\(bos)\",\"eos_token_id\":\(eosId),\"bos_token_id\":\(bosId),\"model\":\"\(modelId)\"}"
                sendResponse(fd: fd, status: 200, body: info, contentType: "application/json")

            case ("POST", "/tokenize"), ("POST", "/v1/tokenize"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":\"missing prompt\"}", contentType: "application/json")
                    return
                }
                let addSpecial = json["add_special_tokens"] as? Bool ?? true
                let ctx = await container.perform { ctx in ctx }
                let tokens = ctx.tokenizer.encode(text: prompt, addSpecialTokens: addSpecial)
                let tokensJson = "[\(tokens.map { String($0) }.joined(separator: ","))]"
                sendResponse(fd: fd, status: 200, body: "{\"tokens\":\(tokensJson)}", contentType: "application/json")

            case ("POST", "/v1/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":\"invalid request\"}", contentType: "application/json")
                    return
                }
                let maxTokens = json["max_tokens"] as? Int ?? 256
                let temperature = json["temperature"] as? Double ?? 0.0
                let isStream = json["stream"] as? Bool ?? false
                await handleCompletions(fd: fd, prompt: prompt, maxTokens: maxTokens,
                                       temperature: Float(temperature), stream: isStream)

            case ("GET", "/health"), ("GET", "/"):
                sendResponse(fd: fd, status: 200, body: "{\"status\":\"ok\"}", contentType: "application/json")

            case ("GET", "/metrics"):
                let m = await promptCache.getMetrics()
                let sc = await promptCache.getSessionCount()
                let body = """
                {"cache":{"requests":\(m.totalRequests),"hits":\(m.cacheHits),"misses":\(m.cacheMisses),"hit_rate":\(String(format:"%.3f",m.hitRate)),"trim_failures":\(m.trimFailures),"evictions":\(m.evictions),"sessions_active":\(sc),"sessions_max":\(await promptCache.maxSessions)},"throughput":{"total_prefill_tokens":\(m.totalPrefillTokens),"total_reused_tokens":\(m.totalReusedTokens),"total_decode_tokens":\(m.totalDecodeTokens),"avg_prefill_tokens_per_request":\(String(format:"%.0f",m.avgPrefillTokens)),"avg_prefill_ms":\(String(format:"%.1f",m.avgPrefillMs)),"avg_decode_tok_per_sec":\(String(format:"%.1f",m.avgDecodeTokensPerSec))}}
                """
                sendResponse(fd: fd, status: 200, body: body.trimmingCharacters(in: .whitespacesAndNewlines), contentType: "application/json")

            case ("OPTIONS", _):
                // CORS preflight
                sendResponse(fd: fd, status: 204, body: "", contentType: "text/plain",
                            extraHeaders: "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n")

            default:
                sendResponse(fd: fd, status: 404, body: "{\"error\":\"not found\"}", contentType: "application/json")
            }
        } catch {
            log("ERROR handling \(method) \(path): \(error)")
            sendResponse(fd: fd, status: 500,
                       body: "{\"error\":\"internal server error\"}",
                       contentType: "application/json")
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - requestStart) * 1000
        log("\(method) \(path) completed in \(String(format: "%.0f", elapsed))ms")
    }

    func handleChat(fd: Int32, request: ChatRequest) async {
        let isStreaming = request.stream ?? false
        let requestId = "chatcmpl-\(UUID().uuidString.prefix(8))"

        do {
            var ctx = await container.perform { ctx in ctx }
            // Convert to tokenizer's expected format
            let messages: [[String: String]] = request.messages.map {
                var d: [String: String] = ["role": $0.role]
                if let c = $0.content { d["content"] = c }
                return d
            }
            // Pass tools if present — the chat template injects tool definitions
            let toolsAny = request.tools?.value as? [Any]
            let tokens: [Int]
            if let toolsAny = toolsAny {
                let toolSpecs: [[String: any Sendable]] = toolsAny.compactMap { $0 as? [String: Any] }.map { tool in
                    // Deep convert to [String: any Sendable]
                    func convert(_ v: Any) -> any Sendable {
                        if let s = v as? String { return s }
                        if let n = v as? Int { return n }
                        if let b = v as? Bool { return b }
                        if let d = v as? Double { return d }
                        if let arr = v as? [Any] { return arr.map { convert($0) } as [any Sendable] }
                        if let dict = v as? [String: Any] {
                            return dict.mapValues { convert($0) } as [String: any Sendable]
                        }
                        return String(describing: v)
                    }
                    return tool.mapValues { convert($0) } as [String: any Sendable]
                }
                tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            } else {
                tokens = try ctx.tokenizer.applyChatTemplate(messages: messages)
            }
            // Prompt caching: reuse KV state from previous requests
            let prefillStart = CFAbsoluteTimeGetCurrent()
            let (reusedCache, newTokens, cacheStatus, sessionId) = await promptCache.fetch(tokens: tokens, model: ctx.model)
            let tokenArray = MLXArray(newTokens.isEmpty ? tokens : newTokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: request.temperature ?? 0.6)
            if let maxTokens = request.max_tokens {
                params.maxTokens = maxTokens
            }

            // Set tool call format if tools are present
            if toolsAny != nil {
                ctx.configuration.toolCallFormat = .xmlFunction
            }

            log("\(cacheStatus.logString) prefill=\(newTokens.count) stream=\(isStreaming) tools=\(toolsAny?.count ?? 0)")

            // Start keepalive task for long prefills (sends SSE comments every 2s)
            // Only for streaming — non-streaming clients don't expect SSE
            let keepaliveTask: Task<Void, Never>?
            if isStreaming && newTokens.count > 1000 {
                // Send SSE headers early so keepalive comments have somewhere to go
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
                _ = header.withCString { write(fd, $0, Int(strlen($0))) }

                keepaliveTask = Task {
                    while !Task.isCancelled {
                        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
                        if Task.isCancelled { break }
                        let comment = ": keepalive\n\n"
                        _ = comment.withCString { write(fd, $0, Int(strlen($0))) }
                    }
                }
            } else {
                keepaliveTask = nil
            }

            if isStreaming {
                // SSE headers (only if not already sent by keepalive setup)
                if keepaliveTask == nil {
                    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
                    _ = header.withCString { write(fd, $0, Int(strlen($0))) }
                }

                // Cancel keepalive once generation starts producing tokens
                keepaliveTask?.cancel()

                var hadToolCall = false
                // Send role chunk immediately so client knows we're alive
                writeSSE(fd: fd, requestId: requestId, role: "assistant", content: "", finishReason: nil)
                // Flush immediately
                _ = "".withCString { _ in fcntl(fd, F_FULLFSYNC) }

                do {
                    // Accumulate ALL text, parse tool calls at the end.
                    // Streaming text is emitted in real-time UNLESS we detect the
                    // start of a tool call, at which point we buffer until complete.
                    var fullText = ""
                    var emittedUpTo = 0  // index in fullText that we've already sent

                    for try await generation in try generate(
                        input: input, cache: reusedCache, parameters: params, context: ctx
                    ) {
                        switch generation {
                        case .chunk(let text):
                            fullText += text

                            // Check if there's a potential tool call starting in the un-emitted portion
                            let unemitted = String(fullText[fullText.index(fullText.startIndex, offsetBy: emittedUpTo)...])

                            if unemitted.contains("<tool_call>") || unemitted.contains("<function=") {
                                // Might be a tool call — check if it's complete
                                if unemitted.contains("</tool_call>") || unemitted.contains("</function>") {
                                    // Complete tool call — parse and emit
                                    if let tc = parseToolCallXML(unemitted) {
                                        hadToolCall = true
                                        emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments)
                                    } else {
                                        writeSSE(fd: fd, requestId: requestId, role: nil, content: unemitted, finishReason: nil)
                                    }
                                    emittedUpTo = fullText.count
                                }
                                // Incomplete — keep buffering, don't emit
                            } else if unemitted.contains("<") {
                                // Any `<` in unemitted text could be start of a tag.
                                // Hold back until we know it's not a tool call.
                                // Emit everything before the `<`, keep the rest buffered.
                                if let ltIdx = unemitted.lastIndex(of: "<") {
                                    let safe = String(unemitted[unemitted.startIndex..<ltIdx])
                                    if !safe.isEmpty {
                                        writeSSE(fd: fd, requestId: requestId, role: nil, content: safe, finishReason: nil)
                                        emittedUpTo += safe.count
                                    }
                                }
                                continue
                            } else {
                                // Safe to emit
                                writeSSE(fd: fd, requestId: requestId, role: nil, content: text, finishReason: nil)
                                emittedUpTo = fullText.count
                            }

                        case .toolCall(let tc):
                            // generate() parsed it — emit directly
                            hadToolCall = true
                            let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                            let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                            let argsRaw = String(data: argsJSON, encoding: .utf8) ?? "{}"
                            emitToolCallSSE(fd: fd, requestId: requestId, name: tc.function.name, arguments: argsRaw)
                        case .info(let info):
                            // Flush any remaining buffered text
                            if emittedUpTo < fullText.count {
                                let remaining = String(fullText[fullText.index(fullText.startIndex, offsetBy: emittedUpTo)...])
                                if let tc = parseToolCallXML(remaining) {
                                    hadToolCall = true
                                    emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments)
                                } else if !remaining.isEmpty {
                                    writeSSE(fd: fd, requestId: requestId, role: nil, content: remaining, finishReason: nil)
                                }
                            }
                            let fr = hadToolCall ? "tool_calls" : "stop"
                            let usageJSON = ",\"usage\":{\"prompt_tokens\":\(info.promptTokenCount),\"completion_tokens\":\(info.generationTokenCount),\"total_tokens\":\(info.promptTokenCount + info.generationTokenCount)}"
                            let finalEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(fr)\"}]\(usageJSON)}\n\ndata: [DONE]\n\n"
                            _ = finalEvent.withCString { write(fd, $0, Int(strlen($0))) }
                        }
                    }
                } catch {
                    // Generation error during streaming — send error SSE event and close cleanly
                    log("ERROR during streaming generation: \(error)")
                    let errMsg = error.localizedDescription.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                    let errEvent = "data: {\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n"
                    _ = errEvent.withCString { write(fd, $0, Int(strlen($0))) }
                }
                // Record timing and save cache
                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, tokens: tokens)
            } else {
                keepaliveTask?.cancel()

                // Non-streaming: collect all text
                var fullText = ""
                var completionTokens = 0
                var toolCalls: [(name: String, args: String)] = []
                for try await generation in try generate(
                    input: input, cache: reusedCache, parameters: params, context: ctx
                ) {
                    switch generation {
                    case .chunk(let text):
                        fullText += text
                        completionTokens += 1
                    case .toolCall(let tc):
                        let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                        let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                        toolCalls.append((name: tc.function.name, args: String(data: argsJSON, encoding: .utf8) ?? "{}"))
                    default: break
                    }
                }

                // Build response with tool_calls if present
                let finishReason = toolCalls.isEmpty ? "stop" : "tool_calls"
                var responseBody: String
                if toolCalls.isEmpty {
                    let response = ChatResponse(
                        id: requestId, object: "chat.completion",
                        created: Int(Date().timeIntervalSince1970), model: modelId,
                        choices: [.init(index: 0, message: .init(role: "assistant", content: fullText),
                                       delta: nil, finish_reason: finishReason)],
                        usage: .init(prompt_tokens: tokens.count, completion_tokens: completionTokens,
                                    total_tokens: tokens.count + completionTokens))
                    responseBody = String(data: try JSONEncoder().encode(response), encoding: .utf8)!
                } else {
                    // Build tool_calls response manually (ChatResponse doesn't have tool_calls field)
                    let tcJSON = toolCalls.enumerated().map { (i, tc) in
                        "{\"id\":\"call_\(UUID().uuidString.prefix(8))\",\"type\":\"function\",\"function\":{\"name\":\"\(tc.name)\",\"arguments\":\(tc.args)}}"
                    }.joined(separator: ",")
                    let content = "\"\(fullText.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r"))\""
                    responseBody = "{\"id\":\"\(requestId)\",\"object\":\"chat.completion\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\(content),\"tool_calls\":[\(tcJSON)]},\"finish_reason\":\"\(finishReason)\"}],\"usage\":{\"prompt_tokens\":\(tokens.count),\"completion_tokens\":\(completionTokens),\"total_tokens\":\(tokens.count + completionTokens)}}"
                }
                sendResponse(fd: fd, status: 200, body: responseBody,
                           contentType: "application/json")
                // Record timing and save cache
                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, tokens: tokens)
            }
        } catch {
            log("ERROR in handleChat: \(error)")
            let errMsg = error.localizedDescription.replacingOccurrences(of: "\"", with: "'")
            let err = "{\"error\":{\"message\":\"\(errMsg)\"}}"
            sendResponse(fd: fd, status: 500, body: err, contentType: "application/json")
        }
    }

    func handleCompletions(fd: Int32, prompt: String, maxTokens: Int, temperature: Float, stream: Bool) async {
        let requestId = "cmpl-\(UUID().uuidString.prefix(8))"
        do {
            let ctx = await container.perform { ctx in ctx }
            let tokens = ctx.tokenizer.encode(text: prompt)
            let tokenArray = MLXArray(tokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: temperature)
            params.maxTokens = maxTokens

            log("completions: \(tokens.count) prompt tokens, max_tokens=\(maxTokens), stream=\(stream)")

            if stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
                _ = header.withCString { write(fd, $0, Int(strlen($0))) }

                let result = try await container.generate(input: input, parameters: params)
                for try await event in result {
                    if let chunk = event.chunk {
                        let escaped = chunk.replacingOccurrences(of: "\\", with: "\\\\")
                            .replacingOccurrences(of: "\"", with: "\\\"")
                            .replacingOccurrences(of: "\n", with: "\\n")
                        let sseData = "{\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\(escaped)\",\"finish_reason\":null}]}"
                        let sse = "data: \(sseData)\n\n"
                        _ = sse.withCString { write(fd, $0, Int(strlen($0))) }
                    }
                    if event.info != nil {
                        let sseData = "{\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}]}"
                        let sse = "data: \(sseData)\n\ndata: [DONE]\n\n"
                        _ = sse.withCString { write(fd, $0, Int(strlen($0))) }
                    }
                }
            } else {
                var fullText = ""
                let result = try await container.generate(input: input, parameters: params)
                var usage: (prompt: Int, completion: Int) = (tokens.count, 0)
                for try await event in result {
                    if let chunk = event.chunk { fullText += chunk; usage.completion += 1 }
                }
                let escaped = fullText.replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                    .replacingOccurrences(of: "\n", with: "\\n")
                let body = "{\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"text\":\"\(escaped)\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":\(usage.prompt),\"completion_tokens\":\(usage.completion),\"total_tokens\":\(usage.prompt + usage.completion)}}"
                sendResponse(fd: fd, status: 200, body: body, contentType: "application/json")
            }
        } catch {
            let errMsg = String(describing: error).replacingOccurrences(of: "\"", with: "'")
            sendResponse(fd: fd, status: 500, body: "{\"error\":{\"message\":\"\(errMsg)\"}}", contentType: "application/json")
        }
    }

    /// Parse XML tool call from text: <function=name><parameter=key>value</parameter></function>
    /// Handles both <tool_call>...<function=...>...</tool_call> and bare <function=...>
    func parseToolCallXML(_ text: String) -> (name: String, arguments: String)? {
        guard let funcStart = text.range(of: "<function=") else { return nil }
        guard let nameEnd = text.range(of: ">", range: funcStart.upperBound..<text.endIndex) else { return nil }

        let funcName = String(text[funcStart.upperBound..<nameEnd.lowerBound])

        // Extract parameters
        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter=", range: search..<text.endIndex) {
            guard let pNameEnd = text.range(of: ">", range: paramStart.upperBound..<text.endIndex) else { break }
            let paramName = String(text[paramStart.upperBound..<pNameEnd.lowerBound])
            guard let paramEnd = text.range(of: "</parameter>", range: pNameEnd.upperBound..<text.endIndex) else { break }
            var value = String(text[pNameEnd.upperBound..<paramEnd.lowerBound])
            // Trim leading/trailing newlines
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }

    /// Emit a tool call as an SSE event
    func emitToolCallSSE(fd: Int32, requestId: String, name: String, arguments: String) {
        let argsEscaped = arguments.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        let tcId = UUID().uuidString.lowercased()
        let tcEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"index\":0,\"id\":\"\(tcId)\",\"type\":\"function\",\"function\":{\"name\":\"\(name)\",\"arguments\":\"\(argsEscaped)\"}}]}}]}\n\n"
        _ = tcEvent.withCString { write(fd, $0, Int(strlen($0))) }
    }

    func writeSSE(fd: Int32, requestId: String, role: String?, content: String?, finishReason: String?) {
        let delta: [String: String?] = ["role": role, "content": content]
        let filteredDelta = delta.compactMapValues { $0 }

        // Build JSON manually to avoid encoding issues with nil
        var deltaJson = "{"
        deltaJson += filteredDelta.map { "\"\($0.key)\":\"\($0.value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r"))\"" }.joined(separator: ",")
        deltaJson += "}"

        let fr = finishReason.map { "\"\($0)\"" } ?? "null"
        let event = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"delta\":\(deltaJson),\"finish_reason\":\(fr)}]}\n\n"
        _ = event.withCString { write(fd, $0, Int(strlen($0))) }
    }

    func sendResponse(fd: Int32, status: Int, body: String, contentType: String, extraHeaders: String = "") {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 204: statusText = "No Content"
        case 400: statusText = "Bad Request"
        case 413: statusText = "Payload Too Large"
        case 404: statusText = "Not Found"
        case 500: statusText = "Internal Server Error"
        default: statusText = "Unknown"
        }
        let response = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.utf8.count)\r\nAccess-Control-Allow-Origin: *\r\n\(extraHeaders)\r\n\(body)"
        _ = response.withCString { write(fd, $0, Int(strlen($0))) }
    }

    enum ServerError: Error {
        case socketCreation, bind(UInt16), listen
    }
}

// MARK: - Entry Point

@main
struct MLXServerApp {
    @MainActor
    static func main() async throws {
        let args = CommandLine.arguments
        let model = args.firstIndex(of: "--model").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        } ?? "mlx-community/gemma-4-e2b-it-4bit"

        let port = args.firstIndex(of: "--port").flatMap { i in
            i + 1 < args.count ? UInt16(args[i + 1]) : nil
        } ?? 8080

        log("Loading model: \(model)")
        let config: ModelConfiguration
        if model.hasPrefix("/") || model.hasPrefix("~") || model.hasPrefix(".") {
            // Local path
            let expandedPath = NSString(string: model).expandingTildeInPath
            config = ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
        } else {
            // HuggingFace model ID
            config = ModelConfiguration(id: model)
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: config) { p in
            if p.fractionCompleted > 0.99 { log("Model loaded") }
        }

        let server = SimpleHTTPServer(port: port, container: container, modelId: model)
        try server.start()
    }
}
