import Foundation
import MLX
import MLXLMCommon

// MARK: - HTTP Server

final class SimpleHTTPServer {
    let port: UInt16
    let container: ModelContainer
    let modelId: String
    let promptCache: ServerPromptCache
    let slotManager: SlotManager
    let slotCount: Int
    let kvScheme: String?
    let kvBits: Int?
    let kvStart: Int?
    let ctxSize: Int?
    let defaultMaxTokens: Int?
    let enableThinking: Bool
    private var serverSocket: Int32 = -1
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    static let maxBodySize = 10 * 1024 * 1024

    static func autoMaxSessions() -> Int {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        let available = max(totalGB - 20, 2)
        let sessions = Int(available / 1.5)
        return max(2, min(sessions, 10))
    }

    init(port: UInt16, container: ModelContainer, modelId: String, slotCount: Int = 4,
         kvScheme: String? = nil, kvBits: Int? = nil, kvStart: Int? = nil,
         ctxSize: Int? = nil, defaultMaxTokens: Int? = nil, enableThinking: Bool = true) {
        self.port = port
        self.container = container
        self.modelId = modelId
        self.slotCount = slotCount
        self.kvScheme = kvScheme
        self.kvBits = kvBits
        self.kvStart = kvStart
        self.ctxSize = ctxSize
        self.defaultMaxTokens = defaultMaxTokens
        self.enableThinking = enableThinking
        let maxSess = SimpleHTTPServer.autoMaxSessions()
        self.promptCache = ServerPromptCache(maxSessions: maxSess, kvScheme: kvScheme)
        self.slotManager = SlotManager(slotCount: slotCount)
        log("Cache: \(maxSess) max sessions (\(ProcessInfo.processInfo.physicalMemory / (1024*1024*1024))GB RAM)")
        log("Slots: \(slotCount) parallel inference slots")
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
        log("Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/completions, POST /v1/messages, GET /health, GET /metrics, GET /slots")

        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global())
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

    // MARK: - Request Routing

    func handleClient(_ fd: Int32) async {
        defer { close(fd) }

        while true {
            let requestStart = CFAbsoluteTimeGetCurrent()
            var method = "?"
            var path = "?"
            var originHeader: String? = nil
            var keepAlive = false

            var headerData = Data()
            var byte: UInt8 = 0
            while headerData.count < 65536 {
                let n = read(fd, &byte, 1)
                guard n == 1 else { return }
                headerData.append(byte)
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
            guard parts.count >= 3 else { return }
            method = String(parts[0])
            path = String(parts[1]).split(separator: "?").first.map(String.init) ?? String(parts[1])
            let httpVersion = String(parts[2]).uppercased()
            keepAlive = (httpVersion == "HTTP/1.1")

            var contentLength = 0
            for line in lines {
                let lower = line.lowercased()
                if lower.hasPrefix("content-length:") {
                    contentLength = Int(lower.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
                }
                if lower.hasPrefix("origin:") {
                    originHeader = String(line.dropFirst(7)).trimmingCharacters(in: .whitespaces)
                }
                if lower.hasPrefix("connection:") {
                    let value = lower.dropFirst(11).trimmingCharacters(in: .whitespaces)
                    if value.contains("close") { keepAlive = false }
                    else if value.contains("keep-alive") { keepAlive = true }
                }
            }
            let corsOrigin = originHeader ?? "*"

            if contentLength > SimpleHTTPServer.maxBodySize {
                sendResponse(fd: fd, status: 413,
                           body: "{\"error\":{\"message\":\"request body too large\",\"type\":\"invalid_request_error\",\"code\":413}}",
                           contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
                if !keepAlive { return }
                continue
            }

            var bodyData = Data()
            if contentLength > 0 {
                bodyData.reserveCapacity(contentLength)
                var remaining = contentLength
                var buf = [UInt8](repeating: 0, count: min(remaining, 65536))
                while remaining > 0 {
                    let toRead = min(remaining, buf.count)
                    let n = read(fd, &buf, toRead)
                    guard n > 0 else { return }
                    bodyData.append(contentsOf: buf[0..<n])
                    remaining -= n
                }
            }
            let bodyStr = String(data: bodyData, encoding: .utf8) ?? ""

            log("\(method) \(path)")

            switch (method, path) {
            case ("GET", "/v1/models"):
                let models = "{\"object\":\"list\",\"data\":[{\"id\":\"\(modelId)\",\"object\":\"model\",\"created\":\(Int(Date().timeIntervalSince1970)),\"owned_by\":\"local\",\"meta\":{\"n_ctx_train\":131072}}]}"
                sendResponse(fd: fd, status: 200, body: models, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("POST", "/v1/chat/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let request = try? JSONDecoder().decode(ChatRequest.self, from: data) else {
                    sendResponse(fd: fd, status: 400,
                               body: "{\"error\":{\"message\":\"invalid request\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
                    if !keepAlive { return }
                    continue
                }
                let isStreaming = request.stream ?? false
                await handleChat(fd: fd, request: request, corsOrigin: corsOrigin, keepAlive: keepAlive)
                if isStreaming { return }

            case ("GET", "/tokenizer_info"), ("GET", "/v1/tokenizer_info"):
                let ctx = await container.perform { ctx in ctx }
                let eos = ctx.tokenizer.eosToken ?? ""
                let bos = ctx.tokenizer.bosToken ?? ""
                let eosId = ctx.tokenizer.eosTokenId ?? -1
                let bosId = ctx.tokenizer.bosToken.flatMap { ctx.tokenizer.convertTokenToId($0) } ?? -1
                let info = "{\"eos_token\":\"\(eos)\",\"bos_token\":\"\(bos)\",\"eos_token_id\":\(eosId),\"bos_token_id\":\(bosId),\"model\":\"\(modelId)\"}"
                sendResponse(fd: fd, status: 200, body: info, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("POST", "/tokenize"), ("POST", "/v1/tokenize"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":{\"message\":\"missing prompt\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
                    if !keepAlive { return }
                    continue
                }
                let addSpecial = json["add_special_tokens"] as? Bool ?? true
                let ctx = await container.perform { ctx in ctx }
                let tokens = ctx.tokenizer.encode(text: prompt, addSpecialTokens: addSpecial)
                let tokensJson = "[\(tokens.map { String($0) }.joined(separator: ","))]"
                sendResponse(fd: fd, status: 200, body: "{\"tokens\":\(tokensJson)}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("POST", "/v1/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":{\"message\":\"invalid request\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
                    if !keepAlive { return }
                    continue
                }
                let maxTokens = json["max_tokens"] as? Int ?? 256
                let temperature = json["temperature"] as? Double ?? 0.0
                let isStream = json["stream"] as? Bool ?? false
                await handleCompletions(fd: fd, prompt: prompt, maxTokens: maxTokens,
                                       temperature: Float(temperature), stream: isStream, corsOrigin: corsOrigin, keepAlive: keepAlive)
                if isStream { return }

            case ("GET", "/health"), ("GET", "/"):
                sendResponse(fd: fd, status: 200, body: "{\"status\":\"ok\"}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("GET", "/metrics"):
                let m = await promptCache.getMetrics()
                let sc = await promptCache.getSessionCount()
                let activeSlots = await slotManager.activeSlotCount()
                let queueDepth = await slotManager.queueDepth()
                let body = """
                {"cache":{"requests":\(m.totalRequests),"hits":\(m.cacheHits),"misses":\(m.cacheMisses),"hit_rate":\(String(format:"%.3f",m.hitRate)),"trim_failures":\(m.trimFailures),"evictions":\(m.evictions),"sessions_active":\(sc),"sessions_max":\(await promptCache.maxSessions)},"throughput":{"total_prefill_tokens":\(m.totalPrefillTokens),"total_reused_tokens":\(m.totalReusedTokens),"total_decode_tokens":\(m.totalDecodeTokens),"avg_prefill_tokens_per_request":\(String(format:"%.0f",m.avgPrefillTokens)),"avg_prefill_ms":\(String(format:"%.1f",m.avgPrefillMs)),"avg_decode_tok_per_sec":\(String(format:"%.1f",m.avgDecodeTokensPerSec))},"slots":{"total":\(slotCount),"active":\(activeSlots),"queue_depth":\(queueDepth)}}
                """
                sendResponse(fd: fd, status: 200, body: body.trimmingCharacters(in: .whitespacesAndNewlines), contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("GET", "/slots"):
                let status = await slotManager.slotStatus()
                let slotsJSON = status.map { s in
                    "{\"id\":\(s.id),\"state\":\"\(s.state)\",\"request_id\":\"\(s.requestId)\",\"prompt_tokens\":\(s.promptTokens),\"generation_tokens\":\(s.genTokens),\"elapsed_ms\":\(String(format:"%.0f",s.elapsed * 1000))}"
                }.joined(separator: ",")
                let qd = await slotManager.queueDepth()
                let body = "{\"slots\":[\(slotsJSON)],\"queue_depth\":\(qd)}"
                sendResponse(fd: fd, status: 200, body: body, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

            case ("POST", "/v1/messages"):
                await handleAnthropicMessages(fd: fd, bodyStr: bodyStr, corsOrigin: corsOrigin, keepAlive: keepAlive)
                let wasStreaming = bodyStr.contains("\"stream\":true") || bodyStr.contains("\"stream\": true")
                if wasStreaming { return }

            case ("OPTIONS", _):
                sendResponse(fd: fd, status: 204, body: "", contentType: "text/plain",
                            extraHeaders: "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n", corsOrigin: corsOrigin, keepAlive: keepAlive)

            default:
                sendResponse(fd: fd, status: 404, body: "{\"error\":{\"message\":\"not found\",\"type\":\"not_found_error\",\"code\":404}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
            }

            let elapsed = (CFAbsoluteTimeGetCurrent() - requestStart) * 1000
            log("\(method) \(path) \(String(format: "%.0f", elapsed))ms")

            if !keepAlive { return }
        }
    }

    // MARK: - Response Helpers

    func sendResponse(fd: Int32, status: Int, body: String, contentType: String, extraHeaders: String = "", corsOrigin: String = "*", keepAlive: Bool = true) {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 204: statusText = "No Content"
        case 400: statusText = "Bad Request"
        case 413: statusText = "Payload Too Large"
        case 404: statusText = "Not Found"
        case 500: statusText = "Internal Server Error"
        case 503: statusText = "Service Unavailable"
        default: statusText = "Unknown"
        }
        let connectionHeader = keepAlive ? "keep-alive" : "close"
        let response = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.utf8.count)\r\nConnection: \(connectionHeader)\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\(extraHeaders)\r\n\(body)"
        _ = response.withCString { write(fd, $0, Int(strlen($0))) }
    }

    // MARK: - SSE Helpers

    func writeSSE(fd: Int32, requestId: String, role: String?, content: String?, finishReason: String?, reasoningContent: String? = nil, requestModel: String? = nil, includeNullContent: Bool = false) {
        func escape(_ s: String) -> String {
            s.replacingOccurrences(of: "\\", with: "\\\\")
             .replacingOccurrences(of: "\"", with: "\\\"")
             .replacingOccurrences(of: "\n", with: "\\n")
             .replacingOccurrences(of: "\r", with: "\\r")
        }
        var parts: [String] = []
        if let role = role { parts.append("\"role\":\"\(escape(role))\"") }
        if let content = content {
            parts.append("\"content\":\"\(escape(content))\"")
        } else if includeNullContent {
            parts.append("\"content\":null")
        }
        if let rc = reasoningContent { parts.append("\"reasoning_content\":\"\(escape(rc))\"") }
        let deltaJson = "{\(parts.joined(separator: ","))}"

        let fr = finishReason.map { "\"\($0)\"" } ?? "null"
        let responseModel = requestModel ?? modelId
        let event = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"delta\":\(deltaJson),\"finish_reason\":\(fr)}]}\n\n"
        _ = event.withCString { write(fd, $0, Int(strlen($0))) }
    }

    func emitToolCallSSE(fd: Int32, requestId: String, name: String, arguments: String, requestModel: String? = nil) {
        let argsEscaped = arguments.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        let tcId = UUID().uuidString.lowercased()
        let responseModel = requestModel ?? modelId
        let tcEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"index\":0,\"id\":\"\(tcId)\",\"type\":\"function\",\"function\":{\"name\":\"\(name)\",\"arguments\":\"\(argsEscaped)\"}}]}}]}\n\n"
        _ = tcEvent.withCString { write(fd, $0, Int(strlen($0))) }
    }

    enum ServerError: Error {
        case socketCreation, bind(UInt16), listen
    }
}
