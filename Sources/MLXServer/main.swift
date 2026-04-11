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

// MARK: - Prompt Cache

/// Caches KV state from previous requests to avoid re-prefilling shared prefixes.
/// When a new request shares the same token prefix as the cached state,
/// only the new tokens need to be prefilled.
final class PromptCache {
    var tokens: [Int] = []
    var cache: [KVCache] = []

    /// Find the common prefix length between cached tokens and new tokens
    func commonPrefixLength(with newTokens: [Int]) -> Int {
        let maxLen = min(tokens.count, newTokens.count)
        for i in 0..<maxLen {
            if tokens[i] != newTokens[i] { return i }
        }
        return maxLen
    }

    /// Get a reusable cache for the given tokens. Returns (cache, tokensToProcess).
    /// If there's a prefix match, returns the cached KV state and only the new tokens.
    func fetch(tokens newTokens: [Int], model: any LanguageModel) -> ([KVCache], [Int]) {
        let prefixLen = commonPrefixLength(with: newTokens)

        if prefixLen > 0 && !cache.isEmpty {
            // We have a matching prefix — trim cache to prefix length
            let trimAmount = tokens.count - prefixLen
            if trimAmount > 0 {
                for c in cache { let _ = c.trim(trimAmount) }
            }
            let remaining = Array(newTokens[prefixLen...])
            log("Cache hit: \(prefixLen)/\(newTokens.count) tokens cached, \(remaining.count) new")
            return (cache, remaining)
        }

        // No match — fresh cache
        log("Cache miss: prefilling \(newTokens.count) tokens from scratch")
        cache = model.newCache(parameters: nil)
        return (cache, newTokens)
    }

    /// Save the current state after generation
    func save(tokens: [Int]) {
        self.tokens = tokens
    }
}

final class SimpleHTTPServer {
    let port: UInt16
    let container: ModelContainer
    let modelId: String
    let promptCache = PromptCache()
    private var serverSocket: Int32 = -1

    init(port: UInt16, container: ModelContainer, modelId: String) {
        self.port = port
        self.container = container
        self.modelId = modelId
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
        log("Endpoints: GET /v1/models, POST /v1/chat/completions")

        while true {
            let client = accept(serverSocket, nil, nil)
            guard client >= 0 else { continue }
            Task { await handleClient(client) }
        }
    }

    func handleClient(_ fd: Int32) async {
        defer { close(fd) }

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
        let method = String(parts[0])
        let path = String(parts[1]).split(separator: "?").first.map(String.init) ?? String(parts[1])

        // Parse Content-Length and read body
        var contentLength = 0
        for line in lines {
            let lower = line.lowercased()
            if lower.hasPrefix("content-length:") {
                contentLength = Int(lower.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
            }
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

        case ("GET", "/health"), ("GET", "/"):
            sendResponse(fd: fd, status: 200, body: "{\"status\":\"ok\"}", contentType: "application/json")

        case ("OPTIONS", _):
            // CORS preflight
            sendResponse(fd: fd, status: 204, body: "", contentType: "text/plain",
                        extraHeaders: "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n")

        default:
            sendResponse(fd: fd, status: 404, body: "{\"error\":\"not found\"}", contentType: "application/json")
        }
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
            let (reusedCache, newTokens) = promptCache.fetch(tokens: tokens, model: ctx.model)
            let tokenArray = MLXArray(newTokens.isEmpty ? tokens : newTokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: request.temperature ?? 0.6)
            if let maxTokens = request.max_tokens {
                params.maxTokens = maxTokens
            }

            // Set tool call format if tools are present
            // Qwen models use XML format (<tool_call>), most others use JSON
            if toolsAny != nil {
                ctx.configuration.toolCallFormat = .xmlFunction
            }

            log("Generating: \(tokens.count) prompt tokens, stream=\(isStreaming), tools=\(toolsAny?.count ?? 0), max=\(request.max_tokens ?? -1), toolFormat=\(String(describing: ctx.configuration.toolCallFormat))")

            if isStreaming {
                // SSE headers
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
                _ = header.withCString { write(fd, $0, Int(strlen($0))) }

                var hadToolCall = false
                // Send role chunk immediately so client knows we're alive
                writeSSE(fd: fd, requestId: requestId, role: "assistant", content: "", finishReason: nil)
                // Flush immediately
                _ = "".withCString { _ in fcntl(fd, F_FULLFSYNC) }

                for try await generation in try generate(
                    input: input, cache: reusedCache, parameters: params, context: ctx
                ) {
                    switch generation {
                    case .chunk(let text):
                        writeSSE(fd: fd, requestId: requestId, role: nil, content: text, finishReason: nil)
                    case .toolCall(let tc):
                        hadToolCall = true
                        // Emit tool call — arguments must be a JSON STRING (not object)
                        let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                        let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                        // Escape the JSON string for embedding in JSON
                        let argsRaw = String(data: argsJSON, encoding: .utf8) ?? "{}"
                        let argsEscaped = argsRaw.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                        let tcId = UUID().uuidString.lowercased()
                        let tcEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"index\":0,\"id\":\"\(tcId)\",\"type\":\"function\",\"function\":{\"name\":\"\(tc.function.name)\",\"arguments\":\"\(argsEscaped)\"}}]}}]}\n\n"
                        _ = tcEvent.withCString { write(fd, $0, Int(strlen($0))) }
                    case .info(let info):
                        let fr = hadToolCall ? "tool_calls" : "stop"
                        let usageJSON = ",\"usage\":{\"prompt_tokens\":\(info.promptTokenCount),\"completion_tokens\":\(info.generationTokenCount),\"total_tokens\":\(info.promptTokenCount + info.generationTokenCount)}"
                        let finalEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(fr)\"}]\(usageJSON)}\n\ndata: [DONE]\n\n"
                        _ = finalEvent.withCString { write(fd, $0, Int(strlen($0))) }
                    }
                }
                // Save cache for next request
                promptCache.save(tokens: tokens)
            } else {
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
                // Save cache for next request
                promptCache.save(tokens: tokens)
            }
        } catch {
            let err = "{\"error\":{\"message\":\"\(error.localizedDescription)\"}}"
            sendResponse(fd: fd, status: 500, body: err, contentType: "application/json")
        }
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
