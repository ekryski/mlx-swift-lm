import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon

// MARK: - Anthropic Messages API Types

struct AnthropicMessagesRequest: Codable {
    let model: String
    let max_tokens: Int
    let messages: [AnthropicMessage]
    let system: AnyCodable?  // string or array of content blocks
    let stream: Bool?
    let temperature: Float?
    let top_p: Float?
    let top_k: Int?
    let stop_sequences: [String]?
    let tools: [AnthropicTool]?
    let tool_choice: AnyCodable?
    let thinking: AnthropicThinking?

    enum CodingKeys: String, CodingKey {
        case model, max_tokens, messages, system, stream, temperature
        case top_p, top_k, stop_sequences, tools, tool_choice, thinking
    }
}

struct AnthropicMessage: Codable {
    let role: String
    let content: AnthropicContent

    enum CodingKeys: String, CodingKey { case role, content }
}

/// Content can be a string or array of content blocks
enum AnthropicContent: Codable {
    case text(String)
    case blocks([AnthropicContentBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
        } else {
            self = .text("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }

    var textContent: String {
        switch self {
        case .text(let s): return s
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let t) = block.type { return t }
                return nil
            }.joined(separator: "\n")
        }
    }
}

struct AnthropicContentBlock: Codable {
    let type: AnthropicBlockType

    enum CodingKeys: String, CodingKey {
        case type
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicCodingKeys.self)
        let typeStr = try container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "type")!)

        switch typeStr {
        case "text":
            let text = try container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "text")!)
            type = .text(text)
        case "tool_use":
            let id = try container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "id")!)
            let name = try container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "name")!)
            let input = try container.decode(AnyCodable.self, forKey: DynamicCodingKeys(stringValue: "input")!)
            type = .toolUse(id: id, name: name, input: input)
        case "tool_result":
            let toolUseId = try container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "tool_use_id")!)
            let content = try? container.decode(String.self, forKey: DynamicCodingKeys(stringValue: "content")!)
            type = .toolResult(toolUseId: toolUseId, content: content ?? "")
        default:
            type = .text("")
        }
    }

    func encode(to encoder: Encoder) throws {
        // Not needed for request parsing
    }
}

enum AnthropicBlockType {
    case text(String)
    case toolUse(id: String, name: String, input: AnyCodable)
    case toolResult(toolUseId: String, content: String)
}

struct DynamicCodingKeys: CodingKey {
    var stringValue: String
    var intValue: Int?
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { self.intValue = intValue; self.stringValue = "\(intValue)" }
}

struct AnthropicTool: Codable {
    let name: String
    let description: String?
    let input_schema: AnyCodable?

    enum CodingKeys: String, CodingKey {
        case name, description, input_schema
    }
}

struct AnthropicThinking: Codable {
    let type: String  // "enabled" or "disabled"
    let budget_tokens: Int?
}

// MARK: - Anthropic Messages Handler

extension SimpleHTTPServer {

    func handleAnthropicMessages(fd: Int32, bodyStr: String, corsOrigin: String, keepAlive: Bool) async {
        guard let data = bodyStr.data(using: .utf8),
              let request = try? JSONDecoder().decode(AnthropicMessagesRequest.self, from: data) else {
            sendAnthropicError(fd: fd, status: 400, type: "invalid_request_error",
                             message: "Invalid request body", corsOrigin: corsOrigin, keepAlive: keepAlive)
            return
        }

        let isStreaming = request.stream ?? false
        let msgId = "msg_\(UUID().uuidString.prefix(12).lowercased())"

        let slot = await slotManager.acquireSlot()
        slot.requestId = msgId
        slot.startTime = CFAbsoluteTimeGetCurrent()
        log("slot[\(slot.id)] acquired for \(msgId) (anthropic)")
        defer { Task { await slotManager.releaseSlot(slot) } }

        await slotManager.acquirePrefill()
        var prefillReleased = false
        func releasePrefillOnce() async {
            if !prefillReleased {
                prefillReleased = true
                await slotManager.releasePrefill()
            }
        }

        do {
            slot.state = .prefilling
            var ctx = await container.perform { ctx in ctx }

            // Convert Anthropic messages to chat template format
            var messages: [[String: String]] = []

            // System prompt is top-level in Anthropic format
            if let sys = request.system {
                if let sysStr = sys.value as? String {
                    messages.append(["role": "system", "content": sysStr])
                } else if let sysArr = sys.value as? [[String: Any]] {
                    let sysText = sysArr.compactMap { $0["text"] as? String }.joined(separator: "\n")
                    if !sysText.isEmpty {
                        messages.append(["role": "system", "content": sysText])
                    }
                }
            }

            for msg in request.messages {
                let content = msg.content.textContent
                messages.append(["role": msg.role, "content": content])
            }

            // Convert Anthropic tools to OpenAI format for template
            var toolSpecs: [[String: any Sendable]]? = nil
            if let tools = request.tools, !tools.isEmpty {
                toolSpecs = tools.map { tool -> [String: any Sendable] in
                    var fn: [String: any Sendable] = ["name": tool.name]
                    if let desc = tool.description { fn["description"] = desc }
                    if let schema = tool.input_schema?.value {
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
                        fn["parameters"] = convert(schema)
                    }
                    return ["type": "function" as any Sendable, "function": fn as any Sendable]
                }
            }

            // Determine thinking mode
            let thinkingEnabled: Bool
            if let thinking = request.thinking {
                thinkingEnabled = (thinking.type == "enabled")
            } else {
                thinkingEnabled = self.enableThinking
            }

            let thinkCtx: [String: any Sendable] = ["enable_thinking": thinkingEnabled]
            let tokens: [Int]
            if let toolSpecs {
                do {
                    tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs, additionalContext: thinkCtx)
                } catch {
                    tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: thinkCtx)
                }
            } else {
                tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: thinkCtx)
            }

            let prefillStart = CFAbsoluteTimeGetCurrent()
            let (reusedCache, newTokens, cacheStatus, sessionId) = await promptCache.fetch(tokens: tokens, model: ctx.model)
            let tokenArray = MLXArray(newTokens.isEmpty ? tokens : newTokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: request.temperature ?? 0.6)
            params.maxTokens = request.max_tokens
            if let topP = request.top_p { params.topP = topP }
            if let topK = request.top_k { params.topK = topK }
            // Default repetition penalty
            params.repetitionPenalty = 1.1
            params.repetitionContextSize = 64
            params.kvScheme = self.kvScheme
            if let bits = self.kvBits { params.kvBits = bits }
            if let start = self.kvStart { params.quantizedKVStart = start }

            slot.promptTokenCount = tokens.count
            slot.state = .generating
            log("slot[\(slot.id)] \(cacheStatus.logString) stream=\(isStreaming) (anthropic)")

            if isStreaming {
                await handleAnthropicStreaming(fd: fd, msgId: msgId, request: request,
                                               input: input, cache: reusedCache, params: params,
                                               ctx: ctx, tokens: tokens, sessionId: sessionId,
                                               prefillStart: prefillStart, corsOrigin: corsOrigin,
                                               thinkingEnabled: thinkingEnabled,
                                               releasePrefill: releasePrefillOnce)
            } else {
                // Non-streaming
                var fullText = ""
                var completionTokens = 0
                var toolCalls: [(name: String, args: String)] = []

                for try await generation in try generate(
                    input: input, cache: reusedCache, parameters: params, context: ctx
                ) {
                    if !prefillReleased { await releasePrefillOnce() }
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

                // Strip thinking blocks
                var reasoningContent: String? = nil
                if let thinkEnd = fullText.range(of: "</think>") {
                    var thinkContent = String(fullText[..<thinkEnd.lowerBound])
                    if let tagEnd = thinkContent.range(of: "<think>") {
                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                    }
                    reasoningContent = thinkContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    fullText = String(fullText[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if thinkingEnabled {
                    // Template injected <think>\n, output is all thinking until </think>
                    if let thinkEnd2 = fullText.range(of: "</think>") {
                        reasoningContent = String(fullText[..<thinkEnd2.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                        fullText = String(fullText[thinkEnd2.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                }

                // Parse XML tool calls from text
                if toolCalls.isEmpty {
                    let parsed = parseAllToolCalls(fullText)
                    if !parsed.isEmpty {
                        for tc in parsed { toolCalls.append((name: tc.name, args: tc.arguments)) }
                        fullText = ""
                    }
                }

                // Build content blocks
                var contentBlocks: [[String: Any]] = []

                if let rc = reasoningContent, !rc.isEmpty {
                    contentBlocks.append(["type": "thinking", "thinking": rc])
                }

                if !fullText.isEmpty {
                    contentBlocks.append(["type": "text", "text": fullText])
                }

                for tc in toolCalls {
                    let inputObj = (try? JSONSerialization.jsonObject(with: Data(tc.args.utf8))) ?? [:]
                    contentBlocks.append([
                        "type": "tool_use",
                        "id": "toolu_\(UUID().uuidString.prefix(8).lowercased())",
                        "name": tc.name,
                        "input": inputObj
                    ])
                }

                if contentBlocks.isEmpty {
                    contentBlocks.append(["type": "text", "text": ""])
                }

                let stopReason: String
                if !toolCalls.isEmpty { stopReason = "tool_use" }
                else if completionTokens >= request.max_tokens { stopReason = "max_tokens" }
                else { stopReason = "end_turn" }

                let response: [String: Any] = [
                    "id": msgId,
                    "type": "message",
                    "role": "assistant",
                    "model": request.model,
                    "content": contentBlocks,
                    "stop_reason": stopReason,
                    "stop_sequence": NSNull(),
                    "usage": [
                        "input_tokens": tokens.count,
                        "output_tokens": completionTokens,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0
                    ]
                ]

                let responseData = try JSONSerialization.data(withJSONObject: response)
                let responseBody = String(data: responseData, encoding: .utf8) ?? "{}"
                sendResponse(fd: fd, status: 200, body: responseBody,
                           contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

                await promptCache.recordTiming(prefillMs: 0, decodeMs: (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000, decodeTokens: completionTokens)
                await promptCache.save(sessionId: sessionId, tokens: tokens)
            }
        } catch {
            await releasePrefillOnce()
            log("ERROR anthropic messages: \(error)")
            sendAnthropicError(fd: fd, status: 500, type: "api_error",
                             message: error.localizedDescription, corsOrigin: corsOrigin, keepAlive: keepAlive)
        }
    }

    // MARK: - Anthropic Streaming

    private func handleAnthropicStreaming(
        fd: Int32, msgId: String, request: AnthropicMessagesRequest,
        input: LMInput, cache: [KVCache], params: GenerateParameters,
        ctx: ModelContext, tokens: [Int], sessionId: UUID,
        prefillStart: CFAbsoluteTime, corsOrigin: String,
        thinkingEnabled: Bool,
        releasePrefill: @escaping () async -> Void
    ) async {
        // SSE headers — Anthropic uses Connection: keep-alive for SSE
        let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\r\n"
        _ = header.withCString { write(fd, $0, Int(strlen($0))) }

        // message_start
        writeAnthropicEvent(fd: fd, event: "message_start",
            data: "{\"type\":\"message_start\",\"message\":{\"id\":\"\(msgId)\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"\(request.model)\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":\(tokens.count),\"output_tokens\":0,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}")

        var prefillDone = false
        var fullText = ""
        var thinkText = ""
        var inThinkBlock = thinkingEnabled
        var contentBlockStarted = false
        var thinkBlockStarted = false
        var blockIndex = 0
        var tokenCount = 0
        var hadToolCall = false

        do {
            for try await generation in try generate(
                input: input, cache: cache, parameters: params, context: ctx
            ) {
                if !prefillDone {
                    prefillDone = true
                    await releasePrefill()
                }
                tokenCount += 1

                switch generation {
                case .chunk(let text):
                    fullText += text

                    if inThinkBlock {
                        if fullText.contains("</think>") {
                            // Think block ended
                            if let range = fullText.range(of: "</think>") {
                                var thinkContent = String(fullText[..<range.lowerBound])
                                if let tagEnd = thinkContent.range(of: "<think>") {
                                    thinkContent = String(thinkContent[tagEnd.upperBound...])
                                }
                                let newThink = String(thinkContent.dropFirst(thinkText.count))
                                if !newThink.isEmpty && thinkBlockStarted {
                                    writeAnthropicDelta(fd: fd, index: blockIndex, type: "thinking_delta", key: "thinking", value: newThink)
                                }

                                // Close think block
                                if thinkBlockStarted {
                                    writeAnthropicEvent(fd: fd, event: "content_block_stop", data: "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}")
                                    blockIndex += 1
                                }

                                let afterThink = String(fullText[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                                fullText = afterThink
                                inThinkBlock = false
                                thinkBlockStarted = false
                                continue
                            }
                        } else {
                            // Still in think block — stream as thinking
                            var thinkContent = fullText
                            if let tagEnd = thinkContent.range(of: "<think>") {
                                thinkContent = String(thinkContent[tagEnd.upperBound...])
                            }
                            let newThink = thinkContent.count > thinkText.count ? String(thinkContent.dropFirst(thinkText.count)) : ""
                            if !newThink.isEmpty {
                                if !thinkBlockStarted {
                                    writeAnthropicEvent(fd: fd, event: "content_block_start",
                                                       data: "{\"type\":\"content_block_start\",\"index\":\(blockIndex),\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}")
                                    thinkBlockStarted = true
                                }
                                writeAnthropicDelta(fd: fd, index: blockIndex, type: "thinking_delta", key: "thinking", value: newThink)
                                thinkText = thinkContent
                            }
                            continue
                        }
                    }

                    // Regular content
                    if !text.isEmpty && !inThinkBlock {
                        if !contentBlockStarted {
                            writeAnthropicEvent(fd: fd, event: "content_block_start",
                                               data: "{\"type\":\"content_block_start\",\"index\":\(blockIndex),\"content_block\":{\"type\":\"text\",\"text\":\"\"}}")
                            contentBlockStarted = true
                        }
                        writeAnthropicDelta(fd: fd, index: blockIndex, type: "text_delta", key: "text", value: text)
                    }

                case .toolCall(let tc):
                    hadToolCall = true
                    // Close text block if open
                    if contentBlockStarted {
                        writeAnthropicEvent(fd: fd, event: "content_block_stop", data: "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}")
                        blockIndex += 1
                        contentBlockStarted = false
                    }

                    let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                    let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                    let argsStr = String(data: argsJSON, encoding: .utf8) ?? "{}"
                    let tcId = "toolu_\(UUID().uuidString.prefix(8).lowercased())"

                    writeAnthropicEvent(fd: fd, event: "content_block_start",
                                       data: "{\"type\":\"content_block_start\",\"index\":\(blockIndex),\"content_block\":{\"type\":\"tool_use\",\"id\":\"\(tcId)\",\"name\":\"\(tc.function.name)\",\"input\":{}}}")
                    let escapedArgs = argsStr.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                    writeAnthropicEvent(fd: fd, event: "content_block_delta",
                                       data: "{\"type\":\"content_block_delta\",\"index\":\(blockIndex),\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\(escapedArgs)\"}}")
                    writeAnthropicEvent(fd: fd, event: "content_block_stop", data: "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}")
                    blockIndex += 1

                case .info(let info):
                    // Close any open blocks
                    if thinkBlockStarted {
                        writeAnthropicEvent(fd: fd, event: "content_block_stop", data: "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}")
                        blockIndex += 1
                    }
                    if contentBlockStarted {
                        writeAnthropicEvent(fd: fd, event: "content_block_stop", data: "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}")
                    }

                    // Check for remaining tool calls in text
                    if !hadToolCall {
                        let parsed = parseAllToolCalls(fullText)
                        if !parsed.isEmpty { hadToolCall = true }
                    }

                    let stopReason = hadToolCall ? "tool_use" : (info.generationTokenCount >= request.max_tokens ? "max_tokens" : "end_turn")

                    writeAnthropicEvent(fd: fd, event: "message_delta",
                                       data: "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"\(stopReason)\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":\(info.generationTokenCount)}}")
                    writeAnthropicEvent(fd: fd, event: "message_stop", data: "{\"type\":\"message_stop\"}")
                }
            }
        } catch {
            log("ERROR anthropic streaming: \(error)")
            let errMsg = error.localizedDescription.replacingOccurrences(of: "\"", with: "'")
            writeAnthropicEvent(fd: fd, event: "error",
                               data: "{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"\(errMsg)\"}}")
        }

        await promptCache.recordTiming(prefillMs: 0, decodeMs: (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000, decodeTokens: tokenCount)
        await promptCache.save(sessionId: sessionId, tokens: tokens)
    }

    // MARK: - Anthropic SSE Helpers

    func writeAnthropicEvent(fd: Int32, event: String, data: String) {
        let sse = "event: \(event)\ndata: \(data)\n\n"
        _ = sse.withCString { write(fd, $0, Int(strlen($0))) }
    }

    func writeAnthropicDelta(fd: Int32, index: Int, type: String, key: String, value: String) {
        let escaped = value.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        let data = "{\"type\":\"content_block_delta\",\"index\":\(index),\"delta\":{\"type\":\"\(type)\",\"\(key)\":\"\(escaped)\"}}"
        writeAnthropicEvent(fd: fd, event: "content_block_delta", data: data)
    }

    func sendAnthropicError(fd: Int32, status: Int, type: String, message: String, corsOrigin: String, keepAlive: Bool) {
        let escapedMsg = message.replacingOccurrences(of: "\"", with: "'")
        let body = "{\"type\":\"error\",\"error\":{\"type\":\"\(type)\",\"message\":\"\(escapedMsg)\"}}"
        sendResponse(fd: fd, status: status, body: body, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
    }
}
