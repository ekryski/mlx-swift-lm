import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon

// MARK: - Chat Completions

extension SimpleHTTPServer {

    func handleChat(fd: Int32, request: ChatRequest, corsOrigin: String = "*", keepAlive: Bool = true) async {
        let isStreaming = request.stream ?? false
        let requestId = "chatcmpl-\(UUID().uuidString.prefix(8))"
        log("params: temp=\(request.temperature ?? -1) max=\(request.effectiveMaxTokens ?? -1) top_p=\(request.top_p ?? -1) top_k=\(request.top_k ?? -1) min_p=\(request.min_p ?? -1) freq=\(request.frequency_penalty ?? -1) pres=\(request.presence_penalty ?? -1) rep=\(request.repeat_penalty ?? -1)")

        let slot = await slotManager.acquireSlot()
        slot.requestId = requestId
        slot.startTime = CFAbsoluteTimeGetCurrent()
        log("slot[\(slot.id)] acquired for \(requestId)")
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

            var messages: [[String: String]] = []
            for msg in request.messages {
                // Pass all roles through natively — Qwen's chat template handles
                // "tool" role with <tool_response> tags. Don't convert to "user".
                var d: [String: String] = ["role": msg.role]
                if let c = msg.content { d["content"] = c }
                messages.append(d)
            }

            let toolsAny = request.tools?.value as? [Any]
            let tokens: [Int]
            if let toolsAny = toolsAny {
                let toolSpecs: [[String: any Sendable]] = toolsAny.compactMap { $0 as? [String: Any] }.map { tool in
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
                // Always enable thinking — template handles it, we strip <think> from output
                let thinkCtx: [String: any Sendable] = ["enable_thinking": self.enableThinking]
                do {
                    tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs, additionalContext: thinkCtx)
                } catch {
                    log("Chat template with tools failed, retrying without tools in template")
                    var toolDesc = "Available tools:\n"
                    for tool in toolSpecs {
                        if let fn = tool["function"] as? [String: Any],
                           let name = fn["name"] as? String {
                            let desc = fn["description"] as? String ?? ""
                            toolDesc += "- \(name): \(desc)\n"
                        }
                    }
                    if var sys = messages.first, sys["role"] == "system" {
                        sys["content"] = (sys["content"] ?? "") + "\n\n" + toolDesc
                        var adjusted = messages
                        adjusted[0] = sys
                        tokens = try ctx.tokenizer.applyChatTemplate(messages: adjusted, tools: nil, additionalContext: thinkCtx)
                    } else {
                        var adjusted = messages
                        adjusted.insert(["role": "system", "content": toolDesc], at: 0)
                        tokens = try ctx.tokenizer.applyChatTemplate(messages: adjusted, tools: nil, additionalContext: thinkCtx)
                    }
                }
            } else {
                let thinkCtx: [String: any Sendable] = ["enable_thinking": self.enableThinking]
                tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: thinkCtx)
            }

            let prefillStart = CFAbsoluteTimeGetCurrent()
            let (reusedCache, newTokens, cacheStatus, sessionId) = await promptCache.fetch(tokens: tokens, model: ctx.model)
            let tokenArray = MLXArray(newTokens.isEmpty ? tokens : newTokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: request.temperature ?? 0.6)
            let maxTokens = request.effectiveMaxTokens
            if let maxTokens { params.maxTokens = maxTokens }
            if let topP = request.top_p { params.topP = topP }
            if let topK = request.top_k { params.topK = topK }
            if let minP = request.min_p { params.minP = minP }

            let repContext = request.repeat_last_n ?? 64
            if let rp = request.repeat_penalty, rp > 0 {
                params.repetitionPenalty = rp
                params.repetitionContextSize = repContext
            }
            if let fp = request.frequency_penalty, fp > 0 {
                params.frequencyPenalty = fp
                params.frequencyContextSize = repContext
            }
            if let pp = request.presence_penalty, pp > 0 {
                params.presencePenalty = pp
                params.presenceContextSize = repContext
            }
            if params.repetitionPenalty == nil && params.frequencyPenalty == nil && params.presencePenalty == nil {
                params.repetitionPenalty = 1.1
                params.repetitionContextSize = repContext
            }
            params.kvScheme = self.kvScheme
            if let bits = self.kvBits { params.kvBits = bits }
            if let start = self.kvStart { params.quantizedKVStart = start }

            // toolCallFormat is auto-detected from model_type during model load
            // (see ToolCallFormat.infer). No need to override here.

            slot.promptTokenCount = tokens.count
            slot.state = .generating
            log("slot[\(slot.id)] \(cacheStatus.logString) stream=\(isStreaming)")

            var keepaliveTimer: DispatchSourceTimer? = nil
            if isStreaming {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\r\n"
                _ = header.withCString { write(fd, $0, Int(strlen($0))) }

                let timer = DispatchSource.makeTimerSource(queue: .global())
                let keepaliveFd = fd
                timer.schedule(deadline: .now() + 2, repeating: 2.0)
                timer.setEventHandler {
                    let chunk = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\n"
                    _ = chunk.withCString { write(keepaliveFd, $0, Int(strlen($0))) }
                }
                timer.resume()
                keepaliveTimer = timer
            }

            if isStreaming {
                var prefillDone = false
                let reqModel = request.model
                var hadToolCall = false

                writeSSE(fd: fd, requestId: requestId, role: "assistant", content: nil, finishReason: nil, requestModel: reqModel, includeNullContent: true)

                do {
                    var fullText = ""
                    var thinkText = ""
                    var emittedUpTo = 0
                    // When thinking is enabled, template injects <think>\n,
                    // so model output starts as thinking content.
                    // When off, still detect <think> tags in case model produces them anyway.
                    var inThinkBlock = self.enableThinking
                    var tokenCount = 0

                    for try await generation in try generate(
                        input: input, cache: reusedCache, parameters: params, context: ctx
                    ) {
                        if !prefillDone {
                            prefillDone = true
                            keepaliveTimer?.cancel()
                            await releasePrefillOnce()
                        }
                        tokenCount += 1

                        switch generation {
                        case .chunk(let text):
                            fullText += text

                            if emittedUpTo == 0 && !inThinkBlock {
                                let trimmed = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
                                if trimmed.hasPrefix("<think") || trimmed.hasPrefix("</think") {
                                    inThinkBlock = true
                                }
                            }
                            if inThinkBlock {
                                if fullText.contains("</think>") {
                                    if let range = fullText.range(of: "</think>") {
                                        var thinkContent = String(fullText[..<range.lowerBound])
                                        if let tagEnd = thinkContent.range(of: "<think>") {
                                            thinkContent = String(thinkContent[tagEnd.upperBound...])
                                        }
                                        let newThink = thinkContent.count > thinkText.count ? String(thinkContent[thinkContent.index(thinkContent.startIndex, offsetBy: thinkText.count)...]) : ""
                                        if !newThink.isEmpty {
                                            writeSSE(fd: fd, requestId: requestId, role: nil, content: nil, finishReason: nil, reasoningContent: newThink, requestModel: reqModel)
                                        }
                                        thinkText = thinkContent
                                        let afterThink = String(fullText[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                                        fullText = afterThink
                                        emittedUpTo = 0
                                        inThinkBlock = false
                                        if fullText.isEmpty { continue }
                                    }
                                } else {
                                    var thinkContent = fullText
                                    if let tagEnd = thinkContent.range(of: "<think>") {
                                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                                    }
                                    let newThink = thinkContent.count > thinkText.count ? String(thinkContent[thinkContent.index(thinkContent.startIndex, offsetBy: thinkText.count)...]) : ""
                                    if !newThink.isEmpty {
                                        writeSSE(fd: fd, requestId: requestId, role: nil, content: nil, finishReason: nil, reasoningContent: newThink, requestModel: reqModel)
                                        thinkText = thinkContent
                                    }
                                    continue
                                }
                            }

                            if hadToolCall {
                                let remaining = String(fullText[fullText.index(fullText.startIndex, offsetBy: emittedUpTo)...])
                                let stripped = remaining
                                    .replacingOccurrences(of: "</minimax:tool_call>", with: "")
                                    .replacingOccurrences(of: "</invoke>", with: "")
                                    .replacingOccurrences(of: "</tool_call>", with: "")
                                    .trimmingCharacters(in: .whitespacesAndNewlines)
                                if stripped.isEmpty {
                                    emittedUpTo = fullText.count
                                    continue
                                }
                            }

                            if emittedUpTo == 0 && !fullText.isEmpty {
                                let trimmed = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
                                if trimmed.isEmpty { continue }
                                if trimmed.count < fullText.count { fullText = trimmed }
                            }

                            let unemitted = String(fullText[fullText.index(fullText.startIndex, offsetBy: emittedUpTo)...])

                            if unemitted.contains("<tool_call>") || unemitted.contains("<function=") || unemitted.contains("<minimax:tool_call>") || unemitted.contains("<invoke name=") {
                                if unemitted.contains("</tool_call>") || unemitted.contains("</function>") || unemitted.contains("</minimax:tool_call>") || unemitted.contains("</invoke>") {
                                    let tcs = parseAllToolCalls(unemitted)
                                    if !tcs.isEmpty {
                                        hadToolCall = true
                                        for tc in tcs { emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments, requestModel: reqModel) }
                                    } else {
                                        writeSSE(fd: fd, requestId: requestId, role: nil, content: unemitted, finishReason: nil, requestModel: reqModel)
                                    }
                                    emittedUpTo = fullText.count
                                }
                            } else if unemitted.contains("<") {
                                let tagPrefixes = ["<tool_call", "<function=", "<minimax:", "<invoke", "</think", "<think", "<parameter"]
                                let hasTagStart = tagPrefixes.contains(where: { unemitted.contains($0) })
                                if hasTagStart {
                                    if let ltIdx = unemitted.lastIndex(of: "<") {
                                        let safe = String(unemitted[unemitted.startIndex..<ltIdx])
                                        if !safe.isEmpty {
                                            writeSSE(fd: fd, requestId: requestId, role: nil, content: safe, finishReason: nil, requestModel: reqModel)
                                            emittedUpTo += safe.count
                                        }
                                    }
                                    continue
                                }
                                writeSSE(fd: fd, requestId: requestId, role: nil, content: unemitted, finishReason: nil, requestModel: reqModel)
                                emittedUpTo = fullText.count
                            } else {
                                writeSSE(fd: fd, requestId: requestId, role: nil, content: text, finishReason: nil, requestModel: reqModel)
                                emittedUpTo = fullText.count
                            }

                        case .toolCall(let tc):
                            hadToolCall = true
                            let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                            let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                            let argsRaw = String(data: argsJSON, encoding: .utf8) ?? "{}"
                            emitToolCallSSE(fd: fd, requestId: requestId, name: tc.function.name, arguments: argsRaw, requestModel: reqModel)

                        case .info(let info):
                            if emittedUpTo < fullText.count {
                                let remaining = String(fullText[fullText.index(fullText.startIndex, offsetBy: emittedUpTo)...])
                                let remainTCs = parseAllToolCalls(remaining)
                                if !remainTCs.isEmpty {
                                    hadToolCall = true
                                    for tc in remainTCs { emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments, requestModel: reqModel) }
                                } else if !remaining.isEmpty {
                                    writeSSE(fd: fd, requestId: requestId, role: nil, content: remaining, finishReason: nil, requestModel: reqModel)
                                }
                            }
                            let maxTok = request.max_tokens ?? Int.max
                            let fr = hadToolCall ? "tool_calls" : (info.generationTokenCount >= maxTok ? "length" : "stop")
                            let responseModel = reqModel ?? modelId
                            let usageJSON = ",\"usage\":{\"prompt_tokens\":\(info.promptTokenCount),\"completion_tokens\":\(info.generationTokenCount),\"total_tokens\":\(info.promptTokenCount + info.generationTokenCount)}"
                            let finalEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(fr)\"}]\(usageJSON)}\n\ndata: [DONE]\n\n"
                            _ = finalEvent.withCString { write(fd, $0, Int(strlen($0))) }
                        }
                    }
                } catch {
                    await releasePrefillOnce()
                    log("ERROR streaming generation: \(error)")
                    let errMsg = error.localizedDescription.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                    let errEvent = "data: {\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}\n\ndata: [DONE]\n\n"
                    _ = errEvent.withCString { write(fd, $0, Int(strlen($0))) }
                }

                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, tokens: tokens)
            } else {
                keepaliveTimer?.cancel()

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

                var reasoningContent: String? = nil
                if let thinkEnd = fullText.range(of: "</think>") {
                    var thinkContent = String(fullText[..<thinkEnd.lowerBound])
                    if let tagEnd = thinkContent.range(of: "<think>") {
                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                    }
                    reasoningContent = thinkContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    fullText = String(fullText[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if fullText.hasPrefix("<think>") {
                    var thinkContent = fullText
                    if let tagEnd = thinkContent.range(of: "<think>") {
                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                    }
                    reasoningContent = thinkContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    if reasoningContent?.isEmpty == true { reasoningContent = nil }
                    fullText = ""
                }

                if toolCalls.isEmpty {
                    let parsed = parseAllToolCalls(fullText)
                    if !parsed.isEmpty {
                        for tc in parsed { toolCalls.append((name: tc.name, args: tc.arguments)) }
                        fullText = ""
                    }
                }

                let responseModel = request.model ?? modelId
                let maxTok = request.max_tokens ?? Int.max
                let finishReason: String
                if !toolCalls.isEmpty { finishReason = "tool_calls" }
                else if completionTokens >= maxTok { finishReason = "length" }
                else { finishReason = "stop" }

                var responseBody: String
                if toolCalls.isEmpty {
                    let response = ChatResponse(
                        id: requestId, object: "chat.completion",
                        created: Int(Date().timeIntervalSince1970), model: responseModel,
                        system_fingerprint: "mlx-swift-v1",
                        choices: [.init(index: 0, message: .init(role: "assistant", content: fullText, reasoning_content: reasoningContent),
                                       delta: nil, finish_reason: finishReason)],
                        usage: .init(prompt_tokens: tokens.count, completion_tokens: completionTokens,
                                    total_tokens: tokens.count + completionTokens))
                    responseBody = String(data: try JSONEncoder().encode(response), encoding: .utf8)!
                } else {
                    let tcJSON = toolCalls.enumerated().map { (i, tc) in
                        // OpenAI spec: arguments must be a JSON-encoded string, not a raw object
                        let argsEscaped = tc.args.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                        return "{\"id\":\"call_\(UUID().uuidString.prefix(8))\",\"type\":\"function\",\"function\":{\"name\":\"\(tc.name)\",\"arguments\":\"\(argsEscaped)\"}}"
                    }.joined(separator: ",")
                    let content = "\"\(fullText.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r"))\""
                    let rcField: String
                    if let rc = reasoningContent {
                        let rcEscaped = rc.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r")
                        rcField = ",\"reasoning_content\":\"\(rcEscaped)\""
                    } else { rcField = "" }
                    responseBody = "{\"id\":\"\(requestId)\",\"object\":\"chat.completion\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\(content)\(rcField),\"tool_calls\":[\(tcJSON)]},\"finish_reason\":\"\(finishReason)\"}],\"usage\":{\"prompt_tokens\":\(tokens.count),\"completion_tokens\":\(completionTokens),\"total_tokens\":\(tokens.count + completionTokens)}}"
                }
                sendResponse(fd: fd, status: 200, body: responseBody,
                           contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)

                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, tokens: tokens)
            }
        } catch {
            log("ERROR handleChat: \(error)")
            let errMsg = error.localizedDescription.replacingOccurrences(of: "\"", with: "'")
            let err = "{\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}"
            sendResponse(fd: fd, status: 500, body: err, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
        }
    }

    // MARK: - Text Completions

    func handleCompletions(fd: Int32, prompt: String, maxTokens: Int, temperature: Float, stream: Bool, corsOrigin: String = "*", keepAlive: Bool = true) async {
        let requestId = "cmpl-\(UUID().uuidString.prefix(8))"

        let slot = await slotManager.acquireSlot()
        slot.requestId = requestId
        slot.startTime = CFAbsoluteTimeGetCurrent()
        slot.state = .generating
        defer { Task { await slotManager.releaseSlot(slot) } }

        do {
            let ctx = await container.perform { ctx in ctx }
            let tokens = ctx.tokenizer.encode(text: prompt)
            let tokenArray = MLXArray(tokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: temperature)
            params.maxTokens = maxTokens
            params.kvScheme = self.kvScheme

            log("completions: \(tokens.count) prompt tokens, max=\(maxTokens), stream=\(stream)")

            if stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\r\n"
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
                sendResponse(fd: fd, status: 200, body: body, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
            }
        } catch {
            let errMsg = String(describing: error).replacingOccurrences(of: "\"", with: "'")
            sendResponse(fd: fd, status: 500, body: "{\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin, keepAlive: keepAlive)
        }
    }
}
