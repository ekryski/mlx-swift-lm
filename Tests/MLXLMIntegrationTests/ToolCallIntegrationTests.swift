// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

/// Integration tests for tool call format auto-detection and end-to-end parsing.
///
/// These tests verify that:
/// 1. Tool call formats are correctly auto-detected from model_type
/// 2. Tool calls are correctly parsed from actual model generation output
///
/// References:
/// - LFM2: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
/// - GLM4: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public class ToolCallIntegrationTests: XCTestCase {

    // MARK: - Model IDs

    static let lfm2ModelId = "mlx-community/LFM2-2.6B-Exp-4bit"
    static let glm4ModelId = "mlx-community/GLM-4-9B-0414-4bit"
    static let mistral3ModelId = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"

    // MARK: - Shared State

    nonisolated(unsafe) static var lfm2Container: ModelContainer?
    nonisolated(unsafe) static var glm4Container: ModelContainer?
    nonisolated(unsafe) static var mistral3Container: ModelContainer?

    // MARK: - Tool Schema

    static let weatherToolSchema: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        ] as [String: any Sendable],
                        "unit": [
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    // MARK: - Setup

    override public class func setUp() {
        super.setUp()

        let lfm2Expectation = XCTestExpectation(description: "Load LFM2")
        let glm4Expectation = XCTestExpectation(description: "Load GLM4")
        let mistral3Expectation = XCTestExpectation(description: "Load Mistral3")

        Task {
            do {
                lfm2Container = try await LLMModelFactory.shared.loadContainer(
                    configuration: .init(id: lfm2ModelId)
                )
            } catch {
                print("Failed to load LFM2: \(error)")
            }
            lfm2Expectation.fulfill()
        }

        Task {
            do {
                glm4Container = try await LLMModelFactory.shared.loadContainer(
                    configuration: .init(id: glm4ModelId)
                )
            } catch {
                print("Failed to load GLM4: \(error)")
            }
            glm4Expectation.fulfill()
        }

        Task {
            do {
                mistral3Container = try await VLMModelFactory.shared.loadContainer(
                    configuration: .init(id: mistral3ModelId)
                )
            } catch {
                print("Failed to load Mistral3: \(error)")
            }
            mistral3Expectation.fulfill()
        }

        _ = XCTWaiter.wait(
            for: [lfm2Expectation, glm4Expectation, mistral3Expectation], timeout: 600)
    }

    // MARK: - LFM2 Tests

    func testLFM2ToolCallFormatAutoDetection() async throws {
        guard let container = Self.lfm2Container else {
            throw XCTSkip("LFM2 model not available")
        }

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .lfm2,
            "LFM2 model should auto-detect .lfm2 tool call format"
        )
    }

    func testLFM2EndToEndToolCallGeneration() async throws {
        guard let container = Self.lfm2Container else {
            throw XCTSkip("LFM2 model not available")
        }

        // Create input with tool schema
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema
        )

        // Generate with tools
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("LFM2 Output: \(result)")
        print("LFM2 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            // Location should contain something related to Tokyo
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    // MARK: - GLM4 Tests

    func testGLM4ToolCallFormatAutoDetection() async throws {
        guard let container = Self.glm4Container else {
            throw XCTSkip("GLM4 model not available")
        }

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .glm4,
            "GLM4 model should auto-detect .glm4 tool call format"
        )
    }

    func testGLM4EndToEndToolCallGeneration() async throws {
        guard let container = Self.glm4Container else {
            throw XCTSkip("GLM4 model not available")
        }

        // Create input with tool schema
        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Paris?"),
            ],
            tools: Self.weatherToolSchema
        )

        // Generate with tools
        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("GLM4 Output: \(result)")
        print("GLM4 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            // Location should contain something related to Paris
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("paris"),
                    "Expected location to contain 'Paris', got: \(location)"
                )
            }
        }
    }

    // MARK: - Mistral3 Tests

    func testMistral3ToolCallFormatAutoDetection() async throws {
        guard let container = Self.mistral3Container else {
            throw XCTSkip("Mistral3 model not available")
        }

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .mistral,
            "Mistral3 model should auto-detect .mistral tool call format"
        )
    }

    func testMistral3EndToEndToolCallGeneration() async throws {
        guard let container = Self.mistral3Container else {
            throw XCTSkip("Mistral3 model not available")
        }

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. When asked about weather, use the get_weather function."
                ),
                .user("What's the weather in Tokyo?"),
            ],
            tools: Self.weatherToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 100
        )

        print("Mistral3 Output: \(result)")
        print("Mistral3 Tool Calls: \(toolCalls)")

        // Verify we got a tool call (model may or may not call the tool)
        if !toolCalls.isEmpty {
            let toolCall = toolCalls.first!
            XCTAssertEqual(toolCall.function.name, "get_weather")
            if let location = toolCall.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("tokyo"),
                    "Expected location to contain 'Tokyo', got: \(location)"
                )
            }
        }
    }

    func testMistral3MultipleToolCallGeneration() async throws {
        guard let container = Self.mistral3Container else {
            throw XCTSkip("Mistral3 model not available")
        }

        let multiToolSchema: [[String: any Sendable]] =
            Self.weatherToolSchema + [
                [
                    "type": "function",
                    "function": [
                        "name": "get_time",
                        "description": "Get the current time in a given timezone",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "timezone": [
                                    "type": "string",
                                    "description":
                                        "The timezone, e.g. America/New_York, Asia/Tokyo",
                                ] as [String: any Sendable]
                            ] as [String: any Sendable],
                            "required": ["timezone"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]

        let input = UserInput(
            chat: [
                .system(
                    "You are a helpful assistant with access to tools. Always use the available tools to answer questions. Call multiple tools in parallel when needed."
                ),
                .user(
                    "What's the weather in Tokyo and what time is it there?"
                ),
            ],
            tools: multiToolSchema
        )

        let (result, toolCalls) = try await generateWithTools(
            container: container,
            input: input,
            maxTokens: 150
        )

        print("Mistral3 Output: \(result)")
        print("Mistral3 Calls: \(toolCalls)")

        // Verify all returned tool calls have valid names from our schema
        let validNames: Set<String> = ["get_weather", "get_time"]
        for toolCall in toolCalls {
            XCTAssertTrue(
                validNames.contains(toolCall.function.name),
                "Unexpected tool call: \(toolCall.function.name)"
            )
        }

        // If the model made multiple calls, verify we got more than one
        if toolCalls.count > 1 {
            print("Successfully parsed \(toolCalls.count) tool calls from Mistral3")
        }
    }

    // MARK: - Helper Methods

    /// Generate text and collect any tool calls
    private func generateWithTools(
        container: ModelContainer,
        input: UserInput,
        maxTokens: Int
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        let result = try await container.perform(nonSendable: input) {
            (context: ModelContext, input) in
            let lmInput = try await context.processor.prepare(input: input)
            let parameters = GenerateParameters(maxTokens: maxTokens)

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            var collectedText = ""
            var collectedToolCalls: [ToolCall] = []

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info:
                    break
                }
            }

            return (collectedText, collectedToolCalls)
        }

        return result
    }

    // MARK: - Qwen3.5 Tests

    /// Minimal test to verify Qwen3.5 4B can produce native tool calls.
    ///
    /// This test isolates the mlx-swift-lm pipeline from Sam to determine
    /// whether the model can generate <tool_call> XML and the library can
    /// parse it correctly with the .qwen35 format.
    ///
    /// Uses Qwen team recommended parameters for thinking mode:
    /// temperature=1.0, top_p=0.95, presence_penalty=1.5, repetition_penalty=1.0, min_p=0.05
    func testQwen35ToolCallGeneration() async throws {
        // Load Qwen3.5 4B — will be downloaded on first run
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        // Verify format auto-detection
        let detectedFormat = await container.configuration.toolCallFormat
        XCTAssertEqual(
            detectedFormat, .qwen35,
            "Qwen3.5 should auto-detect .qwen35 tool call format, got: \(String(describing: detectedFormat))"
        )

        // Simple tool schema
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        // Create input — use .messages() path with enable_thinking
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool. Do not answer without calling the tool first."],
            ["role": "user", "content": "What time is it?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        // Qwen team recommended parameters for thinking mode
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 1.0,
            topP: 0.95,
            minP: 0.05,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            let allTokens = lmInput.text.tokens.asArray(Int32.self)
            print("[Qwen3.5 Test] Prompt tokens: \(allTokens.count)")

            // Decode last 30 tokens of prompt for debugging
            let lastTokens = Array(allTokens.suffix(30))
            let decoded = await context.tokenizer.decode(tokens: lastTokens.map { Int($0) })
            print("[Qwen3.5 Test] Prompt suffix (last 30 tokens): \(decoded)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 Test] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 Test] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 Test] Tool calls: \(collectedToolCalls)")
        print("[Qwen3.5 Test] Text length: \(collectedText.count), tool call count: \(collectedToolCalls.count)")

        // The model should either:
        // 1. Produce a tool call via the .qwen35 parser (ideal)
        // 2. Produce text containing <tool_call> XML (parser not matching)
        // 3. Produce text mentioning it would use a tool (thinking only, no actual call)
        // Any of these tells us the model CAN generate tokens with tools present
        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        XCTAssertTrue(producedOutput, "Qwen3.5 should produce some output (text or tool calls), got 0 tokens")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 Test] ✅ Native tool call detected!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_current_time")
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 Test] ⚠️ Tool call XML in text but not parsed by processor")
        } else {
            print("[Qwen3.5 Test] ⚠️ Model produced text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Test Qwen3.5 with thinking disabled to see if that changes tool call behavior.
    /// Some smaller models may need thinking disabled to focus on structured output.
    func testQwen35ToolCallNoThinking() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool."],
            ["role": "user", "content": "What time is it?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": false]
        )

        // Qwen team: instruct/non-thinking mode for general tasks
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.7,
            topP: 0.8,
            topK: 20,
            minP: 0.0,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5,
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            let allTokens = lmInput.text.tokens.asArray(Int32.self)
            print("[Qwen3.5 NoThink] Prompt tokens: \(allTokens.count)")

            let lastTokens = Array(allTokens.suffix(30))
            let decoded = await context.tokenizer.decode(tokens: lastTokens.map { Int($0) })
            print("[Qwen3.5 NoThink] Prompt suffix (last 30 tokens): \(decoded)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 NoThink] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 NoThink] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 NoThink] Tool calls: \(collectedToolCalls)")
        print("[Qwen3.5 NoThink] Text length: \(collectedText.count), tool call count: \(collectedToolCalls.count)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        // Note: 4B model may produce 0 tokens with thinking disabled — that's expected
        print("[Qwen3.5 NoThink] Produced output: \(producedOutput)")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 NoThink] ✅ Native tool call detected!")
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 NoThink] ⚠️ Tool call XML in text but not parsed")
        } else if collectedText.isEmpty {
            print("[Qwen3.5 NoThink] ⚠️ 0 tokens — 4B model likely requires thinking mode")
        } else {
            print("[Qwen3.5 NoThink] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Test Qwen3.5 with a multi-turn conversation that includes a tool call example
    /// in the history, so the model can see the </think> → <tool_call> pattern.
    func testQwen35ToolCallWithHistory() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        // Multi-turn with a previous tool call in history
        // This gives the model an in-context example of the </think> → <tool_call> pattern
        // NOTE: arguments must be a dictionary, not a JSON string, for the template
        // to render <parameter=key>value</parameter> blocks correctly.
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about weather, you MUST use the get_weather tool."],
            ["role": "user", "content": "What's the weather in Tokyo?"],
            [
                "role": "assistant",
                "reasoning_content": "The user is asking about the weather in Tokyo. I should use the get_weather tool.",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": [
                                "location": "Tokyo"
                            ] as [String: any Sendable]
                        ] as [String: any Sendable]
                    ] as [String: any Sendable]
                ] as [any Sendable]
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"],
            ["role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."],
            ["role": "user", "content": "What about Paris?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        // Qwen team recommended parameters for thinking mode
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 1.0,
            topP: 0.95,
            minP: 0.05,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            let allTokens = lmInput.text.tokens.asArray(Int32.self)
            print("[Qwen3.5 History] Prompt tokens: \(allTokens.count)")

            // Show last 50 tokens to verify template renders tool call + thinking correctly
            let lastTokens = Array(allTokens.suffix(50))
            let decoded = await context.tokenizer.decode(tokens: lastTokens.map { Int($0) })
            print("[Qwen3.5 History] Prompt suffix (last 50 tokens): \(decoded)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 History] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 History] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 History] Tool calls: \(collectedToolCalls)")
        print("[Qwen3.5 History] Text length: \(collectedText.count), tool call count: \(collectedToolCalls.count)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        XCTAssertTrue(producedOutput, "Qwen3.5 should produce some output with history context")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 History] ✅ Native tool call detected with history!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_weather")
            if let location = collectedToolCalls.first?.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("paris"),
                    "Expected location to contain 'Paris', got: \(location)"
                )
            }
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 History] ⚠️ Tool call XML in text but not parsed by processor")
        } else {
            print("[Qwen3.5 History] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Test Qwen3.5 with the CUSTOM template that preserves thinking in history.
    ///
    /// The original template strips <think> tags from assistant messages before
    /// the last user query (the `last_query_index` mechanism). This means the model
    /// never sees the `</think> → <tool_call>` transition pattern in-context.
    ///
    /// The custom template adds: `{%- if reasoning_content or loop.index0 > ns.last_query_index %}`
    /// so that any assistant message with `reasoning_content` gets <think> wrapping,
    /// regardless of position. This gives the model the `</think>\n\n<tool_call>` pattern.
    func testQwen35ToolCallWithCustomTemplate() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        // Multi-turn with previous tool call + reasoning_content
        // NOTE: arguments must be a dictionary for the template to render parameters
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about weather, you MUST use the get_weather tool."],
            ["role": "user", "content": "What's the weather in Tokyo?"],
            [
                "role": "assistant",
                "reasoning_content": "The user is asking about the weather in Tokyo. I should use the get_weather tool to find out.",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": [
                                "location": "Tokyo"
                            ] as [String: any Sendable]
                        ] as [String: any Sendable]
                    ] as [String: any Sendable]
                ] as [any Sendable]
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"],
            ["role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."],
            ["role": "user", "content": "What about Paris?"],
        ]

        // Use the custom template that preserves thinking in history
        var input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        input.chatTemplate = Self.qwen35CustomTemplate

        // Qwen team recommended parameters for thinking mode
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 1.0,
            topP: 0.95,
            minP: 0.05,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            let allTokens = lmInput.text.tokens.asArray(Int32.self)
            print("[Qwen3.5 Custom] Prompt tokens: \(allTokens.count)")

            // Show last 80 tokens to verify thinking IS in history
            let lastTokens = Array(allTokens.suffix(80))
            let decoded = await context.tokenizer.decode(tokens: lastTokens.map { Int($0) })
            print("[Qwen3.5 Custom] Prompt suffix (last 80 tokens):\n\(decoded)")

            // Verify the custom template preserved <think> in history
            let fullPrompt = await context.tokenizer.decode(tokens: allTokens.map { Int($0) })
            let hasThinkInHistory = fullPrompt.contains("<think>\nThe user is asking about the weather")
            print("[Qwen3.5 Custom] Thinking preserved in history: \(hasThinkInHistory)")
            XCTAssertTrue(hasThinkInHistory, "Custom template should preserve <think> in historical assistant messages with reasoning_content")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 Custom] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 Custom] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 Custom] Tool calls: \(collectedToolCalls)")
        print("[Qwen3.5 Custom] Text length: \(collectedText.count), tool call count: \(collectedToolCalls.count)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        XCTAssertTrue(producedOutput, "Qwen3.5 should produce some output with custom template")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 Custom] ✅ Native tool call detected with custom template!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_weather")
            if let location = collectedToolCalls.first?.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("paris"),
                    "Expected location to contain 'Paris', got: \(location)"
                )
            }
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 Custom] ⚠️ Tool call XML in text but not parsed by processor")
        } else {
            print("[Qwen3.5 Custom] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Raw token diagnostic test: captures every token the model generates
    /// to determine exactly what's happening at the thinking → tool call boundary.
    func testQwen35RawTokenDiagnostic() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        // Log the loaded EOS configuration
        let loadedEos = await container.configuration.eosTokenIds
        print("[Token Diag] Loaded eosTokenIds from config: \(loadedEos)")

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool. Do not answer without calling the tool first."],
            ["role": "user", "content": "What time is it?"],
        ]

        var input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        input.chatTemplate = Self.qwen35CustomTemplate

        // Qwen team recommended parameters for thinking mode
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 1.0,
            topP: 0.95,
            minP: 0.05,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5
        )

        var allTokenIds: [Int] = []

        try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            // Use raw token generation with includeStopToken to see the stop token
            let stream = try generateTokens(
                input: lmInput,
                parameters: parameters,
                context: context,
                includeStopToken: true
            )

            for try await tokenGeneration in stream {
                switch tokenGeneration {
                case .token(let tokenId):
                    allTokenIds.append(tokenId)
                case .info(let info):
                    print("[Token Diag] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            // Decode and print each token
            print("[Token Diag] Total tokens generated (including stop): \(allTokenIds.count)")
            print("[Token Diag] Token-by-token breakdown:")
            for (i, tokenId) in allTokenIds.enumerated() {
                let decoded = context.tokenizer.decode(tokens: [tokenId])
                let isEos = tokenId == 248046 || tokenId == 248044
                let isThink = tokenId == 248068 || tokenId == 248069
                let isToolCall = tokenId == 248058 || tokenId == 248059
                let marker = isEos ? " ← EOS" : (isThink ? " ← THINK" : (isToolCall ? " ← TOOL" : ""))
                print("  [\(i)]: \(tokenId) → [\(decoded)]\(marker)")
            }

            // Full decoded text
            let fullText = context.tokenizer.decode(tokens: allTokenIds)
            print("[Token Diag] Full decoded output:\n\(fullText)")
        }
    }

    // MARK: - Qwen3.5 Parameter Profile Tests

    /// Test Qwen3.5 thinking mode with coding-optimized parameters.
    ///
    /// Qwen team recommendation for precise coding tasks:
    /// temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
    func testQwen35ToolCallThinkingCodingProfile() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool."],
            ["role": "user", "content": "What time is it?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        // Qwen team: thinking mode for precise coding tasks
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            minP: 0.0,
            repetitionPenalty: 1.0,
            presencePenalty: 0.0,
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            print("[Qwen3.5 ThinkCoding] Prompt tokens: \(lmInput.text.tokens.asArray(Int32.self).count)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 ThinkCoding] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 ThinkCoding] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 ThinkCoding] Tool calls: \(collectedToolCalls)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        XCTAssertTrue(producedOutput, "Qwen3.5 should produce output with thinking-coding profile")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 ThinkCoding] ✅ Native tool call detected!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_current_time")
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 ThinkCoding] ⚠️ Tool call XML in text but not parsed by processor")
        } else {
            print("[Qwen3.5 ThinkCoding] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Test Qwen3.5 non-thinking mode with general task parameters.
    ///
    /// Qwen team recommendation for instruct/non-thinking general tasks:
    /// temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
    func testQwen35ToolCallNonThinkingGeneralProfile() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool."],
            ["role": "user", "content": "What time is it?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": false]
        )

        // Qwen team: instruct/non-thinking mode for general tasks
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.7,
            topP: 0.8,
            topK: 20,
            minP: 0.0,
            repetitionPenalty: 1.0,
            presencePenalty: 1.5,
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            print("[Qwen3.5 NoThinkGeneral] Prompt tokens: \(lmInput.text.tokens.asArray(Int32.self).count)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 NoThinkGeneral] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 NoThinkGeneral] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 NoThinkGeneral] Tool calls: \(collectedToolCalls)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        // Note: 4B model may produce 0 tokens with thinking disabled
        print("[Qwen3.5 NoThinkGeneral] Produced output: \(producedOutput)")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 NoThinkGeneral] ✅ Native tool call detected!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_current_time")
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 NoThinkGeneral] ⚠️ Tool call XML in text but not parsed")
        } else if collectedText.isEmpty {
            print("[Qwen3.5 NoThinkGeneral] ⚠️ 0 tokens — 4B model may require thinking mode")
        } else {
            print("[Qwen3.5 NoThinkGeneral] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    /// Test Qwen3.5 non-thinking mode with reasoning task parameters.
    ///
    /// Qwen team recommendation for instruct/non-thinking reasoning tasks:
    /// temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0
    func testQwen35ToolCallNonThinkingReasoningProfile() async throws {
        let config = ModelConfiguration(id: "mlx-community/Qwen3.5-4B-8bit")
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            )
        } catch {
            throw XCTSkip("Qwen3.5 4B model not available: \(error)")
        }

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about weather, you MUST use the get_weather tool."],
            ["role": "user", "content": "What is the weather like in London?"],
        ]

        let input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": false]
        )

        // Qwen team: instruct/non-thinking mode for reasoning tasks
        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 1.0,
            topP: 1.0,
            topK: 40,
            minP: 0.0,
            repetitionPenalty: 1.0,
            presencePenalty: 2.0,
        )

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            print("[Qwen3.5 NoThinkReasoning] Prompt tokens: \(lmInput.text.tokens.asArray(Int32.self).count)")

            let stream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    collectedText += text
                case .toolCall(let toolCall):
                    collectedToolCalls.append(toolCall)
                case .info(let info):
                    print("[Qwen3.5 NoThinkReasoning] Info: prompt=\(info.promptTokenCount) generated=\(info.generationTokenCount)")
                }
            }

            return (collectedText, collectedToolCalls)
        }

        print("[Qwen3.5 NoThinkReasoning] Raw text output: [\(collectedText)]")
        print("[Qwen3.5 NoThinkReasoning] Tool calls: \(collectedToolCalls)")

        let producedOutput = !collectedText.isEmpty || !collectedToolCalls.isEmpty
        // Note: 4B model may produce 0 tokens with thinking disabled
        print("[Qwen3.5 NoThinkReasoning] Produced output: \(producedOutput)")

        if !collectedToolCalls.isEmpty {
            print("[Qwen3.5 NoThinkReasoning] ✅ Native tool call detected!")
            XCTAssertEqual(collectedToolCalls.first?.function.name, "get_weather")
            if let location = collectedToolCalls.first?.function.arguments["location"]?.asString {
                XCTAssertTrue(
                    location.lowercased().contains("london"),
                    "Expected location to contain 'London', got: \(location)"
                )
            }
        } else if collectedText.contains("<tool_call>") || collectedText.contains("<function=") {
            print("[Qwen3.5 NoThinkReasoning] ⚠️ Tool call XML in text but not parsed")
        } else if collectedText.isEmpty {
            print("[Qwen3.5 NoThinkReasoning] ⚠️ 0 tokens — 4B model may require thinking mode")
        } else {
            print("[Qwen3.5 NoThinkReasoning] ⚠️ Text but no tool call: \(collectedText.prefix(500))")
        }
    }

    // MARK: - Custom Template

    /// Qwen3.5 custom chat template with two fixes:
    /// 1. Preserve thinking in history when reasoning_content is present
    /// 2. Fix argument iteration for tool calls (HuggingFace fix)
    /// 3. Don't inject empty think block when thinking is disabled
    // swiftlint:disable line_length
    static let qwen35CustomTemplate = """
    {%- set image_count = namespace(value=0) %}\
    {%- set video_count = namespace(value=0) %}\
    {%- macro render_content(content, do_vision_count, is_system_content=false) %}\
        {%- if content is string %}\
            {{- content }}\
        {%- elif content is iterable and content is not mapping %}\
            {%- for item in content %}\
                {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\
                    {%- if is_system_content %}\
                        {{- raise_exception('System message cannot contain images.') }}\
                    {%- endif %}\
                    {%- if do_vision_count %}\
                        {%- set image_count.value = image_count.value + 1 %}\
                    {%- endif %}\
                    {%- if add_vision_id %}\
                        {{- 'Picture ' ~ image_count.value ~ ': ' }}\
                    {%- endif %}\
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\
                {%- elif 'video' in item or item.type == 'video' %}\
                    {%- if is_system_content %}\
                        {{- raise_exception('System message cannot contain videos.') }}\
                    {%- endif %}\
                    {%- if do_vision_count %}\
                        {%- set video_count.value = video_count.value + 1 %}\
                    {%- endif %}\
                    {%- if add_vision_id %}\
                        {{- 'Video ' ~ video_count.value ~ ': ' }}\
                    {%- endif %}\
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}\
                {%- elif 'text' in item %}\
                    {{- item.text }}\
                {%- else %}\
                    {{- raise_exception('Unexpected item type in content.') }}\
                {%- endif %}\
            {%- endfor %}\
        {%- elif content is none or content is undefined %}\
            {{- '' }}\
        {%- else %}\
            {{- raise_exception('Unexpected content type.') }}\
        {%- endif %}\
    {%- endmacro %}\
    {%- if not messages %}\
        {{- raise_exception('No messages provided.') }}\
    {%- endif %}\
    {%- if tools and tools is iterable and tools is not mapping %}\
        {{- '<|im_start|>system\\n' }}\
        {{- "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>" }}\
        {%- for tool in tools %}\
            {{- "\\n" }}\
            {{- tool | tojson }}\
        {%- endfor %}\
        {{- "\\n</tools>" }}\
        {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\
        {%- if messages[0].role == 'system' %}\
            {%- set content = render_content(messages[0].content, false, true)|trim %}\
            {%- if content %}\
                {{- '\\n\\n' + content }}\
            {%- endif %}\
        {%- endif %}\
        {{- '<|im_end|>\\n' }}\
    {%- else %}\
        {%- if messages[0].role == 'system' %}\
            {%- set content = render_content(messages[0].content, false, true)|trim %}\
            {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\
        {%- endif %}\
    {%- endif %}\
    {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\
    {%- for message in messages[::-1] %}\
        {%- set index = (messages|length - 1) - loop.index0 %}\
        {%- if ns.multi_step_tool and message.role == "user" %}\
            {%- set content = render_content(message.content, false)|trim %}\
            {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}\
                {%- set ns.multi_step_tool = false %}\
                {%- set ns.last_query_index = index %}\
            {%- endif %}\
        {%- endif %}\
    {%- endfor %}\
    {%- if ns.multi_step_tool %}\
        {{- raise_exception('No user query found in messages.') }}\
    {%- endif %}\
    {%- for message in messages %}\
        {%- set content = render_content(message.content, true)|trim %}\
        {%- if message.role == "system" %}\
            {%- if not loop.first %}\
                {{- raise_exception('System message must be at the beginning.') }}\
            {%- endif %}\
        {%- elif message.role == "user" %}\
            {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\
        {%- elif message.role == "assistant" %}\
            {%- set reasoning_content = '' %}\
            {%- if message.reasoning_content is string %}\
                {%- set reasoning_content = message.reasoning_content %}\
            {%- else %}\
                {%- if '</think>' in content %}\
                    {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\
                    {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\
                {%- endif %}\
            {%- endif %}\
            {%- set reasoning_content = reasoning_content|trim %}\
            {%- if reasoning_content or loop.index0 > ns.last_query_index %}\
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\
            {%- else %}\
                {{- '<|im_start|>' + message.role + '\\n' + content }}\
            {%- endif %}\
            {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}\
                {%- for tool_call in message.tool_calls %}\
                    {%- if tool_call.function is defined %}\
                        {%- set tool_call = tool_call.function %}\
                    {%- endif %}\
                    {%- if loop.first %}\
                        {%- if content|trim %}\
                            {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\
                        {%- else %}\
                            {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\
                        {%- endif %}\
                    {%- else %}\
                        {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\
                    {%- endif %}\
                    {%- if tool_call.arguments is mapping %}\
                        {%- for args_name in tool_call.arguments %}\
                            {%- set args_value = tool_call.arguments[args_name] %}\
                            {{- '<parameter=' + args_name + '>\\n' }}\
                            {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is iterable and args_value is not string) else args_value | string %}\
                            {{- args_value }}\
                            {{- '\\n</parameter>\\n' }}\
                        {%- endfor %}\
                    {%- endif %}\
                    {{- '</function>\\n</tool_call>' }}\
                {%- endfor %}\
            {%- endif %}\
            {{- '<|im_end|>\\n' }}\
        {%- elif message.role == "tool" %}\
            {%- if loop.previtem and loop.previtem.role != "tool" %}\
                {{- '<|im_start|>user' }}\
            {%- endif %}\
            {{- '\\n<tool_response>\\n' }}\
            {{- content }}\
            {{- '\\n</tool_response>' }}\
            {%- if not loop.last and loop.nextitem.role != "tool" %}\
                {{- '<|im_end|>\\n' }}\
            {%- elif loop.last %}\
                {{- '<|im_end|>\\n' }}\
            {%- endif %}\
        {%- else %}\
            {{- raise_exception('Unexpected message role.') }}\
        {%- endif %}\
    {%- endfor %}\
    {%- if add_generation_prompt %}\
        {{- '<|im_start|>assistant\\n' }}\
        {%- if enable_thinking is defined and enable_thinking is false %}\
        {%- else %}\
            {{- '<think>\\n' }}\
        {%- endif %}\
    {%- endif %}
    """
    // swiftlint:enable line_length
}

// MARK: - JSONValue Extension for Testing

extension JSONValue {
    var asString: String? {
        if case .string(let s) = self {
            return s
        }
        return nil
    }
}
