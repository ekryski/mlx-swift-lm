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

    // MARK: - Shared State

    nonisolated(unsafe) static var lfm2Container: ModelContainer?
    nonisolated(unsafe) static var glm4Container: ModelContainer?

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

        _ = XCTWaiter.wait(for: [lfm2Expectation, glm4Expectation], timeout: 600)
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

    // MARK: - Helper Methods

    /// Generate text and collect any tool calls
    private func generateWithTools(
        container: ModelContainer,
        input: UserInput,
        maxTokens: Int
    ) async throws -> (text: String, toolCalls: [ToolCall]) {
        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            let parameters = GenerateParameters(maxTokens: maxTokens)

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
            ["role": "system", "content": "You are a helpful assistant. When asked about time, use the get_current_time tool."],
            ["role": "user", "content": "What time is it?"],
        ]

        var input = UserInput(
            prompt: .messages(messages),
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        // Generate — use moderate temperature, no presence penalty
        let parameters = GenerateParameters(maxTokens: 200, temperature: 0.7, topP: 0.8)

        var collectedText = ""
        var collectedToolCalls: [ToolCall] = []

        let result = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)

            print("[Qwen3.5 Test] Prompt tokens: \(lmInput.text.tokens.dim(0))")

            // Decode first 100 tokens of prompt for debugging
            let allTokens = lmInput.text.tokens.asArray(Int32.self)
            let lastTokens = Array(allTokens.suffix(20))
            let decoded = await context.tokenizer.decode(tokens: lastTokens.map { Int($0) })
            print("[Qwen3.5 Test] Prompt suffix (last 20 tokens): \(decoded)")

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
            print("[Qwen3.5 Test] ⚠️ Model produced text but no tool call: \(collectedText.prefix(200))")
        }
    }
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
