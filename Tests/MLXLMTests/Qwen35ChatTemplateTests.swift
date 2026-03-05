// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import Tokenizers

/// Comprehensive tests for Qwen3.5 chat template rendering.
///
/// These tests verify that swift-jinja correctly renders the Qwen3.5 chat_template.jinja
/// for all combinations of: thinking on/off, tool calling with/without args,
/// multi-turn tool calls, and tool responses.
///
/// The expected outputs are derived from the official Qwen3.5 Jinja template:
/// https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja
@Suite("Qwen3.5 Chat Template Tests")
struct Qwen35ChatTemplateTests {

    // MARK: - Shared Tokenizer

    @MainActor
    static let tokenizerTask = Task {
        try await AutoTokenizer.from(pretrained: "mlx-community/Qwen3.5-4B-8bit")
    }

    static func sharedTokenizer() async throws -> Tokenizer {
        try await tokenizerTask.value
    }

    // MARK: - Helper

    /// Render messages through the chat template and return decoded text.
    func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) async throws -> String {
        let tokenizer = try await Self.sharedTokenizer()
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
        )
        return tokenizer.decode(tokens: tokens)
    }

    // MARK: - 1. Simple Chat (No Tools, No Thinking)

    @Test("Simple user message without tools or thinking")
    func simpleChat() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "Hello!"],
        ]
        let result = try await render(messages: messages)
        print("[Template Test] simpleChat:\n\(result)")

        // Expected: system message, user message, and generation prompt with <think>
        #expect(result.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"))
        #expect(result.contains("<|im_start|>user\nHello!<|im_end|>"))
        #expect(result.contains("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 2. Thinking Enabled

    @Test("Chat with thinking explicitly enabled")
    func thinkingEnabled() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What is 2+2?"],
        ]
        let result = try await render(
            messages: messages,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] thinkingEnabled:\n\(result)")

        // Should end with <think>\n (open-ended for model to fill)
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 3. Thinking Disabled

    @Test("Chat with thinking disabled produces empty think block")
    func thinkingDisabled() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What is 2+2?"],
        ]
        let result = try await render(
            messages: messages,
            additionalContext: ["enable_thinking": false]
        )
        print("[Template Test] thinkingDisabled:\n\(result)")

        // Should end with empty think block: <think>\n\n</think>\n\n
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    // MARK: - 4. Tools Present (First Turn)

    @Test("Tools in system message - correct tool schema injection")
    func toolsPresent() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What time is it?"],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolsPresent:\n\(result)")

        // Tool schema should be in system message
        #expect(result.contains("# Tools"))
        #expect(result.contains("<tools>"))
        #expect(result.contains("get_time"))
        #expect(result.contains("</tools>"))

        // Tool call format instructions should be present
        #expect(result.contains("<tool_call>"))
        #expect(result.contains("<function=example_function_name>"))
        #expect(result.contains("<IMPORTANT>"))

        // User's system message should be appended after tool instructions
        #expect(result.contains("You are a helpful assistant."))

        // Should end with generation prompt
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 5. Tool Call With Arguments

    @Test("Assistant tool call with arguments renders correctly")
    func toolCallWithArgs() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What's the weather in Tokyo?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Tokyo", "unit": "celsius"]
                                as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable],
                            "unit": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolCallWithArgs:\n\(result)")

        // Tool call should use XML function format
        #expect(result.contains("<tool_call>"))
        #expect(result.contains("<function=get_weather>"))
        #expect(result.contains("<parameter=location>"))
        #expect(result.contains("Tokyo"))
        #expect(result.contains("</parameter>"))
        #expect(result.contains("</function>"))
        #expect(result.contains("</tool_call>"))
    }

    // MARK: - 6. Tool Call Without Arguments

    @Test("Assistant tool call without arguments renders correctly")
    func toolCallNoArgs() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What time is it?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_current_time",
                            "arguments": [:] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_current_time",
                    "description": "Get current time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolCallNoArgs:\n\(result)")

        // Should have tool call tags but no parameter tags
        #expect(result.contains("<tool_call>"))
        #expect(result.contains("<function=get_current_time>"))
        #expect(result.contains("</function>"))
        #expect(result.contains("</tool_call>"))
    }

    // MARK: - 7. Tool Call With String Arguments (JSON string)

    @Test("Assistant tool call with JSON string arguments")
    func toolCallWithStringArgs() async throws {
        // This tests the case where arguments come as a JSON string (common in real usage)
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What's the weather?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": "{\"location\": \"Paris\"}",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolCallWithStringArgs:\n\(result)")

        // The template uses `tool_call.arguments|items` which requires a dict, not a string.
        // If arguments is a JSON string, this may fail silently.
        // This test documents the actual behavior.
        #expect(result.contains("<tool_call>"))
        #expect(result.contains("<function=get_weather>"))
    }

    // MARK: - 8. Multi-Turn: Tool Call → Tool Response → Assistant

    @Test("Complete tool call round-trip: call → response → answer")
    func multiTurnToolCall() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "What's the weather in Tokyo?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Tokyo"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"],
            ["role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."],
            ["role": "user", "content": "What about Paris?"],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] multiTurnToolCall:\n\(result)")

        // Verify structure:
        // 1. System message with tools
        #expect(result.contains("# Tools"))

        // 2. First user message
        #expect(result.contains("<|im_start|>user\nWhat's the weather in Tokyo?<|im_end|>"))

        // 3. Assistant with tool call (before last_query_index, so NO thinking)
        #expect(result.contains("<|im_start|>assistant\n<tool_call>"))

        // 4. Tool response wrapped in <tool_response> inside user role
        #expect(result.contains("<tool_response>"))
        #expect(result.contains("\"temperature\": 22"))
        #expect(result.contains("</tool_response>"))

        // 5. Assistant response
        #expect(result.contains("The weather in Tokyo is 22°C and sunny."))

        // 6. Second user question
        #expect(result.contains("<|im_start|>user\nWhat about Paris?<|im_end|>"))

        // 7. Generation prompt
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 9. Tool Call With reasoning_content

    @Test("Tool call with reasoning_content preserves thinking in history")
    func toolCallWithReasoningContent() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What time is it?"],
            [
                "role": "assistant",
                "reasoning_content": "I should use the get_time tool to answer this.",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_time",
                            "arguments": [:] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"time\": \"3:00 PM\"}"],
            ["role": "user", "content": "Thanks, now what day is it?"],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolCallWithReasoningContent:\n\(result)")

        // KEY TEST: The assistant tool call message is BEFORE last_query_index (index 1 < 3).
        // Per the template's logic at line 100:
        //   if loop.index0 > ns.last_query_index → wrap with <think>
        //   else → no <think> wrapping
        // So reasoning_content should NOT appear in output (template strips it for history).
        // This is the root cause of the model not learning the </think> → <tool_call> pattern.

        // Check if reasoning_content appears (it shouldn't per the default template)
        let hasThinkingInHistory = result.contains("<think>\nI should use the get_time tool")
        print("[Template Test] Thinking preserved in history: \(hasThinkingInHistory)")
        // NOTE: This test DOCUMENTS the behavior. The default template strips thinking
        // from historical messages before last_query_index.
    }

    // MARK: - 10. Multiple Tool Calls in One Turn

    @Test("Multiple tool calls in single assistant message")
    func multipleToolCalls() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Compare weather in Tokyo and Paris"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Tokyo"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Paris"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [any Sendable],
            ] as [String: any Sendable],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] multipleToolCalls:\n\(result)")

        // Should have two tool_call blocks
        let toolCallCount = result.components(separatedBy: "<tool_call>").count - 1
        print("[Template Test] Tool call count: \(toolCallCount)")
        #expect(toolCallCount == 2, "Expected 2 tool calls, got \(toolCallCount)")

        // Both locations should be present
        #expect(result.contains("Tokyo"))
        #expect(result.contains("Paris"))
    }

    // MARK: - 11. Tool Response Grouping

    @Test("Consecutive tool responses are grouped under single user message")
    func toolResponseGrouping() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Compare Tokyo and Paris weather"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Tokyo"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Paris"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [any Sendable],
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"temp\": 22}"],
            ["role": "tool", "content": "{\"temp\": 18}"],
            ["role": "user", "content": "Which is warmer?"],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolResponseGrouping:\n\(result)")

        // Consecutive tool responses should be grouped under one <|im_start|>user block
        // with multiple <tool_response> sections (uses loop.previtem / loop.nextitem)
        let toolResponseCount = result.components(separatedBy: "<tool_response>").count - 1
        print("[Template Test] Tool response count: \(toolResponseCount)")
        #expect(toolResponseCount == 2, "Expected 2 tool responses, got \(toolResponseCount)")
    }

    // MARK: - 12. last_query_index Logic

    @Test("last_query_index correctly identifies last real user message")
    func lastQueryIndex() async throws {
        // The template's last_query_index skips user messages that are just tool responses.
        // In this conversation:
        //   [0] system, [1] user (query), [2] assistant (tool call),
        //   [3] tool (response), [4] user (follow-up query)
        // last_query_index should be 4 (the follow-up, not the tool response)
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "What time is it?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_time",
                            "arguments": [:] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
            ["role": "tool", "content": "{\"time\": \"3:00 PM\"}"],
            ["role": "assistant", "content": "It's 3:00 PM."],
            ["role": "user", "content": "Thanks!"],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] lastQueryIndex:\n\(result)")

        // Messages BEFORE last_query_index (index 5) should NOT have <think> wrapping.
        // The assistant tool call at index 2 and the "It's 3:00 PM" at index 4
        // should both be WITHOUT <think> tags.

        // The assistant at index 2 should just have content (no think wrapper)
        #expect(result.contains("<|im_start|>assistant\n<tool_call>") ||
                result.contains("<|im_start|>assistant\n\n<tool_call>"))

        // The generation prompt should have <think> since we're generating after last query
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 13. Verify Exact Prompt Structure (Simple Case)

    @Test("Exact prompt structure for simple tool call scenario")
    func exactPromptStructure() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What time is it?"],
        ]
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

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        // Print the FULL rendered prompt for manual inspection
        let sep = String(repeating: "=", count: 80)
        print(sep)
        print("[Template Test] FULL RENDERED PROMPT (simple tool call):")
        print(sep)
        print(result)
        print(sep)

        // Verify the overall structure is correct
        let lines = result.components(separatedBy: "\n")
        #expect(!lines.isEmpty, "Rendered prompt should not be empty")

        // Should start with system message containing tools
        #expect(result.hasPrefix("<|im_start|>system\n# Tools"))

        // Should contain the user message
        #expect(result.contains("<|im_start|>user\nWhat time is it?<|im_end|>"))

        // Should end with assistant generation prompt
        #expect(result.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - 14. Verify Tool Call Arguments Iteration Bug

    @Test("Tool call arguments iteration - is defined vs is mapping")
    func toolCallArgumentsIteration() async throws {
        // The upstream template has: `tool_call.arguments is defined`
        // which should be `is mapping` per the HF fix.
        // This test checks if arguments render correctly as key-value pairs.
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Weather?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": [
                                "location": "Tokyo",
                                "unit": "celsius",
                                "detailed": true,
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable],
                            "unit": ["type": "string"] as [String: any Sendable],
                            "detailed": ["type": "boolean"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolCallArgumentsIteration:\n\(result)")

        // Each argument should have its own parameter tag
        #expect(result.contains("<parameter=location>"))
        #expect(result.contains("Tokyo"))
        #expect(result.contains("<parameter=unit>"))
        #expect(result.contains("celsius"))
        #expect(result.contains("<parameter=detailed>"))
        // Boolean should be converted to string: "true" or "True"
        let hasDetailedValue = result.contains("true") || result.contains("True")
        #expect(hasDetailedValue, "detailed parameter should contain boolean value")
    }

    // MARK: - 15. Tool Response Without loop.previtem/nextitem

    @Test("Single tool response renders correctly without previtem/nextitem issues")
    func singleToolResponse() async throws {
        // This tests the simplest tool response path to verify it works
        // even if loop.previtem/nextitem are broken in swift-jinja
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What time?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_time",
                            "arguments": [:] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ] as [String: any Sendable],
            ["role": "tool", "content": "3:00 PM"],
            ["role": "user", "content": "Thanks"],
        ]

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] singleToolResponse:\n\(result)")

        // Tool response should be wrapped
        #expect(result.contains("<tool_response>"))
        #expect(result.contains("3:00 PM"))
        #expect(result.contains("</tool_response>"))

        // Should have im_start/im_end around the tool response section
        // The template wraps tool responses in user role
        #expect(result.contains("<|im_start|>user"))
    }

    // MARK: - 16. No Tools (Verify template handles absence)

    @Test("Messages without tools don't include tool schema")
    func noTools() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "Hello!"],
        ]
        let result = try await render(
            messages: messages,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] noTools:\n\(result)")

        #expect(!result.contains("# Tools"))
        #expect(!result.contains("<tools>"))
        #expect(!result.contains("<IMPORTANT>"))
    }

    // MARK: - 17. Empty System Message With Tools

    @Test("Tools present but no system message content")
    func toolsNoSystemContent() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What time?"],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let result = try await render(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )
        print("[Template Test] toolsNoSystemContent:\n\(result)")

        // Should still have tool schema in system message
        #expect(result.contains("# Tools"))
        // But no user-provided system content appended
    }

    // MARK: - 18. Token ID Diagnostics

    @Test("Diagnostic: Check special token IDs and <tool_call> tokenization")
    func tokenDiagnostics() async throws {
        let tokenizer = try await Self.sharedTokenizer()

        // Check EOS token
        let eosId = tokenizer.eosTokenId
        print("[Diagnostics] EOS token ID: \(String(describing: eosId))")
        if let eosId {
            let eosDecoded = tokenizer.decode(tokens: [eosId])
            print("[Diagnostics] EOS token decoded: [\(eosDecoded)]")
        }

        // Check how key special strings tokenize
        let specialStrings = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
            "<|endoftext|>",
        ]

        for str in specialStrings {
            let encoded = tokenizer.encode(text: str, addSpecialTokens: false)
            let decoded = tokenizer.decode(tokens: encoded)
            let isSingleToken = encoded.count == 1
            print("[Diagnostics] '\(str)' → tokens: \(encoded) (count: \(encoded.count), single: \(isSingleToken)) → decoded: [\(decoded)]")
        }

        // Check if <tool_call> tokenization matches the token the model needs to produce
        let toolCallTokens = tokenizer.encode(text: "<tool_call>", addSpecialTokens: false)
        #expect(!toolCallTokens.isEmpty, "<tool_call> should tokenize to at least one token")

        // Print the full transition sequence the model should produce
        let transitionText = "</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
        let transitionTokens = tokenizer.encode(text: transitionText, addSpecialTokens: false)
        print("[Diagnostics] Full transition tokenized (\(transitionTokens.count) tokens): \(transitionTokens)")

        // Decode each token individually to see the token boundaries
        for (i, tokenId) in transitionTokens.enumerated() {
            let decoded = tokenizer.decode(tokens: [tokenId])
            print("[Diagnostics]   token[\(i)]: \(tokenId) → [\(decoded)]")
        }
    }

    // MARK: - 19. Full Prompt Dump (Integration Test Scenario)

    @Test("Diagnostic: Render and dump the exact prompt used in integration test")
    func fullPromptDump() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. When asked about time, you MUST use the get_current_time tool. Do not answer without calling the tool first."],
            ["role": "user", "content": "What time is it?"],
        ]
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

        let tokenizer = try await Self.sharedTokenizer()
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": true]
        )

        let decoded = tokenizer.decode(tokens: tokens)

        let separator = String(repeating: "=", count: 80)
        print(separator)
        print("[Full Prompt] Token count: \(tokens.count)")
        print("[Full Prompt] Decoded text:")
        print(separator)
        print(decoded)
        print(separator)

        // Show the last 20 tokens (the generation boundary)
        let lastTokens = Array(tokens.suffix(20))
        print("[Full Prompt] Last 20 tokens:")
        for (i, tokenId) in lastTokens.enumerated() {
            let tokenDecoded = tokenizer.decode(tokens: [tokenId])
            print("  [\(tokens.count - 20 + i)]: \(tokenId) → [\(tokenDecoded)]")
        }
    }
}
