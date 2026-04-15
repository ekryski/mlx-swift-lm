import Foundation
import Testing

@testable import Tokenizers

/// Tests for Gemma 4 chat template rendering.
///
/// Verifies that the external `chat_template.jinja` (NOT inline in tokenizer_config.json)
/// is loaded and rendered correctly by swift-jinja. The critical test is that the
/// non-thinking mode produces a `<|channel>thought\n<channel|>` prefill at the end of
/// the prompt — without this prefill, the model generates thinking tokens itself and
/// produces "incoherent" output (the model IS coherent, but it's showing internal
/// chain-of-thought instead of responding).
///
/// Reference: Python `transformers` output for the same template, verified against
/// `mlx-community/gemma-4-26b-a4b-it-4bit` on 2026-04-14.
@Suite("Gemma 4 Chat Template Tests")
struct Gemma4ChatTemplateTests {

    // MARK: - Shared Tokenizer

    /// Load the 26b tokenizer (the model that exhibited "incoherent" output).
    /// Only downloads tokenizer files, not model weights.
    @MainActor
    static let tokenizerTask = Task {
        try await AutoTokenizer.from(pretrained: "mlx-community/gemma-4-26b-a4b-it-4bit")
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

    // MARK: - 1. Template Loads from External .jinja File

    @Test("External chat_template.jinja is loaded — output uses Gemma 4 turn tokens")
    func templateLoadsFromJinja() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Hello!"],
        ]
        let result = try await render(messages: messages)
        print("[Gemma4 Template] templateLoadsFromJinja:\n\(result)")

        // If the .jinja template loaded, output should have Gemma 4 turn tokens
        #expect(result.contains("<|turn>user\n"), "Should use <|turn> format from .jinja template")
        #expect(result.contains("<turn|>"), "Should have turn end markers")
        #expect(result.contains("<|turn>model\n"), "Should have model turn prompt")

        // Should NOT have Qwen-style tokens (wrong template)
        #expect(!result.contains("<|im_start|>"), "Should NOT have Qwen-style im_start tokens")
    }

    // MARK: - 2. Non-Thinking Mode Produces Prefill (CRITICAL)

    @Test("Non-thinking mode produces <|channel>thought prefill — prevents model self-thinking")
    func nonThinkingPrefill() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is your name?"],
        ]
        // No enable_thinking parameter — template default
        let result = try await render(messages: messages)
        print("[Gemma4 Template] nonThinkingPrefill:\n\(result)")

        // CRITICAL: The template MUST produce the thinking prefill at the end.
        // Without this, the model generates <|channel>thought tokens itself and
        // produces "incoherent" output that is actually chain-of-thought reasoning.
        //
        // Expected ending: <|turn>model\n<|channel>thought\n<channel|>
        #expect(
            result.contains("<|channel>thought\n<channel|>"),
            """
            Non-thinking mode MUST include '<|channel>thought\\n<channel|>' prefill.
            Without this prefill, the model generates thinking tokens itself,
            producing output that looks incoherent (it's actually internal reasoning).
            This is the root cause of Gemma 4 26b/31b 'incoherence' bug.

            Got: \(result.suffix(100))
            """
        )

        // The prefill should come after the model turn marker
        #expect(
            result.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"),
            """
            Prompt should end with model turn + thinking prefill.
            Got suffix: \(result.suffix(60))
            """
        )
    }

    // MARK: - 3. Explicit enable_thinking=false Also Produces Prefill

    @Test("Explicit enable_thinking=false also produces thinking prefill")
    func explicitThinkingFalsePrefill() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Hello!"],
        ]
        let result = try await render(
            messages: messages,
            additionalContext: ["enable_thinking": false]
        )
        print("[Gemma4 Template] explicitThinkingFalsePrefill:\n\(result)")

        // enable_thinking=false should produce the same prefill as no parameter
        #expect(
            result.contains("<|channel>thought\n<channel|>"),
            "enable_thinking=false should include thinking prefill"
        )
    }

    // MARK: - 4. Thinking Mode Omits Prefill

    @Test("Thinking mode omits prefill — model thinks freely")
    func thinkingModeNoPrefill() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is 2+2?"],
        ]
        let result = try await render(
            messages: messages,
            additionalContext: ["enable_thinking": true]
        )
        print("[Gemma4 Template] thinkingModeNoPrefill:\n\(result)")

        // With thinking enabled, the prefill should NOT be present
        #expect(
            !result.contains("<|channel>thought\n<channel|>"),
            "Thinking mode should NOT include the thinking prefill"
        )

        // Should end with just the model turn marker (model generates thinking freely)
        #expect(
            result.hasSuffix("<|turn>model\n"),
            "Thinking mode should end with model turn, no prefill"
        )

        // Thinking mode injects <|think|> token in the system block
        #expect(
            result.contains("<|think|>"),
            "Thinking mode should inject <|think|> marker in system block"
        )
    }

    // MARK: - 5. System Prompt Formatting

    @Test("System prompt uses Gemma 4 turn format")
    func systemPromptFormatting() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant. Keep responses concise."],
            ["role": "user", "content": "Hi"],
        ]
        let result = try await render(messages: messages)
        print("[Gemma4 Template] systemPromptFormatting:\n\(result)")

        // System message uses <|turn>system format
        #expect(result.contains("<|turn>system\n"))
        #expect(
            result.contains("You are a helpful assistant. Keep responses concise."),
            "System prompt content should be preserved"
        )
    }

    // MARK: - 6. BOS Token Present

    @Test("Output starts with BOS token")
    func bosToken() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Hello"],
        ]
        let result = try await render(messages: messages)
        print("[Gemma4 Template] bosToken:\n\(result)")

        #expect(result.hasPrefix("<bos>"), "Output should start with <bos> token")
    }

    // MARK: - 7. Multi-turn Conversation

    @Test("Multi-turn conversation preserves turn structure")
    func multiTurn() async throws {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Hi"],
            ["role": "assistant", "content": "Hello! How can I help?"],
            ["role": "user", "content": "What is your name?"],
        ]
        let result = try await render(messages: messages)
        print("[Gemma4 Template] multiTurn:\n\(result)")

        // Both user turns should be present
        #expect(result.contains("<|turn>user\nHi<turn|>"))
        #expect(result.contains("<|turn>user\nWhat is your name?<turn|>"))

        // Assistant turn should use model role
        #expect(result.contains("<|turn>model\n"))

        // Should still end with the thinking prefill
        #expect(
            result.contains("<|channel>thought\n<channel|>"),
            "Multi-turn should still include thinking prefill at the end"
        )
    }
}
