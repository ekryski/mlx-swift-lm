// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        // Find the JSON content between tags
        var text = content

        // Strip tags if present
        if let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if let data = jsonStr.data(using: .utf8),
            let function = try? JSONDecoder().decode(ToolCall.Function.self, from: data)
        {
            return ToolCall(function: function)
        }

        // Fallback: Qwen3.5 models wrap XML function format inside <tool_call> tags:
        //   <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        // Try parsing as XML function format when JSON fails.
        let xmlParser = XMLFunctionParser()
        return xmlParser.parse(content: text, tools: tools)
    }
}
