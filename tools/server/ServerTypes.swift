import Foundation

func log(_ msg: String) {
    FileHandle.standardError.write(Data("[MLXServer] \(msg)\n".utf8))
}

// MARK: - OpenAI Request Types

struct ChatMessage: Codable {
    let role: String
    let content: String?

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        if let str = try? container.decode(String.self, forKey: .content) {
            content = str
        } else {
            content = nil
        }
    }

    enum CodingKeys: String, CodingKey { case role, content }
}

struct StreamOptions: Codable {
    let include_usage: Bool?
}

struct ChatRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let temperature: Float?
    let stream: Bool?
    let stream_options: StreamOptions?
    // Sampling
    let top_p: Float?
    let top_k: Int?
    let min_p: Float?
    let seed: Int?
    let frequency_penalty: Float?
    let presence_penalty: Float?
    let repeat_penalty: Float?
    let repeat_last_n: Int?
    // Tools
    let tools: AnyCodable?
    let tool_choice: AnyCodable?
    // Other
    let stop: AnyCodable?
    let n: Int?
    let logprobs: Bool?
    let top_logprobs: Int?
    let response_format: AnyCodable?

    var effectiveMaxTokens: Int? { max_completion_tokens ?? max_tokens }

    enum CodingKeys: String, CodingKey {
        case model, messages, max_tokens, max_completion_tokens, temperature, stream, stream_options
        case top_p, top_k, min_p, seed, frequency_penalty, presence_penalty
        case repeat_penalty, repeat_last_n
        case tools, tool_choice, stop, n, logprobs, top_logprobs, response_format
    }
}

// MARK: - OpenAI Response Types

struct ChatResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let system_fingerprint: String
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
        let reasoning_content: String?

        init(role: String? = nil, content: String? = nil, reasoning_content: String? = nil) {
            self.role = role
            self.content = content
            self.reasoning_content = reasoning_content
        }

        func encode(to encoder: Swift.Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(role, forKey: .role)
            try container.encodeIfPresent(content, forKey: .content)
            try container.encodeIfPresent(reasoning_content, forKey: .reasoning_content)
        }

        enum CodingKeys: String, CodingKey {
            case role, content, reasoning_content
        }
    }

    struct Usage: Codable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
    }
}

// MARK: - Generic JSON Codable

/// Wraps any JSON value so we can accept arbitrary fields without failing to decode.
struct AnyCodable: Codable {
    let value: Any?
    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([JSONValue].self) {
            value = arr.map { $0.toAny() }
        } else if let dict = try? container.decode([String: JSONValue].self) {
            value = dict.mapValues { $0.toAny() }
        } else {
            value = nil
        }
    }
    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encodeNil()
    }
}

/// Recursive JSON value for preserving tools structure.
enum JSONValue: Codable {
    case string(String), int(Int), double(Double), bool(Bool), null
    case array([JSONValue]), object([String: JSONValue])

    init(from decoder: Swift.Decoder) throws {
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

    func encode(to encoder: Swift.Encoder) throws {
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
