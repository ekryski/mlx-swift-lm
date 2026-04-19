# MLXServer

OpenAI-compatible inference server for [mlx-swift-lm](../../). Drop-in replacement for llama-server and mlx_lm.server — works with OpenCode, Hermes, Claude Code, Continue, and any OpenAI-compatible client.

## Quick Start

```bash
# Build
make metal  # compile Metal shaders (first time only)
swift build -c release --product MLXServer

# Run with a HuggingFace model
.build/arm64-apple-macosx/release/MLXServer --model mlx-community/Qwen3.5-35B-A3B-4bit

# Run with a local model directory
.build/arm64-apple-macosx/release/MLXServer --model /path/to/model --port 8080

# Test
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## Usage

```
MLXServer --model <model> [options]

OPTIONS:
  -m, --model <path|id>    Model path or HuggingFace ID (required)
  --port <port>            Listen port (default: 8080)
  --slots <n>              Parallel inference slots (default: 4)
  --kv <scheme>            KV cache scheme (e.g. turbo4v2)
  --kv-bits <n>            KV cache quantization bits
  --kv-start <n>           Layer to start KV quantization
  -c, --ctx-size <n>       Context size limit
  -n, --n-predict <n>      Default max generation tokens
  --reasoning <on|off>     Enable/disable thinking (default: on)
  -h, --help               Show help
```

### Sampling Parameters (per-request)

All standard OpenAI sampling parameters are supported in the request body:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.6 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | int | — | Top-K filtering |
| `min_p` | float | — | Minimum probability threshold |
| `frequency_penalty` | float | — | Penalize frequent tokens |
| `presence_penalty` | float | — | Penalize repeated tokens |
| `repeat_penalty` | float | 1.1* | Repetition penalty (auto-applied if no other penalty set) |
| `repeat_last_n` | int | 64 | Context window for repetition penalty |
| `max_tokens` | int | — | Max generation tokens |
| `max_completion_tokens` | int | — | Alias for max_tokens (OpenAI) |
| `seed` | int | — | RNG seed for reproducibility |
| `stream` | bool | false | Enable SSE streaming |

## API Endpoints

### Health

```
GET /health
GET /
```

Returns `{"status":"ok"}`.

### Models

```
GET /v1/models
```

Returns the loaded model in OpenAI list format.

### Chat Completions (OpenAI)

```
POST /v1/chat/completions
```

OpenAI-compatible chat completion. Supports streaming (SSE), tool calls, and thinking/reasoning blocks.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "max_tokens": 1000,
  "temperature": 0.7,
  "tools": [...]
}
```

**Streaming response:** Server-Sent Events with `data: {chunk}\n\n` format, terminated by `data: [DONE]\n\n`. Connection is closed after streaming completes.

**Non-streaming response:** Standard OpenAI JSON with `choices[0].message.content`.

### Messages (Anthropic)

```
POST /v1/messages
```

Anthropic Messages API compatible. Supports streaming (SSE with `event:` + `data:` pairs), tool calls, and thinking/reasoning blocks. Works with Claude Code, Anthropic SDKs, and any Anthropic-compatible client.

**Request:** Uses top-level `system` field, `max_tokens` required, content blocks array format.

**Streaming:** Uses `event: message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop` event types.

**Usage with Claude Code:**
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

### Text Completions

```
POST /v1/completions
```

Raw text completion (non-chat). Supports streaming.

### Tokenize

```
POST /tokenize
POST /v1/tokenize
```

**Request:** `{"prompt": "Hello world", "add_special_tokens": true}`
**Response:** `{"tokens": [9906, 1917]}`

### Tokenizer Info

```
GET /tokenizer_info
GET /v1/tokenizer_info
```

Returns EOS/BOS token info.

### Metrics

```
GET /metrics
```

Returns cache hit rates, throughput stats, and slot utilization.

### Slots

```
GET /slots
```

Returns current inference slot states (idle/prefilling/generating).

## Features

### Prompt Caching

Multi-session KV cache with longest-common-prefix (LCP) matching. Subsequent requests that share a prefix with previous requests skip redundant prefill. Sessions are automatically evicted under memory pressure (macOS memory pressure events).

### Parallel Inference

Multiple inference slots allow concurrent requests. Prefill is serialized (one at a time) to avoid GPU contention, but decode runs concurrently across slots.

### Tool Calls

Supports tool call detection and parsing from model output in multiple formats:
- OpenAI XML: `<function=name><parameter=key>value</parameter></function>`
- MiniMax XML: `<invoke name="..."><parameter name="...">value</parameter></invoke>`
- Native `ToolCallFormat.xmlFunction`

### Thinking/Reasoning

`<think>...</think>` blocks from reasoning models (Qwen3.x, etc.) are automatically stripped from `content` and exposed as `reasoning_content` in both streaming and non-streaming responses.

### CORS

Full CORS support with preflight (OPTIONS) handling. `Access-Control-Allow-Origin` echoes the request's `Origin` header, or defaults to `*`.

## Testing

```bash
# Run the test suite against a running server
python3 tools/server/tests/test-server.py --url http://127.0.0.1:8080

# Quick mode (skip slow tool/think tests)
python3 tools/server/tests/test-server.py --quick

# Verbose
python3 tools/server/tests/test-server.py -v
```

## Architecture

```
tools/server/
├── MLXServerApp.swift  # Entry point, CLI arg parsing, model loading
├── ServerTypes.swift   # OpenAI/Anthropic request/response Codable types
├── ServerHTTP.swift    # HTTP server, routing, response/SSE helpers
├── ServerChat.swift    # Chat completions + text completions handlers (OpenAI)
├── ServerMessages.swift # Anthropic Messages API handler (/v1/messages)
├── ServerCache.swift   # Multi-session KV cache with LCP matching
├── ServerSlots.swift   # Parallel inference slot manager
├── ServerTools.swift   # Tool call XML parsing (auto-detected per model)
├── README.md
└── tests/
    └── test-server.py  # API compatibility test suite (78 tests)
```

## Client Configuration

### OpenCode

```json
{
  "provider": "openai",
  "model": "local-model",
  "baseURL": "http://127.0.0.1:8080/v1"
}
```

### Hermes / Continue

Point the OpenAI-compatible endpoint to `http://127.0.0.1:8080/v1`.

### curl

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true,
    "max_tokens": 100
  }'
```
