# MLX Server

A local OpenAI-compatible inference server built in Swift on Apple Silicon. Serves any MLX model with tool calling, streaming, and prompt caching — designed for coding agents like Hermes, opencode, Aider, and Cline.

## Quick Start

```bash
# Build
swift build --product MLXServer -c release

# Serve a local model
.build/release/MLXServer --model ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --port 8080

# Or serve a HuggingFace model (downloads automatically)
.build/release/MLXServer --model mlx-community/gemma-4-e2b-it-4bit --port 8080
```

The server is now running at `http://127.0.0.1:8080`. Point any OpenAI-compatible client at it.

## What It Does

You run a model locally on your Mac. Coding agents connect to it like they would connect to OpenAI or Anthropic — same API, same streaming format, same tool calling. No cloud, no API keys, no latency to a datacenter.

### For Users

- **Drop-in replacement** for OpenAI API in any coding agent
- **Works offline** — no internet needed after model download
- **Private** — your code never leaves your machine
- **Fast** — prompt caching means the second message in a conversation is 41x faster than the first

### For Developers

- OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming)
- Tool calling with automatic XML-to-JSON conversion (Qwen3 format)
- Multi-session KV cache with longest-common-prefix matching
- macOS memory pressure handling
- Zero dependencies beyond mlx-swift-lm itself

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming or non-streaming) |
| `/v1/models` | GET | List available models |
| `/metrics` | GET | Cache hit rates, throughput stats |
| `/health` | GET | Server health check |

## Configuration

### Command Line Options

```
--model <path-or-id>    Model path or HuggingFace ID (required)
--port <number>         Port to listen on (default: 8080)
--host <address>        Host to bind to (default: 127.0.0.1)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NATIVE_PREFILL` | Set to `0` to disable native C++ prefill bridge |

## Agent Setup

### Hermes

Edit `~/.hermes/config.yaml`:
```yaml
model:
  default: qwen3-coder-30b
  provider: custom
  base_url: http://localhost:8080/v1
```

### opencode

Edit `~/.config/opencode/opencode.json`:
```json
{
  "provider": {
    "local-mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local MLX",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "your-model-id": {
          "name": "Your Model Name"
        }
      }
    }
  },
  "model": "local-mlx/your-model-id"
}
```

### Any OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

## How It Works

### Prompt Caching

Coding agents send the same system prompt and tool definitions on every request — typically 10-25K tokens that never change. Without caching, this means 10-15 seconds of redundant processing per turn.

The server caches the KV (key-value) state from previous requests. When a new request arrives with the same prefix, it reuses the cached state and only processes the new tokens.

**What this means in practice:**

| | Without cache | With cache |
|---|---|---|
| First message | 16 seconds | 16 seconds |
| Every message after | 16 seconds | 0.4 seconds |

The cache finds the longest matching prefix across all active sessions using byte-level token comparison. It automatically handles:

- Same agent, next turn (extends the conversation)
- Different agent (creates a new cached session)
- Returning to a previous agent (finds its cached session via KV deep copy)

### Multi-Session Support

The server maintains multiple cached sessions simultaneously — one per agent or conversation. The number of sessions is auto-configured based on your Mac's RAM:

| Mac | RAM | Max Sessions |
|-----|-----|-------------|
| M4 Pro | 24 GB | 2 |
| M4 Max | 64 GB | 10 |
| M4 Max | 128 GB | 10 |

When you switch between Hermes and opencode, both agents keep their cached state. Neither has to re-process their system prompt when you switch back.

### Tool Calling

The server handles tool calling for models that use XML format (like Qwen3-Coder). When the model generates:

```xml
<tool_call>
<function=terminal>
<parameter=command>
ls -la
</parameter>
</function>
</tool_call>
```

The server converts this to standard OpenAI format:

```json
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "terminal",
      "arguments": "{\"command\": \"ls -la\"}"
    }
  }]
}
```

This happens automatically — agents see standard OpenAI tool calls regardless of the model's native format.

### Memory Management

The server monitors macOS memory pressure:

- **Normal**: all sessions cached, maximum performance
- **Warning**: evicts idle sessions, keeps the most recent one
- **Critical**: flushes all cached sessions to free memory

This means the server won't crash your Mac if you're running other memory-intensive apps. It backs off gracefully and re-caches on the next request.

### Native Prefill Bridge

For Gemma 4 models, a native C++ prefill bridge bypasses Swift's MLX binding overhead, achieving 2.4x faster prefill:

| Context | Prefill Speed |
|---------|--------------|
| 512 tokens | 15,000 tok/s |
| 1024 tokens | 20,000 tok/s |
| 2048 tokens | 21,400 tok/s |

This is enabled by default. Disable with `NATIVE_PREFILL=0` if needed.

## Monitoring

### /metrics Endpoint

```bash
curl http://127.0.0.1:8080/metrics | python3 -m json.tool
```

```json
{
  "cache": {
    "requests": 30,
    "hits": 29,
    "misses": 1,
    "hit_rate": 0.967,
    "evictions": 0,
    "sessions_active": 3,
    "sessions_max": 10
  },
  "throughput": {
    "total_prefill_tokens": 54491,
    "total_reused_tokens": 38305,
    "avg_prefill_tokens_per_request": 1816
  }
}
```

**What to look for:**

- `hit_rate` close to 1.0 means caching is working well
- `sessions_active` tells you how many conversations are cached
- `avg_prefill_tokens_per_request` should be low after warmup (means prefix reuse is working)
- `evictions` > 0 means you're running out of cache slots (increase RAM or reduce concurrent agents)

### Request Logs

The server logs every request to stderr:

```
[MLXServer] cache=hit prefix=14832/15200 new=368 prefill=368 stream=true tools=29
[MLXServer] POST /v1/chat/completions completed in 892ms
```

## Architecture

```
Client Request
    |
    v
HTTP Parser (raw sockets, no framework dependencies)
    |
    v
Chat Template (tokenizer.applyChatTemplate with tools)
    |
    v
ServerPromptCache (actor, thread-safe)
    |-- LCP match across cached sessions
    |-- KV deep copy for session isolation
    |-- Auto-configured session count from RAM
    |-- macOS memory pressure eviction
    |
    v
generate() (MLXLMCommon async stream)
    |
    v
Server-side Tool Call Parser
    |-- Buffers text chunks
    |-- Detects <function= or <tool_call> patterns
    |-- Converts XML to OpenAI JSON format
    |
    v
SSE Streaming Response
    |-- Keepalive comments during long prefill
    |-- Usage stats in final chunk
    |-- Graceful error handling
```

## Testing

```bash
# Start server
.build/release/MLXServer --model <your-model> --port 8080

# Run test suite (26 tests)
python3 Sources/MLXServer/test_cache.py
```

Tests cover:
- Cache hits/misses with different system prompts
- Session interleaving (switching between agents)
- Conversation growth (multi-turn)
- LRU eviction at capacity
- Tool call parsing
- Usage stats in responses
- Error handling (malformed requests, oversized bodies)
- Server crash resilience
- Metrics endpoint
- Auto-configured session count

## Limitations

- **Single model**: serves one model at a time (restart to switch)
- **macOS only**: uses Apple Silicon unified memory, Metal GPU
- **Tool format**: currently handles Qwen3 XML format; other formats pass through to the generate() pipeline's built-in parser
- **No authentication**: designed for local use only, no API key validation
- **Model quality**: tool calling reliability depends on the model — some models (like Qwen3-Coder) occasionally answer directly instead of calling tools with large system prompts
