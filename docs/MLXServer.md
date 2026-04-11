# MLX Server

A local OpenAI-compatible inference server built in Swift on Apple Silicon. Serves any MLX model with tool calling, streaming, and prompt caching — designed for coding agents like Hermes, opencode, Aider, and Cline.

## Prerequisites

- **macOS** on Apple Silicon (M1 or later)
- **Xcode Command Line Tools**: `xcode-select --install`
- **An MLX model** — either downloaded locally or a HuggingFace model ID

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ekryski/mlx-swift-lm.git
cd mlx-swift-lm
```

If you're using a local mlx-swift checkout (for development):
```bash
export MLX_SWIFT_PATH=/path/to/your/mlx-swift
```

### 2. Build the metallib (required once)

MLX needs compiled Metal shaders. Run this once after cloning:

```bash
./scripts/build-metallib.sh
```

This creates `.build/arm64-apple-macosx/release/mlx.metallib`. If it fails, make sure `MLX_SWIFT_PATH` is set or run `swift package resolve` first.

### 3. Build the server

```bash
swift build --product MLXServer -c release
```

First build takes a few minutes (compiles MLX, transformers, etc). Subsequent builds are fast.

### 4. Get a model

You have two options:

**Option A: Download an MLX model manually** (recommended for large models)

Use `huggingface-cli` or `mlx_lm` to download:
```bash
# Install huggingface-cli if needed
pip3 install huggingface_hub

# Download a model
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --local-dir ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
```

Or with mlx_lm (also installs Python MLX):
```bash
pip3 install mlx-lm
python3 -c "from mlx_lm.utils import load; load('mlx-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit')"
```

The model will be cached in `~/.cache/huggingface/hub/`.

**Option B: Let the server download it** (convenient for small models)

Just pass the HuggingFace model ID — the server downloads it on first launch:
```bash
.build/release/MLXServer --model mlx-community/gemma-4-e2b-it-4bit --port 8080
```

### 5. Start the server

```bash
# With a locally downloaded model
.build/release/MLXServer --model ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --port 8080

# Or with a HuggingFace ID
.build/release/MLXServer --model mlx-community/gemma-4-e2b-it-4bit --port 8080
```

You should see:
```
[MLXServer] Loading model: ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
[MLXServer] Auto-configured: 10 max cached sessions (128GB RAM)
[MLXServer] Listening on http://127.0.0.1:8080
[MLXServer] Endpoints: GET /v1/models, POST /v1/chat/completions, GET /metrics
```

### 6. Verify it works

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20}'
```

### Choosing a model

| Model | Size | RAM needed | Good for |
|-------|------|-----------|----------|
| `mlx-community/gemma-4-e2b-it-4bit` | ~3 GB | 8 GB+ | Testing, small tasks |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit` | ~18 GB | 32 GB+ | Coding agents with tool calling |
| `mlx-community/Qwen3.5-27B-8bit` | ~28 GB | 48 GB+ | General purpose |

The model must fit in your Mac's unified memory. Check with `system_profiler SPHardwareDataType | grep Memory`.

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

1. Install Hermes if you haven't: [hermes-agent.com](https://hermes-agent.com)

2. Edit `~/.hermes/config.yaml` — set the model and provider:
```yaml
model:
  default: qwen3-coder-30b       # any name you want
  provider: custom
  base_url: http://localhost:8080/v1
```

3. Also set the auxiliary models (Hermes uses these for routing, compression, etc):
```yaml
smart_model_routing:
  cheap_model:
    provider: custom
    model: qwen3-coder-30b
    base_url: http://localhost:8080/v1
    api_key: local

auxiliary:
  vision:
    provider: custom
    model: qwen3-coder-30b
    base_url: http://localhost:8080/v1
    api_key: local
  compression:
    provider: custom
    model: qwen3-coder-30b
    base_url: http://localhost:8080/v1
    api_key: local
```

4. Start the server, then run Hermes:
```bash
# Terminal 1
.build/release/MLXServer --model ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --port 8080

# Terminal 2
hermes
```

### opencode

1. Install opencode if you haven't: [opencode.ai](https://opencode.ai)

2. Edit `~/.config/opencode/opencode.json`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local MLX",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "/Users/yourname/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit": {
          "name": "Qwen3 Coder 30B (local)"
        }
      }
    }
  },
  "model": "local-mlx//Users/yourname/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit"
}
```

> **Note:** The model ID in `models` and `model` must match what the server reports. Use the full path for local models, or the HuggingFace ID for downloaded models.

3. Verify the model shows up:
```bash
opencode models
# Should show: local-mlx//Users/yourname/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
```

4. Start the server, then run opencode:
```bash
# Terminal 1
.build/release/MLXServer --model ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --port 8080

# Terminal 2
opencode
```

### Aider

```bash
aider --openai-api-base http://127.0.0.1:8080/v1 --openai-api-key unused --model any-name
```

### Any OpenAI-compatible client (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="any",  # server ignores this, uses loaded model
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Running multiple agents simultaneously

Both agents connect to the same server on port 8080. The multi-session cache handles them independently — each agent gets its own cached KV state.

```bash
# Terminal 1: server
.build/release/MLXServer --model ~/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit --port 8080

# Terminal 2: hermes
hermes

# Terminal 3: opencode
opencode
```

Switch between them freely. The server caches each agent's system prompt separately, so switching back is fast (no re-processing).

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
