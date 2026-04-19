#!/usr/bin/env python3
"""
MLXServer OpenAI API compatibility test suite.

Tests that MLXServer behaves like llama-server / mlx-lm server / OpenAI API.
Covers: streaming SSE, non-streaming, tool calls, think blocks, keep-alive,
        CORS, endpoints, error handling.

Usage:
    python3 scripts/test-server.py [--url http://127.0.0.1:8080] [--verbose]
"""

import argparse
import json
import http.client
import sys
import time
import urllib.parse
from typing import Optional

# ── Helpers ──────────────────────────────────────────────────────────────

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

passed = 0
failed = 0
skipped = 0
verbose = False

def log(msg):
    if verbose:
        print(f"  {Colors.CYAN}→{Colors.RESET} {msg}")

def ok(name):
    global passed
    passed += 1
    print(f"  {Colors.GREEN}✓{Colors.RESET} {name}")

def fail(name, reason=""):
    global failed
    failed += 1
    detail = f" — {reason}" if reason else ""
    print(f"  {Colors.RED}✗{Colors.RESET} {name}{detail}")

def skip(name, reason=""):
    global skipped
    skipped += 1
    detail = f" — {reason}" if reason else ""
    print(f"  {Colors.YELLOW}⊘{Colors.RESET} {name}{detail}")

def request(base: str, method: str, path: str, body: Optional[dict] = None,
            headers: Optional[dict] = None, stream: bool = False,
            timeout: int = 60) -> tuple:
    """Raw HTTP request. Returns (status, headers_dict, body_str, raw_conn).
    If stream=True, returns the connection for manual reading."""
    parsed = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    payload = json.dumps(body) if body else None
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    resp_headers = dict(resp.getheaders())
    if stream:
        return resp.status, resp_headers, resp, conn
    data = resp.read().decode("utf-8")
    conn.close()
    return resp.status, resp_headers, data, None


def parse_sse_events(resp) -> list:
    """Read SSE events from an HTTP response object."""
    events = []
    buf = ""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buf:
            event_str, buf = buf.split("\n\n", 1)
            for line in event_str.strip().split("\n"):
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        events.append({"done": True})
                        return events
                    try:
                        events.append(json.loads(data))
                    except json.JSONDecodeError:
                        events.append({"raw": data, "parse_error": True})
    return events


# ── Test Categories ──────────────────────────────────────────────────────

def test_health(base):
    print(f"\n{Colors.BOLD}Health & Discovery{Colors.RESET}")

    # GET /health
    status, _, body, _ = request(base, "GET", "/health")
    if status == 200:
        data = json.loads(body)
        if "status" in data:
            ok("GET /health returns status")
        else:
            fail("GET /health", "missing 'status' field")
    else:
        fail("GET /health", f"status={status}")

    # GET /
    status, _, body, _ = request(base, "GET", "/")
    if status == 200:
        ok("GET / returns 200")
    else:
        fail("GET /", f"status={status}")

    # GET /v1/models
    status, _, body, _ = request(base, "GET", "/v1/models")
    if status == 200:
        data = json.loads(body)
        if data.get("object") == "list" and "data" in data and len(data["data"]) > 0:
            model = data["data"][0]
            checks = ["id", "object", "created", "owned_by"]
            missing = [k for k in checks if k not in model]
            if missing:
                fail("GET /v1/models schema", f"missing: {missing}")
            else:
                ok("GET /v1/models — correct schema")
        else:
            fail("GET /v1/models", "bad structure")
    else:
        fail("GET /v1/models", f"status={status}")

    # GET /metrics
    status, _, body, _ = request(base, "GET", "/metrics")
    if status == 200:
        data = json.loads(body)
        if "cache" in data and "throughput" in data:
            ok("GET /metrics — has cache + throughput")
        else:
            fail("GET /metrics", "missing cache or throughput")
    else:
        fail("GET /metrics", f"status={status}")

    # GET /slots
    status, _, body, _ = request(base, "GET", "/slots")
    if status == 200:
        data = json.loads(body)
        if "slots" in data:
            ok("GET /slots — has slots array")
        else:
            fail("GET /slots", "missing slots")
    else:
        fail("GET /slots", f"status={status}")


def test_cors(base):
    print(f"\n{Colors.BOLD}CORS{Colors.RESET}")

    # OPTIONS preflight
    status, hdrs, _, _ = request(base, "OPTIONS", "/v1/chat/completions",
                                  headers={"Origin": "http://localhost:3000"})
    if status == 204:
        ok("OPTIONS returns 204")
    else:
        fail("OPTIONS preflight", f"status={status}")

    acao = hdrs.get("Access-Control-Allow-Origin", hdrs.get("access-control-allow-origin", ""))
    if acao:
        ok(f"CORS Allow-Origin present: {acao}")
    else:
        fail("CORS Allow-Origin missing")

    # Regular request with Origin
    status, hdrs, _, _ = request(base, "GET", "/health",
                                  headers={"Origin": "http://myapp.com"})
    acao = hdrs.get("Access-Control-Allow-Origin", hdrs.get("access-control-allow-origin", ""))
    if acao:
        ok(f"CORS on regular request: {acao}")
    else:
        fail("CORS missing on regular GET")


def test_chat_nonstreaming(base):
    print(f"\n{Colors.BOLD}Chat Completions (non-streaming){Colors.RESET}")

    body = {
        "messages": [{"role": "user", "content": "Say exactly: test123"}],
        "stream": False,
        "max_tokens": 20,
        "temperature": 0.0
    }
    status, hdrs, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body, timeout=60)

    if status != 200:
        fail("non-streaming chat", f"status={status} body={resp_body[:200]}")
        return

    data = json.loads(resp_body)
    log(f"Response: {json.dumps(data, indent=2)[:500]}")

    # Required top-level fields (OpenAI spec)
    required = ["id", "object", "created", "model", "choices", "usage"]
    missing = [k for k in required if k not in data]
    if missing:
        fail("non-streaming schema", f"missing: {missing}")
    else:
        ok("non-streaming — all required fields present")

    # Object type
    if data.get("object") == "chat.completion":
        ok("object = 'chat.completion'")
    else:
        fail("object type", f"got '{data.get('object')}'")

    # ID prefix
    if data.get("id", "").startswith("chatcmpl-"):
        ok("id starts with 'chatcmpl-'")
    else:
        fail("id prefix", f"got '{data.get('id')}'")

    # Choices structure
    if data.get("choices"):
        choice = data["choices"][0]
        if "message" in choice and "finish_reason" in choice:
            ok("choice has message + finish_reason")
        else:
            fail("choice structure", f"keys: {list(choice.keys())}")

        msg = choice.get("message", {})
        if msg.get("role") == "assistant":
            ok("message.role = 'assistant'")
        else:
            fail("message.role", f"got '{msg.get('role')}'")

        if "content" in msg:
            ok(f"message.content present (len={len(msg['content'])})")
        else:
            fail("message.content missing")

        if choice.get("finish_reason") in ("stop", "length"):
            ok(f"finish_reason = '{choice['finish_reason']}'")
        else:
            fail("finish_reason", f"got '{choice.get('finish_reason')}'")
    else:
        fail("no choices in response")

    # Usage
    usage = data.get("usage", {})
    usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    missing_usage = [k for k in usage_fields if k not in usage]
    if missing_usage:
        fail("usage fields", f"missing: {missing_usage}")
    else:
        if usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]:
            ok("usage.total = prompt + completion")
        else:
            fail("usage math", f"{usage['total_tokens']} != {usage['prompt_tokens']} + {usage['completion_tokens']}")

    # Connection header — non-streaming should support keep-alive
    conn_header = hdrs.get("Connection", hdrs.get("connection", ""))
    log(f"Connection header: {conn_header}")


def test_chat_streaming(base):
    print(f"\n{Colors.BOLD}Chat Completions (streaming){Colors.RESET}")

    body = {
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "stream": True,
        "max_tokens": 50,
        "temperature": 0.0
    }
    status, hdrs, resp, conn = request(base, "POST", "/v1/chat/completions",
                                        body=body, stream=True, timeout=60)

    if status != 200:
        fail("streaming chat", f"status={status}")
        if conn:
            conn.close()
        return

    # Check headers
    ct = hdrs.get("Content-Type", hdrs.get("content-type", ""))
    if "text/event-stream" in ct:
        ok("Content-Type = text/event-stream")
    else:
        fail("Content-Type", f"got '{ct}'")

    conn_header = hdrs.get("Connection", hdrs.get("connection", ""))
    if conn_header.lower() == "close":
        ok("Connection: close for SSE")
    else:
        fail("Connection header for SSE", f"got '{conn_header}' (should be 'close')")

    # Parse SSE events
    events = parse_sse_events(resp)
    if conn:
        conn.close()

    if not events:
        fail("no SSE events received")
        return

    log(f"Got {len(events)} SSE events")

    # First event should have role
    first = events[0]
    if isinstance(first, dict) and not first.get("done"):
        delta = first.get("choices", [{}])[0].get("delta", {})
        if delta.get("role") == "assistant":
            ok("first chunk has role='assistant'")
        else:
            fail("first chunk role", f"delta={delta}")

        # Check required fields on chunk
        chunk_required = ["id", "object", "created", "model", "choices"]
        chunk_missing = [k for k in chunk_required if k not in first]
        if chunk_missing:
            fail("chunk schema", f"missing: {chunk_missing}")
        else:
            ok("chunk has all required fields")

        if first.get("object") == "chat.completion.chunk":
            ok("object = 'chat.completion.chunk'")
        else:
            fail("chunk object type", f"got '{first.get('object')}'")
    else:
        fail("first event not a valid chunk")

    # Content chunks should have delta.content
    content_chunks = []
    for e in events:
        if isinstance(e, dict) and not e.get("done"):
            delta = e.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                content_chunks.append(delta["content"])

    if content_chunks:
        full_text = "".join(content_chunks)
        ok(f"got {len(content_chunks)} content chunks, text='{full_text[:80]}...'")
    else:
        fail("no content chunks received")

    # Last non-DONE event should have finish_reason
    last_data = None
    for e in reversed(events):
        if isinstance(e, dict) and not e.get("done"):
            last_data = e
            break

    if last_data:
        fr = last_data.get("choices", [{}])[0].get("finish_reason")
        if fr in ("stop", "length"):
            ok(f"final chunk finish_reason='{fr}'")
        else:
            fail("final chunk finish_reason", f"got '{fr}'")

        # Usage in final chunk (OpenAI includes this)
        if "usage" in last_data:
            ok("usage present in final chunk")
        else:
            fail("usage missing from final chunk (OpenAI includes it)")
    else:
        fail("no final data chunk found")

    # [DONE] sentinel
    if events and events[-1].get("done"):
        ok("stream ends with data: [DONE]")
    else:
        fail("missing [DONE] sentinel")


def test_streaming_connection_close(base):
    """Critical test: after streaming completes, connection must close promptly."""
    print(f"\n{Colors.BOLD}Streaming Connection Lifecycle{Colors.RESET}")

    body = {
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": True,
        "max_tokens": 10,
        "temperature": 0.0
    }
    status, hdrs, resp, conn = request(base, "POST", "/v1/chat/completions",
                                        body=body, stream=True, timeout=30)
    if status != 200:
        fail("streaming request failed", f"status={status}")
        if conn:
            conn.close()
        return

    events = parse_sse_events(resp)

    # After [DONE], the server should close the connection.
    # Try reading more — should get empty (connection closed).
    start = time.time()
    try:
        extra = resp.read(1)  # should return b'' immediately if closed
        elapsed = time.time() - start
        if not extra and elapsed < 2.0:
            ok(f"connection closed promptly after [DONE] ({elapsed:.1f}s)")
        elif not extra:
            fail("connection closed but slowly", f"{elapsed:.1f}s")
        else:
            fail("server sent data after [DONE]", f"got {extra!r}")
    except Exception as e:
        elapsed = time.time() - start
        if elapsed < 2.0:
            ok(f"connection closed after [DONE] ({elapsed:.1f}s)")
        else:
            fail("connection close timeout", f"{elapsed:.1f}s: {e}")
    finally:
        if conn:
            conn.close()


def test_sequential_requests(base):
    """Critical test: multiple requests in sequence must all succeed."""
    print(f"\n{Colors.BOLD}Sequential Requests (regression){Colors.RESET}")

    for i in range(3):
        body = {
            "messages": [{"role": "user", "content": f"What is {i+1}+{i+1}?"}],
            "stream": True,
            "max_tokens": 20,
            "temperature": 0.0
        }
        try:
            status, _, resp, conn = request(base, "POST", "/v1/chat/completions",
                                             body=body, stream=True, timeout=30)
            if status != 200:
                fail(f"sequential request #{i+1}", f"status={status}")
                if conn:
                    conn.close()
                continue

            events = parse_sse_events(resp)
            if conn:
                conn.close()

            content = ""
            for e in events:
                if isinstance(e, dict) and not e.get("done"):
                    delta = e.get("choices", [{}])[0].get("delta", {})
                    content += delta.get("content", "") or ""

            has_done = any(e.get("done") for e in events if isinstance(e, dict))
            if content and has_done:
                ok(f"request #{i+1} completed: '{content[:40]}...'")
            else:
                fail(f"request #{i+1}", f"content={bool(content)} done={has_done}")
        except Exception as e:
            fail(f"request #{i+1}", str(e))


def test_keep_alive_nonstreaming(base):
    """Non-streaming requests on same connection via keep-alive."""
    print(f"\n{Colors.BOLD}Keep-Alive (non-streaming){Colors.RESET}")

    parsed = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)

    for i in range(3):
        body = json.dumps({
            "messages": [{"role": "user", "content": f"Say '{i}'"}],
            "stream": False,
            "max_tokens": 10,
            "temperature": 0.0
        })
        try:
            conn.request("POST", "/v1/chat/completions", body=body,
                        headers={"Content-Type": "application/json", "Connection": "keep-alive"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            if resp.status == 200:
                parsed_data = json.loads(data)
                content = parsed_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                ok(f"keep-alive request #{i+1}: '{content[:30]}...'")
            else:
                fail(f"keep-alive #{i+1}", f"status={resp.status}")
        except Exception as e:
            fail(f"keep-alive #{i+1}", str(e))
            # Reconnect
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)

    conn.close()


def test_completions(base):
    """Test /v1/completions (text completion, not chat)."""
    print(f"\n{Colors.BOLD}Text Completions (/v1/completions){Colors.RESET}")

    # Non-streaming
    body = {"prompt": "The capital of France is", "max_tokens": 10, "temperature": 0.0, "stream": False}
    status, _, resp_body, _ = request(base, "POST", "/v1/completions", body=body, timeout=30)
    if status == 200:
        data = json.loads(resp_body)
        if data.get("object") == "text_completion" and data.get("choices"):
            choice = data["choices"][0]
            if "text" in choice and "finish_reason" in choice:
                ok(f"text completion: '{choice['text'][:40]}...'")
            else:
                fail("text completion choice", f"keys: {list(choice.keys())}")
        else:
            fail("text completion schema", f"object={data.get('object')}")
    else:
        fail("text completion", f"status={status}")

    # Streaming
    body["stream"] = True
    status, hdrs, resp, conn = request(base, "POST", "/v1/completions",
                                        body=body, stream=True, timeout=30)
    if status == 200:
        events = parse_sse_events(resp)
        if conn:
            conn.close()
        texts = []
        for e in events:
            if isinstance(e, dict) and not e.get("done"):
                text = e.get("choices", [{}])[0].get("text", "")
                if text:
                    texts.append(text)
        has_done = any(e.get("done") for e in events if isinstance(e, dict))
        if texts and has_done:
            ok(f"streaming text completion: {''.join(texts)[:40]}...")
        else:
            fail("streaming text completion", f"texts={len(texts)} done={has_done}")
    else:
        fail("streaming text completion", f"status={status}")
        if conn:
            conn.close()


def test_think_blocks(base):
    """Test that <think>...</think> blocks are handled correctly."""
    print(f"\n{Colors.BOLD}Think/Reasoning Blocks{Colors.RESET}")

    # Qwen3.x models with thinking enabled may produce <think> blocks.
    # The server should strip them and optionally expose as reasoning_content.
    body = {
        "messages": [
            {"role": "system", "content": "Think step by step before answering."},
            {"role": "user", "content": "What is 15 * 23?"}
        ],
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.6
    }
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body, timeout=60)
    if status == 200:
        data = json.loads(resp_body)
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content")

        # Content should NOT contain raw <think> tags
        if "<think>" in content:
            fail("think tags leaked into content", f"content starts: {content[:100]}")
        else:
            ok("content free of <think> tags")

        if reasoning:
            ok(f"reasoning_content present (len={len(reasoning)})")
        else:
            log("no reasoning_content (model may not have used thinking)")
            ok("think block test passed (no think block produced)")
    else:
        fail("think block test", f"status={status}")

    # Streaming think test
    body["stream"] = True
    status, _, resp, conn = request(base, "POST", "/v1/chat/completions",
                                     body=body, stream=True, timeout=60)
    if status == 200:
        events = parse_sse_events(resp)
        if conn:
            conn.close()

        content_text = ""
        reasoning_text = ""
        for e in events:
            if isinstance(e, dict) and not e.get("done"):
                delta = e.get("choices", [{}])[0].get("delta", {})
                content_text += delta.get("content", "") or ""
                reasoning_text += delta.get("reasoning_content", "") or ""

        if "<think>" in content_text:
            fail("streaming: think tags in content", f"content starts: {content_text[:100]}")
        else:
            ok("streaming: content free of <think> tags")

        if reasoning_text:
            ok(f"streaming: reasoning_content streamed (len={len(reasoning_text)})")
        else:
            log("streaming: no reasoning_content chunks")
            ok("streaming think test passed (no think block produced)")
    else:
        fail("streaming think test", f"status={status}")
        if conn:
            conn.close()


def test_tool_calls(base):
    """Test tool call handling in both streaming and non-streaming."""
    print(f"\n{Colors.BOLD}Tool Calls{Colors.RESET}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # Non-streaming with tools
    body = {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        "tools": tools,
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.0
    }
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body, timeout=60)
    if status == 200:
        data = json.loads(resp_body)
        msg = data.get("choices", [{}])[0].get("message", {})
        fr = data.get("choices", [{}])[0].get("finish_reason")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            ok(f"tool_calls present ({len(tool_calls)} calls)")
            tc = tool_calls[0]
            tc_checks = []
            if "id" in tc:
                tc_checks.append("id")
            if tc.get("type") == "function":
                tc_checks.append("type=function")
            if "function" in tc and "name" in tc["function"]:
                tc_checks.append(f"name={tc['function']['name']}")
            if "function" in tc and "arguments" in tc["function"]:
                tc_checks.append("arguments")
            ok(f"tool_call structure: {', '.join(tc_checks)}")

            if fr == "tool_calls":
                ok("finish_reason = 'tool_calls'")
            else:
                fail("finish_reason for tool call", f"got '{fr}' (expected 'tool_calls')")
        else:
            # Model might not have called the tool — that's OK, just note it
            skip("model didn't produce tool call", "model-dependent")
    else:
        fail("tool call request", f"status={status}")

    # Streaming with tools
    body["stream"] = True
    status, _, resp, conn = request(base, "POST", "/v1/chat/completions",
                                     body=body, stream=True, timeout=60)
    if status == 200:
        events = parse_sse_events(resp)
        if conn:
            conn.close()

        tool_call_chunks = []
        for e in events:
            if isinstance(e, dict) and not e.get("done"):
                delta = e.get("choices", [{}])[0].get("delta", {})
                if "tool_calls" in delta:
                    tool_call_chunks.append(delta["tool_calls"])

        if tool_call_chunks:
            ok(f"streaming tool_calls: {len(tool_call_chunks)} chunks")
            # Check first chunk has id, type, function.name
            first_tc = tool_call_chunks[0][0] if tool_call_chunks[0] else {}
            if "id" in first_tc:
                ok("streaming tool_call has id")
            if first_tc.get("function", {}).get("name"):
                ok(f"streaming tool_call name: {first_tc['function']['name']}")
        else:
            skip("streaming: model didn't produce tool call", "model-dependent")
    else:
        fail("streaming tool call", f"status={status}")
        if conn:
            conn.close()

    # Tool result round-trip
    body_with_result = {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_123", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_123",
             "content": "{\"temp\": 22, \"condition\": \"sunny\"}"}
        ],
        "stream": False,
        "max_tokens": 100,
        "temperature": 0.0
    }
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions",
                                       body=body_with_result, timeout=60)
    if status == 200:
        ok("tool result round-trip accepted")
    else:
        fail("tool result round-trip", f"status={status} body={resp_body[:200]}")


def test_error_handling(base):
    """Test error responses."""
    print(f"\n{Colors.BOLD}Error Handling{Colors.RESET}")

    # Invalid JSON
    parsed = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    conn.request("POST", "/v1/chat/completions", body="not json",
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8")
    if resp.status == 400:
        ok("invalid JSON → 400")
    else:
        fail("invalid JSON", f"status={resp.status}")
    conn.close()

    # Missing messages
    body = {"stream": False, "max_tokens": 10}
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body)
    if status == 400:
        ok("missing messages → 400")
    else:
        fail("missing messages", f"status={status}")

    # 404
    status, _, _, _ = request(base, "GET", "/v1/nonexistent")
    if status == 404:
        ok("unknown endpoint → 404")
    else:
        fail("404 handling", f"status={status}")

    # Error response format
    if status == 404:
        pass  # already checked
    status, _, resp_body, _ = request(base, "GET", "/v1/nonexistent")
    if status == 404:
        try:
            err = json.loads(resp_body)
            if "error" in err and "message" in err["error"]:
                ok("error response has {error: {message: ...}} format")
            else:
                fail("error format", f"got: {resp_body[:100]}")
        except json.JSONDecodeError:
            fail("error response not JSON", resp_body[:100])


def test_tokenize(base):
    """Test tokenize endpoint."""
    print(f"\n{Colors.BOLD}Tokenize{Colors.RESET}")

    body = {"prompt": "Hello world"}
    status, _, resp_body, _ = request(base, "POST", "/tokenize", body=body, timeout=10)
    if status == 200:
        data = json.loads(resp_body)
        if "tokens" in data and isinstance(data["tokens"], list):
            ok(f"tokenize: {len(data['tokens'])} tokens")
        else:
            fail("tokenize response", f"got: {resp_body[:100]}")
    else:
        fail("tokenize", f"status={status}")

    # With v1 prefix
    status, _, resp_body, _ = request(base, "POST", "/v1/tokenize", body=body, timeout=10)
    if status == 200:
        ok("/v1/tokenize works")
    else:
        fail("/v1/tokenize", f"status={status}")


def test_tokenizer_info(base):
    """Test tokenizer_info endpoint."""
    print(f"\n{Colors.BOLD}Tokenizer Info{Colors.RESET}")

    status, _, resp_body, _ = request(base, "GET", "/tokenizer_info", timeout=10)
    if status == 200:
        data = json.loads(resp_body)
        if "eos_token" in data:
            ok(f"tokenizer_info: eos='{data['eos_token']}'")
        else:
            fail("tokenizer_info", "missing eos_token")
    else:
        fail("tokenizer_info", f"status={status}")


def test_system_message(base):
    """Test that system messages work."""
    print(f"\n{Colors.BOLD}System Message{Colors.RESET}")

    body = {
        "messages": [
            {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
            {"role": "user", "content": "Hello"}
        ],
        "stream": False,
        "max_tokens": 30,
        "temperature": 0.0
    }
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body, timeout=30)
    if status == 200:
        ok("system message accepted")
    else:
        fail("system message", f"status={status}")


def test_multi_turn(base):
    """Test multi-turn conversation."""
    print(f"\n{Colors.BOLD}Multi-Turn Conversation{Colors.RESET}")

    body = {
        "messages": [
            {"role": "user", "content": "My name is Tom."},
            {"role": "assistant", "content": "Nice to meet you, Tom!"},
            {"role": "user", "content": "What is my name?"}
        ],
        "stream": False,
        "max_tokens": 30,
        "temperature": 0.0
    }
    status, _, resp_body, _ = request(base, "POST", "/v1/chat/completions", body=body, timeout=30)
    if status == 200:
        data = json.loads(resp_body)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        if "tom" in content:
            ok(f"multi-turn: model remembered name ('{content[:50]}')")
        else:
            ok(f"multi-turn: request accepted (content: '{content[:50]}')")
    else:
        fail("multi-turn", f"status={status}")


def parse_anthropic_sse(resp) -> list:
    """Parse Anthropic SSE events (event: + data: pairs)."""
    events = []
    buf = ""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buf:
            event_str, buf = buf.split("\n\n", 1)
            event_type = None
            event_data = None
            for line in event_str.strip().split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        event_data = {"raw": line[6:]}
            if event_type and event_data:
                events.append({"event": event_type, **event_data})
                if event_type == "message_stop":
                    return events
    return events


def test_anthropic_nonstreaming(base):
    """Test Anthropic Messages API (non-streaming)."""
    print(f"\n{Colors.BOLD}Anthropic Messages API (non-streaming){Colors.RESET}")

    body = {
        "model": "local-model",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say exactly: test123"}],
        "temperature": 0.0
    }
    hdrs = {"anthropic-version": "2023-06-01"}
    status, _, resp_body, _ = request(base, "POST", "/v1/messages", body=body,
                                       headers=hdrs, timeout=60)
    if status != 200:
        fail("anthropic non-streaming", f"status={status} body={resp_body[:200]}")
        return

    data = json.loads(resp_body)
    log(f"Response: {json.dumps(data, indent=2)[:500]}")

    # Required fields
    required = ["id", "type", "role", "model", "content", "stop_reason", "usage"]
    missing = [k for k in required if k not in data]
    if missing:
        fail("anthropic schema", f"missing: {missing}")
    else:
        ok("all required fields present")

    if data.get("type") == "message":
        ok("type = 'message'")
    else:
        fail("type", f"got '{data.get('type')}'")

    if data.get("id", "").startswith("msg_"):
        ok("id starts with 'msg_'")
    else:
        fail("id prefix", f"got '{data.get('id')}'")

    if data.get("role") == "assistant":
        ok("role = 'assistant'")
    else:
        fail("role", f"got '{data.get('role')}'")

    # Content blocks
    content = data.get("content", [])
    if isinstance(content, list) and len(content) > 0:
        ok(f"content has {len(content)} block(s)")
        # First block may be "thinking" (if reasoning enabled) or "text"
        first = content[0]
        if first.get("type") == "text":
            ok(f"content[0].type = 'text', text='{first.get('text', '')[:40]}'")
        elif first.get("type") == "thinking":
            ok(f"content[0].type = 'thinking' (reasoning enabled)")
            text_blocks = [b for b in content if b.get("type") == "text"]
            if text_blocks:
                ok(f"text block present: '{text_blocks[0].get('text', '')[:40]}'")
        else:
            fail("content[0].type", f"got '{first.get('type')}'")
    else:
        fail("content blocks", "empty or not array")

    if data.get("stop_reason") in ("end_turn", "max_tokens"):
        ok(f"stop_reason = '{data['stop_reason']}'")
    else:
        fail("stop_reason", f"got '{data.get('stop_reason')}'")

    # Usage
    usage = data.get("usage", {})
    if "input_tokens" in usage and "output_tokens" in usage:
        ok(f"usage: in={usage['input_tokens']} out={usage['output_tokens']}")
    else:
        fail("usage fields", f"got {list(usage.keys())}")


def test_anthropic_streaming(base):
    """Test Anthropic Messages API (streaming SSE)."""
    print(f"\n{Colors.BOLD}Anthropic Messages API (streaming){Colors.RESET}")

    body = {
        "model": "local-model",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Count 1 to 5"}],
        "temperature": 0.0
    }
    hdrs = {"anthropic-version": "2023-06-01"}
    status, resp_hdrs, resp, conn = request(base, "POST", "/v1/messages",
                                             body=body, headers=hdrs, stream=True, timeout=60)
    if status != 200:
        fail("anthropic streaming", f"status={status}")
        if conn: conn.close()
        return

    events = parse_anthropic_sse(resp)
    if conn: conn.close()

    if not events:
        fail("no SSE events")
        return

    log(f"Got {len(events)} events")

    # Check event sequence
    event_types = [e.get("event") for e in events]

    if "message_start" in event_types:
        ok("has message_start")
        msg_start = [e for e in events if e.get("event") == "message_start"][0]
        msg = msg_start.get("message", {})
        if msg.get("role") == "assistant":
            ok("message_start has role=assistant")
    else:
        fail("missing message_start")

    if "content_block_start" in event_types:
        ok("has content_block_start")
    else:
        fail("missing content_block_start")

    # Collect text deltas
    text_content = ""
    for e in events:
        if e.get("event") == "content_block_delta":
            delta = e.get("delta", {})
            if delta.get("type") == "text_delta":
                text_content += delta.get("text", "")
    if text_content:
        ok(f"text deltas: '{text_content[:60]}'")
    else:
        # May have reasoning content instead
        ok("streaming completed (content may be in reasoning blocks)")

    if "content_block_stop" in event_types:
        ok("has content_block_stop")
    else:
        fail("missing content_block_stop")

    if "message_delta" in event_types:
        ok("has message_delta")
        msg_delta = [e for e in events if e.get("event") == "message_delta"][0]
        sr = msg_delta.get("delta", {}).get("stop_reason")
        if sr in ("end_turn", "max_tokens", "tool_use"):
            ok(f"message_delta stop_reason='{sr}'")
        else:
            fail("message_delta stop_reason", f"got '{sr}'")
    else:
        fail("missing message_delta")

    if event_types[-1] == "message_stop":
        ok("ends with message_stop")
    else:
        fail("last event", f"got '{event_types[-1]}'")


def test_anthropic_system(base):
    """Test Anthropic system prompt (top-level, not in messages)."""
    print(f"\n{Colors.BOLD}Anthropic System Prompt{Colors.RESET}")

    body = {
        "model": "local-model",
        "max_tokens": 30,
        "system": "Always respond with exactly one word.",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0
    }
    hdrs = {"anthropic-version": "2023-06-01"}
    status, _, resp_body, _ = request(base, "POST", "/v1/messages", body=body,
                                       headers=hdrs, timeout=30)
    if status == 200:
        ok("system prompt accepted")
    else:
        fail("system prompt", f"status={status}")


def test_anthropic_tool_calls(base):
    """Test Anthropic tool calls."""
    print(f"\n{Colors.BOLD}Anthropic Tool Calls{Colors.RESET}")

    body = {
        "model": "local-model",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }],
        "temperature": 0.0
    }
    hdrs = {"anthropic-version": "2023-06-01"}
    status, _, resp_body, _ = request(base, "POST", "/v1/messages", body=body,
                                       headers=hdrs, timeout=60)
    if status == 200:
        data = json.loads(resp_body)
        content = data.get("content", [])
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if tool_uses:
            tc = tool_uses[0]
            ok(f"tool_use block: name={tc.get('name')}")
            if tc.get("id", "").startswith("toolu_"):
                ok("tool_use id starts with 'toolu_'")
            if "input" in tc:
                ok(f"tool_use input present: {tc['input']}")
            if data.get("stop_reason") == "tool_use":
                ok("stop_reason = 'tool_use'")
            else:
                fail("stop_reason", f"got '{data.get('stop_reason')}'")
        else:
            skip("model didn't produce tool_use", "model-dependent")
    else:
        fail("anthropic tool call", f"status={status}")


def test_anthropic_errors(base):
    """Test Anthropic error responses."""
    print(f"\n{Colors.BOLD}Anthropic Errors{Colors.RESET}")

    # Missing max_tokens (required)
    body = {"model": "local-model", "messages": [{"role": "user", "content": "hi"}]}
    hdrs = {"anthropic-version": "2023-06-01"}
    status, _, resp_body, _ = request(base, "POST", "/v1/messages", body=body,
                                       headers=hdrs, timeout=10)
    if status == 400:
        data = json.loads(resp_body)
        if data.get("type") == "error" and "error" in data:
            ok("missing max_tokens → 400 with error object")
        else:
            fail("error format", f"got {resp_body[:100]}")
    else:
        fail("missing max_tokens", f"status={status}")

    # Invalid JSON
    parsed = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    conn.request("POST", "/v1/messages", body="not json",
                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    resp = conn.getresponse()
    resp.read()
    if resp.status == 400:
        ok("invalid JSON → 400")
    else:
        fail("invalid JSON", f"status={resp.status}")
    conn.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global verbose

    parser = argparse.ArgumentParser(description="MLXServer OpenAI API test suite")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Server URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests (tools, think)")
    args = parser.parse_args()
    verbose = args.verbose

    base = args.url.rstrip("/")
    print(f"{Colors.BOLD}MLXServer Test Suite{Colors.RESET}")
    print(f"Target: {base}\n")

    # Check server is up
    try:
        status, _, _, _ = request(base, "GET", "/health", timeout=5)
        if status != 200:
            print(f"{Colors.RED}Server not healthy (status={status}){Colors.RESET}")
            sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}Cannot connect to {base}: {e}{Colors.RESET}")
        sys.exit(1)

    test_health(base)
    test_cors(base)
    test_chat_nonstreaming(base)
    test_chat_streaming(base)
    test_streaming_connection_close(base)
    test_sequential_requests(base)
    test_keep_alive_nonstreaming(base)
    test_completions(base)
    test_tokenize(base)
    test_tokenizer_info(base)
    test_error_handling(base)
    test_system_message(base)
    test_multi_turn(base)

    test_anthropic_nonstreaming(base)
    test_anthropic_streaming(base)
    test_anthropic_system(base)
    test_anthropic_errors(base)

    if not args.quick:
        test_think_blocks(base)
        test_tool_calls(base)
        test_anthropic_tool_calls(base)

    # Summary
    total = passed + failed + skipped
    print(f"\n{Colors.BOLD}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.GREEN}{passed} passed{Colors.RESET}, "
          f"{Colors.RED}{failed} failed{Colors.RESET}, "
          f"{Colors.YELLOW}{skipped} skipped{Colors.RESET} "
          f"({total} total)")

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
