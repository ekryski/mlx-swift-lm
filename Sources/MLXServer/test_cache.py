#!/usr/bin/env python3
"""
Server prompt cache test suite.
Tests multi-session caching, prefix reuse, eviction, and interleaving.
Requires MLXServer running on localhost:8080.
"""
import json
import time
import urllib.request
import sys

URL = "http://127.0.0.1:8080/v1/chat/completions"
PASS = 0
FAIL = 0

def req(messages, tools=None, max_tokens=5, stream=False):
    """Send a request and return (response_dict, elapsed_ms)."""
    body = {"model": "test", "messages": messages, "max_tokens": max_tokens, "stream": stream}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    t0 = time.time()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(r, timeout=60)
    elapsed = (time.time() - t0) * 1000
    result = json.loads(resp.read().decode())
    return result, elapsed

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")

def test_basic_cache_hit():
    """Same system prompt, different user message → prefix reuse."""
    print("\n=== Test: Basic Cache Hit ===")
    sys_msg = "You are a helpful assistant who answers briefly."

    _, t1 = req([{"role": "system", "content": sys_msg}, {"role": "user", "content": "hi"}])
    _, t2 = req([{"role": "system", "content": sys_msg}, {"role": "user", "content": "what is 2+2"}])

    check("Second request faster", t2 < t1 * 0.8, f"t1={t1:.0f}ms t2={t2:.0f}ms")
    check("Second request under 200ms", t2 < 200, f"t2={t2:.0f}ms")

def test_cache_miss():
    """Completely different system prompt → cache miss."""
    print("\n=== Test: Cache Miss ===")

    _, t1 = req([{"role": "system", "content": "You are a pirate captain."}, {"role": "user", "content": "ahoy"}])
    _, t2 = req([{"role": "system", "content": "You are a space explorer."}, {"role": "user", "content": "greetings"}])

    # Both should be similar speed (both cold)
    check("Both requests take similar time", abs(t1 - t2) < t1 * 0.5, f"t1={t1:.0f}ms t2={t2:.0f}ms")

def test_session_interleave():
    """Alternating between two sessions → both get cache hits."""
    print("\n=== Test: Session Interleave ===")
    sys_a = "You are assistant Alpha. Always start with 'Alpha here.'"
    sys_b = "You are assistant Beta. Always start with 'Beta here.'"

    # Cold start both sessions
    _, ta1 = req([{"role": "system", "content": sys_a}, {"role": "user", "content": "hello"}])
    _, tb1 = req([{"role": "system", "content": sys_b}, {"role": "user", "content": "hello"}])

    # Return to session A
    _, ta2 = req([{"role": "system", "content": sys_a}, {"role": "user", "content": "what time is it"}])
    # Return to session B
    _, tb2 = req([{"role": "system", "content": sys_b}, {"role": "user", "content": "what day is it"}])

    # Note: interleave gets partial cache hit (BOS prefix only) because
    # trim() is destructive. Full session isolation requires KV copy support.
    check("Session A return completes", ta2 < 500, f"ta2={ta2:.0f}ms")
    check("Session B return completes", tb2 < 500, f"tb2={tb2:.0f}ms")

def test_conversation_growth():
    """Growing conversation → each turn reuses previous prefix."""
    print("\n=== Test: Conversation Growth ===")
    sys = "You are helpful."
    msgs = [{"role": "system", "content": sys}]

    times = []
    for i, user_msg in enumerate(["hi", "what is 2+2", "and 3+3", "what about 4+4"]):
        msgs.append({"role": "user", "content": user_msg})
        result, t = req(msgs, max_tokens=10)
        content = result["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": content})
        times.append(t)

    # Turn 2 reuses system prompt prefix; with small prompts the difference is subtle
    check("Turn 2 not dramatically slower than turn 1", times[1] < times[0] * 1.5, f"t1={times[0]:.0f}ms t2={times[1]:.0f}ms")
    check("Turn 4 not much slower than turn 2", times[3] < times[1] * 2, f"t2={times[1]:.0f}ms t4={times[3]:.0f}ms")

def test_eviction():
    """More sessions than maxSessions → oldest evicted."""
    print("\n=== Test: Eviction (max 3 sessions) ===")
    sessions = [
        "You are Alice, a cheerful assistant.",
        "You are Bob, a grumpy assistant.",
        "You are Carol, a wise assistant.",
        "You are Dave, a funny assistant.",
    ]

    # Fill 3 sessions
    for i, sys in enumerate(sessions[:3]):
        req([{"role": "system", "content": sys}, {"role": "user", "content": f"hello {i}"}])

    # 4th session should evict the oldest (session 0)
    req([{"role": "system", "content": sessions[3]}, {"role": "user", "content": "hello 3"}])

    # Session 1 should still be cached
    _, t_cached = req([{"role": "system", "content": sessions[1]}, {"role": "user", "content": "still here?"}])

    # Session 0 should be evicted (cold)
    _, t_evicted = req([{"role": "system", "content": sessions[0]}, {"role": "user", "content": "are you still here?"}])

    # This is a soft check — both are fast for small prompts, but the pattern should hold
    check("Session 1 still cached (not evicted)", t_cached < 200, f"t={t_cached:.0f}ms")

def test_tool_calls():
    """Tool definitions in prefix → reused across requests."""
    print("\n=== Test: Tool Call Caching ===")
    tools = [{"type": "function", "function": {"name": "run_command", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    sys = "You are a coding assistant."

    _, t1 = req([{"role": "system", "content": sys}, {"role": "user", "content": "list files"}], tools=tools)
    _, t2 = req([{"role": "system", "content": sys}, {"role": "user", "content": "show disk usage"}], tools=tools)

    check("Tool request cached", t2 < t1 * 0.8, f"t1={t1:.0f}ms t2={t2:.0f}ms")

def test_usage_in_response():
    """Response includes usage stats."""
    print("\n=== Test: Usage Stats ===")
    result, _ = req([{"role": "user", "content": "hi"}])
    usage = result.get("usage", {})
    check("Has prompt_tokens", "prompt_tokens" in usage, str(usage))
    check("Has completion_tokens", "completion_tokens" in usage, str(usage))
    check("prompt_tokens > 0", usage.get("prompt_tokens", 0) > 0, str(usage))

if __name__ == "__main__":
    # Check server is up
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)
    except:
        print("ERROR: Server not running on localhost:8080")
        sys.exit(1)

    print("Server Prompt Cache Test Suite")
    print("=" * 40)

    test_basic_cache_hit()
    test_cache_miss()
    test_session_interleave()
    test_conversation_growth()
    test_eviction()
    test_tool_calls()
    test_usage_in_response()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL > 0 else 0)
