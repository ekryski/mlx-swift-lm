#!/usr/bin/env python3
"""Quick self-draft test via mlx-lm to validate the concept before Swift."""
import sys, time
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
from mlx_lm import load

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"

def greedy(logits):
    return mx.argmax(logits[:, -1], axis=-1)

def main():
    print("loading...", flush=True)
    model, tok = load(MODEL)
    print("loaded", flush=True)

    prompt = "The capital of France is"
    tokens = mx.array(tok.encode(prompt))[None]

    # Warmup
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # --- Baseline: standard autoregressive ---
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    y = greedy(logits)
    mx.eval(y)

    gen_tokens = [y.item()]
    t0 = time.perf_counter()
    for _ in range(64):
        logits = model(y[None], cache=cache)
        y = greedy(logits)
        mx.eval(y)
        gen_tokens.append(y.item())
    t1 = time.perf_counter()
    baseline_tps = 64 / (t1 - t0)
    print(f"baseline: {baseline_tps:.1f} tok/s")
    print(f"text: {tok.decode(gen_tokens[:20])}")

    # --- Self-draft: first K of 40 layers ---
    lm = model.language_model
    inner = lm.model
    total_layers = len(inner.layers)
    draft_frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    draft_layers = max(1, int(total_layers * draft_frac))
    draft_n = 4
    print(f"\nself-draft: {draft_layers}/{total_layers} layers, draft_n={draft_n}")

    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    y = greedy(logits)
    mx.eval(y)

    gen_tokens2 = [y.item()]
    total_accepted = 0
    total_rounds = 0
    t0 = time.perf_counter()

    while total_accepted < 64:
        # Draft N tokens (stateless — no cache to avoid corruption)
        drafts = []
        draft_tok = y
        for _ in range(draft_n):
            # Partial forward: only first draft_layers layers, no cache
            tok_input = mx.array([[draft_tok.item()]])  # [1, 1] int32
            h = inner.embed_tokens(tok_input)
            for i in range(draft_layers):
                h = inner.layers[i](h, mask=None, cache=None)
            h = inner.norm(h)
            if lm.lm_head is not None:
                draft_logits = lm.lm_head(h)
            else:
                draft_logits = inner.embed_tokens.as_linear(h)
            draft_tok = mx.argmax(draft_logits[:, -1], axis=-1).squeeze()
            mx.eval(draft_tok)
            drafts.append(draft_tok.item())

        # Verify: [y, draft0, draft1, draft2, draft3] through full model
        verify_input = mx.array([[y.item()] + drafts])
        verify_logits = model(verify_input, cache=cache)
        # verify_logits[0, i] = logits for position i+1

        accepted = 0
        for i in range(4):
            target = mx.argmax(verify_logits[0, i], axis=-1)
            mx.eval(target)
            target_val = target.item()
            if target_val == drafts[i]:
                gen_tokens2.append(target_val)
                accepted += 1
            else:
                gen_tokens2.append(target_val)
                accepted += 1
                break

        # Bonus token if all matched
        if accepted == 4:
            bonus = mx.argmax(verify_logits[0, 4], axis=-1)
            mx.eval(bonus)
            gen_tokens2.append(bonus.item())
            accepted += 1

        y = mx.array(gen_tokens2[-1])
        total_accepted += accepted
        total_rounds += 1

    t1 = time.perf_counter()
    selfdraft_tps = total_accepted / (t1 - t0)
    avg_accept = total_accepted / total_rounds
    print(f"self-draft: {selfdraft_tps:.1f} tok/s ({total_accepted} tokens in {total_rounds} rounds, avg accept {avg_accept:.1f})")
    print(f"text: {tok.decode(gen_tokens2[:20])}")
    print(f"\nspeedup: {selfdraft_tps/baseline_tps:.2f}x")

if __name__ == "__main__":
    main()
