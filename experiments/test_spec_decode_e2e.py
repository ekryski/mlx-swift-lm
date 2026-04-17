#!/usr/bin/env python3
"""
End-to-end speculative decoding with trained draft head.
Measures ACTUAL tok/s, not just acceptance rate.

1. Load model + train draft head (reuses collection from prior run if cached)
2. Run baseline autoregressive decode → measure tok/s
3. Run speculative decode → measure tok/s
4. Verify output matches (lossless check)
"""
import sys, time, os
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
import numpy as np

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
NUM_GEN = 200  # tokens to generate for speed measurement


class DeepMLP(nn.Module):
    def __init__(self, hidden_dim, inner=4096, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim * 2, inner, bias=False)
        self.layers = [nn.Linear(inner, inner, bias=False) for _ in range(num_layers - 1)]
        self.output_proj = nn.Linear(inner, hidden_dim, bias=False)
        self.norms = [nn.RMSNorm(inner) for _ in range(num_layers)]

    def __call__(self, h):
        x = nn.gelu(self.input_proj(h))
        for layer, norm in zip(self.layers, self.norms[1:]):
            x = x + nn.gelu(layer(norm(x)))
        return self.output_proj(self.norms[0](x))


def baseline_decode(model, tok, prompt, num_tokens):
    """Standard autoregressive decode. Returns (tokens, tok/s)."""
    lm = model.language_model
    inner = lm.model
    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()

    # Prefill
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)

    gen = [y.item()]
    t0 = time.perf_counter()
    for _ in range(num_tokens - 1):
        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)
        gen.append(y.item())
    t1 = time.perf_counter()

    tps = (num_tokens - 1) / (t1 - t0)
    return gen, tps


def spec_decode(model, draft, tok, prompt, num_tokens, num_draft=4):
    """Speculative decode with draft head. Returns (tokens, tok/s, stats)."""
    lm = model.language_model
    inner = lm.model
    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()

    # Prefill
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    y = mx.argmax(logits[:, -1], axis=-1)
    last_h = h[0, -1]
    mx.eval(y, last_h)

    gen = [y.item()]
    total_accepted = 0
    total_rounds = 0
    total_draft_matches = 0

    t0 = time.perf_counter()
    while len(gen) < num_tokens:
        # --- Draft phase: generate N candidate tokens ---
        draft_tokens = []
        draft_tok = y
        for _ in range(num_draft):
            emb = inner.embed_tokens(mx.array([draft_tok.item()]))
            x = mx.concatenate([last_h[None], emb], axis=-1)
            h_pred = draft(x)
            draft_logits = lm.lm_head(h_pred)
            draft_tok = mx.argmax(draft_logits, axis=-1).squeeze()
            mx.eval(draft_tok)
            draft_tokens.append(draft_tok.item())

        # --- Verify phase: run full model on [y, draft0, ..., draftN-1] ---
        verify_input = mx.array([[y.item()] + draft_tokens])
        h = inner.embed_tokens(verify_input)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        verify_logits = lm.lm_head(h)
        # verify_logits[0, i] = logits predicting position i+1

        # --- Accept/reject ---
        accepted = 0
        for i in range(num_draft):
            target_tok = mx.argmax(verify_logits[0, i], axis=-1)
            mx.eval(target_tok)
            target_val = target_tok.item()

            if target_val == draft_tokens[i]:
                gen.append(target_val)
                accepted += 1
                total_draft_matches += 1
            else:
                # Mismatch: accept target's correction
                gen.append(target_val)
                accepted += 1
                break

        # If all drafts matched, get bonus token
        if accepted == num_draft:
            bonus = mx.argmax(verify_logits[0, num_draft], axis=-1)
            mx.eval(bonus)
            gen.append(bonus.item())
            accepted += 1

        # Update state: last accepted token's hidden state
        last_accepted_idx = min(accepted, num_draft)
        last_h = h[0, last_accepted_idx]
        # Set y to last generated token
        y_val = gen[-1]
        y = mx.array(y_val)
        mx.eval(last_h)

        # Trim cache: we verified accepted+1 tokens but only want to keep
        # up to last_accepted_idx+1 positions. The verify already pushed all
        # tokens into cache. Need to trim excess.
        excess = num_draft + 1 - (accepted)
        if excess > 0:
            for c in cache:
                if hasattr(c, 'trim'):
                    c.trim(excess)
                elif hasattr(c, 'offset') and hasattr(c, 'keys') and c.keys is not None:
                    # KVCache: trim by reducing offset and slicing
                    c.offset -= excess
                    if c.keys is not None:
                        c.keys = c.keys[..., :-excess, :]
                        c.values = c.values[..., :-excess, :]

        total_accepted += accepted
        total_rounds += 1

        if len(gen) >= num_tokens:
            break

    t1 = time.perf_counter()
    gen = gen[:num_tokens]
    tps = (len(gen) - 1) / (t1 - t0)
    acceptance = total_draft_matches / (total_rounds * num_draft) if total_rounds > 0 else 0
    avg_accepted = total_accepted / total_rounds if total_rounds > 0 else 0

    stats = {
        "rounds": total_rounds,
        "avg_accepted_per_round": avg_accepted,
        "draft_match_rate": acceptance,
    }
    return gen, tps, stats


if __name__ == "__main__":
    print("=== End-to-End Speculative Decode Benchmark ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]

    # Quick train (20K tokens, 30 epochs — enough for ~30% acceptance)
    print("--- Training draft head (20K tokens, 30 epochs) ---", flush=True)
    prompt_train = "Write a comprehensive analysis of machine learning history and applications."
    tokens = mx.array(tok.encode(prompt_train))[None, :100]
    cache_train = model.make_cache()
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache_train[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    y = mx.argmax(logits[:, -1], axis=-1)
    last_h = h[0, -1]
    mx.eval(y, last_h)

    h_list, tok_list, next_list = [], [], []
    for step in range(20000):
        h_list.append(last_h)
        tok_list.append(y.item())
        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache_train[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        fp = mx.argmax(logits[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(fp, last_h)
        next_list.append(fp.item())
        y = fp.squeeze()
        if (step+1) % 5000 == 0:
            print(f"  collected {step+1}/20000", flush=True)

    h_data = mx.stack(h_list)
    tok_data = mx.array(tok_list)
    targets = mx.array(next_list)
    mx.eval(h_data, tok_data, targets)

    draft = DeepMLP(D, inner=4096, num_layers=3)
    optimizer = optim.Adam(learning_rate=3e-4)
    def loss_fn(m, hb, tb, tgt):
        emb = inner.embed_tokens(tb)
        x = mx.concatenate([hb, emb], axis=-1)
        return nn.losses.cross_entropy(lm.lm_head(m(x)), tgt).mean()
    lg = nn.value_and_grad(draft, loss_fn)

    t0 = time.perf_counter()
    for ep in range(30):
        perm = mx.array(np.random.permutation(20000))
        for j in range(0, 20000, 512):
            idx = perm[j:j+512]
            loss, grads = lg(draft, h_data[idx], tok_data[idx], targets[idx])
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
        if (ep+1) % 10 == 0:
            print(f"  epoch {ep+1}/30: loss={loss.item():.4f}", flush=True)
    print(f"  training: {time.perf_counter()-t0:.0f}s", flush=True)

    # Benchmark
    test_prompt = "Explain the differences between TCP and UDP networking protocols."

    print(f"\n--- Baseline decode ({NUM_GEN} tokens) ---", flush=True)
    baseline_tokens, baseline_tps = baseline_decode(model, tok, test_prompt, NUM_GEN)
    print(f"  {baseline_tps:.1f} tok/s", flush=True)
    print(f"  text: {tok.decode(baseline_tokens[:30])}", flush=True)

    for N in [2, 4, 6, 8]:
        print(f"\n--- Speculative decode N={N} ({NUM_GEN} tokens) ---", flush=True)
        spec_tokens, spec_tps, stats = spec_decode(model, draft, tok, test_prompt, NUM_GEN, num_draft=N)
        print(f"  {spec_tps:.1f} tok/s (speedup: {spec_tps/baseline_tps:.2f}x)", flush=True)
        print(f"  rounds: {stats['rounds']}, avg accepted: {stats['avg_accepted_per_round']:.1f}, match rate: {stats['draft_match_rate']*100:.1f}%", flush=True)

        # Verify lossless
        match = baseline_tokens[:50] == spec_tokens[:50]
        print(f"  first 50 tokens match baseline: {match}", flush=True)
        if not match:
            print(f"  baseline: {baseline_tokens[:20]}", flush=True)
            print(f"  spec:     {spec_tokens[:20]}", flush=True)

    print("\nDONE")
