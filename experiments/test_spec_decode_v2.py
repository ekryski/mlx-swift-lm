#!/usr/bin/env python3
"""
Spec decode v2: proper cache snapshot/restore for GDN hybrid models.

Key fix: snapshot ALL cache state before verify. On partial accept,
restore snapshot and re-feed ONLY accepted tokens through the full model.
This is more expensive but guarantees lossless output.
"""
import sys, time
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import KVCache
import numpy as np
import copy

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
NUM_GEN = 200

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


def snapshot_cache(cache):
    """Deep copy all cache states."""
    snapshots = []
    for c in cache:
        if isinstance(c, KVCache):
            snapshots.append({
                'type': 'kv',
                'keys': mx.array(c.keys) if c.keys is not None else None,
                'values': mx.array(c.values) if c.values is not None else None,
                'offset': c.offset,
            })
        else:
            # ArraysCache / MambaCache — snapshot the state array
            snapshots.append({
                'type': 'arrays',
                'state': [mx.array(s) if s is not None else None for s in c.state] if c.state else None,
            })
    return snapshots


def restore_cache(cache, snapshots):
    """Restore cache from snapshot."""
    for c, snap in zip(cache, snapshots):
        if snap['type'] == 'kv':
            c.keys = snap['keys']
            c.values = snap['values']
            c.offset = snap['offset']
        else:
            if snap['state'] is not None:
                c.state = snap['state']


def full_forward_one_token(model, y, cache):
    """Run one token through full model, return (logits, hidden_state)."""
    lm = model.language_model
    inner = lm.model
    y_in = mx.array([[y.item() if hasattr(y, 'item') else y]])
    h = inner.embed_tokens(y_in)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    return logits[0, -1], h[0, -1]


def spec_decode_v2(model, draft, tok, prompt, num_tokens, num_draft=2):
    """Speculative decode with proper snapshot/restore."""
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
    total_rounds = 0
    total_draft_matches = 0
    total_accepted = 0

    t0 = time.perf_counter()
    while len(gen) < num_tokens:
        # --- Snapshot cache before draft+verify ---
        snap = snapshot_cache(cache)
        # Materialize snapshot
        to_eval = []
        for s in snap:
            if s['type'] == 'kv':
                if s['keys'] is not None: to_eval.extend([s['keys'], s['values']])
            else:
                if s['state']:
                    to_eval.extend([x for x in s['state'] if x is not None])
        if to_eval: mx.eval(to_eval)

        # --- Draft N tokens (no cache — stateless, uses draft head) ---
        draft_tokens = []
        draft_h = last_h
        draft_y = y
        for _ in range(num_draft):
            emb = inner.embed_tokens(mx.array([draft_y.item()]))
            x = mx.concatenate([draft_h[None], emb], axis=-1)
            h_pred = draft(x)
            draft_logits = lm.lm_head(h_pred)
            draft_tok = mx.argmax(draft_logits, axis=-1).squeeze()
            mx.eval(draft_tok)
            draft_tokens.append(draft_tok.item())
            # For next draft: use predicted h as the "hidden state"
            draft_h = h_pred.squeeze()
            draft_y = draft_tok

        # --- Verify: feed [y, draft0, ..., draftN-1] through full model ---
        # This advances the cache by N+1 positions
        verify_input = mx.array([[y.item()] + draft_tokens])
        h = inner.embed_tokens(verify_input)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        verify_logits = lm.lm_head(h)

        # --- Accept/reject ---
        accepted = 0
        for i in range(num_draft):
            target_tok = mx.argmax(verify_logits[0, i], axis=-1)
            mx.eval(target_tok)
            if target_tok.item() == draft_tokens[i]:
                gen.append(target_tok.item())
                accepted += 1
                total_draft_matches += 1
            else:
                gen.append(target_tok.item())
                accepted += 1
                break

        # Bonus token if all matched
        all_matched = (accepted == num_draft)
        if all_matched:
            bonus = mx.argmax(verify_logits[0, num_draft], axis=-1)
            mx.eval(bonus)
            gen.append(bonus.item())
            accepted += 1

        total_accepted += accepted
        total_rounds += 1

        # --- Restore cache and replay accepted tokens ---
        # The verify pushed all N+1 tokens into cache. We need to undo that
        # and only keep the accepted tokens' effect.
        restore_cache(cache, snap)

        # Now replay accepted tokens one-by-one through full model
        replay_tokens = gen[-(accepted):]
        current_y = y
        for rt in replay_tokens:
            _, last_h = full_forward_one_token(model, current_y, cache)
            mx.eval(last_h)
            current_y = mx.array(rt)

        y = mx.array(gen[-1])
        mx.eval(y, last_h)

        if len(gen) >= num_tokens:
            break

    t1 = time.perf_counter()
    gen = gen[:num_tokens]
    tps = (len(gen) - 1) / (t1 - t0)
    stats = {
        "rounds": total_rounds,
        "avg_accepted": total_accepted / total_rounds if total_rounds > 0 else 0,
        "match_rate": total_draft_matches / (total_rounds * num_draft) if total_rounds > 0 else 0,
    }
    return gen, tps, stats


def baseline_decode(model, tok, prompt, num_tokens):
    lm = model.language_model
    inner = lm.model
    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()
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
    return gen, (num_tokens - 1) / (t1 - t0)


if __name__ == "__main__":
    print("=== Spec Decode v2 (snapshot/restore) ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]

    # Quick train
    print("--- Training (20K tokens, 30 epochs) ---", flush=True)
    prompt_train = "Write a detailed analysis of the history of computing and AI."
    tokens = mx.array(tok.encode(prompt_train))[None, :100]
    cache_t = model.make_cache()
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache_t[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    y = mx.argmax(logits[:, -1], axis=-1)
    lh = h[0, -1]
    mx.eval(y, lh)

    hd, td, nd = [], [], []
    for step in range(20000):
        hd.append(lh)
        td.append(y.item())
        yi = mx.array([[y.item()]])
        h = inner.embed_tokens(yi)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache_t[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        fp = mx.argmax(logits[:, -1], axis=-1)
        lh = h[0, -1]
        mx.eval(fp, lh)
        nd.append(fp.item())
        y = fp.squeeze()
        if (step+1) % 5000 == 0:
            print(f"  {step+1}/20000", flush=True)

    h_data = mx.stack(hd); tok_data = mx.array(td); targets = mx.array(nd)
    mx.eval(h_data, tok_data, targets)

    draft = DeepMLP(D)
    opt = optim.Adam(learning_rate=3e-4)
    def lf(m, hb, tb, tgt):
        emb = inner.embed_tokens(tb)
        x = mx.concatenate([hb, emb], axis=-1)
        return nn.losses.cross_entropy(lm.lm_head(m(x)), tgt).mean()
    lg = nn.value_and_grad(draft, lf)
    for ep in range(30):
        perm = mx.array(np.random.permutation(20000))
        for j in range(0, 20000, 512):
            idx = perm[j:j+512]
            loss, grads = lg(draft, h_data[idx], tok_data[idx], targets[idx])
            opt.update(draft, grads)
            mx.eval(draft.parameters(), opt.state)
        if (ep+1) % 10 == 0:
            print(f"  epoch {ep+1}: loss={loss.item():.4f}", flush=True)

    # Benchmark
    test = "Explain the differences between TCP and UDP networking protocols."
    print(f"\n--- Baseline ({NUM_GEN} tokens) ---", flush=True)
    bt, btps = baseline_decode(model, tok, test, NUM_GEN)
    print(f"  {btps:.1f} tok/s", flush=True)

    for N in [2, 4]:
        print(f"\n--- Spec decode N={N} ({NUM_GEN} tokens) ---", flush=True)
        st, stps, stats = spec_decode_v2(model, draft, tok, test, NUM_GEN, num_draft=N)
        print(f"  {stps:.1f} tok/s (speedup: {stps/btps:.2f}x)", flush=True)
        print(f"  rounds: {stats['rounds']}, avg: {stats['avg_accepted']:.1f}, match: {stats['match_rate']*100:.1f}%", flush=True)
        match = bt[:50] == st[:50]
        print(f"  first 50 match baseline: {match}", flush=True)
        if not match:
            # Find first mismatch
            for i in range(min(50, len(bt), len(st))):
                if bt[i] != st[i]:
                    print(f"  diverge at pos {i}: baseline={bt[i]}, spec={st[i]}", flush=True)
                    break

    print("\nDONE")
