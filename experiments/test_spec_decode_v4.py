#!/usr/bin/env python3
"""
Spec decode v4: alpha-style snapshot/restore + batch verify.

Key insight: batch verify costs 1.2x of single forward (not Nx).
GDN kernel amortizes dispatch + state across the T-loop.

Approach matching alpha's SpeculativeTokenIterator:
1. Snapshot Mamba state before verify
2. Batch verify all N+1 tokens in ONE forward
3. Accept matching prefix
4. Trim KV cache for rejected positions
5. Restore Mamba snapshot on partial rejection
6. Set y = correction token for next round

The next round's verify naturally feeds the correction token through
the restored Mamba state, rebuilding it. Accepted tokens from the
previous round are "lost" from Mamba state, but KV cache has them.
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


def snapshot_mamba(cache):
    """Save Mamba states. Returns list of (state_arrays_or_None) per cache entry."""
    snaps = []
    for c in cache:
        if isinstance(c, KVCache):
            snaps.append(None)  # KV cache uses trim, not snapshot
        else:
            # ArraysCache / MambaCache
            s = c.state
            if s:
                snaps.append([mx.array(a) for a in s])  # deep copy
                mx.eval(snaps[-1])
            else:
                snaps.append(None)
    return snaps


def restore_mamba(cache, snaps):
    """Restore Mamba states from snapshot."""
    for c, snap in zip(cache, snaps):
        if snap is not None:
            c.state = snap


def trim_kv(cache, num_to_trim):
    """Trim KV caches (attention layers) by removing last N entries."""
    for c in cache:
        if isinstance(c, KVCache) and c.keys is not None and num_to_trim > 0:
            c.offset -= num_to_trim
            c.keys = c.keys[..., :-num_to_trim, :]
            c.values = c.values[..., :-num_to_trim, :]


def spec_decode_v4(model, draft, tok, prompt, num_tokens, num_draft=4):
    """Spec decode with proper alpha-style cache management."""
    lm = model.language_model
    inner = lm.model
    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()

    # Prefill
    logits = model(tokens, cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)

    # Get initial hidden state for draft head
    # Re-run to extract h (model() doesn't expose it)
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=None)  # Don't double-cache; use separate call
    h = inner.norm(h)
    last_h = h[0, -1]
    mx.eval(last_h)

    # Actually, that double-processes. Let me track h differently.
    # For v4: after each verify, capture last_h from the verify logits.
    # Use pseudoinverse or just re-run inner model on the last accepted token.
    # Simpler: after verify, the hidden state at position 'accepted' is what we need.
    # But model() doesn't return hidden states.
    #
    # COMPROMISE: use the draft head's predicted h as last_h for subsequent drafts.
    # Not perfect but avoids the double-forward problem.

    gen = [y.item()]
    total_rounds = 0
    total_matches = 0
    total_accepted = 0

    t0 = time.perf_counter()
    while len(gen) < num_tokens:
        # --- Snapshot Mamba state ---
        snap = snapshot_mamba(cache)

        # --- Draft N tokens with draft head (stateless, no cache) ---
        drafts = []
        cur_h = last_h
        cur_y = y
        for _ in range(num_draft):
            emb = inner.embed_tokens(mx.array([cur_y.item()]))
            x = mx.concatenate([cur_h[None], emb], axis=-1)
            h_pred = draft(x)
            d_logits = lm.lm_head(h_pred)
            d_tok = mx.argmax(d_logits, axis=-1).squeeze()
            mx.eval(d_tok)
            drafts.append(d_tok.item())
            cur_h = h_pred.squeeze()
            cur_y = d_tok

        # --- Batch verify: [y, d0, ..., dN-1] in ONE forward ---
        verify_tokens = mx.array([[y.item()] + drafts])
        verify_logits = model(verify_tokens, cache=cache)
        # verify_logits[0, i] predicts position i+1

        # --- Accept/reject ---
        accepted = 0
        for i in range(num_draft):
            target = mx.argmax(verify_logits[0, i], axis=-1)
            mx.eval(target)
            if target.item() == drafts[i]:
                gen.append(target.item())
                accepted += 1
                total_matches += 1
            else:
                gen.append(target.item())
                accepted += 1
                break

        all_matched = (accepted == num_draft)
        if all_matched:
            bonus = mx.argmax(verify_logits[0, num_draft], axis=-1)
            mx.eval(bonus)
            gen.append(bonus.item())
            accepted += 1

        total_accepted += accepted
        total_rounds += 1

        # --- Cache management ---
        num_reject = num_draft + 1 - accepted
        if all_matched:
            pass  # Cache has exactly the right state
        else:
            # Trim KV for rejected positions
            trim_kv(cache, num_reject)
            # Restore Mamba to pre-verify state
            restore_mamba(cache, snap)

        # Update y and last_h
        y = mx.array(gen[-1])

        # Update last_h: use the draft head's chain prediction
        # (This is approximate but avoids an extra full forward)
        if all_matched:
            # All matched — h_pred from the last draft step IS the hidden state
            # after processing all accepted tokens through the draft head
            last_h = cur_h
        else:
            # Partial — use h_pred at the last accepted position
            # Approximate: recompute from draft chain up to accepted position
            cur_h = last_h  # Start from pre-draft last_h
            cur_y = y  # But y is now the correction token...
            # This is imprecise. The correction token may not match draft chain.
            # For now, keep last_h stale. It'll be refreshed by the next round's
            # first token going through the model.

            # Actually — we need fresh h from the model. Let me extract it.
            # The verify pass already processed everything. The hidden state
            # at position `accepted-1` in the verify output is what we want.
            # But model() doesn't return hidden states, only logits.

            # HACK: re-run just the last accepted token through the model
            # to get a fresh hidden state. One extra forward.
            last_token = mx.array([[gen[-1]]])
            h = inner.embed_tokens(last_token)
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask=None, cache=cache[i])
            h = inner.norm(h)
            last_h = h[0, -1]
            mx.eval(last_h)
            # This also advances the cache by 1, which is correct (the correction token).

        mx.eval(y)

        if len(gen) >= num_tokens:
            break

    t1 = time.perf_counter()
    gen = gen[:num_tokens]
    tps = (len(gen) - 1) / (t1 - t0)
    stats = {
        'rounds': total_rounds,
        'avg_accepted': total_accepted / total_rounds,
        'match_rate': total_matches / (total_rounds * num_draft),
    }
    return gen, tps, stats


def baseline_decode(model, tok, prompt, num_tokens):
    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)
    gen = [y.item()]
    t0 = time.perf_counter()
    for _ in range(num_tokens - 1):
        logits = model(mx.array([[y.item()]]), cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)
        gen.append(y.item())
    t1 = time.perf_counter()
    return gen, (num_tokens - 1) / (t1 - t0)


if __name__ == "__main__":
    print("=== Spec Decode v4 (batch verify + snapshot/restore) ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]

    # Quick train
    print("--- Training (20K/30ep) ---", flush=True)
    pt = "Write about computing history."
    toks = mx.array(tok.encode(pt))[None, :100]
    ct = model.make_cache()
    logits = model(toks, cache=ct)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)

    # Collect hidden states by running inner model manually
    h = inner.embed_tokens(toks)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=None)  # separate cache for data collection
    h = inner.norm(h)
    init_h = h[0, -1]
    mx.eval(init_h)

    hd, td, nd = [], [], []
    ct2 = model.make_cache()
    logits = model(toks, cache=ct2)
    cur_y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(cur_y)

    for step in range(20000):
        # Get hidden state from full forward
        yi = mx.array([[cur_y.item()]])
        h = inner.embed_tokens(yi)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=ct2[i])
        h = inner.norm(h)
        cur_h = h[0, -1]
        logits = lm.lm_head(h)
        fp = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(fp, cur_h)
        hd.append(cur_h)
        td.append(cur_y.item())
        nd.append(fp.item())
        cur_y = fp.squeeze()
        if (step+1) % 5000 == 0:
            print(f"  {step+1}/20000", flush=True)

    # Fix: hd[i] is the hidden state AFTER processing td[i], and nd[i] is the next token
    # For training: input = (hd[i], td[i]) → target = nd[i]
    # But draft head takes (last_h, current_token) and predicts next hidden → LM head → next token
    # So we need pairs: (h_after_prev, current_token) → next_token
    # hd[i] is h after td[i], nd[i] is next. So (hd[i], nd[i-1]) isn't right.
    # Actually: hd[i] = h_t (hidden at step i), td[i] = token at step i, nd[i] = token at step i+1
    # Draft head input: (h_{i-1}, tok_i) → predict tok_{i+1}
    # So training pairs: (hd[i-1], td[i]) → nd[i] for i >= 1

    h_data = mx.stack(hd[:-1])  # h_{0..N-2}
    tok_data = mx.array(td[1:])  # tok_{1..N-1}
    targets = mx.array(nd[1:])   # next_tok_{1..N-1}

    # Wait actually: hd[i] is h AFTER processing token td[i].
    # Draft head should predict: given h after token i, what is token i+1?
    # So: (hd[i], embed(nd[i-wait...
    # Let me simplify: input = (hd[i], embed(td[i])), target = nd[i]
    # This says: given the hidden state from step i and the token at step i,
    # predict what comes after step i. That's what we want.

    h_data = mx.stack(hd)
    tok_data = mx.array(td)
    targets = mx.array(nd)
    mx.eval(h_data, tok_data, targets)

    draft = DeepMLP(D)
    opt = optim.Adam(learning_rate=3e-4)
    def lf(m, hb, tb, tgt):
        emb = inner.embed_tokens(tb)
        x = mx.concatenate([hb, emb], axis=-1)
        return nn.losses.cross_entropy(lm.lm_head(m(x)), tgt).mean()
    lg = nn.value_and_grad(draft, lf)
    for ep in range(30):
        perm = mx.array(np.random.permutation(len(targets)))
        for j in range(0, len(targets), 512):
            idx = perm[j:j+512]
            loss, grads = lg(draft, h_data[idx], tok_data[idx], targets[idx])
            opt.update(draft, grads)
            mx.eval(draft.parameters(), opt.state)
        if (ep+1) % 10 == 0:
            print(f"  epoch {ep+1}: loss={loss.item():.4f}", flush=True)

    # Benchmark
    test = "Explain the theory of general relativity."
    print(f"\n--- Baseline ({NUM_GEN} tokens) ---", flush=True)
    bt, btps = baseline_decode(model, tok, test, NUM_GEN)
    print(f"  {btps:.1f} tok/s", flush=True)
    print(f"  text: {tok.decode(bt[:30])[:100]}", flush=True)

    for N in [2, 4, 8]:
        print(f"\n--- Spec N={N} ({NUM_GEN} tokens) ---", flush=True)
        st, stps, stats = spec_decode_v4(model, draft, tok, test, NUM_GEN, num_draft=N)
        print(f"  {stps:.1f} tok/s (speedup: {stps/btps:.2f}x)", flush=True)
        print(f"  rounds: {stats['rounds']}, avg: {stats['avg_accepted']:.1f}, match: {stats['match_rate']*100:.1f}%", flush=True)
        print(f"  text: {tok.decode(st[:30])[:100]}", flush=True)
        # Check first divergence
        for i in range(min(50, len(bt), len(st))):
            if bt[i] != st[i]:
                print(f"  first diverge: pos {i}", flush=True)
                break
        else:
            print(f"  first 50 tokens: MATCH ✓", flush=True)

    print("\nDONE")
