#!/usr/bin/env python3
"""
v2 Phase 1: Validate separate GDN draft model fixes Mamba corruption.

Build a tiny model with its OWN GDN-like recurrent layers + attention.
Key: it has its OWN cache state, independent from the target model.

Architecture (micro draft model, ~20M params):
- Embedding: reused from target (frozen)
- 4 layers: 3 simplified GDN-like recurrent + 1 attention
- LM head: reused from target (frozen)
- Own cache: own Mamba state + own KV

The point is NOT acceptance rate (too small to be good).
The point is: does the verification loop produce COHERENT output
when the draft model has its own recurrent state?

If YES → the architecture is viable, scale up for v2.
If NO → something else is wrong beyond Mamba state.
"""
import sys, time
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache
import numpy as np

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"


class MiniRecurrentLayer(nn.Module):
    """Simplified GDN-like recurrent layer for draft model.
    Maintains its own recurrent state (like Mamba).
    """
    def __init__(self, D, state_dim=256):
        super().__init__()
        self.proj_in = nn.Linear(D, state_dim, bias=False)
        self.gate = nn.Linear(D, state_dim, bias=False)
        self.proj_out = nn.Linear(state_dim, D, bias=False)
        self.norm = nn.RMSNorm(D)
        self.state_dim = state_dim

    def __call__(self, x, state=None):
        # x: [B, T, D]
        B, T, D = x.shape
        if state is None:
            state = mx.zeros((B, self.state_dim))

        normed = self.norm(x)
        outs = []
        for t in range(T):
            xt = normed[:, t]  # [B, D]
            g = mx.sigmoid(self.gate(xt))  # [B, state_dim]
            inp = self.proj_in(xt)  # [B, state_dim]
            state = g * state + (1 - g) * inp  # GRU-like update
            out = self.proj_out(state)  # [B, D]
            outs.append(out)

        y = mx.stack(outs, axis=1)  # [B, T, D]
        return x + y, state  # residual


class MiniAttentionLayer(nn.Module):
    """Simple attention layer for draft model."""
    def __init__(self, D, num_heads=8):
        super().__init__()
        self.norm = nn.RMSNorm(D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.num_heads = num_heads
        self.head_dim = D // num_heads

    def __call__(self, x, kv_cache=None):
        B, T, D = x.shape
        normed = self.norm(x)
        q = self.q_proj(normed).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(normed).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(normed).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None and kv_cache[0] is not None:
            k = mx.concatenate([kv_cache[0], k], axis=2)
            v = mx.concatenate([kv_cache[1], v], axis=2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        # Causal mask
        T_q, T_k = q.shape[2], k.shape[2]
        mask = mx.triu(mx.full((T_q, T_k), -1e9), k=T_k - T_q + 1)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        out = self.o_proj(out)

        new_kv = (k, v)
        return x + out, new_kv


class MiniDraftModel(nn.Module):
    """Tiny draft model with own recurrent + attention layers."""
    def __init__(self, D, num_recurrent=3, state_dim=256, num_heads=8):
        super().__init__()
        self.recurrent_layers = [MiniRecurrentLayer(D, state_dim) for _ in range(num_recurrent)]
        self.attn_layer = MiniAttentionLayer(D, num_heads)
        self.norm = nn.RMSNorm(D)

    def __call__(self, x, cache=None):
        """x: [B, T, D]. cache: dict with 'recurrent_states' and 'kv'."""
        if cache is None:
            cache = {'recurrent_states': [None] * len(self.recurrent_layers), 'kv': None}

        new_states = []
        for i, layer in enumerate(self.recurrent_layers):
            x, state = layer(x, cache['recurrent_states'][i])
            new_states.append(state)

        x, kv = self.attn_layer(x, cache['kv'])
        x = self.norm(x)

        cache['recurrent_states'] = new_states
        cache['kv'] = kv
        return x, cache


def spec_decode_separate(target_model, draft_model, embed, lm_head, tok, prompt, num_tokens, num_draft=4):
    """Spec decode with SEPARATE draft model (own cache)."""
    inner = target_model.language_model.model

    tokens = mx.array(tok.encode(prompt))[None]

    # Target prefill
    target_cache = target_model.make_cache()
    target_logits = target_model(tokens, cache=target_cache)
    y = mx.argmax(target_logits[:, -1], axis=-1)
    mx.eval(y)

    # Draft prefill
    draft_cache = {'recurrent_states': [None] * len(draft_model.recurrent_layers), 'kv': None}
    draft_emb = embed(tokens)
    draft_h, draft_cache = draft_model(draft_emb, draft_cache)
    mx.eval([s for s in draft_cache['recurrent_states'] if s is not None])

    gen = [y.item()]
    t0 = time.perf_counter()
    matches = 0
    rounds = 0

    while len(gen) < num_tokens:
        # Snapshot target Mamba
        target_mamba_snap = []
        for c in target_cache:
            if not isinstance(c, KVCache):
                s = c.state
                target_mamba_snap.append([mx.array(a) for a in s] if s else None)
            else:
                target_mamba_snap.append(None)
        mx.eval([x for snap in target_mamba_snap if snap for x in snap])

        # Draft N tokens (draft model has OWN cache — no corruption)
        drafts = []
        for _ in range(num_draft):
            d_emb = embed(mx.array([[y.item() if not drafts else drafts[-1]]]))
            d_h, draft_cache = draft_model(d_emb, draft_cache)
            d_logits = lm_head(d_h)
            d_tok = mx.argmax(d_logits[:, -1], axis=-1)
            mx.eval(d_tok)
            drafts.append(d_tok.item())

        # Verify: batch through target
        verify = mx.array([[y.item()] + drafts])
        v_logits = target_model(verify, cache=target_cache)

        acc = 0
        for i in range(num_draft):
            tt = mx.argmax(v_logits[0, i], axis=-1); mx.eval(tt)
            if tt.item() == drafts[i]:
                gen.append(tt.item()); acc += 1; matches += 1
            else:
                gen.append(tt.item()); acc += 1; break

        if acc == num_draft:
            bonus = mx.argmax(v_logits[0, num_draft], axis=-1); mx.eval(bonus)
            gen.append(bonus.item()); acc += 1

        rounds += 1

        # Restore target Mamba on partial rejection
        reject = num_draft + 1 - acc
        if reject > 0:
            for c, snap in zip(target_cache, target_mamba_snap):
                if snap is not None:
                    c.state = snap
                elif isinstance(c, KVCache) and c.keys is not None:
                    c.offset -= reject
                    c.keys = c.keys[..., :-reject, :]
                    c.values = c.values[..., :-reject, :]

        # Draft cache: on partial rejection, we need to "rollback" draft too.
        # But draft has recurrent state — same problem?
        # KEY DIFFERENCE: draft model is TINY and we can afford to re-process
        # the accepted tokens through it. Or just reset and re-feed.
        # For now: just let draft state drift (it's a weak predictor anyway).
        # The important thing is TARGET state coherency.

        y = mx.array(gen[-1])
        mx.eval(y)

    t1 = time.perf_counter()
    gen = gen[:num_tokens]
    tps = len(gen) / (t1 - t0)
    return gen, tps, matches, rounds


if __name__ == "__main__":
    print("=== v2 Phase 1: Separate GDN Draft Model ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]
    embed = inner.embed_tokens  # shared
    lm_head = lm.lm_head  # shared

    # Build tiny draft model
    draft = MiniDraftModel(D, num_recurrent=3, state_dim=256, num_heads=8)
    params = sum(v.size for _, v in nn.utils.tree_flatten(draft.parameters()))
    print(f"Draft model: {params/1e6:.1f}M params (untrained)\n", flush=True)

    # Baseline
    print("--- Baseline ---", flush=True)
    tokens = mx.array(tok.encode("Explain quantum computing simply."))[None]
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1); mx.eval(y)
    base = [y.item()]
    t0 = time.perf_counter()
    for _ in range(199):
        logits = model(mx.array([[y.item()]]), cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1); mx.eval(y)
        base.append(y.item())
    btps = 199 / (time.perf_counter() - t0)
    print(f"  {btps:.1f} tok/s", flush=True)

    # Spec decode with separate draft (untrained — expect low match but correct output)
    print(f"\n--- Separate draft (untrained, N=2) ---", flush=True)
    gen, tps, matches, rounds = spec_decode_separate(
        model, draft, embed, lm_head, tok,
        "Explain quantum computing simply.", 200, num_draft=2)
    print(f"  {tps:.1f} tok/s (speedup: {tps/btps:.2f}x)", flush=True)
    print(f"  match: {matches}/{rounds*2} = {matches/(rounds*2)*100:.1f}%", flush=True)

    # KEY TEST: is output coherent?
    spec_text = tok.decode(gen[:200])
    base_text = tok.decode(base[:200])
    print(f"\n=== SPEC OUTPUT ===", flush=True)
    print(spec_text[:500], flush=True)
    print(f"\n=== BASELINE OUTPUT ===", flush=True)
    print(base_text[:500], flush=True)

    # Token match
    for i in range(min(50, len(gen), len(base))):
        if gen[i] != base[i]:
            print(f"\nFirst diverge: pos {i}", flush=True)
            break
    else:
        print(f"\nFirst 50: MATCH ✓", flush=True)

    print("\nDONE")
