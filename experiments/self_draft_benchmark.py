#!/usr/bin/env python3
"""
Self-draft speculative decoding viability test for Qwen3.6-35B-A3B.

RESULT: NOT VIABLE. Two independent failure modes kill this approach.

Test methodology:
  Phase 1: Best-case acceptance rate (both models get proper cache, same inputs)
  Phase 2: Per-token timing (draft cost vs full cost)
  Analysis: Theoretical speedup = (1 + N*alpha) / (N*draft_cost/full_cost + 1)

Architecture: 40 layers, attention at [3,7,11,15,19,23,27,31,35,39], rest GDN.
"""

import sys
sys.path.insert(0, "/Users/tom/dev/mlx-lm")

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

MODEL_PATH = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
PROMPT = "Explain the key differences between transformer attention and linear attention mechanisms in modern language models. Be specific about computational complexity and memory usage."


def early_exit_forward(model, tokens, cache, exit_layer):
    """Forward through first exit_layer layers, then norm + lm_head."""
    lm = model.language_model
    inner = lm.model
    hidden = inner.embed_tokens(tokens)
    fa_mask = create_attention_mask(hidden, cache[inner.fa_idx] if inner.fa_idx < exit_layer else None)
    ssm_mask = create_ssm_mask(hidden, cache[inner.ssm_idx] if inner.ssm_idx < exit_layer else None)
    for i in range(exit_layer):
        layer = inner.layers[i]
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=cache[i])
    hidden = inner.norm(hidden)
    if lm.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(hidden)
    return lm.lm_head(hidden)


def phase1_acceptance(model, tokenizer, prompt, exit_layer, max_tokens=150):
    """Best-case acceptance rate measurement."""
    tokens = mx.array(tokenizer.encode(prompt))
    full_cache = model.make_cache()
    ee_cache = model.make_cache()

    full_logits = model.language_model(tokens[None, :], cache=full_cache)
    _ = early_exit_forward(model, tokens[None, :], ee_cache, exit_layer)
    full_tok = mx.argmax(full_logits[:, -1, :], axis=-1)
    mx.eval(full_tok)

    matches = 0
    total = 0
    for _ in range(max_tokens - 1):
        inp = full_tok.reshape(1, 1)
        fl = model.language_model(inp, cache=full_cache)
        full_tok = mx.argmax(fl[:, -1, :], axis=-1)
        eel = early_exit_forward(model, inp, ee_cache, exit_layer)
        ee_tok = mx.argmax(eel[:, -1, :], axis=-1)
        mx.eval(full_tok, ee_tok)
        total += 1
        if full_tok.item() == ee_tok.item():
            matches += 1
        if full_tok.item() == tokenizer.eos_token_id:
            break

    return matches / total if total else 0, total


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    layers = model.language_model.model.layers
    attn_idx = [i for i, l in enumerate(layers) if not l.is_linear]
    print(f"{len(layers)} layers. Attention at: {attn_idx}")

    print("\n=== Phase 1: Acceptance Rate vs Exit Depth ===")
    print(f"{'Exit@':<8} {'Layers skipped':<16} {'Match%':<10}")
    print("-" * 40)
    for el in [4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 39]:
        match_rate, n = phase1_acceptance(model, tokenizer, PROMPT, el, max_tokens=150)
        print(f"{el:<8} {40-el:<16} {match_rate*100:<10.1f}")

    print("\n=== Phase 2: Per-Token Timing ===")
    tokens = mx.array(tokenizer.encode(PROMPT))
    for el in [32, 36, 38, 39]:
        dc = model.make_cache()
        fc = model.make_cache()
        _ = early_exit_forward(model, tokens[None, :], dc, el)
        _ = model.language_model(tokens[None, :], cache=fc)
        mx.eval(mx.array(0))
        tok = mx.array([[1]])
        start = time.perf_counter()
        for _ in range(50):
            l = early_exit_forward(model, tok, dc, el)
            mx.eval(mx.argmax(l[:, -1, :], axis=-1))
        dt = (time.perf_counter() - start) / 50
        start = time.perf_counter()
        for _ in range(50):
            l = model.language_model(tok, cache=fc)
            mx.eval(mx.argmax(l[:, -1, :], axis=-1))
        ft = (time.perf_counter() - start) / 50
        print(f"  Exit@{el}: draft={dt*1000:.1f}ms, full={ft*1000:.1f}ms, "
              f"ratio={dt/ft:.3f}, savings={100*(1-dt/ft):.1f}%")


if __name__ == "__main__":
    main()


"""
═══════════════════════════════════════════════════════════════════════════════
RESULTS (Qwen3.6-35B-A3B-4bit, M-series GPU, greedy decode, 150 tokens)
═══════════════════════════════════════════════════════════════════════════════

Phase 1 — Acceptance Rate:
  Exit@4:   0.7%     Exit@20: 1.3%     Exit@36: 18.1%
  Exit@8:   0.7%     Exit@24: 1.3%     Exit@38: 34.9%
  Exit@12:  0.7%     Exit@28: 2.0%     Exit@39: 71.8%
  Exit@16:  0.7%     Exit@32: 3.4%

Phase 2 — Per-Token Timing:
  Exit@39: draft=9.9ms, full=10.1ms → 1.8% savings (1 layer skipped)
  Exit@38: ~3-4% savings (2 layers skipped)
  Exit@32: ~18-20% savings (8 layers skipped)

Phase 3 — Theoretical Speedup (best case, exit@39, alpha=0.72):
  N=3 drafts: speedup = (1 + 3*0.72) / (3*0.98 + 1) = 3.15 / 3.94 = 0.80x  ← SLOWER
  N=5 drafts: speedup = (1 + 5*0.72) / (5*0.98 + 1) = 4.59 / 5.90 = 0.78x  ← SLOWER

TWO INDEPENDENT FAILURE MODES:
  1. Acceptance cliff: acceptance drops from 72% (skip 1 layer) to 35% (skip 2)
     to 3% (skip 8). The LM head cannot read intermediate hidden states.
  2. Cost cliff: skipping 1 layer saves only 2% per draft token. You need to
     skip many layers for meaningful savings, but then acceptance goes to zero.

The two cliffs are complementary: high acceptance requires near-full depth
(no cost savings), and meaningful cost savings require shallow exit (no
acceptance). There is no sweet spot.

BASELINE PERFORMANCE: 100.5 tok/s (already fast on this hardware)

RECOMMENDATION: Abandon dynamic self-draft for Qwen3.5/3.6. This approach
requires one of:
  a) Trained early-exit heads (lightweight projections at 2-3 depth points)
  b) Layer-skip distillation (expensive training, model-specific)
  c) A separate small draft model (e.g., Qwen3-0.5B → ~$0 to implement)
  d) N-gram / prompt-lookup speculation (already implemented in codebase)
  e) Jacobi-style parallel decoding (orthogonal approach, worth investigating)

Option (c) is the most practical path — the existing SpeculativeTokenIterator
already supports it with zero changes needed.
═══════════════════════════════════════════════════════════════════════════════
"""
