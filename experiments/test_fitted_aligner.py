#!/usr/bin/env python3
"""
Test: fit a linear aligner from prefill hidden states, measure acceptance.

1. Prefill ~2048 tokens, capture h_K and h_40 at every position
2. Solve W = lstsq(h_K, h_40)
3. During decode, draft via LM_head(W @ h_K) instead of full forward
4. Measure acceptance rate vs full model at multiple exit depths
"""
import sys, time
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"

# Calibration text — generic, ships with the tool
CALIB_TEXT = (
    "The transformer architecture revolutionized natural language processing. "
    "It uses self-attention mechanisms to process sequences in parallel. "
    "Each layer applies multi-head attention followed by a feed-forward network. "
    "Modern variants include mixture-of-experts models where only a subset of "
    "parameters are active per token, reducing compute while scaling capacity. "
    "Language models are trained on diverse internet text to predict the next token. "
    "During inference, tokens are generated autoregressively one at a time. "
    "Speculative decoding accelerates this by drafting multiple tokens cheaply "
    "and verifying them in parallel with the full model. "
) * 30  # ~2700 tokens worth


def capture_hidden_states(model, tokens, exit_layer):
    """Forward pass that captures hidden states at exit_layer AND final layer."""
    lm = model.language_model
    inner = lm.model

    h = inner.embed_tokens(tokens)
    h_at_k = None

    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=None)
        if i == exit_layer - 1:
            h_at_k = h

    h_final = inner.norm(h)
    h_at_k_normed = inner.norm(h_at_k)  # apply same norm for fair comparison

    return h_at_k_normed, h_final


def fit_aligner(h_k, h_final):
    """Least-squares: find W such that W @ h_k ≈ h_final."""
    # h_k: [T, D], h_final: [T, D]
    # W = h_final.T @ h_k @ (h_k.T @ h_k)^(-1)
    # Using lstsq: solve h_k @ W.T = h_final for W.T
    # mx.linalg.lstsq not available, use normal equations
    T, D = h_k.shape
    # Add small ridge for stability
    XtX = h_k.T @ h_k + mx.eye(D) * 1e-4  # [D, D]
    XtY = h_k.T @ h_final  # [D, D]
    # Solve XtX @ W.T = XtY → W.T = solve(XtX, XtY)
    W_T = mx.linalg.solve(XtX, XtY, stream=mx.cpu)  # [D, D]
    W = W_T.T  # [D, D]
    mx.eval(W)
    return W


def measure_acceptance(model, W, exit_layer, num_tokens=200):
    """Decode num_tokens, measure how often aligned draft matches full model."""
    lm = model.language_model
    inner = lm.model

    prompt = "The capital of France is"
    tok = model.language_model  # need tokenizer
    # Use the tokenizer from load
    return None  # placeholder


def main():
    print("loading model...", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    total_layers = len(inner.layers)
    print(f"loaded. {total_layers} layers, hidden_dim={inner.layers[0].input_layernorm.weight.shape[0]}", flush=True)

    # Tokenize calibration text
    calib_tokens = mx.array(tok.encode(CALIB_TEXT))[None, :2048]  # [1, T]
    T = calib_tokens.shape[1]
    print(f"calibration: {T} tokens", flush=True)

    # Sweep exit layers
    for exit_layer in [32, 34, 36, 38, 39]:
        print(f"\n=== exit_layer={exit_layer}/{total_layers} (skip {total_layers - exit_layer}) ===", flush=True)

        # Phase 1: capture hidden states during calibration forward
        t0 = time.perf_counter()
        h_k, h_final = capture_hidden_states(model, calib_tokens, exit_layer)
        mx.eval(h_k, h_final)
        t1 = time.perf_counter()
        print(f"  capture: {t1-t0:.2f}s", flush=True)

        # Reshape: [1, T, D] → [T, D]
        h_k_2d = h_k.squeeze(0)
        h_final_2d = h_final.squeeze(0)

        # Phase 2: fit aligner
        t0 = time.perf_counter()
        W = fit_aligner(h_k_2d, h_final_2d)
        t1 = time.perf_counter()
        print(f"  fit aligner: {t1-t0:.3f}s, W shape: {W.shape}", flush=True)

        # Measure fit quality on calibration data
        h_aligned = h_k_2d @ W.T  # [T, D]
        # Get logits from both aligned and original
        if lm.lm_head is not None:
            logits_aligned = lm.lm_head(h_aligned)  # [T, vocab]
            logits_full = lm.lm_head(h_final_2d)  # [T, vocab]
        else:
            logits_aligned = inner.embed_tokens.as_linear(h_aligned)
            logits_full = inner.embed_tokens.as_linear(h_final_2d)

        # Argmax match rate (on calibration data — optimistic but indicative)
        pred_aligned = mx.argmax(logits_aligned, axis=-1)
        pred_full = mx.argmax(logits_full, axis=-1)
        mx.eval(pred_aligned, pred_full)
        match_rate = (pred_aligned == pred_full).astype(mx.float32).mean().item()
        print(f"  calib argmax match: {match_rate*100:.1f}%", flush=True)

        # Raw (no aligner) comparison
        if lm.lm_head is not None:
            logits_raw = lm.lm_head(h_k_2d)
        else:
            logits_raw = inner.embed_tokens.as_linear(h_k_2d)
        pred_raw = mx.argmax(logits_raw, axis=-1)
        mx.eval(pred_raw)
        raw_match = (pred_raw == pred_full).astype(mx.float32).mean().item()
        print(f"  raw (no aligner) match: {raw_match*100:.1f}%", flush=True)
        print(f"  improvement: {raw_match*100:.1f}% → {match_rate*100:.1f}%", flush=True)

        # Phase 3: measure on HELD-OUT tokens (decode)
        # Generate 100 tokens normally, compare draft-with-aligner vs full
        prompt_tokens = mx.array(tok.encode("Explain quantum computing in simple terms."))[None]
        cache = model.make_cache()

        # Prefill prompt
        logits = model(prompt_tokens, cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)

        matches = 0
        raw_matches = 0
        total = 0
        for _ in range(100):
            # Full forward (single token)
            y_input = mx.array([[y.item()]])
            full_logits = model(y_input, cache=cache)
            full_pred = mx.argmax(full_logits[:, -1], axis=-1)

            # What would draft predict? Extract h_k from last forward
            # We need to re-run partial forward for the draft
            h = inner.embed_tokens(mx.array([[y.item()]]))
            for i in range(exit_layer):
                h = inner.layers[i](h, mask=None, cache=None)
            h = inner.norm(h)

            # With aligner
            h_aligned_dec = h.squeeze() @ W.T
            if lm.lm_head is not None:
                draft_logits = lm.lm_head(h_aligned_dec[None])
            else:
                draft_logits = inner.embed_tokens.as_linear(h_aligned_dec[None])
            draft_pred = mx.argmax(draft_logits, axis=-1)

            # Without aligner
            if lm.lm_head is not None:
                raw_logits = lm.lm_head(h.squeeze()[None])
            else:
                raw_logits = inner.embed_tokens.as_linear(h.squeeze()[None])
            raw_pred = mx.argmax(raw_logits, axis=-1)

            mx.eval(full_pred, draft_pred, raw_pred)

            if draft_pred.item() == full_pred.item():
                matches += 1
            if raw_pred.item() == full_pred.item():
                raw_matches += 1
            total += 1

            y = full_pred.squeeze()

        print(f"  decode acceptance (aligner): {matches}/{total} = {matches/total*100:.1f}%", flush=True)
        print(f"  decode acceptance (raw):     {raw_matches}/{total} = {raw_matches/total*100:.1f}%", flush=True)

    print("\nDONE")


if __name__ == "__main__":
    main()
