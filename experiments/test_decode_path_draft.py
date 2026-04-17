#!/usr/bin/env python3
"""
Train draft head on DECODE-PATH hidden states (not prefill).

The key insight: prefill hidden states live in a different distribution
than decode hidden states. EAGLE works because it trains on the decode
trajectory. Let's do the same — run the model autoregressively, collect
(h_t, next_token) pairs from the actual decode path, train the MLP on those.

Plan:
1. Prefill a prompt
2. Decode 2000 tokens autoregressively, saving (h_t, token_{t+1}) at each step
3. Train MLP on those pairs
4. Measure acceptance on a DIFFERENT prompt's decode trajectory
"""
import sys, time
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
import numpy as np

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"

class DraftHead(nn.Module):
    """Tiny MLP: h_t → h_{t+1} prediction, then LM head maps to logits."""
    def __init__(self, D, inner=256):
        super().__init__()
        self.proj_in = nn.Linear(D, inner, bias=False)
        self.proj_out = nn.Linear(inner, D, bias=False)
    def __call__(self, h):
        return self.proj_out(nn.gelu(self.proj_in(h)))


def collect_decode_data(model, tok, prompt, num_tokens=2000):
    """Run autoregressive decode, collect (h_t, next_token) pairs."""
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

    # Decode — collect pairs
    h_list = []
    tok_list = []

    t0 = time.perf_counter()
    for step in range(num_tokens):
        h_list.append(last_h)

        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        full_pred = mx.argmax(logits[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(full_pred, last_h)

        tok_list.append(full_pred.item())
        y = full_pred.squeeze()

        if (step + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    collected {step+1}/{num_tokens} ({(step+1)/elapsed:.0f} tok/s)", flush=True)

    t1 = time.perf_counter()
    print(f"  collected {num_tokens} decode pairs in {t1-t0:.1f}s ({num_tokens/(t1-t0):.0f} tok/s)", flush=True)

    h_train = mx.stack(h_list)  # [N, D]
    targets = mx.array(tok_list)  # [N]
    mx.eval(h_train, targets)
    return h_train, targets


def train_head(draft_head, lm_head, h_train, targets, epochs=5, batch_size=256, lr=1e-3):
    """Train draft head on decode-path data."""
    T = h_train.shape[0]
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(m, hb, tb):
        logits = lm_head(m(hb))
        return nn.losses.cross_entropy(logits, tb).mean()
    lg = nn.value_and_grad(draft_head, loss_fn)

    for ep in range(epochs):
        perm = mx.array(np.random.permutation(T))
        total_loss = 0
        n = 0
        for j in range(0, T, batch_size):
            idx = perm[j:j+batch_size]
            loss, grads = lg(draft_head, h_train[idx], targets[idx])
            optimizer.update(draft_head, grads)
            mx.eval(draft_head.parameters(), optimizer.state)
            total_loss += loss.item()
            n += 1
        if ep == 0 or (ep+1) % 2 == 0:
            print(f"    epoch {ep+1}/{epochs}: loss={total_loss/n:.4f}", flush=True)


def measure_decode_acceptance(model, draft_head, tok, prompt, num_tokens=200):
    """Measure acceptance on a DIFFERENT prompt (held-out)."""
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

    matches = 0
    total = 0
    for _ in range(num_tokens):
        # Draft prediction from last_h
        h_draft = draft_head(last_h)
        draft_logits = lm.lm_head(h_draft[None])
        draft_pred = mx.argmax(draft_logits, axis=-1)

        # Full forward
        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        full_logits = lm.lm_head(h)
        full_pred = mx.argmax(full_logits[:, -1], axis=-1)
        last_h = h[0, -1]

        mx.eval(draft_pred, full_pred, last_h)
        if draft_pred.item() == full_pred.item():
            matches += 1
        total += 1
        y = full_pred.squeeze()

    return matches, total


if __name__ == "__main__":
    print("loading model...", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    D = lm.model.layers[0].input_layernorm.weight.shape[0]
    print(f"loaded. D={D}", flush=True)

    # Phase 1: collect DECODE-PATH training data
    print("\n--- Phase 1: Collect decode-path data ---", flush=True)
    train_prompt = (
        "Write a comprehensive essay about the history of artificial intelligence, "
        "starting from Alan Turing's foundational work through modern deep learning. "
        "Cover key milestones, breakthroughs, and the people behind them."
    )
    h_train, targets = collect_decode_data(model, tok, train_prompt, num_tokens=2000)

    # Phase 2: train + evaluate at different configs
    test_prompts = [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function that implements binary search.",
        "What are the main differences between classical and quantum computing?",
    ]

    for inner_dim in [256, 512]:
        for epochs in [3, 10]:
            print(f"\n--- inner={inner_dim}, epochs={epochs} ---", flush=True)
            draft = DraftHead(D, inner=inner_dim)
            params = sum(v.size for _, v in tree_flatten(draft.parameters()))
            print(f"  params: {params/1e6:.1f}M", flush=True)

            t0 = time.perf_counter()
            train_head(draft, lm.lm_head, h_train, targets, epochs=epochs)
            t1 = time.perf_counter()
            print(f"  train: {t1-t0:.1f}s", flush=True)

            # Calib match (on training data — decode path)
            cp = mx.argmax(lm.lm_head(draft(h_train)), axis=-1)
            mx.eval(cp)
            cm = (cp == targets).astype(mx.float32).mean().item()
            print(f"  calib (decode-path): {cm*100:.1f}%", flush=True)

            # Test on held-out prompts
            for prompt in test_prompts:
                matches, total = measure_decode_acceptance(
                    model, draft, tok, prompt, num_tokens=100)
                short = prompt[:50]
                print(f"  held-out '{short}...': {matches}/{total} = {matches/total*100:.0f}%", flush=True)

    print("\nDONE")
