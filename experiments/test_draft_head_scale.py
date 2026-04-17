#!/usr/bin/env python3
"""
Scale-up: EAGLE-style draft head training on decode-path data.

Key changes vs tiny prototype:
1. 10K decode tokens (not 2K) — ~100s collection
2. Bigger MLP (16M params, 2 hidden layers)
3. Input = concat(h_t, embed(token_t)) — gives token identity context
4. More epochs with LR scheduling
5. Test on 3 held-out prompts with 200 tokens each
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


class DraftHeadV2(nn.Module):
    """2-layer MLP with token embedding concat. ~16M params."""
    def __init__(self, hidden_dim, inner_dim=2048):
        super().__init__()
        # Input: concat(h_t, embed(token_t)) = 2 * hidden_dim
        self.layer1 = nn.Linear(hidden_dim * 2, inner_dim, bias=False)
        self.layer2 = nn.Linear(inner_dim, inner_dim, bias=False)
        self.proj_out = nn.Linear(inner_dim, hidden_dim, bias=False)

    def __call__(self, h_and_emb):
        x = nn.gelu(self.layer1(h_and_emb))
        x = nn.gelu(self.layer2(x))
        return self.proj_out(x)


def collect_decode_data(model, tok, prompt, num_tokens=10000):
    """Autoregressive decode, collect (h_t, token_t, next_token) triples."""
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

    h_list = []
    token_list = []  # current token (for embedding concat)
    next_tok_list = []  # target

    t0 = time.perf_counter()
    for step in range(num_tokens):
        h_list.append(last_h)
        token_list.append(y.item())

        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        full_pred = mx.argmax(logits[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(full_pred, last_h)

        next_tok_list.append(full_pred.item())
        y = full_pred.squeeze()

        if (step + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {step+1}/{num_tokens} ({(step+1)/elapsed:.0f} tok/s)", flush=True)

    t1 = time.perf_counter()
    print(f"  collected {num_tokens} pairs in {t1-t0:.1f}s ({num_tokens/(t1-t0):.0f} tok/s)", flush=True)

    h_data = mx.stack(h_list)
    tok_data = mx.array(token_list)
    targets = mx.array(next_tok_list)
    mx.eval(h_data, tok_data, targets)
    return h_data, tok_data, targets


def train_head(draft, embed_layer, lm_head, h_data, tok_data, targets,
               epochs=20, batch_size=512, lr=3e-4):
    """Train with token embedding concatenation."""
    T = h_data.shape[0]
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, h_batch, tok_batch, target_batch):
        # Get token embeddings
        emb = embed_layer(tok_batch)  # [B, D]
        # Concat hidden state + token embedding
        x = mx.concatenate([h_batch, emb], axis=-1)  # [B, 2*D]
        h_pred = model(x)
        logits = lm_head(h_pred)
        return nn.losses.cross_entropy(logits, target_batch).mean()

    lg = nn.value_and_grad(draft, loss_fn)

    for ep in range(epochs):
        perm = mx.array(np.random.permutation(T))
        total_loss = 0
        n = 0
        for j in range(0, T, batch_size):
            idx = perm[j:j+batch_size]
            loss, grads = lg(draft, h_data[idx], tok_data[idx], targets[idx])
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
            total_loss += loss.item()
            n += 1
        avg = total_loss / n
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}: loss={avg:.4f}", flush=True)


def measure_acceptance(model, draft, tok, prompt, num_tokens=200):
    """Cache-aware acceptance measurement on held-out prompt."""
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
        # Draft: concat(last_h, embed(y)) → MLP → LM head
        emb = inner.embed_tokens(mx.array([y.item()]))  # [1, D]
        x = mx.concatenate([last_h[None], emb], axis=-1)  # [1, 2*D]
        h_pred = draft(x)
        draft_logits = lm.lm_head(h_pred)
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
    print("=== EAGLE-style Draft Head (scale-up) ===", flush=True)
    print("loading model...", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]
    print(f"loaded. D={D}\n", flush=True)

    # Phase 1: collect 10K decode tokens
    print("--- Phase 1: Collect 10K decode-path tokens ---", flush=True)
    prompt = (
        "Write a comprehensive and detailed essay about the complete history "
        "of computing, from Charles Babbage's Analytical Engine through modern "
        "quantum computers. Cover every major milestone, the key people involved, "
        "the technical breakthroughs, and how each era built on the previous one. "
        "Include discussion of hardware, software, networking, AI, and the social "
        "impact of each development."
    )
    h_data, tok_data, targets = collect_decode_data(model, tok, prompt, num_tokens=10000)

    # Phase 2: train + test
    test_prompts = [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function that implements quicksort with detailed comments.",
        "What are the main causes and effects of climate change?",
    ]

    draft = DraftHeadV2(D, inner_dim=2048)
    params = sum(v.size for _, v in tree_flatten(draft.parameters()))
    print(f"\n--- Phase 2: Train ({params/1e6:.1f}M params, 10K samples) ---", flush=True)

    t0 = time.perf_counter()
    train_head(draft, inner.embed_tokens, lm.lm_head,
               h_data, tok_data, targets,
               epochs=20, batch_size=512, lr=3e-4)
    t1 = time.perf_counter()
    print(f"  training time: {t1-t0:.1f}s", flush=True)

    # Calib
    emb_all = inner.embed_tokens(tok_data)
    x_all = mx.concatenate([h_data, emb_all], axis=-1)
    cp = mx.argmax(lm.lm_head(draft(x_all)), axis=-1)
    mx.eval(cp)
    cm = (cp == targets).astype(mx.float32).mean().item()
    print(f"  calib acceptance: {cm*100:.1f}%", flush=True)

    # Held-out
    print(f"\n--- Phase 3: Held-out acceptance (200 tokens each) ---", flush=True)
    for p in test_prompts:
        matches, total = measure_acceptance(model, draft, tok, p, num_tokens=200)
        short = p[:55]
        print(f"  '{short}...': {matches}/{total} = {matches/total*100:.1f}%", flush=True)

    # Save adapter size estimate
    adapter_bytes = params * 2  # bf16
    print(f"\n  adapter size: {adapter_bytes/1e6:.1f} MB (bf16)", flush=True)

    print("\nDONE")
