#!/usr/bin/env python3
"""
v2 Block Draft: parallel token generation via multi-position prediction heads.

Instead of autoregressive drafting (token by token), predict ALL N positions
from the target's hidden state in ONE forward pass.

Architecture (Medusa-style, simplest parallel generation):
- Input: h_t (target's last hidden state) + embed(token_t)
- N parallel heads, each predicting a different future position
- Head_k: MLP(concat(h_t, embed(tok_t))) → logits for position t+k+1
- All heads share the input, run in parallel = ONE forward for N predictions

Training: for each (h_t, tok_t) from decode, supervise head_k with token_{t+k+1}.

Key question: does parallel prediction from h_t alone achieve 85%+ acceptance
at each position? If acceptance decays with distance (position t+8 harder than
t+1), we may need iterative refinement (diffusion).

Phase 1: Train + measure per-position acceptance.
Phase 2: If needed, add refinement steps.
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


class ParallelDraftHead(nn.Module):
    """N independent prediction heads sharing one backbone.

    backbone: shared MLP that processes (h_t, embed(tok_t))
    heads: N separate Linear layers, each predicting a future position
    """
    def __init__(self, D, num_positions=8, backbone_dim=2048):
        super().__init__()
        self.num_positions = num_positions
        # Shared backbone
        self.backbone1 = nn.Linear(D * 2, backbone_dim, bias=False)
        self.backbone2 = nn.Linear(backbone_dim, backbone_dim, bias=False)
        self.backbone_norm = nn.RMSNorm(backbone_dim)
        # Per-position prediction heads
        self.heads = [nn.Linear(backbone_dim, D, bias=False) for _ in range(num_positions)]

    def __call__(self, h_and_emb):
        """Returns list of N hidden states, one per future position."""
        x = nn.gelu(self.backbone1(h_and_emb))
        x = self.backbone_norm(nn.gelu(self.backbone2(x)) + x)  # residual
        return [head(x) for head in self.heads]


def collect_windowed_data(model, tok, prompt, num_tokens, window=8):
    """Collect (h_t, tok_t, [tok_{t+1}, ..., tok_{t+window}]) tuples."""
    lm = model.language_model
    inner = lm.model

    tokens = mx.array(tok.encode(prompt))[None, :100]
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)

    # Collect with manual forward to get hidden states
    h_list = []
    tok_list = []

    t0 = time.perf_counter()
    for step in range(num_tokens + window):
        yi = mx.array([[y.item()]])
        h = inner.embed_tokens(yi)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        fp = mx.argmax(logits[:, -1], axis=-1)
        ch = h[0, -1]
        mx.eval(fp, ch)
        h_list.append(ch)
        tok_list.append(y.item())
        y = fp.squeeze()
        if (step + 1) % 5000 == 0:
            print(f"    {step+1}/{num_tokens+window}", flush=True)

    # Append final token
    tok_list.append(y.item())

    elapsed = time.perf_counter() - t0
    print(f"  collected {num_tokens} pairs in {elapsed:.0f}s ({(num_tokens+window)/elapsed:.0f} tok/s)", flush=True)

    # Build windowed training data
    h_data = mx.stack(h_list[:num_tokens])  # [N, D]
    tok_data = mx.array(tok_list[:num_tokens])  # [N] - current tokens
    # Targets: window of future tokens for each position
    target_windows = []
    for k in range(window):
        target_windows.append(mx.array(tok_list[k+1:num_tokens+k+1]))  # [N]

    mx.eval(h_data, tok_data, *target_windows)
    return h_data, tok_data, target_windows


def train_parallel(draft, embed, lm_head, h_data, tok_data, target_windows,
                   epochs=30, batch_size=512, lr=3e-4):
    """Train all heads simultaneously."""
    T = h_data.shape[0]
    num_pos = len(target_windows)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, h_batch, tok_batch, *tgt_batches):
        emb = embed(tok_batch)
        x = mx.concatenate([h_batch, emb], axis=-1)
        predictions = model(x)  # list of N [B, D]
        total = mx.array(0.0)
        for k, (pred, tgt) in enumerate(zip(predictions, tgt_batches)):
            logits = lm_head(pred)
            ce = nn.losses.cross_entropy(logits, tgt).mean()
            # Weight closer positions higher (they matter more for acceptance)
            weight = 1.0 / (k + 1)
            total = total + ce * weight
        return total

    lg = nn.value_and_grad(draft, loss_fn)

    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = mx.array(np.random.permutation(T))
        total_loss = 0
        n = 0
        for j in range(0, T, batch_size):
            idx = perm[j:j+batch_size]
            args = [draft, h_data[idx], tok_data[idx]] + [tw[idx] for tw in target_windows]
            loss, grads = lg(*args)
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
            total_loss += loss.item()
            n += 1
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}: loss={total_loss/n:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)


def measure_per_position_acceptance(model, draft, tok, prompt, num_tokens=200, num_pos=8):
    """Measure acceptance at each future position independently."""
    lm = model.language_model
    inner = lm.model

    tokens = mx.array(tok.encode(prompt))[None]
    cache = model.make_cache()

    # Prefill + get h
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache[i])
    h = inner.norm(h)
    logits = lm.lm_head(h)
    y = mx.argmax(logits[:, -1], axis=-1)
    last_h = h[0, -1]
    mx.eval(y, last_h)

    # Generate tokens and measure acceptance at each position
    per_pos_matches = [0] * num_pos
    total = 0

    # Collect true future tokens by running baseline
    true_tokens = [y.item()]
    for _ in range(num_tokens + num_pos):
        yi = mx.array([[y.item()]])
        h = inner.embed_tokens(yi)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        y_next = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y_next)
        true_tokens.append(y_next.item())
        y = y_next.squeeze()

    # Now check draft predictions at each position
    # Re-run with fresh cache to get hidden states at each position
    cache2 = model.make_cache()
    h = inner.embed_tokens(tokens)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=None, cache=cache2[i])
    h = inner.norm(h)
    y2 = mx.argmax(lm.lm_head(h)[:, -1], axis=-1)
    last_h2 = h[0, -1]
    mx.eval(y2, last_h2)

    for t in range(num_tokens):
        # Draft all positions from last_h2
        emb = inner.embed_tokens(mx.array([true_tokens[t]]))
        x = mx.concatenate([last_h2[None], emb], axis=-1)
        predictions = draft(x)

        for k in range(num_pos):
            pred_logits = lm.lm_head(predictions[k])
            pred_tok = mx.argmax(pred_logits, axis=-1)
            mx.eval(pred_tok)
            if pred_tok.item() == true_tokens[t + k + 1]:
                per_pos_matches[k] += 1

        total += 1

        # Advance
        yi = mx.array([[true_tokens[t]]])
        h = inner.embed_tokens(yi)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache2[i])
        h = inner.norm(h)
        last_h2 = h[0, -1]
        mx.eval(last_h2)

    return per_pos_matches, total


if __name__ == "__main__":
    print("=== v2 Parallel Draft (Medusa-style) ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]
    NUM_POS = 8

    # Collect 20K windowed training data
    print("--- Collecting 20K decode tokens with 8-position windows ---", flush=True)
    prompt = "Write a comprehensive history of AI from Turing to transformers."
    h_data, tok_data, target_windows = collect_windowed_data(
        model, tok, prompt, num_tokens=20000, window=NUM_POS)

    # Train parallel draft head
    print(f"\n--- Training {NUM_POS}-position parallel draft ---", flush=True)
    draft = ParallelDraftHead(D, num_positions=NUM_POS, backbone_dim=2048)
    params = sum(v.size for _, v in tree_flatten(draft.parameters()))
    print(f"  params: {params/1e6:.1f}M", flush=True)

    train_parallel(draft, inner.embed_tokens, lm.lm_head,
                   h_data, tok_data, target_windows,
                   epochs=30, batch_size=512, lr=3e-4)

    # Measure PER-POSITION acceptance
    print(f"\n--- Per-position acceptance (200 tokens, 3 prompts) ---", flush=True)
    test_prompts = [
        "Explain quantum computing simply.",
        "Write a Python quicksort implementation.",
        "What causes climate change?",
    ]

    total_per_pos = [0] * NUM_POS
    total_count = 0
    for p in test_prompts:
        per_pos, count = measure_per_position_acceptance(
            model, draft, tok, p, num_tokens=200, num_pos=NUM_POS)
        for k in range(NUM_POS):
            total_per_pos[k] += per_pos[k]
        total_count += count
        short = p[:50]
        rates = [f"{per_pos[k]/count*100:.0f}%" for k in range(NUM_POS)]
        print(f"  '{short}': {', '.join(rates)}", flush=True)

    print(f"\n  OVERALL per-position acceptance:", flush=True)
    for k in range(NUM_POS):
        rate = total_per_pos[k] / total_count * 100
        print(f"    pos t+{k+1}: {rate:.1f}%", flush=True)

    # Estimate: probability ALL first K positions match
    print(f"\n  P(all match) for draft block sizes:", flush=True)
    for N in [2, 4, 8]:
        p_all = 1.0
        for k in range(N):
            p_all *= total_per_pos[k] / total_count
        expected_tokens = sum(total_per_pos[k] / total_count for k in range(N)) + 1
        print(f"    N={N}: P(all)={p_all*100:.1f}%, E[tokens]={expected_tokens:.1f}", flush=True)

    print(f"\n  adapter: {params * 2 / 1e6:.0f} MB", flush=True)
    print("DONE")
