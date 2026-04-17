#!/usr/bin/env python3
"""
High-quality draft head: EAGLE-scale training.

- 100K decode tokens (~17 min collection)
- 59M param deep MLP (3-layer, best from prior tests)
- 100 epochs (not 15)
- Cosine LR schedule
- Input: concat(h_t, embed(tok_t))
- Target: next token via frozen LM head

If this crosses 40%+ held-out acceptance, we build the tool.
"""
import sys, time, math
sys.path.insert(0, "/Users/tom/dev/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
import numpy as np

MODEL = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"


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


def collect_decode_data(model, tok, prompts, tokens_per_prompt):
    """Collect from MULTIPLE prompts for diversity."""
    lm = model.language_model
    inner = lm.model
    all_h, all_tok, all_next = [], [], []

    for pi, prompt in enumerate(prompts):
        tokens = mx.array(tok.encode(prompt))[None, :100]
        cache = model.make_cache()
        h = inner.embed_tokens(tokens)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        y = mx.argmax(logits[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(y, last_h)

        t0 = time.perf_counter()
        for step in range(tokens_per_prompt):
            all_h.append(last_h)
            all_tok.append(y.item())
            y_in = mx.array([[y.item()]])
            h = inner.embed_tokens(y_in)
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask=None, cache=cache[i])
            h = inner.norm(h)
            logits = lm.lm_head(h)
            fp = mx.argmax(logits[:, -1], axis=-1)
            last_h = h[0, -1]
            mx.eval(fp, last_h)
            all_next.append(fp.item())
            y = fp.squeeze()

        elapsed = time.perf_counter() - t0
        total = len(all_h)
        print(f"  prompt {pi+1}/{len(prompts)}: +{tokens_per_prompt} tokens ({tokens_per_prompt/elapsed:.0f} tok/s), total={total}", flush=True)

    h_data = mx.stack(all_h)
    tok_data = mx.array(all_tok)
    targets = mx.array(all_next)
    mx.eval(h_data, tok_data, targets)
    return h_data, tok_data, targets


def train(draft, embed_layer, lm_head, h_data, tok_data, targets,
          epochs=100, batch_size=512, lr_max=3e-4, lr_min=1e-5):
    """Train with cosine LR schedule."""
    T = h_data.shape[0]
    steps_per_epoch = (T + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch

    optimizer = optim.Adam(learning_rate=lr_max)

    def loss_fn(m, hb, tb, tgt):
        emb = embed_layer(tb)
        x = mx.concatenate([hb, emb], axis=-1)
        logits = lm_head(m(x))
        return nn.losses.cross_entropy(logits, tgt).mean()
    lg = nn.value_and_grad(draft, loss_fn)

    step = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = mx.array(np.random.permutation(T))
        total_loss = 0
        n = 0
        for j in range(0, T, batch_size):
            # Cosine LR
            progress = step / total_steps
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
            optimizer.learning_rate = mx.array(lr)

            idx = perm[j:j+batch_size]
            loss, grads = lg(draft, h_data[idx], tok_data[idx], targets[idx])
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
            total_loss += loss.item()
            n += 1
            step += 1

        avg = total_loss / n
        if ep == 0 or (ep+1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  epoch {ep+1}/{epochs}: loss={avg:.4f} lr={lr:.1e} ({elapsed:.0f}s)", flush=True)

    print(f"  total training: {time.perf_counter()-t0:.0f}s ({step} steps)", flush=True)


def measure(model, draft, tok, prompt, num_tokens=300):
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
    last_h = h[0, -1]
    mx.eval(y, last_h)

    matches = 0
    for _ in range(num_tokens):
        emb = inner.embed_tokens(mx.array([y.item()]))
        x = mx.concatenate([last_h[None], emb], axis=-1)
        dp = mx.argmax(lm.lm_head(draft(x)), axis=-1)
        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        fp = mx.argmax(lm.lm_head(h)[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(dp, fp, last_h)
        if dp.item() == fp.item():
            matches += 1
        y = fp.squeeze()
    return matches, num_tokens


if __name__ == "__main__":
    print("=== High-Quality Draft Head Training ===\n", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]
    print(f"D={D}\n", flush=True)

    # Phase 1: collect 100K tokens from 5 diverse prompts
    print("--- Phase 1: Collect 100K decode tokens (5 prompts × 20K) ---", flush=True)
    prompts = [
        "Write a comprehensive history of artificial intelligence from 1940 to 2026.",
        "Explain all of quantum mechanics including the math, starting from first principles.",
        "Write a complete implementation of a Redis clone in Python with all data structures.",
        "Analyze the complete works of Shakespeare, covering every play and major sonnet.",
        "Describe the entire history of space exploration from Sputnik through Mars missions.",
    ]
    h_data, tok_data, targets = collect_decode_data(model, tok, prompts, tokens_per_prompt=20000)
    print(f"  total: {h_data.shape[0]} pairs\n", flush=True)

    # Phase 2: train
    print("--- Phase 2: Train (59M params, 100 epochs, cosine LR) ---", flush=True)
    draft = DeepMLP(D, inner=4096, num_layers=3)
    params = sum(v.size for _, v in tree_flatten(draft.parameters()))
    print(f"  params: {params/1e6:.1f}M", flush=True)

    train(draft, inner.embed_tokens, lm.lm_head,
          h_data, tok_data, targets,
          epochs=100, batch_size=512, lr_max=3e-4, lr_min=1e-5)

    # Phase 3: test on 5 held-out prompts
    print(f"\n--- Phase 3: Held-out acceptance (300 tokens each) ---", flush=True)
    test_prompts = [
        "What is the meaning of life according to different philosophical traditions?",
        "Implement a B-tree in C++ with insert, delete, and search operations.",
        "Explain how nuclear fusion works and why it's hard to achieve on Earth.",
        "Write a detailed recipe for French onion soup with wine pairing suggestions.",
        "Compare and contrast democracy and authoritarianism across history.",
    ]
    total_matches = 0
    total_tokens = 0
    for p in test_prompts:
        m, t = measure(model, draft, tok, p, num_tokens=300)
        total_matches += m
        total_tokens += t
        print(f"  '{p[:55]}...': {m}/{t} = {m/t*100:.1f}%", flush=True)

    avg = total_matches / total_tokens * 100
    print(f"\n  OVERALL: {total_matches}/{total_tokens} = {avg:.1f}%", flush=True)
    print(f"  adapter: {params * 2 / 1e6:.0f} MB (bf16)", flush=True)

    # Speedup estimate
    # With near-free drafts (N=4): tokens_per_round ≈ 1/(1-acc) bounded by N
    eff = min(1/(1-avg/100), 5) if avg < 100 else 5
    print(f"  estimated speedup (N=4 drafts): ~{eff:.1f}x", flush=True)

    print("\nDONE")
