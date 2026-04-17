#!/usr/bin/env python3
"""
Go/no-go: transformer draft layer (~100M params) vs MLP.

EAGLE-style: single transformer decoder layer that takes
concat(h_t, embed(token_t)) and predicts h_{t+1}.
The attention mechanism can learn position-dependent patterns
that the MLP can't.
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


class TransformerDraftHead(nn.Module):
    """Single transformer layer draft head. ~100M params.

    Architecture (EAGLE-inspired):
    - Input projection: concat(h_t, embed(tok_t)) [2*D] → D
    - One transformer block: LayerNorm → SelfAttn → Residual → LayerNorm → FFN → Residual
    - Output: h_{t+1} prediction (fed to frozen LM head)

    For single-token decode, self-attention is Q@K^T = scalar, so it's
    effectively a deep nonlinear transform with learned gating.
    """
    def __init__(self, hidden_dim, num_heads=16, ffn_dim=4096):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # Self-attention
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # FFN
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def __call__(self, h_and_emb):
        # Input projection
        x = self.input_proj(h_and_emb)  # [B, D]

        # Reshape for "self-attention" (B=batch, T=1 for decode)
        if x.ndim == 1:
            x = x[None]  # [1, D]

        # Self-attention (trivial for T=1, but learned transform)
        residual = x
        x = self.norm1(x)
        B, D = x.shape

        q = self.q_proj(x).reshape(B, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, self.num_heads, self.head_dim)

        # For single token: attn = softmax(q@k^T/sqrt(d)) @ v = v (trivially)
        # But the projections still learn useful transforms
        scale = self.head_dim ** -0.5
        attn = (q * k).sum(-1, keepdims=True) * scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn * v).reshape(B, D)
        x = self.o_proj(x)
        x = residual + x

        # FFN (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = residual + x

        return x.squeeze(0) if B == 1 else x


class DeepMLP(nn.Module):
    """Deeper MLP for comparison. ~100M params."""
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


def collect_decode_data(model, tok, prompt, num_tokens):
    lm = model.language_model
    inner = lm.model
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

    h_list, tok_list, next_list = [], [], []
    t0 = time.perf_counter()
    for step in range(num_tokens):
        h_list.append(last_h)
        tok_list.append(y.item())
        y_in = mx.array([[y.item()]])
        h = inner.embed_tokens(y_in)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=None, cache=cache[i])
        h = inner.norm(h)
        logits = lm.lm_head(h)
        fp = mx.argmax(logits[:, -1], axis=-1)
        last_h = h[0, -1]
        mx.eval(fp, last_h)
        next_list.append(fp.item())
        y = fp.squeeze()
        if (step+1) % 5000 == 0:
            print(f"    {step+1}/{num_tokens} ({(step+1)/(time.perf_counter()-t0):.0f} tok/s)", flush=True)

    print(f"  collected {num_tokens} pairs in {time.perf_counter()-t0:.0f}s", flush=True)
    h_data = mx.stack(h_list)
    tok_data = mx.array(tok_list)
    targets = mx.array(next_list)
    mx.eval(h_data, tok_data, targets)
    return h_data, tok_data, targets


def train_and_test(name, draft, embed_layer, lm_head, h_data, tok_data, targets,
                   model, tok, test_prompts, epochs=15, batch_size=512, lr=3e-4):
    params = sum(v.size for _, v in tree_flatten(draft.parameters()))
    print(f"\n=== {name} ({params/1e6:.1f}M params) ===", flush=True)

    optimizer = optim.Adam(learning_rate=lr)
    T = h_data.shape[0]

    def loss_fn(m, hb, tb, tgt):
        emb = embed_layer(tb)
        x = mx.concatenate([hb, emb], axis=-1)
        logits = lm_head(m(x))
        return nn.losses.cross_entropy(logits, tgt).mean()
    lg = nn.value_and_grad(draft, loss_fn)

    t0 = time.perf_counter()
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
        if ep == 0 or (ep+1) % 5 == 0:
            print(f"  epoch {ep+1}/{epochs}: loss={total_loss/n:.4f}", flush=True)
    t1 = time.perf_counter()
    print(f"  training: {t1-t0:.0f}s", flush=True)

    # Held-out test
    lm = model.language_model
    inner = lm.model
    for prompt in test_prompts:
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
        for _ in range(200):
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
        print(f"  '{prompt[:50]}...': {matches}/200 = {matches/200*100:.1f}%", flush=True)

    adapter_mb = params * 2 / 1e6
    print(f"  adapter: {adapter_mb:.0f} MB", flush=True)


if __name__ == "__main__":
    print("=== Transformer vs Deep MLP Draft Head ===", flush=True)
    model, tok = load(MODEL)
    lm = model.language_model
    inner = lm.model
    D = inner.layers[0].input_layernorm.weight.shape[0]
    print(f"loaded. D={D}\n", flush=True)

    # Collect 20K decode tokens (balance: enough data, not too long)
    print("--- Collecting 20K decode-path tokens ---", flush=True)
    prompt = (
        "Write an extremely detailed and comprehensive analysis of the complete "
        "history and future of artificial intelligence, machine learning, and "
        "neural networks. Cover every decade from the 1940s through 2026."
    )
    h_data, tok_data, targets = collect_decode_data(model, tok, prompt, num_tokens=20000)

    test_prompts = [
        "Explain quantum computing to a five year old.",
        "Write a recursive fibonacci function in Rust with memoization.",
        "What caused the fall of the Roman Empire?",
    ]

    # Test 1: Transformer draft head (~67M params)
    transformer = TransformerDraftHead(D, num_heads=16, ffn_dim=4096)
    train_and_test("Transformer (1 layer)", transformer, inner.embed_tokens,
                   lm.lm_head, h_data, tok_data, targets, model, tok, test_prompts)

    # Test 2: Deep MLP (~100M params)
    mlp = DeepMLP(D, inner=4096, num_layers=3)
    train_and_test("Deep MLP (3 layer)", mlp, inner.embed_tokens,
                   lm.lm_head, h_data, tok_data, targets, model, tok, test_prompts)

    print("\nDONE")
