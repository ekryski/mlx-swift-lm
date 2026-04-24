# Implementation specs

Design documents for in-flight performance work. Each file is a standalone plan with motivation, concrete code changes, acceptance criteria, and measurement plan so an implementer (human or agent) can work against it without additional context.

## Index

| # | Spec | Target | Expected gain |
|---|---|---|---|
| 001 | [Dense MLP gate+up fusion](001-dense-mlp-gate-up-fusion.md) | `Qwen3NextMLP`, `Gemma4SharedMLP` | +5–10% decode (Qwen3.5 dense), +3–4% (Gemma 4 31B) |
| 002 | [Dense MLP inline activation Metal kernel](002-dense-mlp-inline-activation-kernel.md) | `FusedGateUpMLP` (post-001) | +3–6% decode on top of 001 |
| 003 | [QKV fusion via `batchedQKVQuantizedGEMV`](003-qkv-fusion-batched-qgemv.md) | Qwen3.5 / 3.6 + Gemma 4 attention | +3–5% decode |
| 004 | [`rmsNormQuantizedGEMV` for dense MLP](004-rmsnorm-qgemv-mlp.md) | Qwen3.5 dense, Gemma 4 dense | +3–8% decode on top of 001 |
| 005 | [Turbo KV for `RotatingKVCache`](005-turbo-kv-for-rotating-cache.md) | GPT-OSS-20B, Gemma 4 (all sliding variants) | ~4× KV memory reduction + close GPT-OSS turbo4v2 decode gap vs `ek/tom-eric-moe-tuning` |
| 006 | [KVCache refactor](006-kvcache-refactor.md) | Entire `KVCache` hierarchy | Correctness: makes `kv=turbo*` and `kv=affine*` work uniformly across all models. Supersedes 005. |
| 007 | [Unsloth Dynamic Quant compat](007-unsloth-dynamic-quants.md) | Unsloth `*-UD-MLX-*bit` checkpoints | Correctness: load + run Unsloth Dynamic 2.0 mixed-bit checkpoints. Secondary: lower peak memory vs uniform 4-bit. |

## Ordering

- **001, 002, 004** are the dense-MLP perf stack. **002 and 004 both stack on 001** (they consume the fused `gate_up_proj` weight). Land 001 first, then measure 002 and 004 independently to isolate each contribution; either can ship first.
- **003** is the attention-projection equivalent — orthogonal to the MLP specs. Can ship in parallel.
- **005** was the quick-fix plan for KV quantization wiring: add `RotatingKVCache.toTurboQuantized`, add `maybeTurboQuantizeKVCache`, call both from the generate loop.
- **006 is the cleaner architectural fix** that supersedes 005 — it kills `maybeQuantizeKVCache` entirely and moves lifecycle transitions into the cache classes themselves. If we land 006 we don't need 005.

Suggested ship order: **001 → 003 → 002 → 004 → 006**. Consider 005 only if 006 slips or has to be descoped.

## Conventions

- One spec per PR. Don't bundle.
- Each spec defines its own acceptance criteria. Treat them as the shipping contract: if a number isn't met, the spec is not done.
- KLD checks (where applicable) are a hard gate. Lossy optimizations need quality numbers, not vibes.
- Update the index table above when adding a spec or when one ships.
