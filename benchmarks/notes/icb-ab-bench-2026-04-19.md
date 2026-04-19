# ICB + AB infrastructure benchmark — 2026-04-19

M1 Max 64 GB, macOS 15.7.4, branch `ek/persistent-ab-pilot`,
`summarization` method @ 1024 context, 4-bit weights. Measured after
the step-3 `fast::Quantize::eval_gpu` pipeline-order fix + the
`PersistentGatherFrontAbHandle` input-override landing.

Alpha decode / prefill numbers are taken from
[turbo4v2-decode-regression-2026-04-17.md](turbo4v2-decode-regression-2026-04-17.md);
it didn't capture TTFT or GPU memory — those columns are fresh
measurements on the current branch only.

## Gemma 4 E2B (`mlx-community/gemma-4-e2b-it-4bit`)

| Config | KV | Prefill tok/s | Decode tok/s | Δ vs alpha | TTFT | Peak GPU |
|---|---|---:|---:|---:|---:|---:|
| alpha (pristine)         | none     | 2 906.8 | **99.7**  | —          | n/a     | n/a      |
| ours, no flags           | none     | 2 679.7 | 94.5      | −5.2 %     | 377 ms  | 3.24 GB  |
| ours, `MLX_METAL_AB=1`   | none     | 2 700.7 | 85.9      | −13.8 %    | 375 ms  | 3.24 GB  |
| alpha (pristine)         | turbo4v2 | 2 885.3 | **102.9** | —          | n/a     | n/a      |
| ours, no flags           | turbo4v2 | 2 629.7 | 94.2      | −8.5 %     | 385 ms  | 3.24 GB  |
| ours, `MLX_METAL_AB=1`   | turbo4v2 | 2 628.9 | 85.7      | −16.7 %    | 385 ms  | 3.24 GB  |

## GPT-OSS 20B (`loan-star/gpt-oss-20b-mlx-4Bit`)

| Config | KV | Prefill tok/s | Decode tok/s | Δ vs alpha | TTFT | Peak GPU |
|---|---|---:|---:|---:|---:|---:|
| alpha (pristine)               | none     | 638.7 | **51.7** | —         | n/a     | n/a      |
| ours, no flags                 | none     | 579.3 | 47.0     | −9.1 %    | 1 768 ms | 11.65 GB |
| ours, AB + PersistentAB        | none     | 576.6 | 46.2     | −10.6 %   | 1 777 ms | 11.81 GB |
| alpha (pristine)               | turbo4v2 | 662.2 | **52.2** | —         | n/a     | n/a      |
| ours, no flags                 | turbo4v2 | 547.7 | 46.7     | −10.5 %   | 1 870 ms | 11.65 GB |
| **ours, AB + PersistentAB**    | turbo4v2 | 599.6 | **63.9** | **+22.4 %** | **1 708 ms** | 11.68 GB |

"AB + PersistentAB" means `MLX_METAL_ICB=1 MLX_METAL_AB=1 MLX_PERSISTENT_AB=1`
with the ICB decode-loop **not** enabled. The decode-loop column is
missing from this table by design — see "Known regressions" below.

## Key findings

1. **Baseline regression on our branch (≈ 5–10 %).** A pristine
   `ek/persistent-ab-pilot` checkout with no ICB / AB flags is
   5–10 % slower than `alpha` on every row. Root cause matches the
   diagnosis in the April 17 turbo4v2 regression note:
   `supportIndirectCommandBuffers=YES` on every `MTLComputePipelineDescriptor`
   + a process-global atomic dispatch counter ride along regardless
   of whether an ICB is ever recorded. These should become opt-in
   at pipeline-build time.

2. **AB is net-negative on Gemma 4 E2B at this workload.**
   `MLX_METAL_AB=1` costs an additional 9–10 % on Gemma vs our own
   no-flag baseline (14–17 % vs alpha). Gemma hasn't been tuned for
   AB the way GPT-OSS has; the AB-preamble overhead dominates the
   smaller per-dispatch kernels. Fixing requires either per-op AB
   gating or the same PersistentAb wiring GPT-OSS has (removes the
   transient-allocation cost too).

3. **AB + PersistentAB is a big win on GPT-OSS 20B + turbo4v2
   (+22.4 % vs alpha).** 52.2 → 63.9 tok/s. Turbo4v2 doubles the
   per-layer kernel count (packed-dequant-K and packed-dequant-V
   every step), and AB cuts the encoding cost precisely on those
   hot dispatches. No-quant GPT-OSS doesn't benefit because the
   baseline regression offsets the AB gain.

4. **TTFT essentially flat across configs** (± 50–100 ms, dominated
   by prefill). AB flags don't shift prefill meaningfully at 1024
   tokens — prefill already uses the NativePrefillBridge.

5. **Peak GPU is flat** (± 150 MB). AB's transient AB MTLBuffer per
   layer × step is returned to the pool on command completion.

## Known regressions (not in the tables)

### Gemma 4 E2B + `MLX_PERSISTENT_AB=1` — garbage output

```
$ MLX_METAL_ICB=1 MLX_METAL_AB=1 MLX_PERSISTENT_AB=1 \
    MLX_BENCH_MODEL=gemma4-e2b MLX_BENCH_METHOD=summarization \
    MLX_BENCH_KV=none MLX_BENCH_CONTEXT=1024 \
    swift test --skip-build -c release --filter benchmark
...
[BENCH] Output: -𒂃"{!} dhatunam𒍋𒅓 فونبToGoResult … setPenis ...
```

Gemma's RoPE path isn't wired to any `PersistentRopeFreqsAbHandle` —
only GPT-OSS's `AttentionBlock` reads `MLX_PERSISTENT_AB` and passes
a handle to `MLXFast.ropeAb`. With `MLX_PERSISTENT_AB=1` but no
handle, something is consuming the flag (RMSNorm? SDPA?) and leaving
a stale AB binding on replay. Fix path: plumb persistent-AB handles
through Gemma's `Gemma3nAttention` (or whatever the e2b attention
block is called) the way GPT-OSS does.

### GPT-OSS 20B + full ICB decode-loop at ctx = 1024 — garbage output

```
$ MLX_METAL_ICB=1 MLX_METAL_AB=1 MLX_PERSISTENT_AB=1 \
    MLX_ICB_DECODE_LOOP=1 MLX_ICB_RECORD_STEP=3 \
    MLX_BENCH_MODEL=gpt-oss-20b MLX_BENCH_METHOD=summarization \
    MLX_BENCH_KV=none MLX_BENCH_CONTEXT=1024 \
    swift test --skip-build -c release --filter benchmark
...
[ICB-DECODE] captured 1529 commands, 650 segments, 1631 pin slots,
  handles: 24 SDPA / 0 RoPE / 48 RoPE(freqs)
[BENCH] Generation: 38.9 tok/s (400 tokens)
[BENCH] Output: <|channel|>analysis<|message|>We!!!!!!!!!!!…
```

The just-fixed "WeWeWe" loop (persistent gather AB + step-3 MXFP4
dequant reorder) works at short contexts — `simple` method with
~ 101 prompt tokens produces token-exact match with live greedy —
but regresses at ctx = 1024. Likely a `RotatingKVCache` × record-time
offset interaction: once the sliding-window is exceeded, cache
rotation starts, and our record-time slice_update offsets don't
adapt to the rotated position seen at replay. The same replay path
is fine when no rotation has happened yet.

This is a gating blocker for shipping the decode-loop ICB on real
workloads.

## Recommendation

Priority order for follow-ups:

1. **Decode-loop ICB @ ctx = 1024 fix** — RotatingKVCache × record-
   time offset. Required to ship the decode-loop work.
2. **Baseline ≈ 7 % regression** — gate `supportIndirectCommandBuffers=YES`
   and the dispatch counter behind pipeline-build-time env or API
   so alpha-parity returns for non-ICB code paths.
3. **Gemma 4 E2B PersistentAb wiring** — mirror GPT-OSS's
   `AttentionBlock` on Gemma's e2b attention so `MLX_PERSISTENT_AB`
   is either a no-op (safe) or a real win.
4. **Per-op AB gating for Gemma** — revisit once Gemma is on
   PersistentAb; transient-AB allocation dominates the current
   Gemma AB regression.
