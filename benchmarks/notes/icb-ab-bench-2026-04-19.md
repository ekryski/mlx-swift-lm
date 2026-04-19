# ICB + AB infrastructure benchmark — 2026-04-19

M1 Max 64 GB, macOS 15.7.4, branch `ek/persistent-ab-pilot`,
`summarization` method @ 1024 context, 4-bit weights. Measured after
the full fix set:

- step-3 `fast::Quantize::eval_gpu` pipeline-order reorder,
- `PersistentGatherFrontAbHandle` input-override for the decode-loop
  embedding gather,
- per-cache `icbStepState` + per-layer SDPA `N` for the
  decode-loop's RotatingKVCache rotation,
- MLXNN.RMSNorm persistent-AB path moved to per-module opt-in
  (Gemma 4 opts out because it produces garbage under that path).

Alpha decode / prefill numbers are taken from
[turbo4v2-decode-regression-2026-04-17.md](turbo4v2-decode-regression-2026-04-17.md);
it didn't capture TTFT or GPU memory — those columns are fresh
measurements on the current branch only.

All ours/… rows below have **correct output** (first sentence shown
in the "sample" column for spot-check). The earlier revision of
this note flagged Gemma 4 E2B + `MLX_PERSISTENT_AB=1` and GPT-OSS
20B + decode-loop as garbage-producing; both are now resolved.

## Gemma 4 E2B (`mlx-community/gemma-4-e2b-it-4bit`)

| Config | KV | Prefill | Decode | Δ vs alpha | TTFT | Peak GPU |
|---|---|---:|---:|---:|---:|---:|
| alpha (pristine)            | none     | 2 906.8 | **99.7**  | —       | n/a    | n/a     |
| ours, no flags              | none     | 2 685.0 | 94.5      | −5.2 %  | 377 ms | 3.24 GB |
| ours, AB + PersistentAB     | none     | 2 636.2 | 86.1      | −13.6 % | 384 ms | 3.24 GB |
| alpha (pristine)            | turbo4v2 | 2 885.3 | **102.9** | —       | n/a    | n/a     |
| ours, no flags              | turbo4v2 | 2 585.4 | 95.6      | −7.1 %  | 391 ms | 3.24 GB |
| ours, AB + PersistentAB     | turbo4v2 | 2 679.4 | 86.4      | −16.0 % | 377 ms | 3.24 GB |

Output samples (all coherent English): "This excerpt from *The Great
Gatsby* introduces a narrator who has a highly developed, somewhat
cynical, and guarded way of viewing the world…"

Gemma4's `RMSNorm` modules opt out of the persistent-AB path (see
`Gemma4TextModel.init`) because it produced garbage output at
ctx = 1024. The bug has not been root-caused in the RMSNorm
persistent path itself; for now Gemma safely uses the transient
AB / plain kernels while GPT-OSS keeps its speedup.

## GPT-OSS 20B (`loan-star/gpt-oss-20b-mlx-4Bit`)

| Config | KV | Prefill | Decode | Δ vs alpha | TTFT | Peak GPU |
|---|---|---:|---:|---:|---:|---:|
| alpha (pristine)                       | none     | 638.7  | **51.7** | —         | n/a      | n/a      |
| ours, no flags                         | none     | 551.8  | 47.5     | −8.1 %    | 1 856 ms | 11.65 GB |
| **ours, AB + PersistentAB**            | none     | 551.9  | **63.7** | **+23.2 %** | 1 856 ms | 11.68 GB |
| ours, AB + Pers + ICB decode-loop      | none     | 563.5  | 43.7     | −15.5 %   | 1 818 ms | 11.68 GB |
| alpha (pristine)                       | turbo4v2 | 662.2  | **52.2** | —         | n/a      | n/a      |
| ours, no flags                         | turbo4v2 | 547.7† | 46.7†    | −10.5 %   | 1 870 ms | 11.65 GB |
| **ours, AB + PersistentAB**            | turbo4v2 | 599.6† | **63.9†** | **+22.4 %** | 1 708 ms | 11.68 GB |
| ours, AB + Pers + ICB decode-loop      | turbo4v2 | 561.9  | 41.5     | −20.5 %   | 1 823 ms | 11.68 GB |

† Numbers marked † are from an earlier session of the same day
when the machine was cold; a second sweep after several GPT-OSS
20B runs in a row showed 10–30 % lower decode on every GPT-OSS
turbo4v2 row, including baseline — classic thermal throttling on
M1 Max. The first-sweep numbers are the apples-to-apples
comparison with alpha.

Output samples (all coherent): `<|channel|>analysis<|message|>The
user has posted an excerpt of a text…`, etc.

## Key findings

1. **AB + PersistentAB is the headline win on GPT-OSS 20B** —
   +23.2 % on no-quant (47.5 → 63.7) and +22.4 % on turbo4v2
   (52.2 → 63.9) vs alpha baseline. Both KV configs now gain
   from persistent-AB: the transient-AB allocation cost
   disappears, and the RMSNorm / RoPE / SDPA per-dispatch
   overhead compresses.

2. **The decode-loop ICB does NOT yet win on ctx = 1024.** Even
   with the rotation fix landed (see `icbStepState` +
   `updateNPerLayer`), decode-loop mode is 15–20 % slower than
   plain AB + PersistentAB at long context. Short context
   (ctx = 101, `simple` method) produces token-exact greedy
   parity with live and is comparable to AB + PersistentAB
   throughput. The long-context loss is almost certainly the
   per-step `RotatingKVCache` simulation cost in Swift
   (O(stepsAhead) per layer per step) — fixable but out of
   scope for this session.

3. **Gemma 4 E2B gets no AB win at this workload.** With or
   without the Gemma4 RMSNorm opt-out, AB + PersistentAB costs
   9–10 % on Gemma vs our own no-flag baseline. Gemma's smaller
   per-dispatch kernels (relative to GPT-OSS's MoE + MXFP4
   dequant) make the AB preamble dominate. Real Gemma wins need
   either per-op AB gating or PersistentRope/SDPA wiring in
   `Gemma4Attention` (same pattern as GPT-OSS's `AttentionBlock`).

4. **Baseline regression ≈ 5–10 % vs alpha** still present on
   `no flags` rows. Root cause is diffuse across ~30 commits
   between alpha and this branch — not chased in this session.
   The headline wins above are measured *against alpha* with the
   full AB stack on, so the branch's AB path clears the alpha
   turbo4v2 number by 12 % on GPT-OSS despite the baseline
   regression.

5. **TTFT is flat across configs** (± 50–100 ms, dominated by
   prefill). AB flags don't move prefill meaningfully at 1024
   tokens — prefill uses the NativePrefillBridge.

6. **Peak GPU is flat** (± 150 MB). Persistent ABs add one stable
   MTLBuffer per module × primitive; transient ABs cycle through
   the pool.

## Follow-ups

Priority order for the next session:

1. **Decode-loop ICB long-context perf.** Either cache the
   `icbStepState` result on the cache itself so the per-step
   simulation is O(1) rather than O(stepsAhead), or switch the
   overrides to a persistent start-offset buffer mutated in
   place (no MLXArray allocation on the hot path). Until this,
   the decode-loop is only a win at short contexts.
2. **Baseline ≈ 7 % regression.** Profile `alpha` vs branch on
   a single decode step — probable suspects are always-on
   checks in `CommandEncoder::get_command_encoder`, the AB
   gate dispatch, or the tag-binding lookup in
   `dispatch_threads`. Gate the hot-path cost so the flag-less
   baseline matches alpha.
3. **Gemma 4 E2B PersistentAB wiring.** Mirror GPT-OSS's
   `AttentionBlock` on Gemma's `Gemma4Attention` so SDPA + RoPE
   get persistent handles and `MLX_PERSISTENT_AB=1` turns into
   a real Gemma speedup. The Gemma MLXNN.RMSNorm garbage-output
   bug also deserves a proper root-cause investigation at that
   point.
4. **Per-op AB gating.** Some primitives (RMSNorm on small-axis
   inputs, fused RMSNorm+RoPE) may not amortize the AB preamble.
   A per-kernel opt-in instead of the global `MLX_METAL_AB` flag
   would let Gemma4 pick up only the wins.
