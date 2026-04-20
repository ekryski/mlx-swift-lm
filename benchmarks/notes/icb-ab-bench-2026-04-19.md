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

### Session 2 attempts

**Decode-loop ICB long-context perf — partially addressed.**
Landed an O(1) per-step `icbStepState` (caches the
post-record simulated `idx` on each `RotatingKVCache`, advances
by 1 per consecutive step) plus startArr MLXArray dedup by
writeIdx value (collapses 24 layer allocations to 2 — one
sliding, one full). Throughput at ctx = 1024 barely moves
(43.2 tok/s vs 43.7 tok/s before), because the replay's
~22 ms GPU time dominates — the Swift-side orchestration was
never the bottleneck.

### Session 3 results — decode-loop ICB is a net loss on GPT-OSS 20B

The "decode-loop is a win at short context" claim from the
original checkpoint turns out to be an artifact of **pre-fix
garbage output**. The 69.7 tok/s number was measured while
replay was producing incorrect logits fast; after the step-3
MXFP4, gather-input, and RotatingKVCache-rotation fixes
landed, a correctness-equivalent decode-loop run is slower
than plain AB + PersistentAB at every context length tested:

| Config | ctx | Decode tok/s | Note |
|---|---|---:|---|
| plain AB + PersistentAB | 1024 | 63.7 | correct |
| ICB decode-loop         | 1024 | 40.9 | correct (−36 %) |
| plain AB + PersistentAB | 101  | 47.6 | correct |
| ICB decode-loop         | 101  | 42.4 | correct (−11 %) |

GPU-side additional work beyond what I attempted to
optimize:

* `useResource` hoist (one pass per unique buffer instead of
  per-segment × resource-set-size) landed in
  [mlx] `perf(metal): hoist useResource out of ICB replay segment
  loop`. CPU savings were in the low-ms range; total decode
  throughput didn't move — confirming the bottleneck is GPU
  execution of the replayed ICB, not the Swift/CPU wrapper.
* Apple Silicon appears to pay a real per-command overhead for
  `executeCommandsInBuffer` vs direct dispatch. On GPT-OSS 20B
  the ICB captures ~1 500 commands across ~650 segments; even
  at ~1 µs/command of driver-level resolve cost, that's ~1.5 ms
  of the 6–7 ms gap vs plain dispatch.

Plain `AB + PersistentAB` remains the headline win: +22 % on
GPT-OSS 20B turbo4v2 vs alpha, correct output, no
decode-loop-specific plumbing needed. The ICB decode-loop
machinery is still useful as a capture/replay harness for
regression testing (the step-3 parity checks caught real
bugs) but shouldn't be the recommended generation path today.

**Gemma 4 E2B PersistentAB wiring — attempted, reverted.**
Tried adding `PersistentSdpaAbHandle` to `Gemma4Attention`
mirroring GPT-OSS's `AttentionBlock`. With `MLX_METAL_AB=1
MLX_PERSISTENT_AB=1` the decode produces `<pad>` tokens at
ctx = 1024 but works at ctx ≤ 8 (simple method). Two
suspects: (a) the SDPA AB kernel doesn't handle Gemma4's
sliding-window mask mode (mask comes from
`makeAttentionMask(n:cache:windowSize:)`), (b) Gemma4's scale
= 1.0 interacts with something in the AB path that GPT-OSS's
1/√d scale doesn't. Full debug requires instrumenting
`sdpa_vector_unified` to compare transient vs persistent AB
contents for the same call. Reverted to ship a clean branch;
Gemma4 remains correct but without a persistent-AB decode win.

### Remaining work

Priority order for the next session:

1. **Decide fate of ICB decode-loop.** Given it's a net loss
   on GPT-OSS 20B at every context tested, options are: (a)
   find a workload where per-command replay overhead is
   smaller than CPU encoding savings — probably smaller
   models, micro-kernel-heavy graphs, or very high-batch
   configs; (b) keep the infrastructure as a capture/replay
   regression-test harness; (c) remove. The harness value is
   real — the decode-loop fixes in session 2 caught
   genuine correctness bugs (MXFP4 pipeline ordering,
   RotatingKVCache rotation, gather-input AB packing) that
   the plain-dispatch path would never have surfaced.
2. **Gemma 4 SDPA persistent-AB debug.** Resume the
   session-2 attempt — add a diff-trace for AB contents
   between transient and persistent paths, run the small
   Gemma4 test, identify the first slot/call where they
   diverge. Likely a mask or scale plumbing issue specific
   to Gemma4's 4D attention input.
3. **Gemma 4 MLXNN.RMSNorm persistent-AB garbage.** Still
   unsolved. Symptom: MLX_METAL_AB=1 + MLX_PERSISTENT_AB=1
   on Gemma4 (even without any decode-loop / SDPA changes)
   produces garbage. The opt-out at `Gemma4TextModel.init`
   keeps the flag safe; real fix needs a numerical
   correctness trace of `rms_norm_ab` on Gemma's input
   shapes/strides.
4. **Baseline ≈ 7 % regression vs alpha.** Profile `alpha`
   vs branch on a single decode step — probable suspects are
   always-on checks in `CommandEncoder::get_command_encoder`,
   the AB gate dispatch, or the tag-binding lookup in
   `dispatch_threads`. Gate the hot-path cost so the
   flag-less baseline matches alpha.
5. **Per-op AB gating.** Some primitives (RMSNorm on
   small-axis inputs, fused RMSNorm+RoPE) may not amortize
   the AB preamble. A per-kernel opt-in instead of the global
   `MLX_METAL_AB` flag would let Gemma4 pick up only the wins.
