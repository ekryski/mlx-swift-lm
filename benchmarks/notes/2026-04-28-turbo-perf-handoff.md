# TurboQuant Perf — Next Session Handoff (2026-04-28)

Session goal: close the B-vs-A decode tok/s gap from yesterday's 2026-04-27 doc. Specifically targeting "B path within 5% of A path".

## What we shipped

Single commit on `ek/turbo-kv-perf` (mlx-swift-lm): `2201fda perf(turbo-kv): adaptive flash block size + pre-scaled Q rotation`.

1. **Adaptive TurboFlash block size (H5).** Replaced the static `flashBlockSize = 64` with `adaptiveBlockSize(tokenCount:)` that targets ~32 blocks (was implicitly ~64). Power-of-2, clamped to [16, 256]. M1 Max sweep (Qwen 0.8B turbo4v2, summarization, 200 tok):
   - 1k → block 32: **151 tok/s** (vs 137 @ static 64, **+10%**)
   - 4k → block 128: **124 tok/s** (vs 121 @ static 64, +2-3%)
   - 32k → block 256: **79 tok/s** (vs 53 @ static 64, **+49%**)

   `TURBO_FLASH_BLOCK_SIZE=N` still overrides for one-off measurement (now cached at first read instead of read-per-call — was a 50× perf trap on the hot path).

2. **Pre-scaled Q rotation** (cherry-picked from Tom's PR #93 commit `87822f5`). `qRot = prepareQueries(q) * scale` was matmul + multiply per layer per decode step. New `prepareQueriesScaled(q, scale)` folds `scale` into the cached `(rotationT * scale)` matrix at first use and returns one matmul. Saves ~2 µs/call CPU dispatch + the eltwise pass on GPU.

## Profiling correction

The 2026-04-27 handoff cited an H7 reading: "encode dominates B-path GPU time ~9:1 over the flash kernel itself."

**This does not reproduce on the current pin.** This session's measurement on Qwen 0.8B / 4k summarization, 200 tok:

| Config | decode_steady | tq_encode (CPU dispatch) |
|---|:---:|:---:|
| With encode | 8.23 ms/tok | 33 µs/call (~80 ms total) |
| Encode bypassed (`TURBO_NO_ENCODE=1` debug gate) | 8.03 ms/tok | — |
| Δ | 0.20 ms/tok | ~0.6 ms total |

Encode is **2-3 % of decode time**, not 80-90 %. The TurboFlash kernel itself is now the dominant B-path cost. The H7 reading was likely on an earlier mlx pin where the encode kernel was less efficient, and has since been amortized away.

## H9 (deferred batch encode) — implemented, REGRESSED

Built the full deferred-encode infrastructure: `pendingRawKeysBuf` / `pendingRawValuesBuf` ring of `flushIntervalCached` slots, `appendPendingRaw()` + `flushPendingEncode()`, two-stage merged-softmax attention path (separated `mseScore` + raw-K matmul, concat scores, softmax, `mseWeightedSum` on compressed prefix + raw matmul on pending suffix). Tested on Qwen 0.8B/9B and Qwen 9B at 1k/4k/32k.

**Result**: net regression at every cell tested. The merge attention path is significantly slower than the fused TurboFlash kernel — the encode-dispatch savings (~20 ms total over 200 decode tokens) don't make up for paying the separated path on every attention call.

```
ctx=32768 (32k linear-regime, ideal H9 case):
  H5 only:           79 tok/s
  H5 + H9 (FLUSH=64): 52 tok/s   (-34%)
```

The H9 code is **NOT** committed. Files reverted before commit; the implementation lived through 5 iterations during this session before being abandoned.

If H9 is to come back, it needs **a custom merge kernel** that does the two-stage online softmax in one Metal dispatch instead of mseScore + raw matmul + concat + softmax + mseWeightedSum + raw matmul + add. That's an mlx fork + metallib regen; out of scope for an mlx-swift-lm-only PR.

## Bulk dequant + MLXFast SDPA — also explored, REGRESSED

Wrote `bulkDequantRotated()` using MLX broadcast bit-shift unpacking (no per-dim Swift loop) → codebook gather → norm scale → cast. Then `MLXFast.scaledDotProductAttention(qRot, dequantedK, dequantedV)`. Idea: trade temporary FP16 K/V memory (~4 MB/layer at 4k) for one fast SDPA call on Apple's tuned kernel.

**Result**: also slower than TurboFlash. The bulk-dequant pipeline is 5+ MLX op dispatches (expand_dims, shift, mask, reshape, asType, gather, expand_dims, multiply, asType) per K and per V — the dispatch cost dominates, and SDPA's win over TurboFlash isn't enough.

Code reverted before commit.

## Where we are vs A path

After H5 + pre-scaled rotation (qwen35-0.8b, summarization 200 tok):

| Ctx | A | B | Δ |
|---|:---:|:---:|:---:|
| 1k  | 182 | 143 | -21% |
| 4k  | 177 | 110-124 | -25 to -30% |
| 32k | 97  | 73  | -25% |

Other models with H5 + pre-scaled (100 tok):

| Model | Ctx | A | B | Δ |
|---|---|:---:|:---:|:---:|
| qwen35-9b | 1k | 48 | 44 | -8% |
| qwen35-9b | 4k | 45 | 40 | -12% |
| nemotron-30b-a3b | 1k | 69 | 63 | -8% |
| nemotron-30b-a3b | 4k | 67 | 61 | -9% |

**Larger models are within ~10 % of A — the gap is real but small.** The dramatic gap on Qwen 0.8B is because attention dominates total decode time on a tiny hybrid model (few attn layers, light MLP). For models with substantial GDN / MoE / MLP work, the constant TurboFlash overhead is amortized.

## What would actually close the small-model gap

1. **A custom Metal kernel for fused dequant → matrix-engine SDPA.** The TurboFlash kernel does scalar dequant + dot-product per token; Apple's matrix engine can hammer FP16 dot products much faster but it can't read packed indices directly. A 2-step kernel chain — bulk dequant (memory-bound, ~30 µs at 4k) + MLXFast SDPA (matrix-engine, ~500 µs) — would beat TurboFlash if and only if the dequant fits in one Metal dispatch. The pure-MLX implementation we tried this session uses 9 separate dispatches and that overhead kills it. **Worth a focused day of Metal work in the mlx fork.**
2. **NR0 > 2.** Currently only NR0=2 is instantiated. NR0=4 / NR0=8 might amortize the per-block KV dequant across more queries on small models (totalQ = nQH × L = 8 × 1 = 8 on Qwen 0.8B, easily divisible). Requires adding instantiations to `turbo_flash.metal` in the mlx fork.
3. **Profile the kernel itself with Instruments / xctrace.** Figure out *why* TurboFlash is 2.5× slower than MLXFast SDPA on the same nominal work. Maybe occupancy, register spill, or memory access patterns can be improved without an architectural rewrite.

## Branches and PRs

| Repo | Branch | PR | Status |
|---|---|---|---|
| ekryski/mlx-swift-lm | `ek/turbo-kv-perf` | [#107](https://github.com/ekryski/mlx-swift-lm/pull/107) | adds adaptive block size + pre-scaled Q on top of yesterday's content |

No mlx / mlx-swift changes this session — pin unchanged.

## Local working-tree state at session end

`Libraries/MLXLMCommon/TurboQuantKVCache.swift` has the testing-only `TURBO_USE_BETA` env override re-applied (uncommitted) so B path can be measured without Tom's `--kv …-compact` CLI suffix landing first. Should be reverted before merging #107.

```
+        // B-path testing override — set TURBO_USE_BETA=1 to flip default
+        // `useCompressedAttention=false` to true. Used during the handoff
+        // window before Tom's #99 (`--kv turbo4v2-compact` CLI suffix) lands.
+        let envBeta = (ProcessInfo.processInfo.environment["TURBO_USE_BETA"] ?? "") == "1"
+        self.useCompressedAttention = envBeta || useCompressedAttention
```

## Where to start next session

```bash
git checkout ek/turbo-kv-perf && git pull
git log --oneline -3
# 2201fda perf(turbo-kv): adaptive flash block size + pre-scaled Q rotation
# f8f9996 docs(turbo-kv): next-session handoff doc

# Sanity check: H5 wins at 32k.
TURBO_USE_BETA=1 MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=32768 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady
# Should see ~73-79 tok/s. Pre-H5 baseline at this cell was 52.
```

Then start on item 1 from "What would actually close the small-model gap" — that's the only path to <5 %, and it's an mlx-fork + metallib job, not pure mlx-swift-lm.
