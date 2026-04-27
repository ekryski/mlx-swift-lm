# TurboQuant Perf — Next Session Handoff (2026-04-28)

Session goal: close the B-vs-A decode tok/s gap. Specifically targeting
"B path within 5% of A path." **Met on most cells of the larger-model
end of the matrix.**

## What we shipped

Three commits on `ek/turbo-kv-perf` (mlx-swift-lm only — no mlx /
mlx-swift submodule changes):

1. `2201fda` **adaptive flash block size + pre-scaled Q rotation**
2. `c5ca7a3` **fused bulk-dequant Metal kernel + matrix-engine SDPA path**
3. `a2d95bc` (this doc, originally early-session draft; now superseded)

### 1. Adaptive TurboFlash block size (H5)

Replaced static `flashBlockSize = 64` with `adaptiveBlockSize(tokenCount:)`
targeting ~32 blocks (power-of-2, clamped to [16, 256]). M1 Max sweep:

- 1k → block 32 (151 tok/s vs 137 @ static 64, +10 %)
- 4k → block 128 (124 vs 121, +2 %)
- 32k → block 256 (79 vs 53, **+49 %**)

Also fixed an env-var read on the kernel hot path that was a 50× perf
trap (`flashBlockSizeOverride` is now read once at static init).

### 2. Pre-scaled Q rotation

Cherry-picked from Tom's PR #93 commit `87822f5`. `prepareQueriesScaled`
folds `scale` into the cached `(rotationT * scale)` matrix at first
use, so per-decode-step Q rotation is one matmul instead of matmul +
elementwise multiply.

### 3. Fused bulk-dequant kernel + matrix-engine SDPA

The big win. Replaces the per-step TurboFlash kernel (which does scalar
bit-unpack + dot-product per token) with a two-stage pipeline:

1. **`turbo_bulk_dequant`** — JIT'd via `MLXFast.metalKernel` (no
   metallib regen). One thread per packed `uint32` word; each thread
   emits `32/bits` dim outputs ({16, 8, 4} for bits ∈ {2, 4, 8}). Bit
   unpacking + codebook gather + norm scaling fused into a single
   dispatch. Writes BF16/FP16 output in rotated codec space.

2. **`MLXFast.scaledDotProductAttention`** — same matrix-engine kernel
   A path uses, on the dequanted FP16 K/V.

Trade-off: temporary `B*H*T*D*2`-byte FP16 K/V is allocated per layer
per decode step (freed after SDPA). Memory pressure at 4 K context is
~8 MB / layer / step — well within bounds; falls back transparently
inside MLX's allocator.

Default-on for `keyBits ∈ {2, 4, 8}` and `valueBits ∈ {2, 4, 8}` (all
shipping turbo schemes). `TURBO_DEQUANT_SDPA=0` falls back to
TurboFlash for A/B comparison.

## Final B vs A on M1 Max 64GB (turbo4v2 summarization)

| Model              | 1k          | 4k         | 8k         | 16k        | 32k        |
|--------------------|------------:|-----------:|-----------:|-----------:|-----------:|
| Qwen 0.8B  (A)     | 199.4       | 186.3      | 158.9      | 126.9      | 97.3       |
| Qwen 0.8B  (B)     | 157.1       | 166.2      | 147.1      | 120.9      | 90.3       |
| Δ                  | -21 %       | -11 %      | -7 %       | **-5 %**   | -7 %       |
| Qwen 9B    (A)     | 52.2        | 50.1       | 46.5       | 41.7       | 41.7       |
| Qwen 9B    (B)     | **53.0**    | 48.8       | 45.6       | 40.4       | 34.1       |
| Δ                  | **+1.5 %**  | **-2.6 %** | **-1.9 %** | **-3.1 %** | -18 %      |
| Nemotron 30B-A3B (A)| 74.6       | 72.9       | 71.0       | (TBD)      | (TBD)      |
| Nemotron 30B-A3B (B)| 69.7       | 68.8       | 66.3       | (TBD)      | (TBD)      |
| Δ                  | -6.6 %      | -5.6 %     | -6.6 %     | (TBD)      | (TBD)      |

**Within 5 %** at the bolded cells. Qwen 9B at 1 k actually decodes
*faster* on the compressed cache than on raw fp16 — within run-to-run
noise, but a striking inversion of the prior 30 % gap.

## Why the residual gap differs by model

- **Qwen 0.8B** (hybrid GDN, ~6 attention layers): attention is a
  small fraction of total decode time. The per-step constant overhead
  in B path (Q rotation matmul + V output rotation matmul + 2 dequant
  dispatches + SDPA dispatch) dilates more relative to the cheap
  attention compute. Closing this further needs model-aware fusion —
  e.g. folding `valueRotation` into the output projection's `Wo`
  weight matrix at load time. **Out of scope for the cache layer.**

- **Qwen 9B** (24 attention layers, no GDN): attention dominates. The
  matrix-engine SDPA is fast enough that it compensates for the
  dequant cost. Within ~3 % of A path through 16 k.

- **Qwen 9B at 32 k**: `B*H*T*D*2 = 1*8*32768*128*2 ≈ 67 MB` per layer
  per step of FP16 K (and the same for V). Memory bandwidth becomes
  the bottleneck. At 64 k it'd likely tip back below TurboFlash.
  **TODO**: re-introduce a `tokenCount` threshold for switching back
  to TurboFlash above ~24 k on larger models.

- **Nemotron 30B-A3B**: roughly tracks Qwen 9B trends; not fully
  characterized this session due to bench timeout on long contexts.

## Profiling correction (carried forward from 04-27)

H7's prior "encode dominates 9:1 over flash" reading does not
reproduce on the current pin. Bypassing the encode dispatch entirely
(`TURBO_NO_ENCODE=1` debug gate, removed before commit) saves only
~0.2 ms of an 8 ms decode step on Qwen 0.8B / 4 k — encode is
~2-3 % of decode, not 80-90 %. The TurboFlash kernel itself was the
dominant B-path cost, which is what the new dequant+SDPA path bypasses.

## What was tried and abandoned

- **H9 deferred batch encode** (full implementation: pending ring,
  two-stage merged-softmax attention). The merge attention path is
  inherently slower than fused TurboFlash, so amortizing encode loses
  ground. Reverted before commit.

- **Bulk dequant via pure MLX broadcast ops** (no Metal). 9 dispatches
  per K and per V (expand_dims, shift, mask, reshape, asType, gather,
  expand_dims, multiply, asType) — dispatch overhead killed it.
  Reverted; replaced by the single Metal kernel that ships in c5ca7a3.

- **Threadgroup-shared `norms` load** in the dequant kernel. Saves one
  fp32 load per thread, but the `threadgroup_barrier` overhead eats
  the savings. Net regression of 1-2 % at 4 k. Reverted.

- **Tuning `TURBO_SPARSE_V_THRESHOLD`**. Investigated; the threshold
  only affects the *separated* `mseWeightedSum` kernel (used when no
  TurboFlash kernel matches the bits combo), not the L=1 decode
  fast path. Doesn't help the cells we care about. Skipped.

## What's left for next session

In rough decreasing order of expected payoff:

1. **Smart switching back to TurboFlash on long context for large
   models.** Re-introduce the `tokenCount > THRESHOLD` gate but at a
   higher threshold than the 8 k we tried mid-session — somewhere
   around 24-32 k for 9B and above. Closes the Qwen 9B 32 k -18 %
   regression.

2. **Fold V output rotation into the model's output projection.**
   The current code does `output = matmul(rotated_attn_output,
   valueMSECodec.rotation)` after SDPA. If the model's `Wo` matrix
   is pre-multiplied by `V_rotation` at codec init time, this matmul
   disappears entirely. Saves one matmul per layer per decode step.
   Closes the Qwen 0.8B residual gap (small-model overhead is
   constant per layer, so even one matmul matters).

3. **Async prefill compression** (carried over). `compressRawCache`
   currently runs synchronously inside the *first* decode call,
   inflating user-visible TTFT. Kicking it off concurrently with the
   first decode forward pass would shrink TTFT without affecting
   steady-state decode tok/s.

4. **Eliminate GQA `tile` in the rawKeyMode path**. `MLX.tiled(K,
   [1,1,nRepeats,1,1])` allocates a full repeated K. The non-rawKey
   path now goes through MLXFast SDPA which handles GQA natively;
   rawKeyMode still tiles. Would help turbo0v4 / turbo0v8 schemes.

5. **N-gram speculative decoding** (deferred per user instruction).
   Wired via `MLX_BENCH_NGRAM`. Should compose cleanly with the new
   dequant+SDPA path; multiplies decode tok/s by accept-rate.

6. **Trace inspection of the dequant kernel.** xctrace recorded one
   trace with `Metal System Trace + os_signpost` template at
   `/tmp/mlx-prof/turbo-b.trace` — open in Instruments to see
   per-kernel GPU times and look for low-occupancy / register-spill
   issues. CLI export only returns schema, not row data.

## Branches and PRs

| Repo | Branch | PR | Status |
|---|---|---|---|
| ekryski/mlx-swift-lm | `ek/turbo-kv-perf` | [#107](https://github.com/ekryski/mlx-swift-lm/pull/107) | adds H5, pre-scaled Q, fused dequant kernel + SDPA path |

No mlx / mlx-swift changes — pin unchanged.

## Local working-tree state at session end

`Libraries/MLXLMCommon/TurboQuantKVCache.swift` has the testing-only
`TURBO_USE_BETA` env override re-applied (uncommitted) so B path can
be measured without Tom's `--kv …-compact` CLI suffix landing first.
Should be reverted before merging #107.

## Where to start next session

```bash
git checkout ek/turbo-kv-perf && git pull
git log --oneline -4
# c5ca7a3 perf(turbo-kv): fused bulk-dequant kernel + matrix-engine SDPA path
# a2d95bc docs(turbo-kv): 2026-04-28 perf session handoff (superseded by this doc)
# 2201fda perf(turbo-kv): adaptive flash block size + pre-scaled Q rotation
# f8f9996 docs(turbo-kv): next-session handoff doc

# Sanity check: Qwen 9B 4k should land within ~3 % of A path.
TURBO_USE_BETA=1 MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady
# Should see ~48 tok/s. A-path baseline at this cell is ~50 tok/s.
```

Then start on item 1 (long-context fallback to TurboFlash) for the
Qwen 9B 32k regression.
