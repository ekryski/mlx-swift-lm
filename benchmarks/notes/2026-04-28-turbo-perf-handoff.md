# TurboQuant Perf — Next Session Handoff (2026-04-28)

Session goal: close the B-vs-A decode tok/s gap. Specifically targeting
"B path within 5% of A path." **Met on most cells of the larger-model
end of the matrix.**

## What we shipped

Five commits on `ek/turbo-kv-perf` (mlx-swift-lm only — no mlx /
mlx-swift submodule changes):

1. `2201fda` **adaptive flash block size + pre-scaled Q rotation**
2. `c5ca7a3` **fused bulk-dequant Metal kernel + matrix-engine SDPA path**
3. `b9bfd8b` chore: temporary `TURBO_USE_BETA` env override (superseded)
4. `11d815d` **B path is now the default** (`useCompressedAttention=true`,
   `TURBO_USE_ALPHA=1` forces A, sinks-using models auto-fallback to A,
   README documents all non-bench knobs)
5. `a2d95bc` / `3d9e958` (this doc; latest revision)

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

> Numbers below were captured before commit `11d815d` flipped B to the
> default. The kernel work hasn't changed since, so the per-cell
> tok/s figures still hold; the only thing that's different is that
> the bench command no longer needs `TURBO_USE_BETA=1` to reach B
> (and now needs `TURBO_USE_ALPHA=1` to opt back to A).


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

### Up next (this is the priority order — do these before the longer
### "follow-up" backlog further down)

1. **Redo the regression sweep on the rebased branch.** The current
   sweep started under the old `memoryBytes` bug — the rerun should
   land on `ek/turbo-kv-perf` now that it's rebased on alpha (which
   includes Tom's paged foundation from PR #110) and has the
   `memoryBytes` fix (`b307346`) + the precompiled bulk-dequant
   kernel + the `--prefill-chunk` knob.

   Run:

   ```bash
   ./scripts/benchmark.sh \
     --model qwen35-2b,qwen35-35b-a3b,gpt-oss-20b,nemotron-30b-a3b,gemma4-e2b,gemma4-26b-a4b \
     --method summarization \
     --context 1024,4096,16384,65536 \
     --quant 4bit \
     --kv none,turbo4v2,affine4 \
     --ppl
   ```

   Then move the resulting markdown to `*-bpath.md`, re-run with
   `TURBO_USE_ALPHA=1` for `--kv turbo4v2` only, move the result to
   `*-apath.md`. **If the regression sweep is clean, that's the merge
   gate for #107 + the cross-repo PR set.**

2. **Land [mlx-swift-lm #107](https://github.com/ekryski/mlx-swift-lm/pull/107)
   plus the cross-repo PRs that go with it:**
     - [ekryski/mlx#18](https://github.com/ekryski/mlx/pull/18) — Metal kernel + Primitive + public API
     - [ekryski/mlx-c#9](https://github.com/ekryski/mlx-c/pull/9) — C binding (new this session)
     - [ekryski/mlx-swift#19](https://github.com/ekryski/mlx-swift/pull/19) — Swift binding + submodule bumps + `mlx-generated/metal/turbo_quant.metal`
     - mlx-swift-lm #107 itself (B-path-default + dequant+SDPA + adaptive flash + memoryBytes fix + `--prefill-chunk` knob)

   Merge order: **mlx → mlx-c → mlx-swift → mlx-swift-lm**.

3. **Land [mlx-swift-lm #100](https://github.com/ekryski/mlx-swift-lm/pull/100) — Tom's "fully batched decode + GDN contiguous fix" for Qwen3Next/3.5/3.6.**
   Independent of the perf work in #107; lands once review is clean.

4. **Cut a fresh perf branch off `alpha`** (suggested name
   `ek/turbo-kv-perf-2`) and start the next round of speed-ups +
   memory-reduction work. Two work streams to develop **side-by-side**
   on this branch:

   **(a) Async prefill compression** (carried over from item 4 in
   the legacy backlog below). `compressRawCache` currently runs
   synchronously inside the *first* decode call, inflating user-visible
   TTFT. Kick it off concurrently with the first-token forward pass to
   shrink TTFT without affecting steady-state decode tok/s.

   **(b) Wire in the paged-KV stack from PR #110.** What landed is the
   data-structure foundation only (`PagedKVCache` + `BlockAllocator` +
   tests for forward equivalence vs `KVCacheSimple`). Tom explicitly
   left out of scope: the **Metal paged kernel** that replaces the
   `gather()` decode path, the **model integration** that routes a
   transformer's attention through `PagedKVCache`, and the
   **TurboQuant + paged integration** (paged blocks of compressed
   K/V). All three need to ship for paged to actually move peak GPU.

5. **Once #107 + the cross-repo set + #100 + #110 are all merged,**
   drop into the `ek/turbo-kv-perf-2` branch and pick from the
   legacy backlog below in the original priority order.

   Plus consider — also from [#93](https://github.com/ekryski/mlx-swift-lm/pull/93),
   commit `60bd16d` — a **persistent FP16 dequant cache** alternative
   to our per-step bulk-dequant. Tom's design keeps the dequanted
   K/V resident between decode steps and only appends the new token,
   saving the repeated dequant work each step. The trade-off: it
   doubles steady-state memory (compressed cache + FP16 dequant
   cache co-resident), partially defeating the compression. Worth a
   gated implementation with a memory-budget threshold: dequant-cache
   below a context size, fall back to per-step bulk-dequant above.

### Done in this session (was item 1 last session)

**Port `turbo_bulk_dequant` from JIT to a precompiled Metal kernel —
shipped.** Lives in:

- `mlx/backend/metal/kernels/turbo_quant.metal` —
  `turbo_dequant_rotated<Bits, Dim, PackedWidth, T>` instantiated for
  `bits ∈ {2, 3, 4, 8}` × `dim ∈ {64, 80, 96, 128, 256, 512}` ×
  `T ∈ {bfloat, half}`.
- `mlx/backend/metal/turbo_quant.cpp` + `mlx/fast.cpp` + `mlx/fast.h`
  — host dispatch + public `turbo_bulk_dequant_rotated`.
- `mlx-c/mlx/c/fast.{cpp,h}` + `Source/Cmlx/include/mlx/c/fast.h` —
  C binding `mlx_fast_turbo_bulk_dequant_rotated`.
- `mlx-swift`'s `MLXFast.swift` — Swift binding `MLXFast.turboBulkDequantRotated`.
- `mlx-swift-lm`'s `bulkDequantRotated` now routes to the precompiled
  kernel by default; the JIT version stays behind `TURBO_DEQUANT_JIT=1`
  for A/B regression checking.

Bench (Qwen 0.8B/9B turbo4v2 summarization, M1 Max 64GB, 60 tok):
precompiled is +3-5 % over the JIT'd `MLXFast.metalKernel` equivalent
on Qwen 0.8B short ctx, within run-to-run noise on Qwen 9B and on long
ctx. The bigger benefit is that precompiled avoids the first-dispatch
PSO compile inside user-visible TTFT.

### Legacy backlog (do these on `ek/turbo-kv-perf-2` after the four items above)

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

3. **Eliminate GQA `tile` in the rawKeyMode path**. `MLX.tiled(K,
   [1,1,nRepeats,1,1])` allocates a full repeated K. The non-rawKey
   path now goes through MLXFast SDPA which handles GQA natively;
   rawKeyMode still tiles. Would help turbo0v4 / turbo0v8 schemes.

4. **N-gram speculative decoding** (deferred per user instruction).
   Wired via `MLX_BENCH_NGRAM`. Should compose cleanly with the new
   dequant+SDPA path; multiplies decode tok/s by accept-rate.

7. **Trace inspection of the dequant kernel.** xctrace recorded one
   trace with `Metal System Trace + os_signpost` template at
   `/tmp/mlx-prof/turbo-b.trace` — open in Instruments to see
   per-kernel GPU times and look for low-occupancy / register-spill
   issues. CLI export only returns schema, not row data.

8. **A/B test f32 rotation precision** (cherry-pick from Tom's
   [#93](https://github.com/ekryski/mlx-swift-lm/pull/93) commit
   `6e6de3b`). The current codec keeps `rotation` and `rotationT`
   in bf16 (`MSECodec.init` does `whtRot.asType(.bfloat16)`). Tom
   reported a +0.3 PPL improvement on Qwen 9B from staying in f32 —
   bf16 rounding in the 256×256 hadamard rotation can compound
   across layers. Surgical change: drop the two `.asType(.bfloat16)`
   casts in `Libraries/MLXLMCommon/TurboQuantKVCache.swift:553` and
   `:558`. Low priority — the regression is small and PPLs on this
   branch already look fine in spot checks. Worth a clean A/B if /
   when we hit a quality regression; in the meantime the bf16
   rotation buys a small matmul-throughput edge worth keeping.

## Branches and PRs

| Repo | Branch | PR | Status |
|---|---|---|---|
| ekryski/mlx-swift-lm | `ek/turbo-kv-perf` | [#107](https://github.com/ekryski/mlx-swift-lm/pull/107) | adds H5, pre-scaled Q, fused dequant kernel + SDPA path |

No mlx / mlx-swift changes — pin unchanged.

## Local working-tree state at session end

Clean. As of `11d815d`, `--kv turbo*` already routes to B path by
default — no uncommitted overrides needed. The historic `TURBO_USE_BETA`
hack was committed and then replaced by `TURBO_USE_ALPHA` in the same
PR; `TURBO_USE_BETA` is no longer recognized.

## Where to start next session

```bash
git checkout ek/turbo-kv-perf && git pull
git log --oneline -7
# 11d815d feat(turbo-kv): make B path the default, add TURBO_USE_ALPHA override
# b9bfd8b chore(turbo-kv): TURBO_USE_BETA env override (superseded)
# 3d9e958 docs(turbo-kv): update handoff with fused dequant+SDPA results
# c5ca7a3 perf(turbo-kv): fused bulk-dequant kernel + matrix-engine SDPA path
# a2d95bc docs(turbo-kv): 2026-04-28 perf session handoff (superseded)
# 2201fda perf(turbo-kv): adaptive flash block size + pre-scaled Q rotation
# f8f9996 docs(turbo-kv): next-session handoff doc
```

### Sanity-check decode tok/s (B is the default — no env var needed)

```bash
# Qwen 9B 4k — B path. Should land within ~3 % of A path baseline.
MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady
# Expect ~48 tok/s. tq_value signpost fires (B path active).
```

### A vs B comparison (post-flip)

The default-B flip means an A/B comparison is now `TURBO_USE_ALPHA=1`
vs unset, *not* `TURBO_USE_BETA=1` vs unset. Easy to flip:

```bash
# B path (default): no env var needed.
MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady

# A path (raw fp16): set TURBO_USE_ALPHA=1.
TURBO_USE_ALPHA=1 MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady
```

How to confirm which path actually ran (in case the flag wiring
regresses): grep the `[MLX-PROFILE]` block. B path emits `tq_encode` /
`tq_value` / `tq_rotate` signposts. A path emits `kv_update` / `sdpa`.

```bash
# Should see tq_value 2406 calls on B path; sdpa 2406 on A path.
… | grep -E "tq_|sdpa|kv_update"
```

### Sweep the matrix

```bash
for envvar in "" "TURBO_USE_ALPHA=1"; do
  label=$([ -z "$envvar" ] && echo "B" || echo "A")
  for ctx in 1024 4096 8192 16384 32768; do
    for model in qwen35-0.8b qwen35-9b nemotron-30b-a3b; do
      result=$(env $envvar MLX_BENCH_PROFILE=2 \
        MLX_BENCH_MODEL=$model MLX_BENCH_METHOD=summarization \
        MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 \
        MLX_BENCH_CONTEXT=$ctx MLX_BENCH_NGRAM=0 MLX_BENCH_MAX_TOKENS=60 \
        swift test --skip-build -c release --filter benchmark 2>&1 \
        | grep decode_steady | awk -F'=' '{print $NF}' | tr -d ' ')
      echo "[$model ctx=$ctx $label] $result"
    done
  done
done
```

### Forcing TurboFlash (skip the dequant+SDPA path) for diagnostics

```bash
# B path but with old TurboFlash kernel — useful for bisecting whether
# a regression came from the kernel switch vs something else.
TURBO_DEQUANT_SDPA=0 MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=32768 \
  MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=60 \
  swift test --skip-build -c release --filter benchmark | grep decode_steady
```

Then start on **item 1 — port the dequant kernel from JIT to
precompiled**. Item 2 (long-context TurboFlash fallback) is also a
good candidate if you want a smaller-scope mlx-swift-lm-only PR
first.
