# turbo4v2 decode regression — confined to `ek/metal-icb-prototype`

**2026-04-17, M1 Max 64 GB, macOS, summarization @ 1024 ctx, 4-bit weights.**

## The finding in one sentence

Turbo4v2 KV quantization is at parity with no-quant on pristine `alpha`
for both Gemma 4 E2B and GPT-OSS 20B. The "turbo4v2 slower than no-quant"
regression is produced entirely by the `mlx-swift` pin on
`ek/metal-icb-prototype`.

## Numbers

Pristine rebuild on a fresh worktree off `origin/alpha` (`fbc6cd3`),
full `make clean-all && make` from scratch, then
`./scripts/benchmark.sh --model gemma4-e2b,gpt-oss-20b --kv none,turbo4v2
--method summarization --context 1024`:

| Branch                  | Model         | KV       | Decode tok/s | Prefill tok/s |
|-------------------------|---------------|----------|-------------:|--------------:|
| `alpha`                 | Gemma 4 E2B   | none     |         99.7 |       2 906.8 |
| `alpha`                 | Gemma 4 E2B   | turbo4v2 |    **102.9** |       2 885.3 |
| `alpha`                 | GPT-OSS 20B   | none     |         51.7 |         638.7 |
| `alpha`                 | GPT-OSS 20B   | turbo4v2 |     **52.2** |         662.2 |
| `ek/metal-icb-prototype`| Gemma 4 E2B   | none     |        101.8 |       2 888.2 |
| `ek/metal-icb-prototype`| Gemma 4 E2B   | turbo4v2 |         93.9 |       2 728.0 |
| `ek/metal-icb-prototype`| GPT-OSS 20B   | none     |         64.0 |         580.8 |
| `ek/metal-icb-prototype`| GPT-OSS 20B   | turbo4v2 |         46.5 |         589.0 |

Reading the table row-wise:

- **alpha**: turbo4v2 ≥ no-quant on both models (+3.2 % Gemma, +1.0 %
  GPT-OSS). Matches historical `ek/tom-eric-moe-tuning` behaviour.
- **ICB branch**: turbo4v2 −7.8 % on Gemma, −27 % on GPT-OSS vs the
  branch's *own* no-quant row.

Reading column-wise (absolute tok/s branch-over-branch):

- Turbo4v2 decode is **11 % slower** on ICB than alpha for GPT-OSS
  (46.5 vs 52.2) and **9 % slower** for Gemma (93.9 vs 102.9).
- No-quant decode on ICB is actually **faster** than alpha for GPT-OSS
  (64.0 vs 51.7). That asymmetry is why the intra-branch turbo-vs-none
  gap looks so wide on GPT-OSS: no-quant got a lift from something on
  the ICB branch that turbo4v2 did not benefit from (or was actively
  hurt by).

## What's actually on `ek/metal-icb-prototype` that `alpha` doesn't have

`Package.swift` on ICB branch:

```
.package(url: ".../mlx-swift", branch: "ek/metal-icb-prototype"),
```

That mlx-swift branch adds these commits over `alpha` tip `c92f80d`:

```
4c072fd chore: bump mlx submodule (dispatch-during-record counter)
4154daf chore: bump mlx submodule (ICB skip counter + logging)
7816b96 chore: bump mlx submodule (tolerate pre-pipeline set_input)
3b9174c chore: bump mlx submodule (tolerant set_buffer)
6087b2f feat(GPU): unconditional abort on IndirectCommandBuffer.record throw
e9dab2f feat(GPU): IndirectCommandBuffer Swift API
ab8374a feat(GPU): Swift wrapper for kernel-name log
c81aafb feat(GPU): expose Metal dispatch counter
db983a5 feat(MLXFast): add .slidingWindow(size:) SDPA mask mode
```

The underlying `mlx` submodule bumps bring in:

- `c1a9aead feat(metal): build every compute pipeline with ICB support enabled`
- `beaf6b04 feat(metal): wire IndirectCommandRecorder into CommandEncoder`
- `34684111 fix(metal): make dispatch counter process-global, atomic`
- `be2d6b87 feat(metal): expose cumulative dispatch counter`
- `0671b2fc feat(sdpa): add optional window_size for sliding-causal attention`
- `8ef47843 feat: enable sdpa_vector BD=512 for Gemma 4 decode`
- (plus the sdpa-sliding-window + Qwen35-prefill work)

### Why turbo4v2 is the one that regresses

The turbo4v2 decode path emits roughly **2×** the per-layer kernels of
the no-quant path (packed-dequant-K and packed-dequant-V run every
step in addition to the same attention / sdpa kernels). Any *fixed
per-dispatch* cost added anywhere in `CommandEncoder` gets multiplied
by how many dispatches the workload emits per token.

Two commits in the mlx submodule plausibly add fixed per-dispatch cost
to the *non-ICB* path (i.e. what a normal decode step still takes on
this branch because ICB recording is not active during generate):

1. **`c1a9aead`** — `supportIndirectCommandBuffers = YES` on every
   `MTLComputePipelineDescriptor`. Apple's documentation notes this
   disables pipeline-level optimisations on Apple Silicon even when the
   pipeline is never bound inside an ICB. This is a pay-always tax,
   bigger for dispatch-heavy workloads.
2. **`34684111` / `be2d6b87`** — a process-global atomic dispatch
   counter incremented on every Metal dispatch. Low ns cost per
   dispatch, but unconditional and visible on a per-dispatch hot loop.

The ratio between the Gemma regression (8 %) and the GPT-OSS
regression (27 % intra-branch, 11 % vs alpha absolute) is consistent
with a dispatch-proportional overhead: GPT-OSS has substantially more
per-token kernels than Gemma 4 E2B (confirmed by the
`MLX_BENCH_DISPATCH_AUDIT` counts in the ICB projection doc: Gemma ~274
kernels/token, GPT-OSS ~1 542).

### Why no-quant on GPT-OSS is *faster* on the ICB branch

Not investigated in depth here, but the candidates are:

- `8ef47843 feat: enable sdpa_vector BD=512 for Gemma 4 decode` —
  probably unrelated to GPT-OSS.
- Something in the sdpa-sliding-window path (`0671b2fc`) — GPT-OSS uses
  a 128-token sliding window. Plausibly the new primitive is cheaper
  per step for the window-reset shape.
- MoE kernel wiring on the new mlx tip.

These only help the non-quant path; they don't help turbo4v2 because
turbo4v2 goes through `quantizedScaledDotProductAttention` in
[Libraries/MLXLMCommon/KVCache.swift](Libraries/MLXLMCommon/KVCache.swift),
not through `MLXFast.scaledDotProductAttention`. So turbo4v2 sees only
the per-dispatch tax and none of the upside.

## Recommendation

1. **Do not block ICB work.** The ICB CPU-encoding speedups reported
   earlier today are unchanged. What this note flags is a *cost paid by
   non-ICB decode* while the ICB branch's mlx-swift pin is in place.

2. **Throughput benchmarks that care about KV-quant parity should run
   on `alpha`**, not on `ek/metal-icb-prototype`. Publish decode tok/s
   tables from alpha worktrees until the mlx-side fix lands.

3. **Permanent fix lives in `mlx`, not `mlx-swift-lm`**:

   - Gate `supportIndirectCommandBuffers = YES` behind a runtime flag,
     flipped on only while an `IndirectCommandRecorder` is active.
     Default back to `NO` so the normal decode path is untaxed.
   - Move the process-global dispatch counter behind an env-var opt-in
     (same pattern as the kernel-name log already has). Only bench /
     audit runs need it.

4. **Re-benchmark turbo4v2 on the ICB branch** once the two mlx changes
   above land to confirm we're back to parity with alpha.

## Raw measurement state

- Worktree: `.claude/worktrees/alpha-turbo4v2-repro` off `origin/alpha`
  `fbc6cd3`.
- Dependencies: `make clean-all` followed by `make` — `swift package
  reset` purged cached `.build/` so the build genuinely pulled
  `mlx-swift@alpha` and its `mlx` / `mlx-c` submodules from GitHub.
- Full log: `/tmp/bench-alpha.log`.
- Benchmark command:
  `./scripts/benchmark.sh --model gemma4-e2b,gpt-oss-20b --kv
  none,turbo4v2 --method summarization --context 1024`.
