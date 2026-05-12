# Implementation plan — speculative decoding rollout

**Last updated:** 2026-05-09
**Owner:** Eric (with research help from Claude)

This is the running implementation order across [specs 013–037](.) plus the forked [`ekryski/CoreML-LLM`](https://github.com/ekryski/CoreML-LLM) integration. Items in earlier tiers should ship before items in later tiers because of measurement dependencies, code dependencies, or risk-management ordering — not because later items are less important.

**Update note (2026-05-09 — late):** Reordering pass on the post-hoc spec wave. After review, the priority order for the post-Tier-1 next-step picks is:

1. **Spec 030 — Native MTP / EAGLE-3 draft heads** (promoted from Tier 4 to **Tier 2 row 10**) — largest pure-decode win, mostly model-loader work, lossless, compounds multiplicatively with everything below.
2. **Spec 036 — DuoAttention** (Tier 2 row 10d, unchanged) — slots directly onto PR #186 windowed turbo cache for streaming heads.
3. **Spec 034 — Quest decode-side K-side top-k** (Tier 2 row 10c, unchanged) — composes orthogonally with DuoAttention (Quest applies to the retrieval-head cache DuoAttention identifies).
4. **Spec 037 — TEAL activation thresholding** (promoted from Tier 4 to **Tier 2 row 10e**) — MLP-side bandwidth saving, composes with TurboQuant.
5. **Hybrid model porting** (Granite 4.0-H / Qwen3-Next / Kimi Linear) — Tier 3+; ships after the four post-hoc wins above. The four post-hoc specs are universal across model families; the hybrid kernel port is per-family work.

**Update note (2026-05-09):** Three new spec additions from the post-quadratic-attention research review (see [`papers/beyond-quadratic-attention-on-apple-silicon.md`](../papers/beyond-quadratic-attention-on-apple-silicon.md)) — spec 035 (Quest K_max/K_min selector, parked behind 034), spec 036 (DuoAttention retrieval/streaming head split, Tier 2 candidate), spec 037 (TEAL activation thresholding, Tier 4 candidate). Also reconciles 031–034 sparse-attention specs that landed via PR #188 into the tier roadmap.

**Reorder note (2026-04-29):** Tier 1 and Tier 3 swapped after PR #154 merged spec 023 (Leviathan accept/reject). The deterministic infrastructure wins (state replay, prefix cache, deterministic-stretch, n-gram cache) clear ahead of the more experimental DFlash + Mirror SD bets. Rationale: state replay alone unblocks 5+ models for n-gram + Leviathan paths; prefix cache is universal multi-turn TTFT win; both have phase-1 scaffolds landed (#143, #144) and just need the kernels + iterator wiring. DFlash needs `z-lab/Qwen3.5-*-DFlash` model availability; Mirror SD needs CoreML-LLM Swift Package + ANE measurement infra. Lower risk to land deterministic wins first.

## Tier 0 — land what's already in flight (this week)

These should merge first so the rest of the plan has a clean baseline.

| # | Item | What | Status |
|---|---|---|---|
| 1 | **013 close-out** | Land multi-candidate / strict-greedy / adaptive defaults; the AR-batch optimisation; the auto-routing in `MLXLMCommon.generate(...)`; the fix for the `for token in iterator` copy bug; the new realistic prompt set | ✅ Merged (PR #113) |
| 2 | **018 — bench per-prompt sweep mode** | `--method ngram-spot` (1 prompt × N config matrix) and `--method ngram-sweep-summary` (sweep with per-category best-cell roll-up) | ✅ Merged (PR #140) |
| 3 | **023 — Leviathan accept/reject sampling for n-gram** | Lift the `temperature == 0` requirement on n-gram speculative decoding via accept/reject sampling. Includes batched p-value optimisation and default-on at `temp != 0`. | ✅ Merged (PR #154) |
| 4 | **Spec-decode eval harness** | `scripts/spec-decode-sweep.sh` (parameterised sweep) + `scripts/spec-decode-compare.py` (A/B compare) + regime-classified prompt set | ✅ In PR #154 |

## Tier 1 — orthogonal infrastructure (next 4-6 weeks)

These benefit every decoder family and unlock measurable wins on already-supported models. Land in parallel.

| # | Item | What | Projected win | Status |
|---|---|---|---|---|
| 5 | **020 — state-replay rollback at the cache layer** | Generalised the GDN rollback primitive so any speculative decoder works on hybrid SSM/Mamba models. `StateReplayCache` protocol + `SSMStateCache` conformance + native `gated_delta_step_record` / `state_replay` Metal kernels + n-gram iterator wiring. Mamba (Nemotron-H, Jamba) cleanly opts out via per-instance `canStateReplay = false` — future work to land their kernel variant tracked in spec 020's §"Mamba / Mamba 2 follow-up (post-MVP)". | Unlocked n-gram speculative decoding on Qwen 3.5 / 3.6 GDN hybrids. **Measured on M1 Max 64GB:** Qwen3.5-35B-A3B-4bit D=12 adapt+strict **1.64× baseline** (92% accept), D=4 **1.37×** (83% accept). Mamba families fall back to vanilla `TokenIterator` until kernel variant lands. | ✅ Shipped 2026-05-11 ([PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143) + cross-repo chain [mlx#26](https://github.com/ekryski/mlx/pull/26) / [mlx-c#14](https://github.com/ekryski/mlx-c/pull/14) / [mlx-swift#25](https://github.com/ekryski/mlx-swift/pull/25)) |
| 6 | **017 — prefix KV cache** | Snapshot target KV state at request end keyed on stable prefix; hydrate on next request with the same prefix. All five phases consolidated: Phase 1 (`PrefixKVCache` + `PrefixKey` + LRU + stats), Phase 1B (per-class `serialise()` / `hydrate(from:)` + `generate()` wiring with stream wrapper + defence-in-depth `quantisationKindMismatch` guard), Phase 2 (`LastAssistantOpenerPolicy` for Qwen / Gemma 4 / GPT-OSS), Phase 3 (hybrid GDN+attention via spec 020 state-replay — `SSMStateCache` snapshots), Phase 4 (opt-in disk persistence at `~/.cache/mlx-swift-lm/prefix/`, off by default). **Opt-in for v1** (`prefixCacheEnabled = false` default — flipped to default-on then reverted same day after bench-surfaced limitations with `--kv turbo4v2`, see follow-ups [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) + [#197](https://github.com/ekryski/mlx-swift-lm/issues/197)). When the caller opts in, `prefixCachePolicy` auto-resolves via `AssistantOpener.detect(forModelID:)` → `LastAssistantOpenerPolicy` for Qwen / Gemma / GPT-OSS families (Identity fallback), and `prefixCacheModelID` auto-resolves from `ModelContext.configuration.name`. Cross-session sharing via `BlockAllocator.retain()` tracked in [#133](https://github.com/ekryski/mlx-swift-lm/issues/133). | **Measured 2026-05-12 (M1 Max 64GB, --kv none, realistic 4-turn chat — actual model replies sanitised & fed back):** **Qwen3.5-35B-A3B ~4.3× TTFT** (2186ms→499ms) with prefill rate climbing 17→596 tok/s (~35×). Qwen3.5-0.8B ~2.6× (318→3346 tok/s prefill). Gemma 4 E2B ~2.5×. Gemma 4 26B-A4B ~1.2× (long captured replies eat the win). Gemma 4 31B ~1.8× peak (noisy). GPT-OSS-20B: no benefit (window=128 sliding-window wrap — documented limitation). | ✅ Merged 2026-05-12 ([PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144)) — **opt-in v1, default-on gated on [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) + [#197](https://github.com/ekryski/mlx-swift-lm/issues/197)** |

## Tier 2 — compounding optimisations

These compose with Tier 1 for additional lift on specific workloads. Each is bounded effort.

| # | Item | What | Projected win | Best workload fit | Status |
|---|---|---|---|---|---|
| 7 | **022 — deterministic-stretch acceleration** | Chat-template state machine + bigram fallback drafter. Highest single-target win is GPT-OSS harmony channel transitions. | **+15–30% on GPT-OSS**, +5% across other model families | Harmony / channel-format models, structured-output generation | 🚧 Phase 1 scaffold landed ([PR #145](https://github.com/ekryski/mlx-swift-lm/pull/145) — `ChatTemplateGrammar` protocol + `BigramTable`). Phase 2 (per-family grammars + bigram corpus) open. |
| 8 | **016 — cross-request n-gram cache** | Persist the PLD lookup table across requests on the same model. Three-tier (`nc_context` / `nc_dynamic` / `nc_static`) per llama.cpp. | **+10–30%** on multi-turn chat on top of base PLD | Repeated-template generation, agent loops | 🚧 Phases 1-2 landed ([PR #146](https://github.com/ekryski/mlx-swift-lm/pull/146) — registry + tiered cache). Phase 4 (three-tier draft selection in iterator) open. |
| 9 | **014 Phase 1** — tree attention with K=2 root branches | Verify multiple candidate continuations in one forward via tree attention masks. Composes with multi-candidate. | **+15–25%** on input-grounded prompts where PLD already wins | Document QA, code editing | 🚧 Phase 1 scaffold landed ([PR #147](https://github.com/ekryski/mlx-swift-lm/pull/147) — `DraftTree` primitives). Phase 2 (MLX wiring + iterator integration) open. |
| 10 | **019 — PLD+ attention-weighted span selection** | Hidden-state cosine selection (Phase 1, model-agnostic) then induction-head attention scoring (Phase 2, per-model). | Higher accept rate on multi-candidate hits; +5-15% combined with #9 | Document QA, code editing | 🚧 Phase 1 scaffold landed ([PR #148](https://github.com/ekryski/mlx-swift-lm/pull/148) — selector protocol + cosine helper). Phase 2 (per-model conformance + iterator integration) open. |
| 10a | **030 — Native MTP / EAGLE-3 draft heads** (promoted 2026-05-09 from Tier 4) | Variant A: stop stripping `mtp.*` at sanitize on DeepSeek-V4 / Qwen3.5 / Qwen3-Next / MiMo / GLM-4-MoE; ship `MTPSelfSpeculativeTokenIterator` + custom converter (`scripts/mtp_convert.py`). Variant B: companion EAGLE-style assistant draft + `AssistantDraftRegistry`. **First in the post-Tier-1 priority order** — largest pure-decode win, lossless, compounds multiplicatively with every other post-hoc spec below. | **1.4–2.2× decode** depending on family; assistant-draft path on Gemma 4 is decoder-agnostic (~2× at temp 0) | Universal — every supported model family | 📋 Spec written. Variant A on hybrid Qwen unblocked by spec 020 (shipped 2026-05-11). Variant B + Variant A on DeepSeek-V4 ship independently. Not started. |
| 10b | **036 — DuoAttention retrieval/streaming head split** | Per-head classification (calibration pass on synthetic NIAH) tags each attention head as **retrieval** (full KV) or **streaming** (sink+window). Streaming heads reuse PR #186's `TurboQuantizedKVCache(maxSize:)` directly; retrieval heads keep full cache. Two-cache-per-layer dispatch (Option A) lands first; optional ragged-shape Metal kernel (Option C) in phase 5. Composes orthogonally with 034/035: Quest applies only to the retrieval-head cache. | **1.5–2.2× decode + 1.7–2.6× memory at long context** ([paper](https://arxiv.org/abs/2410.10819): MHA 2.18× / 2.55×, GQA 1.50× / 1.67×); calibration is one-time per model. Stacks multiplicatively with 030 and 034. | Long-context chat, RAG, doc QA — every workload where #186 windowed eviction already helps | 📋 Spec written ([PR #189](https://github.com/ekryski/mlx-swift-lm/pull/189)). Not started. |
| 10c | **034 — decode-side K-side top-k (Quest / RetrievalAttention)** | Per-layer per-step K-side top-k selection on a paged or contiguous cache. Three selectors: (1) block-mean LSH, (2) H2O heavy-hitter retention, (3) recency+sinks baseline. V1 ships on `StandardKVCache` with dense SDPA on gathered slots; V2 uses spec 033's fused block-sparse kernel; phase 7 wires the TurboQuant Path B fast path. Composes with 036: Quest applies to the retrieval-head cache DuoAttention identifies. Spec 035 (Tier 4) is a refinement. | **5–8× long-context decode** (at 32–128K, k≈2048); composes multiplicatively with 030 / 031 / 032 / 036 | Long-context decode, retrieval workloads | 📋 Spec written ([PR #188](https://github.com/ekryski/mlx-swift-lm/pull/188)). Not started. |
| 10d | **037 — TEAL activation thresholding on `FusedGateUpMLP`** (promoted 2026-05-09 from Tier 4) | Training-free magnitude thresholding before `down_proj`: prune entries of `silu(gate) * up` below per-tensor calibrated threshold τ_l → ~50% MLP activation sparsity at <1% PPL drift. Two phases: (1) threshold-and-mask hook + `scripts/teal_calibrate.py`; (2) block-sparse Metal kernel for `(masked_act, down_proj) → out`. Composes orthogonally with TurboQuant. Best on dense models — MoE already has expert-level sparsity. | Paper: **1.53–1.8× decode on H100**. Honest M-series estimate: **1.2–1.4× decode** for the MLP fraction of total time (MLP share is smaller at long context on Apple Silicon). | Dense models (Qwen 3.5 dense tiers, Gemma 4 31B) | 📋 Spec written ([PR #189](https://github.com/ekryski/mlx-swift-lm/pull/189)). Not started. |
| 10e | **031 — vertical-slash sparse prefill** | A-shape + vertical-slash patterns via existing SDPA primitives — no new kernel. Calibration sidecar per model selects head patterns; prefill runs masked SDPA over the sparse pattern. Composes multiplicatively with 032. | **4–7× prefill at 128K** | Long-context prefill, RAG, doc QA | 📋 Spec written ([PR #188](https://github.com/ekryski/mlx-swift-lm/pull/188)). Not started. |
| 10f | **032 — speculative prefill** | Drafter-scored span selection during prefill (PFlash-style); drafter shared with spec 015 DFlash. No new kernel. Stacks on top of 031. | **8–12× TTFT at 128K**; 30–80× when stacked with 031 | Long-context TTFT | 📋 Spec written ([PR #188](https://github.com/ekryski/mlx-swift-lm/pull/188)). Not started. |

## Tier 3 — primary speedup paths (more experimental)

The two highest-leverage features in the headline numbers, but with significant external dependencies. Land after Tier 1 + Tier 2 stabilise.

| # | Item | What | Projected win | Rationale | Status |
|---|---|---|---|---|---|
| 11 | **015 phases 1–3** — DFlash on GPU | Port DFlash's Python reference (`bstnxbt/dflash-mlx engine-v2`) to MLX-Swift. Phase 1 (protocol surface + iterator scaffold), phase 2 (real draft model from `z-lab/Qwen3.5-*-DFlash`), phase 3 (hybrid GDN state-replay rollback — refactor onto the `StateReplayCache` protocol from spec 020). | **2.4–4.4× on Qwen 3.5/3.6** | Standalone win once draft-model availability lands. Spec 020 (Tier 1 #5) is no longer a blocker (shipped 2026-05-11) — phase 3 now just refactors onto the protocol. | 🚧 Phase 1 scaffold landed ([PR #141](https://github.com/ekryski/mlx-swift-lm/pull/141)). Phase 2 (draft model) + phase 3 (refactor onto StateReplayCache) open. |
| 11a | **033 — block-sparse SDPA Metal kernel** | Foundational Metal kernel that takes `(Q, K, V, block_adjacency)` and runs SDPA only over the live blocks. Multi-month, 4-repo PR chain (mlx + mlx-c + mlx-swift + mlx-swift-lm). Enables MInference / FlashPrefill / FlexPrefill / XAttention / TriangleMix patterns; consumed by spec 034 phase 6, spec 035 V2, spec 036 phase 5. **No standalone end-user win** — purely an enabler. | **0× direct** (kernel only); unlocks ~15–30% extra on top of 034/035/036 V1 paths by eliminating the gather memory bandwidth | Foundational; consumers measure the lift | 📋 Spec written ([PR #188](https://github.com/ekryski/mlx-swift-lm/pull/188)). Not started. |
| 12 | **025 — ANE+GPU concurrency primitives** | Race-free cross-device state primitives + concurrency measurement harness for any ANE-offloaded work. Codifies the architectural lessons from the retired AB/ICB track (override-bound fresh allocation, never mutate persistent storage across an async device boundary). 4 phases: measurement harness, buffer lifecycle audit, reference `ANEDraftLoop` primitive, hand-off. **No end-user feature** — pure de-risking infrastructure. See [`sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md`](../../sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md) for the prior-art reasoning. | **0× direct** (de-risking only) | Executes immediately before 021 Phase 1A. Makes the concurrent-execution decision point (#2 below) a clean measurement of the *concurrency hypothesis* rather than a tangle of integration code + concurrency. | 📋 Spec written. Not started. |
| 13 | **021 Phase 1A** — Mirror SD spike | Add `ekryski/CoreML-LLM` as a Swift Package dependency. Glue their `MirrorSpeculativeLoop` to our MLX target via a thin `SpeculativeTarget` adapter. **Includes the Core-ML-vs-private-ANE-API benchmark** to characterise dispatch overhead. **Now depends on spec 025** for the primitives + measurement harness. | **3–5× projected** (Apple Mirror SD paper headline) | High projected win but high integration risk: needs CoreML-LLM Swift Package dep + spec 025 primitives. | 🚧 Phase 1A scaffold landed ([PR #142](https://github.com/ekryski/mlx-swift-lm/pull/142) — protocol + registry + vocab gate). Phase 1B (real Core ML draft + integration) + phase 2 (full iterator) open; gated on spec 025. |

After Phase 1A measurement: if pass, immediately schedule **021 Phase 2** (integrated `MirrorSpeculativeTokenIterator`) — adds 3–4 weeks but lands the headline win on Qwen 3.5/3.6.

## Tier 4 — nice to have / per-model effort

Defer until Tiers 0–3 are solid. Each is bounded effort but per-model rather than universal.

| # | Item | What | When | Status |
|---|---|---|---|---|
| 13 | **015 phases 4–6 + 021 Variant C** — DFlash-on-ANE | Port DFlash draft to Core ML; run on ANE while target verifies on GPU. Composes Mirror SD parallelism × DFlash K=16 amortisation. | After Tier 3 ships and DFlash-on-GPU is stable. **4–7× projected** on Qwen 3.5-27B. | 📋 Spec written. Blocked on Tier 3 #11 + spec 021. Not started. |
| 14 | **014 Phases 2–4** — variable-K tree, bifurcating-on-tight-margin, full suffix-tree merging | Each composes additively with phase 1. Phase 4 = SuffixDecoding port; CoreML-LLM has reference (`SuffixTree.swift` + `SuffixSpeculativeEngine.swift`). | After 014 phase 1 lands and is measured. | 📋 Spec written. Blocked on Tier 2 #9 phase 1 completion. Not started. |
| 15 | **024 — KV cache write fusion** | Eliminate the 60 `copy_bfloat16` dispatches per decode token on Gemma4-E2B (and proportional copies on every other model). Recommended path: extend `mlx`'s `SliceUpdate::eval_gpu` to handle strided source + fix donation — three-repo PR (mlx + mlx-c + mlx-swift) but transparent to all consumers. | After Tier 3 stabilises. **+3–8% decode tok/s** per model, scales with non-shared layer count. Supersedes the deferred `ek/gemma4-e2b-kv-copy-fusion` investigation. | 📋 Spec written. Not started. |
| 16 | **GPT-OSS-20B attention-sinks on TurboQuant path B** | Four-repo cross-cutting PR chain to make the compressed-domain decode path (`-compact` suffix, `useCompressedAttention=true`) coherent on attention-sinks models. Path B + non-sinks works coherently, path B + sinks is wired but disabled because output is incoherent despite kernel math being right on paper. Debug behind `MLX_TURBO_FORCE_BETA_SINKS=1`. | After Tier 3 stabilises. **GPT-OSS-only, path-B-only** — path A is the speed default; this is for users who explicitly need long-context KV compression on sinks models and accept path B's 30–60%-of-path-A decode cost. Outstanding work: identify the divergence between path A (coherent) and path B (incoherent) on GPT-OSS-20B with same sinks tensor. Sliding-window + graph-fuser already ruled out. | 🚧 Four-repo PR chain OPEN: [mlx#16](https://github.com/ekryski/mlx/pull/16) (kernel + C++ primitive — pass2 folds per-head sink logit into cross-block softmax), [mlx-c#8](https://github.com/ekryski/mlx-c/pull/8) (C ABI + 5 missing decls), [mlx-swift#18](https://github.com/ekryski/mlx-swift/pull/18) (Swift wrapper + submodule bumps), [mlx-swift-lm#99](https://github.com/ekryski/mlx-swift-lm/pull/99) (path B CLI surface + sinks routing). Umbrella: [#130](https://github.com/ekryski/mlx-swift-lm/issues/130). |
| 17 | **026 — Profile-guided Morton-order expert weight reordering** | Calibration corpus → co-selection matrix → permutation generator (greedy / spectral / Morton) → per-layer `sanitize()` integration. Permanently reorders MoE expert weights so frequently co-selected experts are adjacent in memory. Eliminates the sort-vs-no-sort tradeoff at small batch sizes. **Experimental** — generalization across corpora is the load-bearing risk. | After Tier 3 stabilises. **+25% prefill at T=128 on MoE models** (recover unsorted-path penalty); +0–5% at T≥1024; +0–3% at decode. Zero runtime cost — all work is offline + once at sanitize(). | 📋 Spec written. Not started. |
| 18 | **027 — Adaptive per-layer mixed-precision quantization framework** | Recipe-driven framework for per-module bit-widths via JSON sidecar + glob-pattern matching. Generalizes the ad-hoc Unsloth UD-MLX pattern (closes [#74](https://github.com/ekryski/mlx-swift-lm/issues/74)) and unlocks recipes like `int4-lm-head` for bf16 models. **Experimental** — value depends on recipe library quality; bf16 audience is small. | After Tier 3 stabilises. **+20× LM head decode** for users who opt into `int4-lm-head` on bf16 models (small audience but huge per-user impact). Closes #74 as a side effect. | 📋 Spec written. Not started. |
| 19 | **028 — Quadratic / chunkwise WY GatedDeltaNet prefill** | Parallelize the GDN recurrence via chunkwise Woodbury-Young + short-context quadratic-attention reformulation. **Research-grade** — a prior quadratic-attention experiment regressed at Dk=128 (cause inconclusive). Multi-month work: Python reference → naive Metal port → fused kernel → quadratic-vs-chunked dispatch → per-model integration. | After spec-decode (Tier 1–3) stabilises. **+5–15× prefill on Qwen 3.5 family** if the kernel works; could regress if it doesn't. Highest research bet in the plan. | 📋 Spec written. Not started. |
| 20 | **029 — ANE-offloaded LM head + Gemma 4 PLE projection** | Use spec 025's ANE+GPU concurrency primitives to overlap fixed-shape projections (LM head, Gemma 4 PLE) on ANE while the GPU does next-layer work. **Hard dependency on spec 025 phase 1 measurement passing** — if Apple Silicon doesn't run ANE+GPU concurrently, this spec dies (along with spec 021). | After spec 025 phase 1 + 2 land. **+5–15% per-token on Gemma 4 E2B** (LM head + PLE both offloaded); +3–8% on other large-vocab models (LM head only); no win on Qwen 3.5 (smaller vocab). | 📋 Spec written. Blocked on Tier 3 #12 (spec 025) measurement gate. Not started. |
| 21 | **fp16 vs bf16 runtime dtype audit + conversion** | Apple Silicon's Metal SIMD natively supports fp16 + fp32; bf16 is a software conversion via [`bf16.h`](../../mlx-swift/Source/Cmlx/mlx-generated/metal/bf16.h) that adds compute overhead in hot kernels. Audit every `bfloat16` site (compute vs storage), convert (b)-class compute sites to fp16, full bench matrix vs `alpha` (decode tok/s + prefill tok/s + PPL/KLD on `qwen35-{0.8,9}b`, `gemma4-{e2b,26b-a4b}`, `gpt-oss-20b`, `nemotron-cascade-2-30b-a3b` × `--kv {none, affine4, turbo4v2}`). Decision gate: ≥5% mean decode tok/s improvement with no PPL/KLD regression. | After Tier 1 ships. **Bounded research, 12-20h.** Sub-5% → publish bench as paper-track artifact, don't merge — this is opt-in upside, not a baseline change. | 📋 Tracked in [#162](https://github.com/ekryski/mlx-swift-lm/issues/162). Not started. |
| 22 | **015 phase 1+2 (Gemma 4 only) — DFlash on Gemma 4** | Spec 015 reordered 2026-05-08: Gemma 4 (full attention + SWA) ships before the hybrid Qwen path because no GDN state-replay is required. Z-lab now publishes `gemma-4-{31B, 26B-A4B}-it-DFlash` drafts; `bstnxbt/dflash-mlx` `main` branch carries the Gemma 4 target adapter. Ships standalone — does not block on spec 020. | Slots in alongside Tier 3 #11 (Qwen path) but can land independently. **2.4–5.8× projected on Gemma 4** per z-lab card. | 📋 Spec reordered 2026-05-08. Not started. |
| 23 | **035 — Quest K_max/K_min selector (refinement of 034)** | Drop-in replacement for spec 034's "block-mean LSH" selector with the original Quest paper's elementwise K_max / K_min upper bound. Tighter score bound → better NIAH retention at smaller k → smaller K-side compute budget for same retrieval fidelity. Costs 2× page metadata (~6% of K-cache vs ~3%). Implementation reuses 034 phases 1–3 (block-summary infrastructure, selector dispatch, NIAH harness); only the per-block scorer differs. **Ships only if 034 V1's measured NIAH curve doesn't already hit the target operating point.** | Marginal lift over 034 — **+10–25% extra k-budget headroom** at the same NIAH retention, or matched k-budget with cleaner failure mode on multi-hop reasoning. Bench-driven decision. Long-context reasoning, multi-hop retrieval where 034 V1 falls short. After 034 V1 ships and NIAH curves are measured. | 📋 Spec written ([PR #189](https://github.com/ekryski/mlx-swift-lm/pull/189)). Blocked on 034 V1 NIAH measurement. Not started. |
| 24 | **038 — Active KV cache SSD offload** | InfiniGen-style mid-generation page-out of cold KV pages to SSD with predictor-driven prefetch. Lets a 128K-context request run on a 16-24 GB Mac by parking streaming-head + Quest-rejected pages on NVMe. **Disjoint from spec 017** (which is cross-request prefill skip; this is single-request memory overflow). Phased: (1) heuristic prefetch on DuoAttention's retrieval flag, (2) trained predictor per model, (3) Quest integration, (4) observability. Spec: [`038-active-kv-cache-ssd-offload.md`](038-active-kv-cache-ssd-offload.md). | **Memory reduction at long context, not throughput.** Target: 128K Qwen 3.5-9B on 32 GB Mac (currently OOMs), at ≥85% of all-in-memory decode tok/s. Multi-month build — only justified if long-context single-request use cases matter to the user base. After #127/#128/#129 and spec 036 phase 1 land. Long-context decode on memory-constrained Macs. | 📋 Spec written 2026-05-12 ([PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144)). Blocked on paged KV ([#127](https://github.com/ekryski/mlx-swift-lm/issues/127) + [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) + [#129](https://github.com/ekryski/mlx-swift-lm/issues/129)) + DuoAttention ([spec 036](036-duoattention-retrieval-streaming-head-split.md)) phase 1. Not started. |

## Issue-tracked perf backlog

Granular perf work, each in its own GitHub issue. No fixed ordering — pick up in priority order based on size labels (S/M/L/XL) and current focus area. Runs in parallel with the tier roadmap above. Issues that map onto a spec-level item are listed under that spec's tier row; the categories below cover everything else.

**Status legend** (used across all issue subtables):
- 🟢 OPEN — Not started.
- 🔵 OPEN — Active / partial progress.
- 🟡 OPEN — Blocked on prerequisite work.
- 🟣 OPEN — Investigation only, decision pending.
- ⚪ OPEN — Stale / re-bench needed; may be obsolete.
- ✅ CLOSED.

### Decode tok/s

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#114](https://github.com/ekryski/mlx-swift-lm/issues/114) | Long-context fallback to TurboFlash for large models | S | 🟢 OPEN |
| [#115](https://github.com/ekryski/mlx-swift-lm/issues/115) | QKV batched fusion via `MLXFast.batchedQKVQuantizedGEMV` | M | 🟢 OPEN |
| [#116](https://github.com/ekryski/mlx-swift-lm/issues/116) | Gate+Up MLP fusion | M | 🟢 OPEN |
| [#117](https://github.com/ekryski/mlx-swift-lm/issues/117) | RMSNorm + GEMV fusion for MLP and attention | L | 🟢 OPEN |
| [#118](https://github.com/ekryski/mlx-swift-lm/issues/118) | Fold V output rotation into Wo at codec init | S | 🟢 OPEN |
| [#119](https://github.com/ekryski/mlx-swift-lm/issues/119) | `compile()` coverage gaps on Qwen 3.5 | S | 🟢 OPEN |
| [#120](https://github.com/ekryski/mlx-swift-lm/issues/120) | Eliminate GQA tile in rawKeyMode path | M | 🟢 OPEN |
| [#121](https://github.com/ekryski/mlx-swift-lm/issues/121) | NR0 > 2 in TurboFlash kernel (multi-repo) | M | 🟢 OPEN |
| [#122](https://github.com/ekryski/mlx-swift-lm/issues/122) | f32 rotation precision A/B (low priority — only if quality regression surfaces) | S | 🟣 OPEN — gated on quality-regression evidence |
| [#157](https://github.com/ekryski/mlx-swift-lm/issues/157) | Int4 LM head quantization for bf16-weight models — **>+50% decode** on bf16 Gemma 4 / Qwen 3.5 (small audience, big per-user impact). Standalone version of spec 027's `int4-lm-head` recipe. | S | 🟢 OPEN |
| [#158](https://github.com/ekryski/mlx-swift-lm/issues/158) | Float16 V accumulator in TurboFlash pass2 kernel (separate from the bf16 output dtype that landed in `74024a24`) — ~5–10% attention kernel improvement | S | 🟢 OPEN |
| [#159](https://github.com/ekryski/mlx-swift-lm/issues/159) | Symbolic sliding-window SDPA mask for GPT-OSS-20B (parity with Gemma 4 PR #55) — 5–10% decode at 4k+ ctx | S | 🟢 OPEN |
| [#160](https://github.com/ekryski/mlx-swift-lm/issues/160) | Persistent FP16 dequant cache for TurboQuant path A — keep dequant resident across decode steps; trades steady-state memory for ≥5% decode at long context | M | 🟢 OPEN |
| [#161](https://github.com/ekryski/mlx-swift-lm/issues/161) | TokenRing → CPU `Set<Int>` for presence/repetition penalties — **investigation first** since current code claims "no CPU←GPU sync" but the audit doc disagrees; 2–5% decode if confirmed | S | 🟣 OPEN — investigation first |

### Prefill / TTFT

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#123](https://github.com/ekryski/mlx-swift-lm/issues/123) | Adaptive GatedDeltaNet `evalInterval` tuning | S | 🟢 OPEN |
| [#124](https://github.com/ekryski/mlx-swift-lm/issues/124) | NemotronH peak-memory regression mystery (investigation) | M | 🟣 OPEN — investigation |
| [#125](https://github.com/ekryski/mlx-swift-lm/issues/125) | Async prefill compression for TurboQuant KV | M | 🟢 OPEN |
| [#126](https://github.com/ekryski/mlx-swift-lm/issues/126) | Tokenization / chat-template caching | S | 🟢 OPEN |
| [#156](https://github.com/ekryski/mlx-swift-lm/issues/156) | Verify non-fused `gated_delta_step` Metal kernel correctness — kernel + ops fallback both still in tree; active path unclear; if kernel is broken, remove or fix; if fixed, switch dispatch and measure prefill win on Qwen 3.5 family | M | 🟣 OPEN — investigation first |

### Memory / paged KV

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) | Metal paged-attention kernel for `PagedKVCache` | L | 🟢 OPEN — prerequisite for Tier 4 row 24 (spec 038) |
| [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) | Wire `PagedKVCache` into Qwen 3 / Gemma 4 / etc model factories | M | 🟡 OPEN — gated on [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) |
| [#129](https://github.com/ekryski/mlx-swift-lm/issues/129) | TurboQuant + paged integration | L | 🟡 OPEN — gated on [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) + [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) |

### Speculation umbrellas (concrete paths in the tier roadmap above)

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#132](https://github.com/ekryski/mlx-swift-lm/issues/132) | EAGLE / MEDUSA-style draft decoding. EAGLE-3 covered for free by spec 021 (Mirror SD); MEDUSA still needs draft-head training infra and is independent of 021. | XL | 🟡 OPEN — EAGLE-3 gated on spec 021 (Tier 3 #13). MEDUSA independent. |
| [#133](https://github.com/ekryski/mlx-swift-lm/issues/133) | Prefix caching across sessions (referenced from Tier 1 row #6) — implements spec 017 phase 3 via `BlockAllocator.retain()` | L | 🔵 OPEN — partially addressed by spec 017 phase 4 (L2 disk cache shipped 2026-05-12 in [PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144), opt-in via `MLX_PREFIX_CACHE_DISK=1`). Concurrent-process safety via `BlockAllocator.retain()` + actor wrapping still pending. |

### Spec 017 prefix-cache follow-ups (default-on gates)

These two issues are the blockers between spec 017's opt-in v1 ship and a future default-on flip. Both surfaced during PR #144 bench validation against `--kv turbo4v2`.

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) | Support compressed-mode TurboQuant in `serialise()` / `hydrate(from:)` (spec 017 phase 1B+) — currently `TurboQuantizedKVCache.isCompressed == true` causes silent snapshot refusal, so Qwen 3.5 / 3.6 / NemotronH get zero cache benefit under `--kv turbo4v2`. Plan: extend `metaState` with seed / rotatingIdx / isCompressed; drop the guard in `serialiseTurbo`; round-trip test. | S | 🟢 OPEN — ~1 day; clean extension of existing per-class serialise pattern. Blocks spec 017 default-on flip. |
| [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) | Gemma 4 26B-A4B / 31B prefix-cache lookup miss under `--kv turbo4v2` — snapshots successfully insert (cache_bytes grows turn over turn) but every lookup misses. Doesn't repro on Gemma 4 E2B (same code path) or on the same models under `--kv none`. Suspected: tokenization non-determinism across turn counts, or `compressionAlgorithm` leaks into key derivation. | M | 🟣 OPEN — investigation first; bug. Blocks spec 017 default-on flip. |
| [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) | TurboQuant + sliding-window eviction produces incoherent output on Gemma 4 specifically — adjacent issue, confused initial spec 017 bench analysis but separate from the prefix cache itself. Gemma 4 + `--kv turbo4v2` runs unquantized today because `makeAttentionCache(...)` falls through to `StandardKVCache`. Multi-day investigation. | L | 🟡 OPEN — gates Gemma 4 + real TurboQuant, but **doesn't block spec 017 default-on**. |

### Batching follow-ups (to PR #138's `generateBatched`)

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#149](https://github.com/ekryski/mlx-swift-lm/issues/149) | TurboQuant decode regresses 0.60× at long-context B>1 | M | 🟢 OPEN |
| [#150](https://github.com/ekryski/mlx-swift-lm/issues/150) | Variable-length prompts + per-sequence EOS for `generateBatched` | M | 🟢 OPEN |
| [#151](https://github.com/ekryski/mlx-swift-lm/issues/151) | Continuous batching — admit prompts while decode is in flight | L | 🟡 OPEN — gated on [#150](https://github.com/ekryski/mlx-swift-lm/issues/150) |

### Investigation / cleanup

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#134](https://github.com/ekryski/mlx-swift-lm/issues/134) | xctrace inspection of dequant kernel (informs [#121](https://github.com/ekryski/mlx-swift-lm/issues/121)) | S | 🟣 OPEN — investigation, informs [#121](https://github.com/ekryski/mlx-swift-lm/issues/121) |
| [#155](https://github.com/ekryski/mlx-swift-lm/issues/155) | `--dispatch-audit` flag for in-process Metal dispatch counting (CI-friendly counter; complements `MLX_METAL_PROFILE`). Salvageable from the abandoned AB/ICB track — see post-mortem in `sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md`. | S | 🟢 OPEN |

### Architecture proposals

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#73](https://github.com/ekryski/mlx-swift-lm/issues/73) | Refactor how we do KVCache to make it cleaner | L | ✅ CLOSED — shipped as spec 006 (KV cache type consolidation, PRs #163–#166). `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache` / `SSMStateCache` hierarchy is the current state. |
| [#74](https://github.com/ekryski/mlx-swift-lm/issues/74) | Add support for Unsloth dynamic quants (UD-MLX checkpoints) | M | 🟡 OPEN — generalised under Tier 4 row 18 (spec 027). Closes as a side effect when 027 ships. |

### Open questions / pending verification

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#77](https://github.com/ekryski/mlx-swift-lm/issues/77) | Investigate dual usage of quant/dequant values in TurboFlash kernel (mineable from PR #93's `60bd16d`) | M | 🟣 OPEN — investigation, low priority |
| [#89](https://github.com/ekryski/mlx-swift-lm/issues/89) | TurboQuant decode regression in `generate()` path but NOT in bridge-direct path — may be obsolete after recent PRs; needs re-bench | M | ⚪ OPEN — re-bench needed; may be obsolete after `74024a24` + spec 006 cleanups |

### Bugs

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#61](https://github.com/ekryski/mlx-swift-lm/issues/61) | `Gemma3TextModel` crashes when `hiddenLayers < slidingWindowPattern` | S | 🟢 OPEN |

### Spec-decode policy

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#153](https://github.com/ekryski/mlx-swift-lm/issues/153) | n-gram speculative decoding: auto-disengage on known regression regimes (small/fast models + paraphrastic workloads). Subsumes spec 023 phase 3. | M | 🟢 OPEN — subsumes spec 023 phase 3 (deferred) |

### Tier 4 follow-ups

| Issue | Topic | Size | Status |
|---|---|---|---|
| [#162](https://github.com/ekryski/mlx-swift-lm/issues/162) | fp16 vs bf16 runtime dtype audit + conversion (Tier 4 row 21) | M | 🟢 OPEN — tracked in Tier 4 row 21 |

## Dependency graph (the tight ones only)

```
013 (✅ shipped) ─┬─► 018 (✅ shipped) ─► everything else (measurement plumbing)
                  │
                  ├─► 023 (✅ shipped) ─── n-gram + sampling
                  │
                  ├─► 022 (Tier 2, phase 1 scaffold landed)
                  ├─► 016 (Tier 2, phase 1-2 scaffold landed)
                  ├─► 014 phase 1 (Tier 2, phase 1 scaffold landed)
                  └─► 019 phase 1 (Tier 2, phase 1 scaffold landed)

020 (✅ shipped 2026-05-11 — phases 1+2+3 consolidated)
  └─► ✅ n-gram on Qwen 3.5/3.6 GDN (now live; auto-routing predicate
      flipped from canTrimPromptCache to canRollbackPromptCache).
      Nemotron-H / Jamba opt out via per-cache canStateReplay=false
      until Mamba kernel variant lands (future work).
  └─► 015 phase 3 (hybrid GDN path for DFlash — unblocked)
  └─► 016 phase 3 (hybrid path for ngram cache — unblocked)
  └─► 017 phase 3 (hybrid prefix-cache snapshots — unblocked)

015 phases 1-3 (Tier 3) ─┬─► 015 phases 4-6 ─┬─► 021 Variant C
                         │                    │
                         └─► 020 prerequisite ✅ landed; phase 3 now
                             refactors onto StateReplayCache protocol

025 (Tier 3 prework) ─► 021 Phase 1A (Tier 3) ─► 021 Phase 2 ─► 021 Variants B/C/D
                                                                  │
                                                                  └─► Tier 4 row 13 (DFlash-on-ANE) reuses 025 primitives

031 ─┬─► 032 (drafter-scored span selection composes on top)
     │     │
     │     └─► drafter shared with 015 (DFlash)
     │
     └─► sparse-prefill calibration reused by 034 selector (head pattern signal)

PR #186 (windowed turbo cache) ─► 036 streaming-head path (Tier 2 10d)
                                    │
                                    └─► spec 020 state-replay (hybrid models)

#127 + #128 + #129 (paged KV cache backlog) ─┬─► 034 V2 (paged variant)
                                              │
                                              ├─► 035 (Quest K_max/K_min, Tier 4 row 23)
                                              │     │
                                              │     └─► reuses 034 phases 1–3
                                              │
                                              └─► 038 active KV SSD offload (Tier 4 row 24)
                                                    │
                                                    └─► also needs 036 (DuoAttention)
                                                        to flag pages safe to spill

033 block-sparse SDPA Metal kernel (Tier 3 11a) ─┬─► 034 phase 6 (fused fast path)
                                                  │
                                                  ├─► 035 V2 (fused per-page sparse)
                                                  │
                                                  └─► 036 phase 5 (ragged per-head fused SDPA)

037 (TEAL, Tier 4 row 25) ─── independent of KV-cache work; pairs with FusedGateUpMLP + TurboQuant
```

## What we're not doing (at least not yet)

- **EAGLE-3 standalone port.** CoreML-LLM has it (`MirrorSpeculativeLoop.swift` runs an EAGLE-3 draft). Once Mirror SD is wired in via 021, EAGLE-3 comes along for free. Skip the standalone port.
- **MTP heads independent of CoreML-LLM.** Same reasoning — `MtpSpeculativeEngine.swift` is shipped upstream. Wire it in via 021's iterator surface, don't reinvent.
- **Private ANE API direct shim.** Measured in 021 Phase 1A but not shipped. Decision deferred until measurement comes back.
- **Cross-vocabulary speculative decoding** (different tokenizers for draft and target). Available in CoreML-LLM (`CrossVocabSpeculativeEngine.swift`) but lower priority than same-tokenizer paths.
- **Spec 023 phase 2** (unify Leviathan + greedy paths). Closed out without implementation — see spec 023 for the three correctness/perf risks.
- **Spec 023 phase 3** (self-disengage on chronically-low accept). Subsumed by issue #153 — should apply uniformly to greedy and Leviathan when implemented.
- **Spec 035 ahead of spec 034 V1 measurement.** Spec 035 is the original Quest paper's K_max/K_min selector; spec 034 V1 ships block-mean LSH. Both target the same K-side decode bandwidth. Don't implement both selectors in parallel — ship 034 V1, measure its NIAH retention curve, only then decide whether 035's tighter (but more storage-expensive) bound is worth the extra metadata maintenance.

## Decision points

The plan assumes pass at each measurement gate. Re-evaluate if any of these come back negative:

1. ✅ **020 phase 2's `SSMStateCache` state-replay matches a sequential reference** — **gate passed (2026-05-11)**. State kept in fp32 throughout the GDN recurrence eliminates bf16 drift; replay kernel inherits the fp32 reference. Locked by `SSMStateCacheStateReplayTests::rollback(acceptedPrefix:) re-folds first k delta log entries via the GDN recurrence` + 6 sibling equivalence tests. Off-by-one bug found and fixed (`be3ecff`) during bench validation — Qwen3.5-35B-A3B D=12 acceptance jumped 34% → 92%.
2. **021 Phase 1A's concurrent-execution check** — Apple Silicon truly running ANE + GPU in parallel without serialisation through XPC / IOSurface / shared queues. **Now operationalised by spec 025's Phase 1 measurement harness** — failure surfaces as a clean primitive-level diagnostic before 021 integration code is written. Failure still kills 021 + Tier 4 row 13, but the cost of finding out drops from ~1–2 weeks of integration debugging to ~3–5 days of a focused harness.
3. **015 Phase 3's state-replay correctness on real Qwen 3.5 models** — same numerical-equivalence concern, scoped to DFlash. **Tier 1 #5 prerequisite ✅ landed**; phase 3 implementation is now blocked only on DFlash phase 2 (draft-model availability), not on the rollback primitive.

## Status snapshot — at this commit

- ✅ **Tier 0 complete.** 013 + 018 + 023 + eval harness shipped.
- ✅ Specs 014–037 written.
- ✅ Two survey papers in `papers/`: [`speculative-decoding-on-apple-silicon.md`](../papers/speculative-decoding-on-apple-silicon.md) (decode throughput) and [`beyond-quadratic-attention-on-apple-silicon.md`](../papers/beyond-quadratic-attention-on-apple-silicon.md) (sparse attention + sub-quadratic architectures + adaptive compute).
- ✅ Phase-1 scaffolds landed for all Tier 1 and Tier 2 items (#141–#148, #143, #144).
- ✅ Specs 031–034 (sparse-attention prefill/decode roadmap) landed on alpha via PR #188 (2026-05-09); now slotted into Tier 2 rows 10c / 10e / 10f + Tier 3 row 11a.
- ✅ Specs 035 / 036 / 037 (Quest K_max/K_min refinement, DuoAttention, TEAL) landed in PR #189 (2026-05-09); slotted as Tier 4 row 24 / Tier 2 row 10b / Tier 2 row 10d respectively after the 2026-05-09 reordering review.
- ✅ Spec 030 promoted from Tier 4 to Tier 2 row 10a (MTP / EAGLE-3 draft heads — largest pure-decode win, first in the post-Tier-1 priority order).
- ✅ **Tier 1 row #5 — state-replay (spec 020)** shipped 2026-05-11 via [PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143) + cross-repo chain. GDN coverage (Qwen 3.5 / 3.6) live; Mamba opts out for now (kernel variant tracked as future work in spec 020).
- ✅ **Tier 1 row #6 — prefix KV cache (spec 017)** all phases consolidated and ready for merge 2026-05-12 via [PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144). Measured ~4.3× TTFT speedup on Qwen3.5-35B-A3B (2186ms → 499ms cold→warm) under realistic 4-turn chat with captured replies; ~2.5× on Qwen3.5-0.8B / Gemma 4 E2B; ~1.5-1.8× on dense Gemma 4 26B-A4B / 31B; GPT-OSS-20B no-benefit (documented limitation).
- ✅ **Issue [#73](https://github.com/ekryski/mlx-swift-lm/issues/73) closed** — KVCache refactor shipped as spec 006 (PRs #163–#166).
- 🔜 Tier 2 phase-1 scaffolds need their phase-2 integration work; runs in parallel with Tier 1.
- 🔜 **Post-Tier-1 priority order (the post-hoc spec wave):**
  1. **Spec 030 — Native MTP / EAGLE-3 draft heads** — largest universal decode win, mostly model-loader work, lossless. Variant B (Gemma 4 assistant draft) + Variant A on DeepSeek-V4 ship independently. Variant A on hybrid Qwen is **now unblocked** since spec 020 shipped 2026-05-11 — no longer waits on Tier 1.
  2. **Spec 036 — DuoAttention** — slots onto PR #186 windowed turbo cache for streaming heads; ~3–4 weeks for V1.
  3. **Spec 034 — Quest decode-side top-k** — V1 cache-agnostic on `StandardKVCache`; composes with spec 036 (Quest applies to the retrieval-head cache DuoAttention identifies); ~4–6 weeks.
  4. **Spec 037 — TEAL activation thresholding** — block-sparse Metal kernel + per-model calibration; ~3–4 weeks.
  5. **Hybrid model porting** (Granite 4.0-H / Qwen3-Next / Kimi Linear) — Tier 3+; ships after the four post-hoc wins above.
- 🔜 Tier 3 (DFlash + Mirror SD phase-2 work) blocked on external pieces (model availability, CoreML-LLM dep).
- 🔜 Tier 3 spec 033 (block-sparse SDPA Metal kernel) is the long-pole kernel for 034 V2 / 035 V2 / 036 phase 5 — multi-month 4-repo chain.
- 📋 **Issue-tracked perf backlog** (36 open issues catalogued in the section above) — granular work that runs in parallel with the tier roadmap. Pick by size label (S/M/L/XL) and current focus area.
