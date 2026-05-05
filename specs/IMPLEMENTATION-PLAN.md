# Implementation plan — speculative decoding rollout

**Last updated:** 2026-05-04
**Owner:** Eric (with research help from Claude)

This is the running implementation order across [specs 013–029](.) plus the forked [`ekryski/CoreML-LLM`](https://github.com/ekryski/CoreML-LLM) integration. Items in earlier tiers should ship before items in later tiers because of measurement dependencies, code dependencies, or risk-management ordering — not because later items are less important.

**Reorder note (2026-04-29):** Tier 1 and Tier 3 swapped after PR #154 merged spec 023 (Leviathan accept/reject). The deterministic infrastructure wins (tape-replay, prefix cache, deterministic-stretch, n-gram cache) clear ahead of the more experimental DFlash + Mirror SD bets. Rationale: tape-replay alone unblocks 5+ models for n-gram + Leviathan paths; prefix cache is universal multi-turn TTFT win; both have phase-1 scaffolds landed (#143, #144) and just need the kernels + iterator wiring. DFlash needs `z-lab/Qwen3.5-*-DFlash` model availability; Mirror SD needs CoreML-LLM Swift Package + ANE measurement infra. Lower risk to land deterministic wins first.

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

| # | Item | What | Projected win |
|---|---|---|---|
| 5 | **020 — tape-replay rollback at the cache layer** | Generalise the GDN rollback primitive so any speculative decoder works on hybrid SSM/Mamba models. Phase 1 (protocol + dispatch helpers) landed in [PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143); phases 2–3 (`MambaCache` conformance + Metal kernel + iterator wiring) need to land. | Unlocks n-gram + Leviathan on Qwen 3.5 / 3.6 / Nemotron-H / Jamba — currently those auto-fall-back to plain `TokenIterator` |
| 6 | **017 — prefix KV cache** | Snapshot target KV state at request end keyed on stable prefix; hydrate on next request with the same prefix. Phase 1 (in-memory cache + key + LRU + stats) landed in [PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144); phase 1B (concrete `KVCache.serialise()` / `hydrate(from:)` per cache type) + phase 2 (chat-aware stable-prefix policy) need to land. Phase 3 (cross-session sharing via `BlockAllocator.retain()`) tracked in [#133](https://github.com/ekryski/mlx-swift-lm/issues/133). CoreML-LLM has working reference (`PrefixCache.swift`, `PrefixKVCache.swift`). | **2–10× TTFT** on multi-turn chat (decoder-agnostic) |

## Tier 2 — compounding optimisations

These compose with Tier 1 for additional lift on specific workloads. Each is bounded effort.

| # | Item | What | Projected win | Best workload fit |
|---|---|---|---|---|
| 7 | **022 — deterministic-stretch acceleration** | Chat-template state machine + bigram fallback drafter. Phase 1 (`ChatTemplateGrammar` protocol + `BigramTable`) landed in [PR #145](https://github.com/ekryski/mlx-swift-lm/pull/145); phase 2 (per-family grammars + bigram corpus) needs to land. Highest single-target win is GPT-OSS harmony channel transitions. | **+15–30% on GPT-OSS**, +5% across other model families | Harmony / channel-format models, structured-output generation |
| 8 | **016 — cross-request n-gram cache** | Persist the PLD lookup table across requests on the same model. Three-tier (`nc_context` / `nc_dynamic` / `nc_static`) per llama.cpp. Phases 1-2 (registry + tiered cache) landed in [PR #146](https://github.com/ekryski/mlx-swift-lm/pull/146); phase 4 (three-tier draft selection in iterator) needs to land. | **+10–30%** on multi-turn chat on top of base PLD | Repeated-template generation, agent loops |
| 9 | **014 Phase 1** — tree attention with K=2 root branches | Verify multiple candidate continuations in one forward via tree attention masks. Composes with multi-candidate. Phase 1 (`DraftTree` primitives) landed in [PR #147](https://github.com/ekryski/mlx-swift-lm/pull/147); phase 2 (MLX wiring + iterator integration) needs to land. | **+15–25%** on input-grounded prompts where PLD already wins | Document QA, code editing |
| 10 | **019 — PLD+ attention-weighted span selection** | Hidden-state cosine selection (Phase 1, model-agnostic) then induction-head attention scoring (Phase 2, per-model). Phase 1 (selector protocol + cosine helper) landed in [PR #148](https://github.com/ekryski/mlx-swift-lm/pull/148); phase 2 (per-model conformance + iterator integration) needs to land. | Higher accept rate on multi-candidate hits; +5-15% combined with #9 | Document QA, code editing |

## Tier 3 — primary speedup paths (more experimental)

The two highest-leverage features in the headline numbers, but with significant external dependencies. Land after Tier 1 + Tier 2 stabilise.

| # | Item | What | Projected win | Rationale |
|---|---|---|---|---|
| 11 | **015 phases 1–3** — DFlash on GPU | Port DFlash's Python reference (`bstnxbt/dflash-mlx engine-v2`) to MLX-Swift. Phase 1 (protocol surface + iterator scaffold) landed in [PR #141](https://github.com/ekryski/mlx-swift-lm/pull/141); phases 2 (real draft model from `z-lab/Qwen3.5-*-DFlash`) + 3 (hybrid GDN tape-replay rollback — depends on Tier 1 #5) need to land. | **2.4–4.4× on Qwen 3.5/3.6** | Standalone win once draft-model availability + tape-replay lands. Spec 020 (Tier 1 #5) is a hard prerequisite for phase 3. |
| 12 | **025 — ANE+GPU concurrency primitives** | Race-free cross-device state primitives + concurrency measurement harness for any ANE-offloaded work. Codifies the architectural lessons from the retired AB/ICB track (override-bound fresh allocation, never mutate persistent storage across an async device boundary). 4 phases: measurement harness, buffer lifecycle audit, reference `ANEDraftLoop` primitive, hand-off. **No end-user feature** — pure de-risking infrastructure. See [`sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md`](../../sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md) for the prior-art reasoning. | **0× direct** (de-risking only) | Executes immediately before 021 Phase 1A. Makes the concurrent-execution decision point (#2 below) a clean measurement of the *concurrency hypothesis* rather than a tangle of integration code + concurrency. |
| 13 | **021 Phase 1A** — Mirror SD spike | Add `ekryski/CoreML-LLM` as a Swift Package dependency. Glue their `MirrorSpeculativeLoop` to our MLX target via a thin `SpeculativeTarget` adapter. Phase 1A scaffold (protocol + registry + vocab gate) landed in [PR #142](https://github.com/ekryski/mlx-swift-lm/pull/142); phase 1B (real Core ML draft + integration) + phase 2 (full iterator) need to land. **Includes the Core-ML-vs-private-ANE-API benchmark** to characterise dispatch overhead. **Now depends on spec 025** for the primitives + measurement harness. | **3–5× projected** (Apple Mirror SD paper headline) | High projected win but high integration risk: needs CoreML-LLM Swift Package dep + spec 025 primitives. |

After Phase 1A measurement: if pass, immediately schedule **021 Phase 2** (integrated `MirrorSpeculativeTokenIterator`) — adds 3–4 weeks but lands the headline win on Qwen 3.5/3.6.

## Tier 4 — nice to have / per-model effort

Defer until Tiers 0–3 are solid. Each is bounded effort but per-model rather than universal.

| # | Item | What | When |
|---|---|---|---|
| 13 | **015 phases 4–6 + 021 Variant C** — DFlash-on-ANE | Port DFlash draft to Core ML; run on ANE while target verifies on GPU. Composes Mirror SD parallelism × DFlash K=16 amortisation. | After Tier 3 ships and DFlash-on-GPU is stable. **4–7× projected** on Qwen 3.5-27B. |
| 14 | **014 Phases 2–4** — variable-K tree, bifurcating-on-tight-margin, full suffix-tree merging | Each composes additively with phase 1. Phase 4 = SuffixDecoding port; CoreML-LLM has reference (`SuffixTree.swift` + `SuffixSpeculativeEngine.swift`). | After 014 phase 1 lands and is measured. |
| 15 | **024 — KV cache write fusion** | Eliminate the 60 `copy_bfloat16` dispatches per decode token on Gemma4-E2B (and proportional copies on every other model). Recommended path: extend `mlx`'s `SliceUpdate::eval_gpu` to handle strided source + fix donation — three-repo PR (mlx + mlx-c + mlx-swift) but transparent to all consumers. | After Tier 3 stabilises. **+3–8% decode tok/s** per model, scales with non-shared layer count. Supersedes the deferred `ek/gemma4-e2b-kv-copy-fusion` investigation. |
| 16 | **GPT-OSS-20B attention-sinks on TurboQuant path B** | Four-repo cross-cutting PR chain to make the compressed-domain decode path (`-compact` suffix, `useCompressedAttention=true`) coherent on attention-sinks models. Umbrella issue: [#130](https://github.com/ekryski/mlx-swift-lm/issues/130). Kernel + ABI + Swift wrapper + user-facing plumbing all OPEN; path B + non-sinks works coherently, path B + sinks is wired but disabled because output is incoherent despite kernel math being right on paper. PRs: [mlx#16](https://github.com/ekryski/mlx/pull/16) (kernel + C++ primitive — pass2 folds per-head sink logit into cross-block softmax), [mlx-c#8](https://github.com/ekryski/mlx-c/pull/8) (C ABI + 5 missing decls), [mlx-swift#18](https://github.com/ekryski/mlx-swift/pull/18) (Swift wrapper + submodule bumps), [mlx-swift-lm#99](https://github.com/ekryski/mlx-swift-lm/pull/99) (path B CLI surface + sinks routing). Debug behind `MLX_TURBO_FORCE_BETA_SINKS=1`. | After Tier 3 stabilises. **GPT-OSS-only, path-B-only** — path A is the speed default; this is for users who explicitly need long-context KV compression on sinks models and accept path B's 30–60%-of-path-A decode cost. Outstanding work: identify the divergence between path A (coherent) and path B (incoherent) on GPT-OSS-20B with same sinks tensor. Sliding-window + graph-fuser already ruled out. |
| 17 | **026 — Profile-guided Morton-order expert weight reordering** | Calibration corpus → co-selection matrix → permutation generator (greedy / spectral / Morton) → per-layer `sanitize()` integration. Permanently reorders MoE expert weights so frequently co-selected experts are adjacent in memory. Eliminates the sort-vs-no-sort tradeoff at small batch sizes. **Experimental** — generalization across corpora is the load-bearing risk. | After Tier 3 stabilises. **+25% prefill at T=128 on MoE models** (recover unsorted-path penalty); +0–5% at T≥1024; +0–3% at decode. Zero runtime cost — all work is offline + once at sanitize(). |
| 18 | **027 — Adaptive per-layer mixed-precision quantization framework** | Recipe-driven framework for per-module bit-widths via JSON sidecar + glob-pattern matching. Generalizes the ad-hoc Unsloth UD-MLX pattern (closes [#74](https://github.com/ekryski/mlx-swift-lm/issues/74)) and unlocks recipes like `int4-lm-head` for bf16 models. **Experimental** — value depends on recipe library quality; bf16 audience is small. | After Tier 3 stabilises. **+20× LM head decode** for users who opt into `int4-lm-head` on bf16 models (small audience but huge per-user impact). Closes #74 as a side effect. |
| 19 | **028 — Quadratic / chunkwise WY GatedDeltaNet prefill** | Parallelize the GDN recurrence via chunkwise Woodbury-Young + short-context quadratic-attention reformulation. **Research-grade** — a prior quadratic-attention experiment regressed at Dk=128 (cause inconclusive). Multi-month work: Python reference → naive Metal port → fused kernel → quadratic-vs-chunked dispatch → per-model integration. | After spec-decode (Tier 1–3) stabilises. **+5–15× prefill on Qwen 3.5 family** if the kernel works; could regress if it doesn't. Highest research bet in the plan. |
| 20 | **029 — ANE-offloaded LM head + Gemma 4 PLE projection** | Use spec 025's ANE+GPU concurrency primitives to overlap fixed-shape projections (LM head, Gemma 4 PLE) on ANE while the GPU does next-layer work. **Hard dependency on spec 025 phase 1 measurement passing** — if Apple Silicon doesn't run ANE+GPU concurrently, this spec dies (along with spec 021). | After spec 025 phase 1 + 2 land. **+5–15% per-token on Gemma 4 E2B** (LM head + PLE both offloaded); +3–8% on other large-vocab models (LM head only); no win on Qwen 3.5 (smaller vocab). |

## Issue-tracked perf backlog

Granular perf work, each in its own GitHub issue. No fixed ordering — pick up in priority order based on size labels (S/M/L/XL) and current focus area. Runs in parallel with the tier roadmap above. Issues that map onto a spec-level item are listed under that spec's tier row; the categories below cover everything else.

### Decode tok/s

| Issue | Topic | Size |
|---|---|---|
| [#114](https://github.com/ekryski/mlx-swift-lm/issues/114) | Long-context fallback to TurboFlash for large models | S |
| [#115](https://github.com/ekryski/mlx-swift-lm/issues/115) | QKV batched fusion via `MLXFast.batchedQKVQuantizedGEMV` | M |
| [#116](https://github.com/ekryski/mlx-swift-lm/issues/116) | Gate+Up MLP fusion | M |
| [#117](https://github.com/ekryski/mlx-swift-lm/issues/117) | RMSNorm + GEMV fusion for MLP and attention | L |
| [#118](https://github.com/ekryski/mlx-swift-lm/issues/118) | Fold V output rotation into Wo at codec init | S |
| [#119](https://github.com/ekryski/mlx-swift-lm/issues/119) | `compile()` coverage gaps on Qwen 3.5 | S |
| [#120](https://github.com/ekryski/mlx-swift-lm/issues/120) | Eliminate GQA tile in rawKeyMode path | M |
| [#121](https://github.com/ekryski/mlx-swift-lm/issues/121) | NR0 > 2 in TurboFlash kernel (multi-repo) | M |
| [#122](https://github.com/ekryski/mlx-swift-lm/issues/122) | f32 rotation precision A/B (low priority — only if quality regression surfaces) | S |
| [#157](https://github.com/ekryski/mlx-swift-lm/issues/157) | Int4 LM head quantization for bf16-weight models — **>+50% decode** on bf16 Gemma 4 / Qwen 3.5 (small audience, big per-user impact). Standalone version of spec 027's `int4-lm-head` recipe. | S |
| [#158](https://github.com/ekryski/mlx-swift-lm/issues/158) | Float16 V accumulator in TurboFlash pass2 kernel (separate from the bf16 output dtype that landed in `74024a24`) — ~5–10% attention kernel improvement | S |
| [#159](https://github.com/ekryski/mlx-swift-lm/issues/159) | Symbolic sliding-window SDPA mask for GPT-OSS-20B (parity with Gemma 4 PR #55) — 5–10% decode at 4k+ ctx | S |
| [#160](https://github.com/ekryski/mlx-swift-lm/issues/160) | Persistent FP16 dequant cache for TurboQuant path A — keep dequant resident across decode steps; trades steady-state memory for ≥5% decode at long context | M |
| [#161](https://github.com/ekryski/mlx-swift-lm/issues/161) | TokenRing → CPU `Set<Int>` for presence/repetition penalties — **investigation first** since current code claims "no CPU←GPU sync" but the audit doc disagrees; 2–5% decode if confirmed | S |

### Prefill / TTFT

| Issue | Topic | Size |
|---|---|---|
| [#123](https://github.com/ekryski/mlx-swift-lm/issues/123) | Adaptive GatedDeltaNet `evalInterval` tuning | S |
| [#124](https://github.com/ekryski/mlx-swift-lm/issues/124) | NemotronH peak-memory regression mystery (investigation) | M |
| [#125](https://github.com/ekryski/mlx-swift-lm/issues/125) | Async prefill compression for TurboQuant KV | M |
| [#126](https://github.com/ekryski/mlx-swift-lm/issues/126) | Tokenization / chat-template caching | S |
| [#156](https://github.com/ekryski/mlx-swift-lm/issues/156) | Verify non-fused `gated_delta_step` Metal kernel correctness — kernel + ops fallback both still in tree; active path unclear; if kernel is broken, remove or fix; if fixed, switch dispatch and measure prefill win on Qwen 3.5 family | M |

### Memory / paged KV

| Issue | Topic | Size |
|---|---|---|
| [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) | Metal paged-attention kernel for `PagedKVCache` | L |
| [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) | Wire `PagedKVCache` into Qwen 3 / Gemma 4 / etc model factories | M |
| [#129](https://github.com/ekryski/mlx-swift-lm/issues/129) | TurboQuant + paged integration | L |

### Speculation umbrellas (concrete paths in the tier roadmap above)

| Issue | Topic | Size |
|---|---|---|
| [#132](https://github.com/ekryski/mlx-swift-lm/issues/132) | EAGLE / MEDUSA-style draft decoding. EAGLE-3 covered for free by spec 021 (Mirror SD); MEDUSA still needs draft-head training infra and is independent of 021. | XL |
| [#133](https://github.com/ekryski/mlx-swift-lm/issues/133) | Prefix caching across sessions (referenced from Tier 1 row #6) — implements spec 017 phase 3 via `BlockAllocator.retain()` | L |

### Batching follow-ups (to PR #138's `generateBatched`)

| Issue | Topic | Size |
|---|---|---|
| [#149](https://github.com/ekryski/mlx-swift-lm/issues/149) | TurboQuant decode regresses 0.60× at long-context B>1 | M |
| [#150](https://github.com/ekryski/mlx-swift-lm/issues/150) | Variable-length prompts + per-sequence EOS for `generateBatched` | M |
| [#151](https://github.com/ekryski/mlx-swift-lm/issues/151) | Continuous batching — admit prompts while decode is in flight | L |

### Investigation / cleanup

| Issue | Topic | Size |
|---|---|---|
| [#134](https://github.com/ekryski/mlx-swift-lm/issues/134) | xctrace inspection of dequant kernel (informs [#121](https://github.com/ekryski/mlx-swift-lm/issues/121)) | S |
| [#155](https://github.com/ekryski/mlx-swift-lm/issues/155) | `--dispatch-audit` flag for in-process Metal dispatch counting (CI-friendly counter; complements `MLX_METAL_PROFILE`). Salvageable from the abandoned AB/ICB track — see post-mortem in `sam/planning/performance-notes/ab-icb-postmortem-2026-05-04.md`. | S |

### Architecture proposals

| Issue | Topic | Size |
|---|---|---|
| [#73](https://github.com/ekryski/mlx-swift-lm/issues/73) | Refactor how we do KVCache to make it cleaner | L |
| [#74](https://github.com/ekryski/mlx-swift-lm/issues/74) | Add support for Unsloth dynamic quants (UD-MLX checkpoints) | M |

### Open questions / pending verification

| Issue | Topic | Size |
|---|---|---|
| [#77](https://github.com/ekryski/mlx-swift-lm/issues/77) | Investigate dual usage of quant/dequant values in TurboFlash kernel (mineable from PR #93's `60bd16d`) | M |
| [#89](https://github.com/ekryski/mlx-swift-lm/issues/89) | TurboQuant decode regression in `generate()` path but NOT in bridge-direct path — may be obsolete after recent PRs; needs re-bench | M |

### Bugs

| Issue | Topic | Size |
|---|---|---|
| [#61](https://github.com/ekryski/mlx-swift-lm/issues/61) | `Gemma3TextModel` crashes when `hiddenLayers < slidingWindowPattern` | S |

### Spec-decode policy

| Issue | Topic | Size |
|---|---|---|
| [#153](https://github.com/ekryski/mlx-swift-lm/issues/153) | n-gram speculative decoding: auto-disengage on known regression regimes (small/fast models + paraphrastic workloads). Subsumes spec 023 phase 3. | M |

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

020 (Tier 1, phase 1 scaffold landed)
  └─► n-gram + Leviathan on Qwen 3.5/3.6 / Nemotron-H / Jamba (auto-routing
      predicate change)
  └─► 015 phase 3 (hybrid GDN path for DFlash)
  └─► 016 phase 3 (hybrid path for ngram cache)
  └─► 017 phase 3 (hybrid prefix-cache snapshots)

015 phases 1-3 (Tier 3) ─┬─► 015 phases 4-6 ─┬─► 021 Variant C
                         │                    │
                         └─► depends on 020 phase 2-3

025 (Tier 3 prework) ─► 021 Phase 1A (Tier 3) ─► 021 Phase 2 ─► 021 Variants B/C/D
                                                                  │
                                                                  └─► Tier 4 row 13 (DFlash-on-ANE) reuses 025 primitives
```

## What we're not doing

- **EAGLE-3 standalone port.** CoreML-LLM has it (`MirrorSpeculativeLoop.swift` runs an EAGLE-3 draft). Once Mirror SD is wired in via 021, EAGLE-3 comes along for free. Skip the standalone port.
- **MTP heads independent of CoreML-LLM.** Same reasoning — `MtpSpeculativeEngine.swift` is shipped upstream. Wire it in via 021's iterator surface, don't reinvent.
- **Private ANE API direct shim.** Measured in 021 Phase 1A but not shipped. Decision deferred until measurement comes back.
- **Cross-vocabulary speculative decoding** (different tokenizers for draft and target). Available in CoreML-LLM (`CrossVocabSpeculativeEngine.swift`) but lower priority than same-tokenizer paths.
- **Spec 023 phase 2** (unify Leviathan + greedy paths). Closed out without implementation — see spec 023 for the three correctness/perf risks.
- **Spec 023 phase 3** (self-disengage on chronically-low accept). Subsumed by issue #153 — should apply uniformly to greedy and Leviathan when implemented.

## Decision points

The plan assumes pass at each measurement gate. Re-evaluate if any of these come back negative:

1. **020 phase 2's `MambaCache` tape-replay matches a sequential reference** — bf16 numerical drift could compromise rollback correctness. If we can't get bit-exact-enough replay, hybrid models stay on `TokenIterator` indefinitely.
2. **021 Phase 1A's concurrent-execution check** — Apple Silicon truly running ANE + GPU in parallel without serialisation through XPC / IOSurface / shared queues. **Now operationalised by spec 025's Phase 1 measurement harness** — failure surfaces as a clean primitive-level diagnostic before 021 integration code is written. Failure still kills 021 + Tier 4 row 13, but the cost of finding out drops from ~1–2 weeks of integration debugging to ~3–5 days of a focused harness.
3. **015 Phase 3's tape-replay correctness on real Qwen 3.5 models** — same numerical-equivalence concern, scoped to DFlash. Depends on Tier 1 #5 landing first.

## Status snapshot — at this commit

- ✅ **Tier 0 complete.** 013 + 018 + 023 + eval harness shipped.
- ✅ Specs 014–029 written.
- ✅ Paper at `papers/speculative-decoding-on-apple-silicon.md` covers the full landscape.
- ✅ Phase-1 scaffolds landed for all Tier 1 and Tier 2 items (#141–#148, #143, #144).
- 🔜 **Tier 1 work — tape-replay #143 phases 2-3 + prefix cache #144 phase 1B** is the next concrete step.
- 🔜 Tier 2 phase-1 scaffolds need their phase-2 integration work; runs in parallel with Tier 1.
- 🔜 Tier 3 (DFlash + Mirror SD phase-2 work) blocked on external pieces (model availability, CoreML-LLM dep).
- 📋 **Issue-tracked perf backlog** (36 open issues catalogued in the section above) — granular work that runs in parallel with the tier roadmap. Pick by size label (S/M/L/XL) and current focus area.
