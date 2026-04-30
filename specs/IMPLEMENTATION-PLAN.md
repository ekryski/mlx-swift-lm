# Implementation plan — speculative decoding rollout

**Last updated:** 2026-04-29
**Owner:** Eric (with research help from Claude)

This is the running implementation order across [specs 013–023](.) plus the forked [`ekryski/CoreML-LLM`](https://github.com/ekryski/CoreML-LLM) integration. Items in earlier tiers should ship before items in later tiers because of measurement dependencies, code dependencies, or risk-management ordering — not because later items are less important.

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
| 6 | **017 — prefix KV cache** | Snapshot target KV state at request end keyed on stable prefix; hydrate on next request with the same prefix. Phase 1 (in-memory cache + key + LRU + stats) landed in [PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144); phase 1B (concrete `KVCache.serialise()` / `hydrate(from:)` per cache type) + phase 2 (chat-aware stable-prefix policy) need to land. CoreML-LLM has working reference (`PrefixCache.swift`, `PrefixKVCache.swift`). | **2–10× TTFT** on multi-turn chat (decoder-agnostic) |

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
| 12 | **021 Phase 1A** — Mirror SD spike | Add `ekryski/CoreML-LLM` as a Swift Package dependency. Glue their `MirrorSpeculativeLoop` to our MLX target via a thin `SpeculativeTarget` adapter. Phase 1A scaffold (protocol + registry + vocab gate) landed in [PR #142](https://github.com/ekryski/mlx-swift-lm/pull/142); phase 1B (real Core ML draft + integration) + phase 2 (full iterator) need to land. **Includes the Core-ML-vs-private-ANE-API benchmark** to characterise dispatch overhead. | **3–5× projected** (Apple Mirror SD paper headline) | High projected win but high integration risk: needs CoreML-LLM Swift Package dep + ANE measurement infra. |

After Phase 1A measurement: if pass, immediately schedule **021 Phase 2** (integrated `MirrorSpeculativeTokenIterator`) — adds 3–4 weeks but lands the headline win on Qwen 3.5/3.6.

## Tier 4 — nice to have / per-model effort

Defer until Tiers 0–3 are solid. Each is bounded effort but per-model rather than universal.

| # | Item | What | When |
|---|---|---|---|
| 13 | **015 phases 4–6 + 021 Variant C** — DFlash-on-ANE | Port DFlash draft to Core ML; run on ANE while target verifies on GPU. Composes Mirror SD parallelism × DFlash K=16 amortisation. | After Tier 3 ships and DFlash-on-GPU is stable. **4–7× projected** on Qwen 3.5-27B. |
| 14 | **014 Phases 2–4** — variable-K tree, bifurcating-on-tight-margin, full suffix-tree merging | Each composes additively with phase 1. Phase 4 = SuffixDecoding port; CoreML-LLM has reference (`SuffixTree.swift` + `SuffixSpeculativeEngine.swift`). | After 014 phase 1 lands and is measured. |

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

021 Phase 1A (Tier 3) ─► 021 Phase 2 ─► 021 Variants B/C/D
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
2. **021 Phase 1A's concurrent-execution check** — Apple Silicon truly running ANE + GPU in parallel without serialisation through XPC / IOSurface / shared queues. Failure here kills 021 entirely (everything in Tier 4 item 13 too).
3. **015 Phase 3's tape-replay correctness on real Qwen 3.5 models** — same numerical-equivalence concern, scoped to DFlash. Depends on Tier 1 #5 landing first.

## Status snapshot — at this commit

- ✅ **Tier 0 complete.** 013 + 018 + 023 + eval harness shipped.
- ✅ Specs 014–023 written.
- ✅ Paper at `papers/speculative-decoding-on-apple-silicon.md` covers the full landscape.
- ✅ Phase-1 scaffolds landed for all Tier 1 and Tier 2 items (#141–#148, #143, #144).
- 🔜 **Tier 1 work — tape-replay #143 phases 2-3 + prefix cache #144 phase 1B** is the next concrete step.
- 🔜 Tier 2 phase-1 scaffolds need their phase-2 integration work; runs in parallel with Tier 1.
- 🔜 Tier 3 (DFlash + Mirror SD phase-2 work) blocked on external pieces (model availability, CoreML-LLM dep).
