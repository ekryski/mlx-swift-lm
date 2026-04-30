# Implementation plan — speculative decoding rollout

**Last updated:** 2026-04-29
**Owner:** Eric (with research help from Claude)

This is the running implementation order across [specs 013–022](.) plus the forked [`ekryski/CoreML-LLM`](https://github.com/ekryski/CoreML-LLM) integration. Items in earlier tiers should ship before items in later tiers because of measurement dependencies, code dependencies, or risk-management ordering — not because later items are less important.

## Tier 0 — land what's already in flight (this week)

These should merge first so the rest of the plan has a clean baseline.

| # | Item | What | Why now |
|---|---|---|---|
| 1 | **013 close-out** | Land multi-candidate / strict-greedy / adaptive defaults; the AR-batch optimisation; the auto-routing in `MLXLMCommon.generate(...)`; the fix for the `for token in iterator` copy bug; the new realistic prompt set | Already implemented + tested; just needs to merge. Everything downstream measures against it. |
| 2 | **018 — bench per-prompt sweep mode** | `--method ngram-spot` (1 prompt × N config matrix) and `--method ngram-sweep-summary` (sweep with per-category best-cell roll-up) | Smallest spec. Replaces the ad-hoc shell loops we used during 013 debugging. Used by every subsequent measurement step in this plan. |

## Tier 1 — primary speedup paths (next 4–6 weeks)

The two highest-leverage features in the entire roadmap. Run in parallel — they don't share code paths.

| # | Item | What | Projected win | Rationale |
|---|---|---|---|---|
| 3 | **015 phases 1–3** — DFlash on GPU | Port DFlash's Python reference (`bstnxbt/dflash-mlx engine-v2`) to MLX-Swift. Phases: (1) stub draft + linear verify, (2) real draft model from `z-lab/Qwen3.5-*-DFlash`, (3) hybrid GDN tape-replay rollback | **2.4–4.4× on Qwen 3.5/3.6** | Standalone win; no cross-framework risk. Produces the Swift-side DFlash reference needed for spec 021's Variant C. |
| 4 | **021 Phase 1A** — Mirror SD spike | Add `ekryski/CoreML-LLM` as a Swift Package dependency. Glue their `MirrorSpeculativeLoop` to our MLX target via a thin `SpeculativeTarget` adapter. Measure end-to-end on Qwen 3.5-27B-4bit. **Includes the Core-ML-vs-private-ANE-API benchmark** to characterise dispatch overhead. | **3–5× projected** (Apple Mirror SD paper headline) | One week of integration glue. Apple's published prior art; CoreML-LLM has the working reference implementation. The single highest-impact single spec in the roadmap if Phase 1A measurement passes. |

After Phase 1A measurement: if pass, immediately schedule **021 Phase 2** (integrated `MirrorSpeculativeTokenIterator`) — adds 3–4 weeks but lands the headline win on Qwen 3.5/3.6.

## Tier 2 — orthogonal infrastructure (parallel with Tier 1)

These benefit every decoder family. Land any time after Tier 0 — they don't block Tier 1, but Tier 1 doesn't block them either.

| # | Item | What | Projected win |
|---|---|---|---|
| 5 | **020 — tape-replay rollback at the cache layer** | Generalise the GDN rollback primitive used in DFlash so it works for any speculative decoder, including PLD. Unblocks PLD on Qwen 3.5/3.6 (currently falls back to baseline on hybrid models). | Unlocks PLD on the entire hybrid-model family + cleans up DFlash's hybrid path |
| 6 | **017 — prefix KV cache** | Snapshot target KV state at request end keyed on stable prefix; hydrate on next request with the same prefix. CoreML-LLM has working reference (`PrefixCache.swift`, `PrefixKVCache.swift`) — port the design. | **2–10× TTFT** on multi-turn chat (decoder-agnostic) |

## Tier 3 — compounding optimisations

These compose with Tiers 1 and 2 for additional lift on specific workloads.

| # | Item | What | Projected win | Best workload fit |
|---|---|---|---|---|
| 7 | **022 — deterministic-stretch acceleration** | Chat-template state machine + bigram fallback drafter. Highest single-target win is GPT-OSS harmony channel transitions. | **+15–30% on GPT-OSS**, +5% across other model families | Harmony / channel-format models, structured-output generation |
| 8 | **016 — cross-request n-gram cache** | Persist the PLD lookup table across requests on the same model. Three-tier (`nc_context` / `nc_dynamic` / `nc_static`) per llama.cpp. | **+10–30%** on multi-turn chat on top of base PLD | Repeated-template generation, agent loops |
| 9 | **014 Phase 1** — tree attention with K=2 root branches | Verify multiple candidate continuations in one forward via tree attention masks. Composes with multi-candidate. | **+15–25%** on input-grounded prompts where PLD already wins | Document QA, code editing |
| 9b | **023 — Leviathan accept/reject sampling for n-gram** | Lift the `temperature == 0` requirement on n-gram speculative decoding. Standard accept/reject formula simplifies for n-gram because draft `q` is degenerate. | **1.2–1.4× at temperature > 0** on input-grounded prompts (which are the majority of real-world calls) | Sampling-based chat, RAG, creative writing |

## Tier 4 — nice to have / per-model effort

Defer until Tiers 0–3 are solid. Each is bounded effort but per-model rather than universal.

| # | Item | What | When |
|---|---|---|---|
| 10 | **015 phases 4–6 + 021 Variant C** — DFlash-on-ANE | Port DFlash draft to Core ML; run on ANE while target verifies on GPU. Composes Mirror SD parallelism × DFlash K=16 amortisation. | After Tier 1 ships and DFlash-on-GPU is stable. **4–7× projected** on Qwen 3.5-27B. |
| 11 | **019 — PLD+ attention-weighted span selection** | Hidden-state cosine selection (Phase 1, model-agnostic) then induction-head attention scoring (Phase 2, per-model). | After 014 Phase 1 ships — composes with tree attention's multi-candidate path. |
| 12 | **014 Phases 2–4** — variable-K tree, bifurcating-on-tight-margin, full suffix-tree merging | Each composes additively with phase 1. Phase 4 = SuffixDecoding port; CoreML-LLM has reference (`SuffixTree.swift` + `SuffixSpeculativeEngine.swift`). | After 014 phase 1 lands and is measured. |

## Dependency graph (the tight ones only)

```
013 (closed) ─┬─► 018 ─► everything else (measurement plumbing)
              │
              ├─► 022 (chat template state machine)
              ├─► 016 (ngram cache, llama.cpp port)
              └─► 014 phase 1 (tree attention)

015 phases 1-3 ─┬─► 015 phases 4-6 ─┬─► 021 Variant C
                │                    │
                └─► 020 ─► 017 hybrid path
                          └─► 016 hybrid path

021 Phase 1A ─► 021 Phase 2 ─► 021 Variants B/C/D

020 ─► PLD on Qwen 3.5/3.6 (auto-routing predicate change)
```

## What we're not doing

- **EAGLE-3 standalone port.** CoreML-LLM has it (`MirrorSpeculativeLoop.swift` runs an EAGLE-3 draft). Once Mirror SD is wired in via 021, EAGLE-3 comes along for free. Skip the standalone port.
- **MTP heads independent of CoreML-LLM.** Same reasoning — `MtpSpeculativeEngine.swift` is shipped upstream. Wire it in via 021's iterator surface, don't reinvent.
- **Private ANE API direct shim.** Measured in 021 Phase 1A but not shipped. Decision deferred until measurement comes back.
- **Cross-vocabulary speculative decoding** (different tokenizers for draft and target). Available in CoreML-LLM (`CrossVocabSpeculativeEngine.swift`) but lower priority than same-tokenizer paths.

## Decision points

The plan assumes pass at each measurement gate. Re-evaluate if any of these come back negative:

1. **018 lands and confirms 013's defaults are right** — if the per-prompt sweep finds a better default than `adaptive=on, strict-greedy=on`, flip them.
2. **021 Phase 1A's concurrent-execution check** — Apple Silicon truly running ANE + GPU in parallel without serialisation through XPC / IOSurface / shared queues. Failure here kills 021 entirely (everything in Tier 4 item 10 too).
3. **015 Phase 3's tape-replay correctness on real Qwen 3.5 models** — bf16 numerical drift in the replay kernel. If we can't get bit-exact-enough replay, the rollback is too lossy and DFlash's lossless guarantee is compromised.
4. **020's MambaCache tape-replay matches a sequential reference** — same numerical-equivalence concern, broader scope.

## Status snapshot — at this commit

- ✅ 013 implemented + tested (multi-candidate, dominance, strict-greedy, adaptive, AR-batch, auto-routing). Defaults flipped to on for adaptive + strict-greedy. 13/13 tests green. Awaiting merge.
- ✅ Specs 014–022 written.
- ✅ Paper at `papers/speculative-decoding-on-apple-silicon.md` covers the full landscape with benchmarks.
- 🔜 Tier 0 work (013 merge + 018 bench mode) is the next concrete step.
- 🔜 Tier 1 work — DFlash on GPU and Mirror SD spike — runs in parallel after Tier 0.
