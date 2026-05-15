# 022 — Deterministic-stretch acceleration ("free-token short-circuit")

- **Status:** 🚧 Phase 1 scaffold landed ([PR #145](https://github.com/ekryski/mlx-swift-lm/pull/145) — `ChatTemplateGrammar` protocol + `BigramTable`). Phase 2 (per-family grammars + bigram corpus) **not started.**
- **Branch:** Phase 1 merged via PR #145; Phase 2 gets a fresh branch off alpha.
- **Depends on:** spec 013 (n-gram iterator). Composes naturally with spec 016 (cross-request cache) and spec 014 (tree attention) but doesn't require them.

## The premise

In any LLM output, some token positions are **structurally determined** before the model runs:

- After `<|im_start|>`, the next tokens are `assistant\n` or `user\n` per the chat template.
- After `<|channel|>` (GPT-OSS harmony), the next token is `analysis` or `final` or similar from a known small set; the token after THAT is `<|message|>`.
- Inside a Markdown numbered list, after `1. {content}\n`, if the model is continuing the list the next tokens are `2. ` with probability ~1.
- Inside a code block opened with triple-backtick + language, after the body and a `\n`, the closing tokens are typically the matching triple-backtick + newline.
- In repeated-template generation (the recipe-bulk / blog-series workloads), after the first instance the template separator tokens (`---\n\n# `) recur with high determinism.

For these positions the next K tokens are knowable from a CPU-side state machine — no forward pass needed to *predict* them. A forward pass is still needed to *commit* them to the KV cache, but **one forward pass can commit K of them at once** instead of K sequential decodes.

This is structurally the same as PLD's verify-batch trick, but with a ~100%-accept-rate oracle instead of a frequency-based heuristic. The wins compound:
- Frequent small wins (chat template boundaries fire 2-4× per turn).
- Occasional large wins (templated bulk generation: spec gives K=12+ effective draft length).

## Two complementary mechanisms

### A. Chat-template state machine (deterministic, narrow)

A small per-tokenizer-family state machine that knows the chat template's grammar. At any decode step, the iterator can ask: "given the last N emitted tokens, what's the next deterministic stretch (if any)?"

For Qwen / Gemma / GPT-OSS / Llama, the relevant deterministic stretches are short (1-6 tokens) but happen reliably:

```
After EOS detection in assistant turn:    stop (no more emit)
After </think> (Qwen / Gemma 4 thinking): the next 0-3 tokens are usually empty or whitespace
After <|channel|> (GPT-OSS harmony):       <channel-name> <|message|>  → 2-3 tokens
After "1. " / "2. " (in numbered list):    rest of "N. " template if continuing — speculative
```

The state machine emits deterministic tokens as drafts to the iterator's verify path. The verifier still runs and confirms. If the model disagrees (e.g., a Gemma 4 thinking-budget processor forced an early `</think>`), the draft is rejected and we fall through to AR.

**This is lossless and bounded — at most ~15-20 tokens per turn come through this path. The win per turn is ~3-5 forwards saved (mostly at turn boundaries and within structured-emission stretches).**

### B. Common-bigram fallback drafter (probabilistic, broad)

When the n-gram lookup misses (PLD finds no match in prompt + history), the iterator currently falls through to AR. Instead, query a small bigram table built from a frequency analysis of the model's output corpus:

```
P(next | last 2 tokens)
```

Threshold at high probability (`P > 0.95`) — extract the deterministic-ish next token and propose it as a 1-token draft. For multi-token drafts, walk the bigram chain greedily until probability drops below threshold or K tokens accumulated.

The bigram table is small (~100K-1M entries), CPU-resident, and lookup is sub-µs. It's keyed on the same tokenizer as the target so token IDs match directly.

**This is "lossless when verify accepts" — verify is still mandatory; the bigram is just providing the drafts. Wins where PLD misses but local statistics still predict.**

The two mechanisms compose: A fires at template boundaries (high-confidence drafts), B fires inside content stretches when PLD misses (medium-confidence drafts). PLD itself fires when the prompt provides verbatim continuations.

```
proposeDraft strategy ladder:
    1. NGramLookup hit (verbatim from prompt + history)         ← spec 013
    2. Cross-request NGramCache hit (history from prior turns)  ← spec 016
    3. Chat-template state machine hit                          ← this spec, mechanism A
    4. Common-bigram fallback                                   ← this spec, mechanism B
    5. AR fallback (no draft, single autoregressive step)       ← spec 013
```

## Why this isn't "skip the forward pass entirely"

Two correctness traps to call out, since the framing "auto-accept without verify" is tempting:

**Trap 1 — KV cache divergence.** The forward pass writes K/V to the cache for every input token. Skipping the forward pass means the cache lacks K/V entries for tokens we've emitted. The next forward pass attends over a cache that's missing recent context — output diverges from baseline immediately. Workarounds (replay the missing tokens later) convert "skip" into "batch-verify-later" — which is what this spec actually proposes.

**Trap 2 — heuristic divergence.** Even tokens that are "100% deterministic from the chat template" can be wrong. A thinking-budget processor can force an early `</think>` insertion before the model would have emitted it. A logit-mask processor (banned tokens, repetition penalty) can shift the model's argmax away from the bigram's prediction. We never get to assume the heuristic is right; we always verify.

Both mechanisms in this spec sidestep both traps by **routing drafts through the existing verify pipeline**. The "free" framing is correct in the sense that we save the *sequential* forward passes that would have decoded K tokens one at a time; we still pay the *one* forward pass that verifies them as a batch.

## Design

### A.1 — Chat-template state machine

Define a small grammar per tokenizer family:

```swift
public protocol ChatTemplateGrammar {
    /// Given the just-emitted token and current chat state, return the
    /// deterministic continuation if any. Nil if no deterministic stretch
    /// applies at this position.
    func deterministicContinuation(
        afterToken token: Int,
        state: ChatTemplateState
    ) -> [Int]?
}

public struct ChatTemplateState {
    let phase: Phase              // .userTurn / .assistantTurn / .thinking / .channelMarker
    let recentTokens: [Int]       // last 8 tokens for n-gram-style lookback
    let chatTemplateConfig: ChatTemplate  // tokenizer-resolved special token IDs
}
```

Per-family implementations live in `Libraries/MLXLMCommon/ChatTemplateGrammar/`:

- `Qwen35Grammar.swift` — handles `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`.
- `Gemma4Grammar.swift` — handles `<start_of_turn>model\n`, `<end_of_turn>`.
- `GPTOSSGrammar.swift` — harmony channel: after `<|channel|>` the next token is in `{analysis, final, commentary}`, after that `<|message|>`.
- `LlamaGrammar.swift` — `<|start_header_id|>assistant<|end_header_id|>\n\n`.

These resolve at iterator construction (the `Tokenizer` already provides the special token IDs we need). The grammar object is queried inside `proposeDraft` after PLD/cache misses but before bigram fallback:

```swift
if let detTokens = chatGrammar.deterministicContinuation(...) {
    return detTokens  // 100% accept-rate drafts
}
```

The deterministic draft is treated identically to a PLD draft by the verify path; if the model disagrees (which it shouldn't for genuinely deterministic stretches but can with logit processors), the standard accept walk handles it.

### A.2 — When the state machine fires

By construction, only at known structural boundaries:

- Just after the assistant-turn opener has been processed (the template's `\n` after `assistant`). Not very useful — the next token is content, not template.
- Just after a thinking-end marker (`</think>` for Qwen 3.5, channel transitions for GPT-OSS). The state machine can predict the small handful of newlines / blanks before content begins.
- At the END of a turn — after the model emits its EOS-like token, the state machine knows the turn is over and can short-circuit final cleanup. Bounded; mostly cosmetic.

The biggest single win is **GPT-OSS harmony channel transitions**, where after every `<|channel|>` the iterator can propose a 3-token deterministic draft: `<channel-name>` `<|message|>` `\n`. Today that's three full forward passes; with the state machine, one. Per turn there can be 5-15 channel transitions, so this saves ~10-40 forward passes per turn on harmony-format models.

### B.1 — Common-bigram fallback

Build a CPU-side bigram table per tokenizer-family. Source data: a few hundred MB of representative text generated by the target model (or a similar one in the same family) on diverse prompts. For each ordered pair `(token_{i-1}, token_i)`, record:

```
table[(t_{i-1}, t_i)] = top-1 next token + its frequency
```

Filter to entries where top-1 has frequency ≥ 0.95 of all observed continuations. The result is a sparse map: for the tokens that *do* have a high-confidence bigram successor, we have one. Tokens that don't — sentence-level content tokens — aren't in the table.

Lookup at decode time:

```swift
func bigramDraft(maxK: Int, recentTokens: [Int]) -> [Int] {
    var draft: [Int] = []
    var prev = recentTokens.suffix(2).asArray()
    for _ in 0..<maxK {
        guard let next = bigramTable.confidentNext(after: prev) else { break }
        draft.append(next)
        prev = [prev.last!, next]
    }
    return draft
}
```

Walk the chain greedily until we either hit `maxK` or the table doesn't have a confident successor. Drafts of length 1-3 are common; 4+ are rare for non-templated content.

### B.2 — Bigram corpus + table size

Build artifact: `Resources/bigrams/{tokenizer-family}.bin`. Compact binary: ~100K-1M entries × 12 bytes = 1-12 MB per tokenizer family. Loaded at iterator construction.

Build pipeline (offline tool, similar to the static n-gram cache from spec 016 phase 3):

```bash
swift run mlx-build-bigram-table \
    --tokenizer-family qwen3 \
    --corpus path/to/qwen-outputs/*.txt \
    --threshold 0.95 \
    --output Resources/bigrams/qwen3.bin
```

Run once per tokenizer family. The result ships with the package.

### C — Routing

In `NGramSpeculativeTokenIterator.speculateRound`, the `proposeDraft` ladder becomes:

```swift
let draft: [Int]
if let det = chatGrammar?.deterministicContinuation(
    afterToken: lastEmitted, state: chatState) {
    draft = det
} else if let pld = lookup.proposeDraft(maxDraft: budget, ...) {
    draft = pld
} else if let bigram = bigramTable?.bigramDraft(maxK: budget, recentTokens: recent) {
    draft = bigram
} else {
    draft = []  // AR fallback path
}
```

Order: state machine first (highest confidence), then PLD (input-grounded — the existing path), then bigram (frequency-based), then AR. Each layer's misses cascade to the next.

Behind a feature flag — `MLX_NGRAM_FREE_TOKENS=1` — to allow A/B comparison. Default ON once measured.

## Expected impact

### Mechanism A (chat-template state machine)

Per-turn savings:

| Model family | Template boundaries / turn | Forwards saved | Per-turn impact at 30 tok/s baseline |
|---|---|---|---|
| Qwen 3.5 (think mode) | 4-6 (think open/close, EOS) | ~5 | ~150 ms saved → ~5% |
| GPT-OSS (harmony) | 10-30 (every channel transition) | ~15-40 | ~500-1300 ms saved → ~15-40% |
| Gemma 4 (chat) | 2-3 (turn boundaries) | ~3 | ~100 ms saved → ~3% |
| Llama / generic (no thinking) | 1-2 | ~1-2 | <2% |

GPT-OSS in harmony mode is the standout — the channel-marker transitions are *frequent* and the deterministic stretches are 2-3 tokens each, so the savings compound. On a single 200-token turn we estimate **20-30% throughput improvement** from mechanism A alone on GPT-OSS.

### Mechanism B (common-bigram fallback)

Bigram fallback only fires when PLD misses and we'd otherwise go AR. Win is bounded by the rate of bigram hits with K ≥ 1.

A rough estimate from spot-checking the recipe-bulk and code-refactor outputs:

- ~30% of generated tokens are in deterministic-bigram regions (after `,`, `.`, `\n`, function-name closing parens, etc.)
- Of those, average bigram chain length when it fires is 2-3 tokens
- So roughly 30% × 2.5 = ~75% of decodes get *some* bigram draft

That's a lot, but most of the bigram drafts are short. Net throughput impact:

```
forwards_per_emit_baseline    ≈ 1.0
forwards_per_emit_w_bigram    ≈ 1.0 / (1 + 0.3 × 1.5)  ≈ 0.69
=> ~45% throughput improvement on workloads where PLD misses heavily
```

That's optimistic; real number is probably half that after accounting for the per-round verify overhead. Still: meaningful on creative-writing prompts where today PLD is net-negative.

### Combined with other specs

When all the spec-decode pieces are layered:

- PLD + cross-request cache (spec 016)
- Multi-candidate + adaptive + strict-greedy (already shipped)
- Chat-template state machine (this spec, A)
- Bigram fallback (this spec, B)
- Tree attention (spec 014)

…on a target like Gemma 4 26B A4B serving a multi-turn chat, the conservative estimate is **2-3× over baseline** — comparable to DFlash's lower bound but achieved via composable, model-agnostic mechanisms with no per-target draft training.

## Implementation phases

1. **Phase 1 — Chat-template state machine for the four supported families.** ~600 lines of Swift across one core module + per-family grammars. No external dependencies. Land first; gain GPT-OSS harmony win immediately.

2. **Phase 2 — Bigram table builder + Qwen 3 / Gemma 4 / GPT-OSS / Llama tables.** Offline tool + 4× small binary resources. Build pipeline runs in CI on a representative corpus.

3. **Phase 3 — Routing wiring + bench harness category.** New `--method free-tokens` benchmark mode that compares baseline / PLD / PLD+state-machine / PLD+bigram / PLD+both side-by-side on the prompt suite.

4. **Phase 4 — Composition with tree attention (spec 014).** When tree drafting is available, the state machine and bigram can each contribute one root branch in the K-way tree. Effectively widens the draft-source set.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/ChatTemplateGrammar.swift` (new) | `ChatTemplateGrammar` protocol, `ChatTemplateState`, registry. |
| `Libraries/MLXLMCommon/ChatTemplateGrammar/Qwen35.swift` (new) | Per-family grammar. |
| `Libraries/MLXLMCommon/ChatTemplateGrammar/Gemma4.swift` (new) | Per-family grammar. |
| `Libraries/MLXLMCommon/ChatTemplateGrammar/GPTOSSHarmony.swift` (new) | Per-family grammar — the highest-leverage one. |
| `Libraries/MLXLMCommon/ChatTemplateGrammar/Llama.swift` (new) | Per-family grammar. |
| `Libraries/MLXLMCommon/BigramTable.swift` (new) | `BigramTable` loader + lookup. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Wire grammar + bigram into `proposeDraft` ladder. Track which source produced each draft for telemetry. |
| `Sources/Tools/MLXBuildBigramTable/` (new) | Offline tool. |
| `Resources/bigrams/{family}.bin` (new) | Binary tables, 1-12 MB each. |
| `Tests/MLXLMTests/ChatTemplateGrammarTests.swift` (new) | Per-family grammar tests + an integration test that confirms the state machine fires on a chat-template-bounded prompt. |
| `Tests/MLXLMTests/BigramTableTests.swift` (new) | Lookup correctness, threshold, build-tool round-trip. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | `--method free-tokens` plumbing + per-source draft attribution in `[BENCH] Spec decode` line. |

## Open questions

1. **Bigram corpus provenance.** Where does the build corpus come from? Options: (a) generate ourselves on a fixed prompt set, (b) use a public dump (RedPajama, OpenWebText), (c) per-deployment corpus tuned to the user's domain. (a) is the simplest path for a shippable default.

2. **Bigram table refresh.** As models get retrained / fine-tuned, their output distributions shift slightly. The bigram table is a frozen artifact; how often does it need rebuilding? Probably once per major model family release.

3. **Tokenizer family identity.** Two models nominally in the same family (e.g., Qwen 3.5-9B vs Qwen 3.5-35B-A3B) might have slight tokenizer differences. The bigram table is keyed by token ID; it's only valid for tokenizers with identical vocab. Plumb a vocab-equivalence check at table load time, fail loudly on mismatch.

4. **Logit processor interaction.** Repetition penalty / banned-token / thinking-budget processors can override the model's natural argmax. The state machine's "deterministic" claim only holds in the absence of strong processors. Detect at iterator construction whether a processor is active and disable mechanism A's high-confidence drafts in that case (or limit it to the most-deterministic stretches like template literal boundaries).

5. **Per-language grammars.** The chat-template grammars are language-agnostic (template tokens are the same regardless of the user's language), but bigram tables are language-conditioned. If a user's chat is mostly in Mandarin, an English-corpus-built table is useless. Future: per-language tables auto-selected from prompt heuristics.

## What this spec is NOT

- **Not "skip the verify forward pass entirely."** The verify forward still runs; the win is from doing fewer of them by drafting more aggressively (with high-confidence drafts) at structural boundaries.
- **Not lossy.** Output remains byte-identical to baseline at temperature 0. The verify path catches any state-machine or bigram overreach.
- **Not a separate iterator.** It composes into the existing `NGramSpeculativeTokenIterator.proposeDraft` ladder. No new top-level routing.
- **Not a replacement for PLD.** PLD wins on input-grounded content; this spec wins on structural / template content. They're additive.

## References

- [Speculative Streaming: Fast LLM Inference Without Auxiliary Models (Apple ML, 2024)](https://machinelearning.apple.com/research/llm-inference) — the "fuse the drafter into the target" framing; this spec is a CPU-side analogue.
- [Lookahead Decoding (Fu et al., 2024, arXiv:2402.02057)](https://arxiv.org/abs/2402.02057) — n-gram-based parallel-decode with deterministic-position drafts; closest published cousin to mechanism B.
- [llama.cpp `ngram-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.cpp) — the static-corpus bigram cache as a "validator", structurally similar to mechanism B's table.
- [GPT-OSS harmony format](https://github.com/openai/harmony) — the channel-transition grammar that's the highest-value target for mechanism A.
