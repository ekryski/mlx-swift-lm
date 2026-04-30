# N-gram speculative-decoding evaluation prompt set

Curated workloads for evaluating speculative-decoding changes. Each subdirectory is one workload class; the **regime classification** below tells you what kind of speedup or regression to expect from n-gram and Leviathan paths.

Use with `scripts/spec-decode-sweep.sh` and `scripts/spec-decode-compare.py` to evaluate any speculative-decoding PR against a known baseline.

## Regime classification

The data from PR #113 (greedy n-gram close-out), PR #154 (Leviathan), and several related sweeps cluster prompts into three regimes by **expected accept rate**, which is the main predictor of speedup magnitude.

### Regurgitative — high accept (50-80%), strong speedup

Output that re-quotes from the prompt, fills in templates, or follows a known structure. The lookup hits often and hits in long runs (5-15 token continuations).

| Directory | Prompt count | Notes |
|---|---|---|
| `qa-requote/` | 3 | Bug report Q&A, recipe Q&A, policy Q&A — output literally re-quotes prompt content |
| `recipe-bulk/` | 1 | Five-soup recipe rewrite — heavy structural repetition |
| `code-refactor/` | 3 | Refactor Python dataclass / JS callbacks / C → Rust — keeps the prompt's variable names / structure |
| `summarization/` | 3 | Summarize tech-news / academic abstract / meeting notes — heavy phrasal re-use |
| `email-chain/` | 1 | Customer-followup email chain — heavy template re-use |

**Expected speedup** on big weight-bandwidth-bound models (Gemma 4 26B A4B class, GPT-OSS-20B class): **1.3–1.6× greedy / 1.1–1.3× Leviathan**.

### Structured — medium accept (30-60%), moderate speedup

Output follows a structure but with novel content per cell. The lookup hits inconsistently — sometimes long runs, sometimes one-token hits.

| Directory | Prompt count | Notes |
|---|---|---|
| `code-completion/` | 3 | Fibonacci / binary-search / SQL update — completes patterns the model has seen |
| `chat-instruction/` | 3 | Photosynthesis / TCP vs UDP / better-sleep — short structured answers |
| `pm-template/` | 1 | Three RFCs — heavy template + novel content |
| `multi-turn-code/` | 1 | Test-table expansion — multi-turn-grounded |

**Expected speedup**: **1.1–1.3× greedy / 1.0–1.2× Leviathan**.

### Paraphrastic — low accept (<20%), neutral or regression

Output is generative, with few repeated n-grams. The lookup almost never hits useful matches.

| Directory | Prompt count | Notes |
|---|---|---|
| `open-generation/` | 3 | Haiku / story / explainer — creative generation |
| `blog-series/` | 1 | Three tutorial posts — long-form prose |

**Expected speedup**: **0.85–1.05×** — usually a small regression because every verify cycle pays K+1 forward overhead while accepting ~0.

## Adding new prompts

When adding a prompt, classify it into one of the three regimes by inspecting:

1. **Source coverage**: how much of the output is re-quoted from the prompt? Higher → more regurgitative.
2. **Structural repetition**: bulleted lists, numbered items, JSON, code blocks → tend toward structured.
3. **Novel content density**: opinion, creative writing, generation from scratch → paraphrastic.

Test your classification: run `scripts/spec-decode-sweep.sh` with a quick (n=3 trial) sweep on Gemma 4 26B A4B and check whether the n-gram accept rate matches the regime's expected band.

## Standard eval cell-set for spec-decode PRs

The canonical 6-cell × 5-trial sweep used in the Leviathan analysis (`benchmarks/gemma4-leviathan-broad-sweep-analysis.md`) is:

```
TI@0:0:0:                                      # baseline greedy
NGgreedy@0:3:0:                                # n-gram greedy (control)
TI@0.6:0:0.6:                                  # baseline sampling
NGlev@0.6:3:0.6:MLX_NGRAM_LEVIATHAN=1          # n-gram + Leviathan @ moderate temp
TI@1.0:0:1.0:                                  # baseline sampling, higher temp
NGlev@1.0:3:1.0:MLX_NGRAM_LEVIATHAN=1          # n-gram + Leviathan @ higher temp
```

Pass this verbatim to `--cells` for cross-PR comparability. Each cell's role:

- **TI@T cells** are the matching baseline — speedup ratios are computed against the same-temperature TI cell.
- **NGgreedy@0** is the greedy-n-gram control. If a new spec PR doesn't change greedy behaviour, this cell should match the prior baseline within trial variance.
- **NGlev@T cells** exercise the Leviathan accept/reject path at non-greedy temperatures.

For a spec PR that introduces a new env var or knob, add additional cells that toggle it, e.g.
`NGtreatment@0.6:3:0.6:MLX_NGRAM_LEVIATHAN=1;MY_NEW_KNOB=1`.

## See also

- `scripts/spec-decode-sweep.sh` — the canonical sweep harness
- `scripts/spec-decode-compare.py` — A vs B compare on two sweep logs
- `benchmarks/gemma4-leviathan-broad-sweep-analysis.md` — example of a full multi-model multi-prompt evaluation
- `Libraries/MLXLMCommon/Documentation.docc/speculative-decoding.md` — user-facing doc with the regime classification + supported-models table
