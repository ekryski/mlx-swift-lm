# 014 — Tree attention for n-gram speculative decoding

- **Status:** 🚧 Phase 1 scaffold landed ([PR #147](https://github.com/ekryski/mlx-swift-lm/pull/147) — `DraftTree` primitives). Phase 2 (MLX wiring + iterator integration) and Phases 2–4 (variable-K tree / bifurcating-on-tight-margin / full suffix-tree merging — Tier 4 row 14) open. **Not started** on those phases.
- **Branch:** Phase 1 merged via PR #147; later phases get fresh branches off alpha.
- **Depends on:** the iterator and auto-routing from spec 013, plus the
  multi-candidate / strict-greedy work that just landed on
  `ek/ngram-speculative-v2`

## Problem

The current `NGramSpeculativeTokenIterator` verifies a **single linear draft**
per round: `[y, d_0, d_1, ..., d_{k-1}]` reshaped to `[1, k+1]` and run through
the target in one forward pass. The first mismatch ends acceptance for that
round.

This is a strict subset of what speculative decoding can do. EAGLE / EAGLE-2 /
EAGLE-3, Medusa, and llama.cpp's own draft-model path all verify a **tree of
candidate continuations** in a single forward pass: at the same draft position
they run two or more candidate tokens in parallel, attention masks ensure each
candidate only sees its own ancestors, and the verifier picks the longest
matching path through the tree. This raises the per-round expected accepted
length without proportionally raising verifier cost — the dominant factor on
M-series silicon is loading weights, not the per-token attention work.

For our prompt-lookup setting the natural tree shapes are narrow but deep:

- **Top-K continuations from `NGramLookup`.** When a key n-gram has multiple
  prior occurrences with different first-tokens, propose all of them as
  parallel branches at depth 0, then linear continuations beneath each. (The
  multi-candidate change just shipped only ever picks one of them.)
- **Branch on tight-margin verify positions.** When the strict-greedy guard
  fires at depth `i` (top-1 vs top-2 logit margin below `ε`), instead of
  rejecting the rest of the draft, expand a 2-way branch at that position
  using the top-1 and top-2 tokens as siblings.
- **Suffix-tree / trie merging.** `NGramLookup` already stores all positions
  per n-gram. Build the merged trie of all next-`m` tokens after every prior
  occurrence and verify the union in a single forward pass. Mirrors
  SuffixDecoding's main contribution but operating on the per-request
  prompt+history rather than a global corpus.

End goal: 1.5–2.5× more committed tokens per verifier forward on
input-grounded workloads where multi-candidate already lifts accept rate
into the 50–75% range.

## Design

### 1. Drafted tree representation

```swift
struct DraftTree {
    /// Flattened DFS order of nodes (excluding the root, which is the
    /// already-committed `y`). `tokens[i]` is the candidate token at node `i`.
    var tokens: [Int]
    /// `parent[i]` is the index of node `i`'s parent in `tokens`, or `-1` if
    /// `i` is a root child (its parent is the implicit `y` slot at position 0
    /// of the verify input).
    var parent: [Int]
    /// `depth[i]` is the distance from the root of the tree (= 1 + parent's
    /// depth). Equivalent to the number of accepted tokens along this path
    /// before node `i`.
    var depth: [Int]
}
```

The flat-DFS representation is what lets a single attention forward verify
the whole tree. Each row in the verifier batch corresponds to one node; the
attention mask is built from the parent links.

### 2. Verifier forward

Build the verify input as `[1, 1 + tree.tokens.count]` — `y` at position 0,
followed by tree nodes in DFS order. The position id of node `i` is
`prev_offset + depth[i]` (children share their parent's depth + 1 — siblings
have the **same** position id, which differs from a linear sequence). The
attention mask is the standard causal mask **AND** an additional ancestor
mask: node `i` may attend to `y` plus its ancestors (and itself). This is the
"tree attention" mask.

Construction:
```
mask[i, 0] = 1                    // every node sees y
mask[i, i+1] = 1                  // self-attention
mask[i, j+1] = 1 if j is an ancestor of i
```

The position ids ensure RoPE rotates each node according to its depth, not
its DFS position — siblings get the same rotary position so the model treats
them as alternative completions of the same prefix.

### 3. Mask construction in MLX-swift

`MLXFast.scaledDotProductAttention` accepts `mask: .array(MLXArray)` of
shape `[..., L_q, L_k]` (or `[L_q]` causal). For a tree of `T = 1 + n_nodes`
positions, build a `[T, T]` boolean mask in Swift from the `parent` array,
materialize as `MLXArray` of dtype `bool` (or `-inf` float mask), and pass
through.

Per-layer KV update is the awkward bit: each layer's `update()` appends `T`
positions to the cache, but only the accepted-path positions are real. After
verify, `trim` by `(T - accepted_path_length)`. Matches the linear-draft
trim path; just trims more.

### 4. Position ids

For pure-attention models (Llama / Phi / Qwen 2-3 dense), MLX's RoPE applies
per-position. For Gemma 4, GPT-OSS, etc., position ids must be passed
explicitly. We need to extend the model's `callAsFunction` path to accept
explicit position ids when the iterator wants them; the default path (linear
positions = `prev_offset + i`) becomes a special case.

### 5. Acceptance: longest-matching-path

After verify, `mainTokens` has shape `[T]`. The verifier's argmax at the
root position (index 0) predicts what should follow `y` — call this `p_0`.
Walk the tree DFS:
- Among root children, find the one whose token equals `p_0`. If none, accept
  zero drafts and emit `p_0`.
- If found, descend to that child and find the prediction at *that* position.
  Repeat until either a level mismatches or the leaf is reached.
- The bonus token is `mainTokens[idx]` where `idx` is the verify position of
  the last accepted node.

This is `match_acceptance_length` from dflash-mlx (`acceptance.py`)
generalised to a tree.

### 6. Tree shape policies

Three concrete policies, picked at iterator construction:

- `linear` (today's behaviour): one path of depth `D`. No tree.
- `topK_root_x_linear`: `K` root branches (top-K continuations from
  `NGramLookup.proposeDraft`), each followed by a linear continuation of
  depth `D - 1`. Total nodes: `K * D`.
- `bifurcating_on_tight_margin`: build a linear path of depth `D`. At each
  position, after a fast logit-margin estimate (e.g. from a prior pass or
  from a draft model if available), if the margin is tight insert a
  **sibling** node carrying the top-2 token. Conceptually a beam of width 2
  at uncertain positions. Total nodes ≤ `2 * D`.

Policy `topK_root_x_linear` is the cheapest to ship — no branch-decision
logic, just K parallel `NGramLookup` continuations.

### 7. Cost model

Compared to the linear baseline:

| Path | Verify input size | Worst-case nodes wasted on full reject |
|---|---|---|
| Linear (today) | `1 + D` | `D` |
| topK × linear | `1 + K*D` | `K*D` |
| Bifurcating | `1 + ≤2D` | `2D` |

For Gemma 4 26B A4B at 4-bit, weight-loading dominates and the verifier
forward cost is roughly flat for input lengths in [1, ~32]. So `K=2, D=4`
(8 verify-input tokens) costs about the same as `K=1, D=4` (5 verify-input
tokens) but should accept on average ~1.6× more tokens per round per the
EAGLE-3 results. On models where attention scales nontrivially with input
length (long context, large K, large D) the cost gap widens.

### 8. Numerical-drift interaction with strict greedy

Tree verify shares the multi-position numerical-drift behaviour that motivated
the strict-greedy guard. In tree mode the guard becomes a per-node check: a
match at a tight-margin node is treated as a non-match (forces the tree walk
to stop). This composes — strict-greedy + tree is just strict-greedy applied
along the chosen path.

## Implementation phases

### Phase 1 — `topK_root_x_linear` only, K=2

- Extend `NGramLookup.proposeDraft` to return up to `K` candidate first
  tokens with their best linear continuations.
- Generalise `speculateRound` to build the verify input as a flat tree with
  `K=2`, generate the tree mask + position ids, do one forward.
- Walk the tree to find the longest matching path.
- Trim, emit, set `y` for next round.

Land behind `MLX_NGRAM_TREE=topk2`. Default off.

Acceptance criteria:
- Output sequence-equal to linear path on `NGramSpeculativeTests`'
  greedy-equality test.
- `≥ +10%` over linear path on the verbatim-extraction prompt and the new
  multi-turn prompt set (spec 015 below).

### Phase 2 — variable K (`topK_root_x_linear` for K ∈ {2, 3, 4})

Same surface, just generalises the tree builder. Probably needs the env
knob to control K and a small policy that backs off K when the lookup
returns fewer candidates.

### Phase 3 — `bifurcating_on_tight_margin`

Requires the strict-greedy guard's logit margin per position. Extends the
linear path with a sibling at any tight-margin position. Reuses the same
flat-tree representation.

Acceptance criteria:
- Resolves the D=4-on-summarisation regression where the iterator currently
  diverges from baseline output (today, the strict-greedy guard handles this
  by refusing to accept; with tree the alternative branch can rescue accept
  rate).

### Phase 4 — full suffix-tree merging

Pull in all prior occurrences of the key n-gram, merge their continuation
prefixes into a trie, prune by frequency × depth, verify the trie. SuffixDecoding-style.

This is the highest-leverage change but also the most invasive. Defer until
phases 1-3 land.

## Files touched by this work

| File | What |
|---|---|
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | New `DraftTree` struct, tree-aware `speculateRound` branch, flat-DFS verifier path, longest-matching-path acceptance walk. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` (NGramLookup) | `proposeDraftTree` returning `(Int, [Int])` tuples — first-token plus continuation. |
| New: `Libraries/MLXLMCommon/SpeculativeTreeMask.swift` | Mask + position id builders. Probably worth its own file given how non-trivial the indexing is. |
| `Tests/MLXLMTests/NGramSpeculativeTreeTests.swift` (new) | Unit tests for tree mask construction; integration test for greedy-equality with `TokenIterator`. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | Plumb a `MLX_NGRAM_TREE` knob through `BenchEnv` and the `simple` / `summarization` / `ngram-sweep` paths. |

## Out of scope

- VLM / cross-attention models. Tree attention shape is the same but the
  position id surgery for cross-attention is its own thing.
- Hybrid SSM / Mamba models (Qwen 3.5 / 3.6). The Mamba layers can't roll
  back partially, so any tree of depth > 1 forces the SuffixDecoding-style
  full-state checkpoint per branch — out of scope until 014 itself lands on
  pure attention.
- Non-greedy temperature > 0. Tree attention combines with stochastic
  acceptance (llama.cpp's `speculative.cpp` does this), but the resample-on-
  reject logic is its own follow-up to land alongside the temperature > 0
  path of the linear iterator.

## Open questions

1. **Position ids.** Pure-attention models in this repo don't expose a
   "feed your own positions" entry point. Adding one is mechanical but
   touches every model implementation (Gemma 4, GPT-OSS, Llama, Phi,
   Qwen 2/3 dense). Worth a side spec (017?) before tackling tree attention.
2. **Mask format.** `MLXFast.scaledDotProductAttention` accepts an array
   mask, but for tree shape we need a `[L_q, L_k]` mask where the structure
   isn't pure causal. Confirm MLX-swift accepts this for all attention
   backends (regular SDPA, TurboQuant compressed, sinks attention).
3. **Tree depth budget.** What does the optimal `(K, D)` envelope look like
   on M1 Max for Gemma 4 26B A4B and GPT-OSS 20B? Probably `(2, 4)` is the
   right starting default, with adaptive scaling à la spec 013 §2.

## References

- EAGLE-3: https://arxiv.org/abs/2503.01840
- SuffixDecoding: https://arxiv.org/pdf/2411.04975
- llama.cpp `speculative.cpp` tree-drafting (`p_split`, `n_seq_dft`):
  https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative/speculative.cpp
- Medusa (parallel heads + tree attention):
  https://arxiv.org/abs/2401.10774
