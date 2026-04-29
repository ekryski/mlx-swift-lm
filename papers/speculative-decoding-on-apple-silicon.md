# Speculative decoding on Apple Silicon: a working engineer's tour

**Author:** Eric Kryski
**Hardware target:** Apple M1 / M5 Max, 16–512 GB unified memory
**Software target:** [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)
**Date:** 2026-04-28
**Status:** Working draft

> *Why does the same speculative-decoding trick give 3.5× on a server GPU and 1.0× on a MacBook? And what do we actually do about it?*

This is a working engineer's pass at speculative decoding — the family of throughput-optimisation tricks that let a language model emit more than one token per forward pass. We start with the basic shape, walk through the three families that matter today (n-gram / prompt-lookup, draft-model, block-diffusion), and call out the crossover techniques (tree attention, prefix caching, tape-replay rollback) that compose across all of them. Throughout, we annotate where the published headline numbers come from and where Apple Silicon's quirks change the calculus.

The narrative is grounded in [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm), where we shipped the n-gram path end-to-end and started porting block-diffusion. Numbers are real M1 Max measurements, not server extrapolations.

---

## 1. Why decode is slow

Single-stream LLM decode is **memory-bandwidth bound**. The bottleneck on a 9B-parameter model isn't matmul throughput — it's the time to stream every weight from VRAM (Apple's Unified Memory) to the on-chip caches once per token. For a 4-bit-quantised 9B model that's roughly 4.5 GB / token. On an M1 Max with ~400 GB/s memory bandwidth, the **theoretical ceiling** is `400 / 4.5 ≈ 89 tok/s` if you could stream weights perfectly with zero overhead.

We don't get 89. **Our practical maximum, measured on Qwen 3.5-9B-4bit (4-bit weights, fp16 KV cache, no compression), is ~54 tok/s at 256-token context.** Decode tok/s holds within ~10% of peak through 8 K context, then drops more sharply: 41.6 tok/s at 16 K – 32 K, 34.8 at 64 K, 26.7 at 128 K — the KV-cache reads start meaningfully contending with weight streams past 32 K. The 4-bit-quant quality cost (Gen KLD vs. bf16 baseline) is small (≤ 0.5) and not strongly context-dependent — most cells land in the 0.08–0.28 range:

| Ctx | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT | Gen PPL | Gen KLD | KV cache | GPU peak |
|----:|--------------:|-------------:|-------------:|-----:|--------:|--------:|---------:|---------:|
| 128 | 204 | 49.8 | 50.0 | 0.5 s | 1.882 | 0.738 | 29 MB | 5.20 GB |
| 256 | 254 | **53.7** | **54.2** | 0.9 s | 1.750 | 0.437 | 33 MB | 5.58 GB |
| 512 | 270 | 53.3 | 53.6 | 1.8 s | 1.582 | 0.512 | 41 MB | 5.78 GB |
| 1024 | 289 | 53.0 | 53.4 | 3.5 s | 1.711 | 0.339 | 57 MB | 5.87 GB |
| 2048 | 292 | 52.2 | 52.6 | 7.0 s | 1.936 | 0.281 | 89 MB | 5.91 GB |
| 4096 | 294 | 50.3 | 50.7 | 13.8 s | 2.007 | 0.084 | 153 MB | 5.98 GB |
| 8192 | 290 | 46.6 | 47.1 | 28.2 s | 1.919 | 0.185 | 281 MB | 6.13 GB |
| 16384 | 276 | 41.6 | 41.9 | 59.2 s | 2.074 | 0.167 | 537 MB | 6.30 GB |
| 32768 | 254 | 41.6 | 41.9 | 128.7 s | 2.149 | 0.196 | 1.02 GB | 6.84 GB |
| 65536 | 213 | 34.8 | 34.9 | 307.7 s | 1.917 | 0.184 | 2.02 GB | 8.08 GB |
| 131072 | 164 | 26.7 | 26.7 | 798.1 s | 1.715 | 0.079 | 4.02 GB | 10.45 GB |

*M1 Max 64 GB, `mlx-community/Qwen3.5-9B-4bit`, KV cache `none` (no compression), `summarization` benchmark, Apr 28 2026 sweep with `--ppl --kld` (KLD baseline = `mlx-community/Qwen3.5-9B-MLX-bf16`). Full per-config detail at [`benchmarks/m1-max-64gb-2026-04-28.md`](../benchmarks/m1-max-64gb-2026-04-28.md).*

#### Benchmark methodology and parameters

**Method:** [`summarization`](../benchmarks/README.md#summarization). Pre-built prompt files containing excerpts from *The Great Gatsby* sized to each target context length are passed as the user message; the model generates a summary up to the per-cell `max-tokens` cap. The context limit is enforced via `RotatingKVCache(maxSize: ctx)`, matching what a real chat / RAG deployment would set as `max_kv`. The harness records prefill throughput, decode throughput (warmup-included average and tokens-11-onwards "steady" rate), TTFT, peak GPU memory, and KV-cache footprint per run.

**Quality metrics**:
- **Gen PPL** — per-token perplexity computed from the model's own log-probabilities at the sampled token, accumulated across the generation, then `exp(-mean(logprobs))`. Lower = the model is more confident in its own outputs at that context size.
- **Gen KLD** — KL divergence vs the highest-fidelity baseline that fits in GPU memory (bf16 → 8bit fallback). Computed by **forced decode** of the generated tokens through the baseline model with no KV-cache compression, recording per-token logprob, then `mean(target_logprob − baseline_logprob)`. Always ≥ 0; values near 0 mean negligible quality cost vs the baseline distribution. For 4-bit / no-KV-quant runs this is purely the **weight-quantization cost**.

**Generation parameters** (Qwen 3.5 family defaults, identical across cells):

| Parameter | Value |
|---|---|
| `max-tokens` | 400 |
| `temperature` | 1.0 |
| `top-p` | 0.95 |
| `top-k` | 20 |
| `min-p` | 0.0 |
| `repetition-penalty` | 1.0 (off) |
| `presence-penalty` | 1.5 |
| `prefill-step-size` | 1024 |
| Speculative decoding | none (baseline) |
| `MLX_MAX_OPS_PER_BUFFER` | 500 (M1 Max default) |

**Hardware / build**: Apple M1 Max (`applegpu_g13s`), 64 GB unified memory, macOS 15.7.4. `ekryski/mlx-swift-lm` at branch `alpha`. Built `release` config via `make build-tests`.

**Reproducing**:

```bash
./scripts/benchmark.sh --model qwen35-9b --quant 4bit --kv none \
    --method summarization --ppl --kld
```

This drives `MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none MLX_BENCH_METHOD=summarization MLX_BENCH_PPL=1 MLX_BENCH_KLD=1 swift test --skip-build -c release --filter benchmark` per cell. See [`benchmarks/README.md`](../benchmarks/README.md) for the full env-var contract and per-method assertions.

That ~54 tok/s peak is **61% of the bandwidth ceiling** — the remaining 39% is RoPE, attention, normalisation, KV-cache reads, sampling, and per-step kernel-launch overhead. Throughput holds within ~10% of peak through 8 K context, drops about 23% by 32 K, 35% by 64 K, and 50% by 128 K as the KV cache grows past the 1–4 GB scale.

(Note Qwen 3.5-9B is hybrid GatedDeltaNet — only ~25% of layers carry positional KV state, which is why a 32 K-token KV cache is just 1 GB instead of the ~8 GB a pure-attention 9B would need. That same architectural choice is what makes it tricky for speculative decoding without spec 020's tape-replay rollback.)

So when we say "spec decode wins when the verify forward is free relative to a single-token forward," the floor we're racing against is ~51 tok/s at the sweet-spot context, not the textbook 89.

The compute units are mostly idle. If we could *use* them — process more tokens per weight-stream — we'd get more tok/s essentially for free.

Speculative decoding is the family of tricks that does exactly that. The shape:

```
   ┌────────────┐                 ┌─────────────┐
   │   draft    │  K cheap tokens │   target    │  verify K in 1 forward
   │  (cheap)   │ ──────────────► │  (slow)     │ ──┐
   └────────────┘                 └─────────────┘   │
                                                    ▼
                                             accept N where draft matches target's argmax
                                             emit N + 1 tokens (N drafts + the target's
                                             next token at position N)
```

The accept count `N` ranges 0..K. Even at `N=1` you're at break-even (one target forward, one emitted token, just like baseline). At `N=K` you've emitted `K+1` tokens for one target forward — the maximum win.

The catch: the target's verify-pass cost grows mildly with `K+1`, and there's overhead in the draft step itself. The trade lives at:

```
expected_speedup = (E[N] + 1) / (1 + draft_cost / target_decode_cost + verify_overhead(K))
```

When `draft_cost` is small (much smaller draft model, or no draft model at all) and `verify_overhead` is small (memory-bandwidth-bound regime where (K+1)-token forward costs almost the same as 1-token), every accepted draft is ~1.0× lift. When the draft is expensive or the verify pass scales, you can lose to baseline even at high accept rates.

**Apple Silicon's specific quirk** is that small models are *too fast for spec decode to help*. Decode on Gemma 4 E2B 4-bit on an M1 Max runs ~110 tok/s — already memory-bound. The verify pass on `K+1=5` tokens isn't 5× more bandwidth (the weights are streamed once), but it does pay 5× the per-token compute and 5× the kernel-launch overhead. And the verify pass *must sync to the CPU* before it can decide accept/reject, blowing the asyncEval pipeline that single-token decode depends on.

This is the unifying theme of everything that follows: **speculative decoding wins when the verify forward is "free" relative to a single-token forward**. That happens on big models, on quantised models with cold caches, and on MoE models where the expert subset is the same. It mostly doesn't happen on small dense models on fast Metal GPUs.

---

## 2. N-gram (prompt-lookup) speculative decoding

### 2.1 The basic algorithm

The simplest form has no draft model at all. You look at the just-decoded token, find a place earlier in the prompt or generation history where the same token (or n-gram) appeared, and propose its continuation as the draft.

```
Prompt: "Q: What's the capital of France? A: The capital of France"
                                            └──── n-gram of size 4 ────┘
                                                                        │
                                                              search history for ↓
History contains:                                                       │
  "...The capital of France is Paris..."                                │
                                          ↓ continuation after the match
                                          ▼
Draft: ["is", "Paris", "."]
```

The lookup is a CPU-side dictionary: `n-gram → list of historical positions where it occurred`. At each decode step:

1. Hash the last `n` accepted tokens.
2. Look them up.
3. If a hit, return the next `K` tokens as the draft.
4. Run the target on `[y_last, draft[0], ..., draft[K-1]]` in one forward.
5. Accept the longest matching prefix.

Total extra cost: a few hundred microseconds of Swift hash-table lookup. The `K=4` to `K=12` draft sizes pay off when the draft and target agree.

### 2.2 Where it wins

The original [Prompt Lookup Decoding by Saxena (2023)](https://github.com/apoorvumang/prompt-lookup-decoding) — which named the technique — and the [vLLM reproduction](https://docs.vllm.ai/en/v0.6.6/usage/spec_decode.html) report **2-4× speedup on input-grounded generation**: summarisation with extractive quoting, document QA where the answer is verbatim in the document, code editing, multi-turn chat where context accumulates. They share a property: the model literally copies entity names, code chunks, or exact phrases from input to output. The lookup almost always finds the right span.

On creative open generation (write me a poem about clouds), accept rates collapse — the model isn't repeating anything in the prompt — and PLD becomes net-negative as the failed verify-batches dominate.

### 2.3 What we shipped in mlx-swift-lm

`NGramSpeculativeTokenIterator` (in [`Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift`](../Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift)). Key design points:

- **Multi-size fallback ladder.** When the configured n-gram size misses, try `n-1`, `n-2`, …, down to `minNgramSize`. Mirrors llama.cpp's [`ngram-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.cpp) ladder (`LLAMA_NGRAM_MIN=1`, `LLAMA_NGRAM_MAX=4`).

- **Multi-candidate selection (default-on).** When a key n-gram has multiple prior occurrences with different continuations, group by first token, count occurrences, pick the most-frequent group. Llama.cpp's [`ngram-map.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-map.cpp) caps this at 4 candidates per key (`COMMON_NGRAM_MAX_VALUES=4`). We don't cap.

- **Dominance gate (off-by-default).** Refuse to draft when the winning candidate doesn't dominate (`max_count > 2 * sum_others`). Trades recall for precision — useful on noisy patterns, hurts on regurgitative prompts. We expose it as `MLX_NGRAM_DOMINANCE=1`.

- **Strict-greedy guard (default-on).** At verify time, compute the top-1 vs top-2 logit margin per position. If `margin < 0.5` and the draft happens to match top-1, refuse the match — it could be a batched-vs-sequential argmax flip from numerical drift. This eliminates the failure mode where `D≥4` produces output that diverges from baseline at temperature 0. The guard adds one tensor allocation (sort along axis -1) folded into the same `eval()` as the main argmax sample, so zero extra GPU sync.

- **Adaptive draft length (default-on).** Track rolling acceptance rate over the last 4 verify rounds. If rate ≥ 0.7, expand `D` by 1.5×. If ≤ 0.3, halve `D`. Otherwise hold steady. Floor at `ngramDraftMin` (1 by default), ceiling at the configured `maxNgramDraftTokens`. Inspired by EAGLE-3's instance-adaptive depth and llama.cpp's `n_accepted` tracking.

- **AR-batch for fallback path.** When the lookup misses, run 4 autoregressive forwards async-pipelined, sync once at the end. This recovers most of the eager-`.item()` overhead that cost ~5 ms/token on the original implementation.

Settings exposed via `GenerateParameters` (`ngramSize`, `maxNgramDraftTokens`, `ngramDraftMin`, `ngramMinHits`, `minNgramSize`) and via env knobs (`MLX_NGRAM_*`) for diagnostic A/B work.

The full set of env knobs and their defaults:

| Knob | Default | Effect |
|---|---|---|
| `MLX_NGRAM_MULTI_CANDIDATE` | on | Frequency-based candidate selection |
| `MLX_NGRAM_DOMINANCE` | off | Reject ambiguous candidates (`max > 2 * sum_others`) |
| `MLX_NGRAM_STRICT_GREEDY` | on | Top-1/top-2 margin check before accept |
| `MLX_NGRAM_STRICT_EPSILON` | 0.5 | Tight-margin threshold (logit units) |
| `MLX_NGRAM_ADAPTIVE` | on | Adaptive draft length |
| `MLX_NGRAM_ADAPTIVE_HI` | 0.7 | Accept-rate threshold to expand |
| `MLX_NGRAM_ADAPTIVE_LO` | 0.3 | Accept-rate threshold to shrink |
| `MLX_NGRAM_AR_BATCH` | 4 | Async-batch size for AR fallback |
| `MLX_NGRAM_FORCE_AR` | off | Diagnostic: force AR every round |
| `MLX_NGRAM_DEBUG` | off | Per-round trace logging |

### 2.4 The Apple Silicon penalty

Real numbers, M1 Max 64 GB, all 4-bit, all temperature 0:

| Model | Workload | Baseline | Best PLD config | Speedup |
|---|---|---|---|---|
| Gemma 4 E2B (~2B) | QA-requote | 90 tok/s | n=2 D=2 | **+1%** |
| Gemma 4 E4B (~4B) | QA-requote | 62 tok/s | n=3 D=4 | **−25%** |
| Gemma 4 26B A4B (MoE 4B active) | QA-requote | 27 tok/s | n=3 D=2 | **+25%** |
| Gemma 4 26B A4B | recipe-bulk (templated, 1200 tok) | 27 tok/s | adaptive+strict | **+1%** |
| Gemma 4 31B (dense) | QA-requote | 14.5 tok/s | n=3 D=2 | **−12%** |
| GPT-OSS 20B (MoE) | code refactor | 78.7 tok/s | n=3 D=2 | **−14%** |
| Qwen 3.5 9B (hybrid GDN) | any | — | falls back to TokenIterator (MambaCache) | parity |

The MoE win is real but narrow. Dense models *lose* to PLD across the board: their decode is fast enough that the verify overhead doesn't amortise.

The reason MoE wins is structural: at decode time only a small subset of experts is active per token, but the **cost of switching expert subsets** between rounds is small relative to a full forward. The `K+1`-token verify hits roughly the same expert subset as the next single-token forward would have hit, so verify cost is ~1× decode rather than ~K×. Combined with high accept rates on input-grounded prompts (~60% on Gemma 4 26B A4B QA), this is enough to net out positive.

The dense-model loss is structural for the opposite reason: the verify forward genuinely scales with K because every expert is "active." On a 31B dense, `K=4` verify costs ~4× decode. Even at 60% accept (`E[N]+1 ≈ 3`), that's at best break-even — and after sync overhead, a loss.

Everywhere we don't win, **output is byte-identical to baseline at temperature 0**. Strict-greedy preserves correctness; the question is purely throughput.

### 2.5 Optimisations: the literature

What the literature says is worth doing on top of base PLD:

| Technique | Source | What it does | Status in mlx-swift-lm |
|---|---|---|---|
| Multi-size fallback ladder | [llama.cpp ngram-cache](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.cpp) | Try size n, n-1, ... | shipped |
| Multi-candidate (k=4) | [llama.cpp ngram-map](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-map.cpp) | Pick most-frequent first token | shipped |
| Dominance gate | llama.cpp ngram-map | Refuse ambiguous patterns | shipped (off-default) |
| Min-hits filter | llama.cpp ngram-map | Require ≥k prior occurrences | shipped |
| Per-value `n_accepted` adaptive depth | llama.cpp ngram-map | Cap next round at last accepted count for that key | not yet — **easy follow-up** |
| Cross-request `nc_dynamic` cache | llama.cpp ngram-cache | Persist lookup across requests | [spec 016](../specs/016-cross-request-ngram-cache.md) |
| Static cache from corpus (`nc_static`) | llama.cpp ngram-cache | Validator from a large pre-built cache | spec 016 phase 3 |
| Save / load to disk | llama.cpp ngram-cache | Persistence | spec 016 phase 3 |
| Adaptive draft length | EAGLE-3 / llama.cpp | Per-rolling-window K scaling | shipped |
| Suffix tree | [SuffixDecoding (NeurIPS 2025)](https://arxiv.org/abs/2411.04975) | Generalises n-gram tables to arbitrary-depth suffix lookup | not yet |
| Per-request + global suffix trees | SuffixDecoding | Two-tier (current + previous outputs) | spec 016 implements analogue |
| Frequency-based candidate scoring | SuffixDecoding | `D(N)` = expected accepted token count | not yet |
| Speculation tree builder | SuffixDecoding | Greedy expansion to MAX_SPEC = α·p | spec 014 (tree attention) |
| Tree attention verification | Medusa / EAGLE / llama.cpp `speculative.cpp` | Verify multiple candidate paths in one forward | [spec 014](../specs/014-tree-attention-spec-decode.md) |
| Stochastic acceptance for T>0 | llama.cpp `speculative.cpp` | Accept with prob `min(1, p_target/p_draft)` | not yet (we're greedy-only) |
| Attention-weighted span selection | [PLD+ (arXiv:2412.01447)](https://arxiv.org/abs/2412.01447) | Score candidates by induction-head attention | [spec 019](../specs/019-pld-plus-attention-weighted-span.md) |
| Hidden-state cosine selection | PLD+ | Cosine on layer-9-13 hidden states | spec 019 phase 1 |
| Strict-greedy margin guard | (this work) | Top-1/top-2 margin check at verify | shipped |

### 2.6 Steering acceptance rate

A workload-aware question that doesn't get enough attention in the literature: **can you increase acceptance by changing what's around the iterator?**

Yes, several ways:

**(a) Prompt structure.** Instructions like "preserve exact phrasing", "quote the source verbatim", or few-shot examples that show verbatim copying tilt the model toward regurgitation. Anecdotal but consistent — a 10-15 percentage-point lift in accept rate is achievable on document-QA prompts by adding "answer using the exact wording from the document" to the system prompt.

**(b) RAG over PLD.** When you're already injecting retrieved passages into the prompt, you've handed PLD an obvious source of high-frequency n-grams. The compounding is large: RAG inflates the n-gram table by the size of retrieved content, and the model's answer regurgitates that content. We have an internal datapoint where adding 4K tokens of retrieved context lifted PLD speedup on Gemma 4 26B A4B from +25% to +40% on the same query.

**(c) Token-class auto-accept.** Punctuation, whitespace, and common connectors ("the", "and", ".", " ") have very tight argmax margins on every model. They're either trivially-accepted (the model would predict the same token without the draft) or they're inherently unpredictable from local context. Auto-accept the trivial ones without verify. Practically: when the draft token has cumulative bigram probability > 0.95 in the model's training distribution, skip verify. Doesn't apply at scale because we don't have those statistics, but a per-tokenizer "common connector" set is cheap.

**(d) Chat-template-token short-circuit.** When the iterator predicts a chat-template special token (`<|im_end|>`, `<|im_start|>`, the assistant-opener sequence), it's deterministic given the chat state. Skip verify and emit. Bounded benefit (handful of tokens per turn), but it's free.

**(e) LoRA / SFT for higher copy rate.** A LoRA fine-tune on input-grounded outputs (training the model to copy from input more aggressively) measurably lifts PLD accept rates. Out of scope for an inference engine, but worth noting for application teams.

**(f) Document-aware indexing.** When the user uploads a doc, PLD can build the n-gram table once at upload time and reuse it across all queries against that doc. The lookup table is small (few hundred KB for a 64K-token doc) and cheap to maintain. This is what the cross-request cache (spec 016) gives us at the engine level.

**(g) Steer the drafter, not the model.** llama.cpp's three-tier cache (context lax / dynamic strict / static validator) is essentially a steering mechanism: lower-bar drafting from the active context, higher-bar from older context, and a "would the static corpus also predict this?" sanity check. The same idea applied to RAG would be: low-bar draft from the retrieved passage, high-bar from the rest of the prompt, sanity-check against a token-frequency baseline.

### 2.7 The verify-sync problem

This is the deepest issue and the one I'd most like a clever answer to.

The accept/reject decision needs the verify forward's argmax, which lives on the GPU. Reading it back to CPU is a sync barrier — the CPU waits until the GPU finishes the verify forward. Single-token decode avoids this (TokenIterator's "previousY trick": sync on the *previous* token's value while the next forward is in flight). Spec decode can't, because the next forward's input depends on accepting the current.

Approaches that exist or are plausible:

- **GPU-side accept/reject.** Compute the accepted-prefix length on the GPU (vector mask + cumulative product, à la dflash-mlx's [`acceptance.py:match_acceptance_length`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/engine/acceptance.py)). Only sync the *count*, not the per-position decisions. Saves a per-step sync on long drafts. We don't currently do this; cheap to add.

- **Pipelined verify.** Build verify N+1's forward graph speculatively assuming all of verify N accepted, then commit or rollback. This is what JIT engines do with branch prediction. Costs verify-N+1's forward when accepted is partial, but saves the sync on full-accept. vLLM v0.18 reports this as ["zero-bubble async scheduling" with spec decode compatibility](https://docs.vllm.ai/en/latest/features/speculative_decoding/), 3.5× verifier-latency improvement.

- **Speculative streaming.** Emit the draft tokens to the consumer optimistically. If verify rejects, send a rollback event. Useful only when the consumer can handle rollback (e.g., a structured-output parser that validates incrementally; not useful for "render to terminal").

- **Hidden-state forwarding (Medusa / MTP / EAGLE).** Instead of running a *separate* draft model, attach prediction heads to the target. The target's forward produces both the next-token logits and the K future-token predictions in one go. Eliminates the draft pass entirely, makes the verify pass free (it's the same forward as the next single-token decode). See §3.4 below.

- **Continuous batching.** When you have multiple concurrent streams, you can mix verify and decode work in the same batch. Doesn't help single-stream inference (our case) but is the dominant production setup at scale.

The right answer for a single-stream desktop inference engine is probably (1) GPU-side accept and (4) hidden-state forwarding (i.e. switching from PLD to MTP / Medusa where the model has the head for it). For multi-stream serving on Apple Silicon, (5) plus (1).

A speculative thought we haven't tried: **dual-stream pipelining**. Submit verify N's forward as a single Metal command buffer, asyncEval, and *immediately* submit verify N+1's forward (assuming full accept). When verify N's result comes back, if accepted, commit verify N+1's work; if rejected, cancel it via Metal's `MTLCommandBuffer.cancel()`. The cancel path wastes verify-N+1 GPU time, but speculatively-correct cases get full pipelining. Apple's GPU command-buffer API supports this, and MLX's lazy-eval graph could be made to track speculative branches. Not implemented; would be a research project.

### 2.8 Where PLD doesn't fit

PLD is fundamentally limited to **input-grounded generation** because it can only draft from observed history. Open generation, abstractive summarisation, creative writing, math reasoning, code generation from a sparse spec — all have low n-gram overlap between prompt and output.

For those workloads, you need a real draft model.

---

## 3. Draft-model speculative decoding

### 3.1 The basic algorithm

Same shape as PLD, but the draft tokens come from a small auxiliary model (~10× smaller than the target):

```
   ┌──────────────────┐                    ┌─────────────────────┐
   │  draft model     │  K tokens          │  target model       │
   │  (~1B params)    │  in K forwards     │  (~10B params)      │
   │                  │ ─────────────────► │                     │
   └──────────────────┘  (or 1 forward     │  verify K in 1      │
                          for parallel    │  forward            │
                          draft heads)     └─────────────────────┘
```

Draft and target share a tokenizer, so their token streams are directly comparable. Every accepted draft token is "free" — it cost a draft-model forward (cheap by ratio) but no target forward.

The math: if draft is `d`× cheaper than target, and accept rate is `p`, expected speedup is

```
speedup = (E[N] + 1) / (1 + K·d⁻¹ + verify_overhead)
```

For `d = 0.1` (draft is 10× cheaper) and `p = 0.8`, `K = 4` gives `(3.4) / (1.4 + ε) ≈ 2.4×`. Real numbers are typically 1.5-2.5× across model families.

### 3.2 Key implementation pieces

The reference for an MLX-Swift implementation is `SpeculativeTokenIterator` in [`Libraries/MLXLMCommon/Evaluate.swift`](../Libraries/MLXLMCommon/Evaluate.swift). It works (verified by the integration test). The interesting design points:

- **Draft autoregresses K times.** Each draft step is a single-token forward through the draft model. K draft forwards + 1 verify forward = K+1 forwards total per round.

- **Verify in one forward.** Concatenate `[y_last, draft[0], ..., draft[K-1]]`, run the target on it, sample argmax per position.

- **Accept walk.** Linear scan: for each position, check if draft[i] equals target's argmax at position i. Stop at first mismatch. Emit accepted drafts plus the target's argmax at the mismatch position (the bonus).

- **Cache trim.** Both draft and target caches have appended K+1 entries during their respective forwards. After acceptance, trim by `(K - accepted)` on both. For non-trimmable Mamba layers, snapshot/restore (or [tape-replay](../specs/020-tape-replay-rollback-generalised.md) — see §6).

### 3.3 Optimisations

**Tree drafting.** Instead of one linear draft path, draft a *tree*: when the draft's top-1 logit is close to its top-2, branch into two parallel paths. Verify all paths in one forward via tree attention masks. Llama.cpp's [`speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative/speculative.cpp) implements this with `p_split` (probability split threshold) and `n_seq_dft` (max parallel sequences). Spec 014 ports this idea to mlx-swift-lm.

**Stochastic acceptance for T > 0.** At temperature > 0, "matches argmax" doesn't apply. Instead: accept draft token `t` with probability `min(1, p_target(t) / p_draft(t))`. On reject, sample from the residual distribution (`max(0, p_target - p_draft)`, normalised). This produces a token sequence distributionally indistinguishable from running the target alone at the same temperature. Llama.cpp's `speculative.cpp` implements this exactly.

**Vocab match enforcement.** Draft and target must share the tokenizer at least up to a small Hamming distance. Llama.cpp checks vocab token equality up to `SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128` and refuses if they differ. We should do the same.

**EAGLE / EAGLE-2 / EAGLE-3.** A line of work where the draft is a small head trained on top of the target's penultimate hidden states. EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)) reports 3.0–6.5× speedup over baseline on production-sized models; ~20-40% over EAGLE-2 via a "training-time test" technique that aligns draft training to inference distribution. The draft is much smaller than a standalone draft model (just a couple of attention layers) and its forward fuses with the target's, so verify cost is essentially the same as a single-token target forward. **This is the highest-leverage technique in the spec-decode literature today.** vLLM has shipped EAGLE-3 with CUDA graphs.

**Verify-specialised kernels.** The verify forward has shape `[1, K+1]` going through quantised matmuls. Stock quantised matmul kernels are tuned for `M=1` decode. dflash-mlx ships a custom Metal SIMDgroup-MMA kernel (`verify_qmm`) for the `M=K+1` case that's ~2× faster on MoE and dense ≥40-layer models. Spec 015 includes this as a phase-6 deferred item — large effort, large speedup.

**Continuous batching with speculation mixed in.** vLLM's "Arctic Inference" reports verifier-latency reduction from ~1.34 ms to ~0.38 ms (~3.5×) by running greedy verification in a tight CUDA kernel rather than a rejection sampler. Same idea applies to Metal — a fused argmax + accept-walk kernel would save significant per-round overhead. Out of scope for v1 of the iterator but worth noting.

### 3.4 The "draft in the same forward" family: Medusa, MTP

A different lineage that often gets lumped in with draft-model spec decode: instead of a separate draft model, attach **multiple decoding heads** to the target. Each head predicts the token at offset `i` from the current position. Run all heads in parallel during the target's forward; verify with the next forward.

[Medusa (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) is the canonical reference: 4-5 extra heads on top of the last hidden state, tree attention to verify multiple top-k predictions per head per position. Reports 2.2-3.6× speedup on Vicuna-7B.

[Multi-Token Prediction (MTP) heads](https://github.com/Blaizzy/mlx-lm/pull/15) generalise this — DeepSeek-V4-Flash and Qwen 3.5 ship native MTP heads trained jointly with the target. The mlx-lm PR #15 implements this for DeepSeek-V4 with up to 2× speedup per backbone forward.

The advantage over a separate draft model: **no separate forward at all**. The draft is a few extra projections in the target's last layer. The disadvantage: the target must be trained or fine-tuned with the draft heads — you can't run Medusa-style decode on an arbitrary off-the-shelf model.

In practice, Medusa / MTP ship as model variants (Medusa-Vicuna, DeepSeek-V4-MTP). For our model zoo, the relevant ones are Qwen 3.5 / 3.6 (which have MTP heads, but we haven't wired them up) and DeepSeek-V4-Flash (which we don't ship).

---

## 4. Block-diffusion: DFlash and friends

### 4.1 The big idea

Both PLD and conventional draft-model speculation produce the K draft tokens **autoregressively** — even if the draft model is small, you still pay K forward passes through it. DFlash ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)) breaks that constraint: the draft model is trained as a **block diffusion model** that produces 16 candidate tokens in *one* forward pass.

The mechanic: feed the draft model 16 mask tokens conditioned on the target's last hidden state, run several "denoising" steps inside one forward, get back 16 token predictions in parallel. Total draft cost: ~2-3× a single-token target forward, but emits 16 candidates instead of 1.

Then verify those 16 in one target forward, accept the longest matching prefix.

### 4.2 The headline numbers

From the [bstnxbt/dflash-mlx engine-v2 README](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2), measured on Apple M5 Max:

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|---|---|---|---|---|---|
| Qwen3.5-4B | 1024 | 53.8 tok/s | 182.9 tok/s | **3.40×** | 86.4% |
| Qwen3.5-9B | 1024 | 31.0 tok/s | 135.3 tok/s | **4.37×** | 89.6% |
| Qwen3.5-27B-4bit | 1024 | 33.6 tok/s | 79.0 tok/s | **2.37×** | 90.0% |
| Qwen3.5-35B-A3B-4bit | 1024 | 143.0 tok/s | 248.9 tok/s | **1.76×** | 89.3% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.3 tok/s | 300.3 tok/s | **2.20×** | 91.0% |

These are the largest sustained Apple-Silicon speedups in the speculative-decoding literature today. They dwarf what we got from PLD (max ~+25% on the same models), and the high accept rate explains why: the draft model is *trained* to align with the target, so it doesn't produce drafts that the target rejects.

The catch: DFlash requires a custom-trained draft model per target. The [z-lab](https://huggingface.co/z-lab) drafts cover the Qwen 3.5 / 3.6 family; nothing else.

### 4.3 What's hard about DFlash

Three pieces beyond the draft model itself:

**(a) Hidden-state capture.** The draft model's cross-attention reads target hidden states from a configured set of layers. The target has to expose those — for our codebase that's a per-model Swift surface (`DFlashTargetModel` protocol; see [SwiftLM's reference port](https://github.com/SharpAI/SwiftLM/tree/main/Sources/DFlash)).

**(b) Tape-replay rollback for hybrid models.** Qwen 3.5 / 3.6 are hybrid GatedDeltaNet + attention. The GatedDeltaNet layers can't trim positionally — there's no "rewind to position N" because the recurrent state is a function of the whole input. Snapshotting the full state per cycle is too expensive at long context. The fix: record an **innovation tape** during the verify forward (the per-step recurrent updates), and on partial accept replay only the accepted prefix's updates onto a snapshot taken at round entry.

**(c) Verify-specialised int4 qmm.** The 17-token verify forward (1 last-committed + 16 draft) goes through quantised matmuls of shape `[1, 17, V]`. Stock `quantized_matmul` is M=1-tuned. DFlash ships a custom Metal SIMDgroup-MMA kernel for M=17 that's roughly 2× faster on MoE and large dense models.

We have specs for all three: [015](../specs/015-dflash-diffusion-speculative-decoding.md) covers (a) and (b); the verify kernel is phase 6 and gets its own follow-up.

### 4.4 The deep insight: tape-replay generalises

Tape-replay rollback isn't specific to DFlash. **Any speculative decoder that needs to roll back a non-trimmable cache can use it.**

The canonical case where this matters: PLD on Qwen 3.5 / 3.6. Today our auto-routing falls back to TokenIterator at parity on these models because their MambaCache layers are non-trimmable. With tape-replay rollback at the cache layer, MambaCache becomes "trimmable in the speculation-decoder sense" — the iterator records innovations during verify, replays the accepted prefix's innovations on partial-accept, discards the rest.

This means: **with spec 020 shipped, PLD on Qwen 3.5 / 3.6 just works**, without any of DFlash's draft-model machinery. We get the correctness baseline of PLD across the whole model zoo, including the hybrid family. The speedup will still be modest (PLD on input-grounded prompts), but it's strictly more code-path coverage than today.

The same primitive feeds spec 017's prefix KV cache: with tape-replay the MambaCache becomes snapshot-able for the prefix-cache use case.

This is the highest-leverage architectural change in the speculative-decoding stack for this codebase.

### 4.5 Other block-diffusion / multi-token-parallel approaches

**Lookahead Decoding** (Fu et al., 2024) — generates multiple tokens at multiple future positions in parallel via an n-gram prediction loop, then verifies. Doesn't need a draft model. Less effective than EAGLE-3 in published numbers; included for completeness.

**Parallel Prompt Decoding** ([arXiv:2405.18628](https://arxiv.org/abs/2405.18628)) — adds learnable prompt tokens at the end of the input that act as parallel-prediction probes. Reports 2.0–3.0× on consumer hardware, training-light.

**Block diffusion in general** ([Kaminsky et al., 2024](https://arxiv.org/abs/2406.06564)) — a broader category. DFlash is the first one to ship at production speed on Apple Silicon as far as we know.

---

## 4½. Cross-compute-unit speculative decoding (Mirror SD)

A different axis from the three families above: **what if the draft and the target run on different physical compute units?** Apple Silicon ships with two: the GPU (where MLX runs) and the Apple Neural Engine (where Core ML runs). Today, conventional speculative decoding leaves the ANE idle 100% of the time during inference.

Apple ML Research published [Mirror Speculative Decoding (arXiv:2510.13161)](https://arxiv.org/abs/2510.13161) in January 2026, formalising this pattern. The trick:

1. **Target on GPU, draft on NPU.** The two compute units run in parallel.
2. **Early-exit signal.** After roughly N/2 layers, the target emits a top-κ token-and-logprob signal to the draft as a small message (`O(B·κ)`). The draft starts working immediately on this hint while the target finishes its remaining layers.
3. **Two rendezvous points per step:** at early-exit (target → draft signal) and at final verification (draft → target draft tokens).

The result is two pipelines hiding each other's latency. Reported numbers from the paper (Qwen3-14B / 32B target on M2 Ultra × 8 + NPU): **2.8–5.8× wall-time speedup**, 30% relative improvement over EAGLE-3.

The single-stream consumer-hardware version is what [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM)'s `MirrorSpeculativeLoop.swift` implements. The same repo ships:

- Qwen 3.5 0.8B / 2B on Core ML / ANE (the first hybrid SSM+attention LLMs running on Core ML — they ported GatedDeltaNet to Core ML for this).
- Swift implementations of MTP, EAGLE-3, Lookahead, SuffixDecoding, cross-vocab speculative decoding.
- A working `PrefixCache` + `PrefixKVCache`.

In other words: **the entire Apple Silicon speculative-decoding stack we've been describing in this paper is already implemented and shipping as an open-source Swift package.** The piece mlx-swift-lm adds is integrating it with our MLX-backed targets.

For the cost model on a Qwen 3.5-27B-4bit target paired with a Qwen 3.5-0.8B Core ML draft on the same SoC:

- Without Mirror SD: GPU draft + GPU verify, ~110 ms per round, ~138 tok/s (matches dflash-mlx's published 2.37×).
- With Mirror SD (sequential within burst): max(t_ANE_draft, t_GPU_verify) ≈ 84 ms, ~1.5× over the GPU-only DFlash baseline.
- With Mirror SD (with early-exit signal, full pipelining): ~50-60 ms wall, projected 3-5× over baseline. This is the published Mirror SD region.

The ANE also draws ~10× less power than the GPU, so the **tokens-per-watt** improvement is much larger than the throughput improvement — particularly relevant for iOS / unplugged-laptop deployments.

For mlx-swift-lm specifically, [spec 021](../specs/021-ane-offloaded-draft-model.md) lays out the integration plan. The most consequential decision is **build vs depend vs reimplement**: CoreML-LLM has done the hard parts and is Apache 2.0 licensed; depending on it gets us to working Mirror SD in days rather than weeks.

## 5. Cross-cutting: what helps all three families

Three techniques that compose with PLD, draft-model spec, and DFlash equally:

### 5.1 Prefix KV cache (spec 017)

Keep the target's KV state at the end of each request, keyed on a stable prefix. Next request with the same prefix: hydrate, prefill only the suffix. Headline numbers from the literature:

- vLLM Automatic Prefix Caching (APC): 2-10× TTFT improvement on multi-turn chat, [vLLM docs](https://docs.vllm.ai/en/stable/design/prefix_caching/).
- dflash-mlx prefix cache: ~1.5-2× wall-clock improvement on agentic workloads (TTFT compounds with per-turn decode speedup).

For mlx-swift-lm: TTFT on Gemma 4 26B A4B with a 4K-token system prompt drops from ~7s on cold turn to ~1s on cached turn. Decoder-agnostic — works the same for TokenIterator, n-gram, DFlash.

The "stable prefix" detail: chat templates end with `<|im_start|>assistant\n`-style boilerplate that changes every turn. Trim the trailing assistant-opener from the cache key so the same key matches across turns of a conversation. dflash-mlx implements this in [`compute_stable_prefix_len`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/cache/policies.py); we'd port the design.

### 5.2 Tape-replay rollback (spec 020)

Discussed in §4.4. Once you have it as a primitive at the cache layer, every speculative decoder benefits:

- PLD on Qwen 3.5 / 3.6 works (instead of falling back to baseline).
- DFlash's hybrid-rollback is a thin wrapper rather than a custom code path.
- Prefix cache works on hybrid models.
- Future tree-attention rollback (spec 014 phase 4) is a natural extension to multi-path replay.

### 5.3 GPU-side accept/reject

Currently we sync to CPU after every verify forward to decide accepted count. That's the dominant per-round overhead on small fast models. Compute the accept count on the GPU, sync only the integer (a single-element tensor read), reuse `mainTokens` and the trimmed cache directly. ~50% of the per-round sync overhead disappears.

dflash-mlx already does this in [`acceptance.py:match_acceptance_length`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/engine/acceptance.py) (vectorised cumprod over the equality mask). We don't yet — straightforward follow-up.

### 5.4 Workload routing

Different decoders fit different workloads. A complete inference engine routes:

```
                    user request
                          │
                          ▼
              has stable prefix in cache?
              ┌──── yes ────┐
              │             ▼
              │        hydrate prefix
              │             │
              ▼             ▼
       short prompt? (<256 tokens, <256 max-tokens output)
       ┌──── yes ────┐
       │             ▼
       │        TokenIterator (fast path; no spec overhead)
       │             │
       ▼             ▼
       target has DFlash draft?
       ┌──── yes ────┐
       │             ▼
       │        DFlashSpeculativeTokenIterator
       │
       ▼
       greedy (T=0) and trimmable / tape-replay-able cache?
       ┌──── yes ────┐
       │             ▼
       │        NGramSpeculativeTokenIterator
       │
       ▼
       fall back to TokenIterator
```

dflash-mlx implements roughly this — short prompts and hybrid-without-draft fall back to baseline; the dispatcher in [`serve.py:_serve_single`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/serve.py) and [`engine/spec_epoch.py:stream_dflash_generate_impl`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/engine/spec_epoch.py) is the reference pattern.

---

## 6. Empirical takeaways from our M1 Max work

### 6.1 The Apple Silicon speed/cost shape

In rough numbers, on M1 Max 64 GB:

- **Baseline single-token decode** is memory-bandwidth bound. Per-step latency is `model_size / bandwidth + per-step overhead`. The per-step overhead is 1-3 ms (kernel launch, RoPE, norm, sample).
- **Multi-token verify (K+1 ≤ ~16)** is *almost* the same memory cost as one decode (weights stream once), but the per-token compute scales linearly. For dense small models, the compute *matters* — a 5-token verify takes ~2-3× one-token decode. For MoE models, the compute matters less because the active expert subset is small.
- **Sync barriers** add ~3-5 ms per occurrence. On the asyncEval pipelined decode path, we sync once per token (TokenIterator's previousY trick). On any spec-decode path, we sync once per *round* (need int values to decide accept/reject), which is the verify-batch frequency.
- **Hidden-state capture** is essentially free if it's already on the forward path.

Implication: spec decode wins when you have **few syncs per token emitted**. That requires either (a) a high accept rate (more emits per verify, hence per sync), or (b) cheap verify forwards relative to single decodes.

The MoE win on Gemma 4 26B A4B comes from (b) — verify cost barely scales. The dense-model losses come from (a) being insufficient to overcome (b)'s failure.

### 6.2 The output-divergence failure mode

When `K ≥ 4` and the model is at temperature 0, batched-vs-sequential numerical drift can flip the verify forward's argmax to coincidentally match the draft. The iterator accepts the draft, but the *correct* greedy token would have been different. Output diverges from the baseline.

This is real and easily reproducible on Gemma 4 E2B at K=4 on summarisation prompts. Our strict-greedy guard solves it: refuse to accept matches whose top-1/top-2 logit margin is below 0.5. The guard preserves byte-identical output to baseline at the cost of some throughput on tight-margin verifies.

(With strict-greedy default-off, K=4 produced "Flirt." as the entire output of a QA prompt where baseline produced 30 coherent tokens. Memorable bug. Default flipped to on after this.)

### 6.3 Acceptance is workload-bound, not config-bound

You can tune K, n, hits, dominance, multi-candidate, adaptive, and strict-greedy all you want — on a workload with no input/output overlap, accept rate stays low. On a workload with high overlap, even the simplest config wins.

The variation across our four sweep prompts on Gemma 4 26B A4B at the same config:

| Prompt | Best config | Accept | Speedup |
|---|---|---|---|
| qa-requote/01-bug-report | n=3 D=2 | 60% | +25% |
| code-refactor/02-js-callbacks | n=4 D=2 | 50% | +18% |
| recipe-bulk/01-five-soups | adaptive+strict | 39% | +1% |
| open-generation/* | (any) | <20% | parity to −15% |

Same iterator, same model, same hardware. The workload is the variable.

### 6.4 What the bench harness should report

Every per-cell line should include accept rate. Today our `[BENCH] Spec decode: X/Y accepted (Z%)` line does this. A useful addition: **pre-EOS accepted-tokens-per-round** (the running average accept count, not the global accept rate). This catches the "accepted a lot early then collapsed at content phase" pattern.

Spec 018 folds the per-prompt sweep into the harness so we stop scripting it by hand.

---

## 7. The roadmap

In rough priority order, what we'd do next on this stack — significantly reordered after the discovery of `john-rocky/CoreML-LLM` and Apple's Mirror SD paper:

1. **Spec 015 phases 1-3 — DFlash port on GPU.** Standalone 2.4× on Qwen 3.5/3.6, no cross-framework risk. Validates the Swift-side DFlash architecture for later composition with Mirror SD.

2. **Spec 021 Phase 1A — Mirror SD measurement spike via CoreML-LLM dependency.** One week of integration glue, validates 3-5× projected speedup on Qwen 3.5-27B. The single highest-impact spec in the roadmap if it pans out. Includes a side-by-side benchmark of CoreML-LLM's Core ML path vs a private-ANE-API direct path to characterise dispatch overhead.

3. **Spec 020 — tape-replay rollback at the cache layer.** Unblocks PLD on Qwen 3.5/3.6 (orthogonal to Mirror SD — Mirror SD's rollback is on the draft side, not the target's KV cache). Feeds spec 017's hybrid-model prefix cache.

4. **Spec 017 — prefix KV cache.** Multi-turn TTFT improvement. Benefits every decoder. CoreML-LLM has a working reference implementation we can port.

5. **Spec 022 — deterministic-stretch acceleration (chat-template state machine + bigram fallback).** GPT-OSS harmony channel transitions are the highest-value single target. Pure-CPU, no GPU/ANE work.

6. **Spec 014 phase 1 — tree attention with K=2 root branches.** Combines with multi-candidate; meaningful lift on input-grounded workloads.

7. **Spec 016 — cross-request n-gram cache.** Smaller win than prefix cache; mechanical port from llama.cpp.

8. **Spec 015 phase 4-6 + Spec 021 Variant C — DFlash-on-ANE composition.** 4-7× speedup region. Requires DFlash draft → Core ML conversion. Mostly relevant for Qwen 3.5/3.6 production deployments where the headline number matters.

9. **Spec 019 — PLD+ attention-weighted span selection.** Per-model effort; deferred behind tree attention.

10. **Spec 018 — bench harness per-prompt sweep mode.** Smallest spec, large quality-of-life improvement.

11. **MTP heads for Qwen 3.5/3.6.** CoreML-LLM has a working `MtpSpeculativeEngine`; if Qwen 3.5 ships native MTP heads (as Qwen 3 reportedly does), this composes with everything above for additional lift.

---

## 8. Acknowledgments and references

The implementation work in [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) is mine; the speculative-decoding family is several years of community work I've stood on.

Foundational:
- [Speculative decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) — the original.
- [Fast Inference from Transformers via Speculative Decoding (Chen et al., 2023)](https://arxiv.org/abs/2302.01318) — the rejection-sampling framing for non-greedy.

PLD line:
- [Prompt Lookup Decoding (Saxena, 2023)](https://github.com/apoorvumang/prompt-lookup-decoding)
- [Simon Willison's writeup (2024)](https://simonwillison.net/2024/Jan/23/prompt-lookup-decoding/)
- [PLD+ (Singh et al., 2024, arXiv:2412.01447)](https://arxiv.org/abs/2412.01447) — attention-weighted span selection
- [SuffixDecoding (Oliaro et al., NeurIPS 2025, arXiv:2411.04975)](https://arxiv.org/abs/2411.04975) — suffix-tree generalisation

Draft-model line:
- [Medusa (Cai et al., 2024, arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — multi-head parallel drafts
- [EAGLE-3 (Li et al., 2025, arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — training-time test, 3-6.5× speedup
- [llama.cpp speculative decode](https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative) — production reference
- [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) — production reference
- [Multi-Token Prediction (DeepSeek-V4, mlx-lm PR #15)](https://github.com/Blaizzy/mlx-lm/pull/15)

Block-diffusion line:
- [DFlash (Chen et al., 2026, arXiv:2602.06036)](https://arxiv.org/abs/2602.06036)
- [bstnxbt/dflash-mlx engine-v2](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2) — Python reference impl
- [SharpAI/SwiftLM DFlash port](https://github.com/SharpAI/SwiftLM/tree/main/Sources/DFlash) — Swift port in progress

Cross-compute-unit / Apple Silicon line:
- [Mirror Speculative Decoding (Apple ML Research, arXiv:2510.13161)](https://arxiv.org/abs/2510.13161) — the paper that names the cross-device-parallelism pattern
- [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM) — Apache 2.0 Swift package with Mirror SD, EAGLE-3, MTP, Lookahead, SuffixDecoding, PLD, and Qwen 3.5 hybrid SSM+attn on Core ML
- [Speculative Streaming (Apple ML Research, arXiv:2402.11131)](https://arxiv.org/abs/2402.11131) — fuse drafter into target via MTP heads
- [Recurrent Drafter (Apple ML Research)](https://machinelearning.apple.com/research/recurrent-drafter) — earlier conceptual ancestor

Crossover techniques:
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — design doc
- [Mamba (Gu & Dao, 2024, arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — the recurrence we want to roll back
- [GatedDeltaNet (Yang et al., 2024, arXiv:2412.06464)](https://arxiv.org/abs/2412.06464) — Qwen 3.5's variant

llama.cpp specifics (the ngram-cache.cpp / ngram-map.cpp / speculative.cpp triad referenced throughout):
- [llama.cpp ngram-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.cpp)
- [llama.cpp ngram-map.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-map.cpp)
- [llama.cpp speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)
- [llama.cpp examples/speculative-simple](https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/speculative-simple.cpp)
