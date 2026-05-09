# Beyond quadratic attention: a 2026 field survey for Apple Silicon inference

**Author:** Eric Kryski
**Hardware target:** Apple M1 / M5 Max, 16–512 GB unified memory
**Software target:** [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)
**Date:** 2026-05-08
**Status:** Working draft — research survey

> *Softmax attention is O(N²). Speculative decoding helps decode but not the underlying forward. What else is there — and which of it actually composes with a quantised, MoE-aware, GatedDeltaNet-capable inference stack on Apple Silicon?*

This is a working engineer's pass at the post-2024 landscape of techniques that reduce or replace the quadratic attention cost. The field is large and noisy: most papers claim 2–10× and most of those claims do not survive an honest implementation. This survey ranks techniques by what *actually* shipped in vLLM / SGLang / TRT-LLM / llama.cpp / MLX in 2025–2026, what failed, and what is too early to call.

The prioritisation throughout is grounded in [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)'s current stack: TurboQuant Int4/Int8, windowed KV eviction (PR #186), GatedDeltaNet, and a post-spec-006 KV cache hierarchy. The Apple Silicon constraints — unified memory, ~400 GB/s bandwidth on M1 Max, GEMM-friendly Metal kernels, no tensor cores — change which 2026-era techniques are worth the engineering hours.

The companion paper, [speculative-decoding-on-apple-silicon.md](speculative-decoding-on-apple-silicon.md), covers decode throughput. This one covers everything else: sparser attention over the KV cache, sub-quadratic alternatives to softmax attention, adaptive per-token compute, and the genuinely weird ideas (Kuramoto oscillators, diffusion LMs, JEPA, test-time training).

---

## 1. Why this matters on Apple Silicon specifically

Two facts dominate the landscape on M-series chips:

1. **Decode is memory-bandwidth bound, not compute bound.** Streaming a 4-bit-quantised 9B model from unified memory at ~400 GB/s tops out around 89 tok/s in theory and ~54 tok/s measured on M1 Max ([speculative-decoding-on-apple-silicon.md §1](speculative-decoding-on-apple-silicon.md)). Anything that reduces *bytes-per-token-generated* converts directly to throughput.

2. **Past 32 K context, KV-cache reads start to contend with weight streams.** On Qwen 3.5-9B-4bit, decode drops from 54 → 41 tok/s at 32 K, 35 at 64 K, 27 at 128 K. The KV cache itself becomes 1–4 GB of bandwidth pressure. Anything that shrinks **what is read per query** at decode time wins long-context throughput regardless of model size.

These two facts collapse the entire research field into a small number of useful axes:

- **Read fewer KV bytes per query** → top-k attention, page-bound selection, eviction, head specialisation, low-rank latent compression
- **Read fewer weight bytes per token** → MoE (architecture), activation sparsity (post-hoc), self-spec drafts (skip layers in the draft)
- **Read fewer steps per emitted token** → speculative decoding (draft heads, MTP, EAGLE-3), diffusion LMs (parallel decode of N positions)
- **Replace quadratic attention entirely** → SSM/linear hybrids that interleave a small fraction of attention layers

The "exotic" ideas (Kuramoto, FFT, JEPA, Hopfield) mostly do not yet improve any of these axes at LM scale. They are tracked in §6 because the conceptual shifts may matter later, not because they ship today.

---

## 2. TL;DR ranked for an mlx-swift-lm-style stack

| Rank | Technique | What it does | Composes with TurboQuant + windowed KV? | Effort |
|---:|---|---|---|---|
| 1 | **DuoAttention** | per-head split into "retrieval" (full KV) vs "streaming" (sink+window) via a calibration pass | Yes — slots into the windowed-KV hierarchy | Med (Metal kernel for ragged per-head shapes) |
| 2 | **Native MTP / EAGLE-3 draft heads** | model-native draft heads (DeepSeek-V3, Qwen3, GLM-4.5) | Yes — model-loader + spec-decode infra | Med (model-by-model loader work) |
| 3 | **Quest** | per-page top-k attention over full KV via min/max bounds | Yes — small page-metadata addition | Med-Low |
| 4 | **Hybrid models** (Granite-4-H, Qwen3-Next, Kimi Linear) | mostly Mamba2/GDN with ~10–25% attention layers | Yes — extends the GDN path you already ship | High (chunkwise-parallel kernels in Metal) |
| 5 | **TEAL activation thresholding** | training-free magnitude sparsity in MLPs at decode | Yes — bandwidth-bound regime is the M-series sweet spot | Med-High (Metal kernels) |
| 6 | **NoPE-hybrid position encoding** | interleave NoPE layers with RoPE for length generalisation | Mostly orthogonal | Low (model-side) — but it's a research project, not a perf win |
| 7 | **Sigmoid / gated attention** | softmax replacement validated at scale (Apple, Qwen NeurIPS 2025 best paper) | Replaces attention math | Research-grade port project |

Everything below in §§3–6 is the supporting evidence and the failure modes.

---

## 3. Sparser attention over the KV cache

Two classes of technique live here, with very different deployment trade-offs.

### 3.1 Pretrained-native sparse attention

These are big wins but cannot be grafted onto existing dense checkpoints without quality loss.

**MLA — Multi-head Latent Attention** (DeepSeek V2/V3, Dec 2024) — compresses K and V into a low-rank latent vector (~512-dim) per token and decompresses on demand. Memory drops 10–30× per decode step; FLOPs are slightly higher due to the extra projection but irrelevant when bandwidth-bound. Lossless when trained natively. Footgun: the absorbed-projection RoPE trick interacts badly with naive position-encoding rewrites. Apple Silicon fit is excellent — bandwidth saving is exactly what M-series needs. ([arXiv 2412.19437](https://arxiv.org/pdf/2412.19437))

**NSA — Native Sparse Attention** (DeepSeek, Feb 2025, ACL 2025 Best Paper) — three parallel branches per layer: compressed (coarse pooled blocks), selected (top-k learned blocks), sliding (local window), each gated. Asymptotic O(N·k) for the dominant selected branch, with FlashAttention-class constants because it is **block-sparse with block size 64–128** — exactly the tile shape Metal's `simdgroup_async_copy` likes. Near-lossless on RULER and reasoning when natively pretrained. Post-hoc grafting works poorly. ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089), [PyTorch impl](https://github.com/lucidrains/native-sparse-attention-pytorch))

**MoBA — Mixture of Block Attention** (Moonshot/Kimi, Feb 2025) — KV blocks routed like MoE experts via a top-k gate per query. Reports 6.5× at 1 M, 16× at 10 M tokens. Critically, MoBA supports **per-layer fall-back to full attention**, enabling cheap fine-tuning conversion of dense models — the only one of the natively-sparse class with a credible conversion path. In production at Kimi long-context. ([arXiv 2502.13189](https://arxiv.org/abs/2502.13189), [GitHub](https://github.com/MoonshotAI/MoBA))

**DSA — DeepSeek Sparse Attention** (V3.2 Sep 2025, V4 2026) — production successor to NSA. A small **lightning indexer** scores tokens, then a hard top-k selector picks them; selected attention runs at full precision. V4 reports 27% of V3.2 FLOPs/token and 10% KV-cache occupancy at 1 M context. SGLang v0.5.9 and vLLM both have day-0 support. Requires native pretraining. ([V3.2 paper](https://arxiv.org/abs/2512.02556), [vLLM day-0 support](https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference))

For mlx-swift-lm, none of these can be retrofitted onto, say, Qwen3-32B without retraining. They matter as **reasons to prefer hosting models that already ship with them** (DeepSeek-V3, V3.2, V4; Kimi K2 with MoBA).

### 3.2 Post-hoc sparse attention (no retraining)

These work on any pretrained model and are the realistic 2026 wins for an inference engine.

**Quest — query-aware page-level top-k** (MIT, ICML 2024) — per-page min/max K bounds let the query upper-bound the *maximum possible* attention score per page; load only top-k pages, run exact attention over them. **Exact attention over a learned-at-runtime subset** — no quality loss in expectation. ~2.2× attention speedup, ~7× e2e at 32 K context. Pure bandwidth saving — *the* Apple Silicon sweet spot. The cleanest drop-in for sparse decode. ([arXiv 2406.10774](https://arxiv.org/abs/2406.10774), [GitHub](https://github.com/mit-han-lab/Quest))

**DuoAttention** (MIT, ICLR 2025) — profiles attention heads with a short calibration pass on synthetic needle-in-haystack tasks and labels each head as **retrieval** (full KV) or **streaming** (sink + window only). Memory roughly halves on MHA, ~1.6× on GQA. Decode 2.18× / 1.50× (MHA / GQA), prefill 1.73× / 1.63×. Llama-3-8B at **3.3 M context on a single A100** with quant. Per-head calibration is done once at deploy time; runtime is straightforward. ([arXiv 2410.10819](https://arxiv.org/abs/2410.10819), [GitHub](https://github.com/mit-han-lab/duo-attention), [project page](https://hanlab.mit.edu/projects/duo-attention))

For mlx-swift-lm specifically, DuoAttention is the cleanest "skip weights per token" win that survives honest benchmarking on SwiGLU models. The streaming-head path is exactly the windowed-KV plumbing that landed in PR #186. The calibration pass is a one-time per-model cost.

### 3.3 Token eviction

Eviction trades quality for memory and bandwidth.

- **H2O** ([arXiv 2306.14048](https://arxiv.org/abs/2306.14048)) — cumulative-attention "heavy hitter" oracle. Breaks under causal masking; recent tokens get wrongly evicted. Mostly research-only in 2026.
- **SnapKV** ([arXiv 2404.14469](https://arxiv.org/abs/2404.14469)) — scores tokens via attention from a recent observation window. SOTA among uniform-budget evictors. Fixes most H2O failures.
- **Scissorhands** ([arXiv 2305.17118](https://arxiv.org/abs/2305.17118)) — assumes "importance persistence." 2025 *Taming Fragility* paper ([arXiv 2510.13334](https://arxiv.org/abs/2510.13334)) shows it fails at <10% budget.
- **PyramidKV** ([arXiv 2406.02069](https://arxiv.org/html/2406.02069v1)) — non-uniform per-layer budget (more cache in early layers). +5–10% over SnapKV.
- **SqueezeAttention** ([arXiv 2404.04793](https://arxiv.org/abs/2404.04793)) — 2D budget — clusters layers by cosine similarity, allocates accordingly. Orthogonal to per-token methods. ~70% reduction at <1.2% quality loss on Mistral-7B.
- **KVPress** (NVIDIA, [GitHub](https://github.com/NVIDIA/kvpress)) — production-quality unifying library.

**Critical failure mode:** every uniform-budget evictor degrades on multi-hop reasoning and long needle-in-a-haystack because of *token importance recurrence* — a CoT token that was unimportant at write time often gets re-attended much later in the chain. The 2026 line that addresses this:

- **LazyEviction** ([arXiv 2506.15969](https://arxiv.org/html/2506.15969v1)) — defers eviction decisions for reasoning chains.
- **ForesightKV** — same idea, foresight-based.

If you serve reasoning models, prefer **Quest** (top-k over a fully retained cache) over any greedy eviction — it eliminates the token-recurrence failure mode by construction.

### 3.4 Hardware-aware sparsity — what actually wins

The dominant lesson from NSA/DSA: **block size ≥ 32 tokens, contiguous, page-aligned**. Unstructured sparsity (per-token random masks) loses on every accelerator including Apple GPUs because it kills coalesced loads and wastes the 32-wide simdgroup.

On Apple Silicon, the win comes from `simdgroup_async_copy` overlapping K/V loads with QK matmul ([Philip Turner's metal-flash-attention](https://github.com/philipturner/metal-flash-attention)) — block-sparse fits this naturally; per-token sparse does not. Page sizes of 16–64 tokens are the sweet spot, matching M-series cache line + threadgroup tile.

---

## 4. Architectures that don't pay O(N²)

The story of 2025–2026 is decisive: **pure non-attention models keep losing the quality battle, but hybrids that interleave a tiny fraction of attention layers with linear/SSM blocks now match dense transformers at 30B+ scale.**

### 4.1 State Space Models

Mamba2 (Dao & Gu, 2024, [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)) is the only SSM still serious in 2026, mostly because **Structured State Space Duality (SSD)** reformulated it as batched GEMMs — exactly the shape Metal/CUDA likes. S4/S5 are historical. Pure Mamba2 lags transformers by ~10 MMLU points at 7B but matches on commonsense (HellaSwag, ARC, PIQA, WinoGrande). Recall and ICL are the persistent weaknesses. **No serious pure-Mamba checkpoint exists above ~8B in 2026** — everyone went hybrid. RecurrentGemma (Griffin) is alive at 2B/9B; Google never released the larger 14B variant.

### 4.2 Hybrid attention + SSM — the pragmatic winners

The pattern that consistently works: **mostly SSM/linear, with ~1 attention layer per 6–8 blocks, often sliding-window**.

| Model | Org | Size | Mix | Attention placement |
|---|---|---|---|---|
| Jamba 1.6 / 1.7 | AI21 | 52B / 398B MoE | 1:7 attn:Mamba, MoE every 2 | Interleaved |
| Zamba2 | Zyphra | 1.2B–7B | 6 Mamba2 + 1 shared attn (ABAB) | Shared global attn |
| Samba | Microsoft | 3.8B | Mamba–MLP–SWA–MLP | Sliding window |
| Hymba | NVIDIA | ~1.5B | **Parallel** SSM+attn heads in same layer | Every layer |
| Nemotron-H | NVIDIA | 8B / 47B / 56B | 92% Mamba2, 8% attn (10 of 118 layers) | Sparse interleave |
| Granite 4.0-H | IBM | 3B / 7B / **32B-A9B MoE** | Mostly Mamba2, few attn | Sparse interleave |
| Falcon-H1 | TII | 0.5B–34B | **Parallel** attn + Mamba2 heads | Every block |
| LFM2 / LFM2-24B-A2B | Liquid AI | 0.35B–24B MoE | LIV convolutions + GQA | 6 attn of 16 blocks |
| MiniMax-01 | MiniMax | **456B / A45.9B** | 7:1 Lightning Attention : softmax | Every 8 layers |

Nemotron-H-56B matches Llama-3.1-70B / Qwen2.5-72B at **3× decode throughput**; Granite 4.0-H-Small (32B / A9B) reduces long-context RAM by >70%; MiniMax-01 hits 4 M-token context at GPT-4o-class quality. All Apache/MIT-style on HuggingFace.

### 4.3 Linear attention with state expansion / fast weights

This is the most active research area in 2026 and has produced the best hybrids.

**Gated DeltaNet** (NVIDIA, ICLR 2025) — DeltaNet's delta-rule update + Mamba2-style gating; chunkwise-parallel over sequence length using batched GEMMs. Outperforms Mamba2 and DeltaNet on language modeling, recall, and length extrapolation. Now embedded in **Qwen3-Next** (80B / A3B, 3:1 GDN:full-attn) and **Kimi Linear** (Moonshot, 48B / A3B, **Kimi Delta Attention** = GDN + finer-grained channel-wise gating, 3:1 ratio). Kimi Linear's headline claim: first linear-attn variant to **beat full softmax under fair comparison** on short, long, and RL-post-training regimes, with 6× decode at 1 M tokens and 75% KV-cache reduction. ([Gated DeltaNet, arXiv 2412.06464](https://arxiv.org/abs/2412.06464); [Kimi Linear, arXiv 2510.26692](https://arxiv.org/pdf/2510.26692))

**RWKV-7 "Goose"** (March 2025) — generalised delta rule with vector-valued gating + in-context learning rates. Pretrained 0.19B–2.9B at Apache-2.0 on HF; the 2.9B is multilingual SOTA at its size despite undertraining. No 30B+ RWKV-7 exists publicly. Recurrent decode O(1), training parallel. ([arXiv 2503.14456](https://arxiv.org/abs/2503.14456))

**TTT / Titans** (Sun et al., NeurIPS 2025; Google Research) — neural long-term memory as an MLP whose weights update at test time on a "surprise" loss (gradient with momentum). Combined with sliding-window short-term attention; demonstrated >2 M context on needle-in-haystack. **No public weights yet**; this is still research direction, not ship-it-tomorrow. ([TTT, arXiv 2407.04620](https://arxiv.org/abs/2407.04620); [Titans, arXiv 2501.00663](https://arxiv.org/abs/2501.00663))

### 4.4 Hyena / FFT / spectral — mostly dead for LM

Hyena, StripedHyena, Monarch Mixer, Based, M2 are all alive in **genomics** (Evo / Evo2 use StripedHyena2 for DNA) but largely abandoned for language. Together AI's StripedHyena-7B was the high-water mark; nobody scaled it for LM. The community moved on once Mamba2's SSD made SSMs equally hardware-friendly without FFT plumbing. FFT also fits Apple GPUs poorly compared to GEMM. The 2025 "Convolutional Multi-Hybrid LMs at Scale" paper ([arXiv 2503.01868](https://arxiv.org/pdf/2503.01868)) extended the line but no LM at scale shipped on it.

**MonarchAttention** ([arXiv 2505.18698](https://arxiv.org/html/2505.18698v1)) is the most pragmatic spectral idea — *zero-shot conversion* of pretrained softmax attention to Monarch-structured attention with hardware-friendly cost. Not a from-scratch architecture but a retrofit; worth tracking.

### 4.5 RetNet — niche

RetNet (Microsoft, 2023) had three forms (parallel/recurrent/chunkwise) but never produced a competitive open checkpoint at scale. The June 2025 survey ([arXiv 2506.06708](https://arxiv.org/abs/2506.06708)) reads as a retrospective. Treat as historically important for the recurrent/chunked dual form.

### 4.6 Diffusion / non-autoregressive LMs — different cost model

**LLaDA-8B** (Renmin U., Feb 2025) — masked-diffusion transformer trained from scratch; competitive with Llama-3-8B in-context learning, beats GPT-4o on the reversal-curse task. **LLaDA-MoE** (1.4B active) matches Qwen2.5-3B-Instruct.

**Mercury 2** (Inception Labs, late 2025) — first commercial diffusion LLM; **~1000 tok/s** vs Claude 4.5 Haiku ~89 and GPT-5 Mini ~71 at parity quality on reasoning. **The only non-AR architecture with a real commercial speed advantage today.** Closed weights.

**SEDD** (Lou et al., ICML 2024 Best Paper) — score-entropy framing for discrete diffusion; beat GPT-2 at matched size, 25–75% perplexity reduction over prior diffusion LMs.

Cost model is fundamentally different: parallel decode of N tokens per refinement step, ~10–20 steps. Quality at scale beyond ~10B is unproven. **For Apple Silicon this is interesting** because parallel decode plays well with GPU saturation, but you still need quadratic attention internally for now.

### 4.7 Chunkwise-parallel form — why it matters for prefill

The chunkwise form splits the sequence into chunks of size C, runs the **parallel form within each chunk** (GEMM-friendly) and the **recurrent form across chunks** (carries state). Mamba2 SSD, GLA, DeltaNet, Gated DeltaNet, KDA, RWKV-7, RetNet all have it. Without it, prefill of long contexts collapses to O(N) sequential steps and you lose to FlashAttention. With it, **TFLA (Tiled Flash Linear Attention) kernels are reportedly faster than FlashAttention-3 at long sequences and >2× faster than Mamba2 kernels** ([arXiv 2503.14376](https://arxiv.org/pdf/2503.14376), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)).

**This is the single most important property for Metal/Apple Silicon:** GEMM is what `mlx::matmul` does well; pointwise scans are where you bleed performance. Anything without chunkwise-parallel form is not worth porting.

### 4.8 The honest quality gap in 2026

**No pure non-attention model has matched a same-size dense softmax transformer** on the hard suite (MMLU-Pro, GPQA, AIME, RepoQA, multi-hop QA at >32 K). Every leaderboard win in this family — Nemotron-H-56B, Granite 4.0-H-32B, Jamba Large 1.7, Qwen3-Next-80B, Kimi Linear-48B, MiniMax-01-456B — is a **hybrid that keeps 8–25% of layers as attention**.

The recurring failure mode is multi-hop retrieval and multi-turn long context: RULER >128 K still favours full attention; SSM hybrids degrade in multi-turn RepoQA / Math. Pure-Mamba lags ~10 MMLU points; pure-RWKV-7 is competitive only at <3B; pure-LLaDA needs AR plan-conditioning to close reasoning gaps. **Hybrids ≈ transformers on quality with 2–6× decode and 70%+ KV-cache savings — which is the actual win.**

### 4.9 What I would ship today on mlx-swift-lm

If the goal is "competitive, deployable, Metal-friendly, 30B+ class": **Granite 4.0-H-Small (32B / A9B)** or **Qwen3-Next-80B-A3B**. Both Apache 2.0, both MoE-active 3–9B (perfect for Apple unified memory), both ship with chunkwise-parallel kernels you can port from FLA.

- **Granite is the conservative pick** — Mamba2/SSD is the most studied SSM kernel on the planet, IBM publishes integration recipes, and the architecture is essentially "Llama with most attention layers swapped for SSD".
- **Qwen3-Next is the aggressive pick** — Gated DeltaNet is harder to implement (delta rule + chunkwise gating), but you already understand GDN, the codebase already has hooks for it, and you would land on a model that beats Qwen3-32B at 10% the training cost and 10× the throughput.
- **For research / experimentation: Kimi Linear-48B-A3B** — the 3:1 KDA:full-attn pattern with 75% KV-cache reduction is the cleanest bet for "can I fit a 48B model on a 64–128 GB Mac with 1 M-token context?"

I would not invest in pure RWKV-7, pure Mamba, RetNet, Hyena, or diffusion LMs for a production inference target in 2026.

---

## 5. Skipping work per token

The user's framing question — **do we need to sample every layer weight every time?** — has a definitive 2026 answer.

### 5.1 The verdict on dense-model contextual sparsity

Right intuition, wrong era. The 2023 line (Deja Vu, PowerInfer, MoEfication) showed ~80% of FFN neurons are dead per token *for ReLU-era models*. **SwiGLU killed easy activation sparsity** — the smooth gate spreads activity across all neurons. On modern SwiGLU stacks:

- **Deja Vu** ([arXiv 2310.17157](https://arxiv.org/abs/2310.17157)) — 2× on OPT-175B in 2023; <1.2× on modern SwiGLU.
- **CATS** ([Stanford blog](https://scalingintelligence.stanford.edu/blogs/cats/)) — ~15% latency improvement, custom kernels.
- **TEAL** ([arXiv 2408.14690](https://arxiv.org/abs/2408.14690)) — training-free magnitude thresholding on hidden states. **40–50% sparsity, 1.53–1.8× wall-clock decode** on Llama-2/3 + Mistral 7B–70B (ICLR 2025). Composes with quantisation. The most pragmatic of the bunch — open kernel at [github.com/FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL). Not in vLLM/SGLang mainline. **Memory-bandwidth bound decoding is exactly where this should win on Apple Silicon.**
- **TurboSparse / ProSparse** ([arXiv 2406.05955](https://arxiv.org/pdf/2406.05955)) — 85–90% inactive neurons but **needs continued pretraining** to swap activation function back to ReLU². Powers PowerInfer-2.

### 5.2 The industry's actual answer is MoE

The industry's real answer to "stop sampling every weight every time" is **bake the sparsity into the architecture**. DeepSeek-V3 fires 5.5% of its weights per token (37B of 671B). Kimi K2 fires 32B of 1T. Qwen3-235B-A22B picks 8 of 128. ([Sebastian Raschka's architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) is the best single overview.)

That *is* "stop sampling every weight every time" — at the gate level, learned at pretraining time, with a routing predictor that is much smarter than any post-hoc activation predictor. **Production reality: vLLM, SGLang, TRT-LLM, llama.cpp, MLX all have first-class MoE.**

For memory-constrained inference, **PowerInfer-2** ([powerinfer.ai/v2](https://powerinfer.ai/v2/)) offloads 50–75% of FFN/expert weights to NAND on smartphones, achieving 11.68 tok/s on Mixtral-47B (29× over llama.cpp). The Apple Silicon analogue would be unified-memory paging with a learned predictor — interesting research direction but the unified-memory bandwidth math is different from NAND, so the win is smaller.

### 5.3 Speculative decoding evolution — mature

Covered in detail in [speculative-decoding-on-apple-silicon.md](speculative-decoding-on-apple-silicon.md). Headline 2025–2026 results:

- **EAGLE-3** (NeurIPS 2025) — reuses target features through a lightweight draft head. ~80% acceptance peak, ~40–60% real workload. **2.5× in vLLM**, 1.81× at batch 2 / 1.38× at batch 64 in SGLang. ([Red Hat writeup](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding))
- **MTP heads (DeepSeek-V3)** — acceptance >80% on MTP1, 1.8× generation throughput, up to **60% higher output throughput** in SGLang. ([LMSYS blog 2025-07-17](https://www.lmsys.org/blog/2025-07-17-mtp/))
- **SpecExec** ([Together AI](https://www.together.ai/blog/specexec)) — designed for offload — 10–18× on consumer GPUs with 70B + RAM/SSD offload.
- **vLLM V1 dropped LLM-draft Medusa** in favour of EAGLE / n-gram / Medusa heads. ([vLLM docs](https://docs.vllm.ai/en/latest/features/spec_decode/))

For mlx-swift-lm: **MTPLX** ([github.com/youssofal/MTPLX](https://github.com/youssofal/MTPLX)) already proves native MTP works on Apple Silicon (2–2.5× decode at temp 0.6). DeepSeek-V3, Qwen3, GLM-4.5 ship with native MTP heads — supporting them is mostly a model-loader change once you have spec-decode infra.

### 5.4 Self-speculative / layer-skip drafts

**LayerSkip** (Meta, 2024, [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)) — same model drafts using a subset of its own layers, verified by the full forward. 1.86× on Llama-7B, 76–98% acceptance depending on exit layer. **Lossless** (verifier is full model). Catch: **needs layer-dropout finetuning** to make early-exit logits trustworthy. Without it, acceptance collapses. Merged into HF Transformers (Nov 2024) and torchtune (Dec 2024). 2025 follow-ups: DEL (COLM 2025) and CLaSp (ACL 2025) add context-aware exit selection without retraining.

### 5.5 What is dead

- **CALM / SkipDecode / true confidence-based early exit** — KV-cache contradiction. A token exited at layer 8 has no K/V at layers 9–32; future tokens needing to attend to it get garbage. **Batch-size-1 only.** ([arXiv 2407.20272](https://arxiv.org/pdf/2407.20272))
- **Mixture of Depths (DeepMind 2024, [arXiv 2404.02258](https://arxiv.org/abs/2404.02258))** and its 2025 successor **Mixture-of-Recursions ([arXiv 2507.10524](https://arxiv.org/abs/2507.10524))** — paper-only above ~1.7B. No production framework ships it.
- **Deja Vu / MoEfication-style activation predictors on SwiGLU** — eaten by MoE.

### 5.6 Test-time compute (o1/R1) — opposing pressure

Reasoning models generate 10–100× more tokens per query and are the dominant 2025 trend. 2025 work showed longer CoT does *not* monotonically help — correct answers are often shorter than wrong ones ([arXiv 2502.12215](https://arxiv.org/abs/2502.12215)). For an inference engineer this means **decode speedups now compound massively** — a 2× spec-decode win on a 30 K-token reasoning trace saves real wall-clock. Spec decoding + reasoning is the hottest combo in vLLM/SGLang.

### 5.7 Skeptical scoreboard

| Technique | Paper claim | Honest 2026 win | In vLLM/SGLang/TRT-LLM | In MLX |
|---|---|---|---|---|
| EAGLE-3 / MTP spec | 3–6× | 1.5–2.5× | Yes | EAGLE: yes (mlx-community); MTP: MTPLX |
| MoE (architecture) | 10× | 5–18× | Yes | Yes |
| LayerSkip self-spec | 2× | 1.5–1.8× | HF only | No |
| TEAL activation sparsity | 1.8× | 1.4–1.7× decode | No | No |
| DuoAttention | 2× decode + memory | Real on long ctx | No (ref only) | No |
| Deja Vu / CATS | 2–6× | <1.2× on SwiGLU | No | No |
| Mixture of Depths | 2× | unproven >2B | No | No |
| CALM early exit | 3× | batch=1 only | No | No |
| PowerInfer-2 | 29× | 22–29× (NAND-bound only) | No | No |

---

## 6. Outside the box

The user asked specifically about softmax alternatives, FFT-based math, geometric/Cartesian formulations, and Kuramoto oscillators / "weights as oscillators on a mesh". Honest field map for 2026.

### 6.1 Softmax alternatives — bluntness of softmax

Softmax has known pathologies (entropy collapse, attention-sink artifacts, normalisation across irrelevant tokens). The 2026 state of play:

- **Sigmoid attention** — Apple's "Theory, Analysis, and Best Practices for Sigmoid Self-Attention" ([arXiv 2409.04431](https://machinelearning.apple.com/research/sigmoid-self-attention)) proved it is a universal approximator with better regularity than softmax. **FlashSigmoid is ~17% faster than FlashAttention2** on H100, works across language/vision/speech *if* you stabilise the early-training large-norm regime. Sample-complexity follow-up at [arXiv 2502.00281](https://arxiv.org/abs/2502.00281).
- **Gated attention** — Qwen, **NeurIPS 2025 Best Paper** ([writeup](https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/)) — systematic exploration of attention gating; the most rigorous validation that softmax has alternatives that work at scale.
- **Squared-ReLU attention** — Primer (Google, 2021, [arXiv 2109.08668](https://arxiv.org/pdf/2109.08668)) showed squared-ReLU is just better for LMs at fixed compute. Wortsman et al. 2023 ([arXiv 2309.08586](https://arxiv.org/pdf/2309.08586)) showed plain ReLU divided by sequence length matches softmax in ViT.
- **Polynomial attention** ([arXiv 2410.18613](https://arxiv.org/abs/2410.18613)) — softmax's win is implicit Frobenius regularisation; well-chosen polynomials with √N scaling reproduce it.
- **Modern Hopfield** — Ramsauer & Hochreiter ([arXiv 2008.02217](https://arxiv.org/abs/2008.02217)) showed transformer attention *is* the update rule of a continuous-state Hopfield net. So "Hopfield attention" isn't really an alternative — it's a re-interpretation that explains why softmax works (energy minimisation, exponential capacity). 2025 work on continuous-time Hopfield memories ([arXiv 2502.10122](https://arxiv.org/abs/2502.10122)) revives this as a way to compress KV caches into continuous memories rather than discrete patterns. Worth tracking for KV-compression intuitions, not as a softmax replacement.

**Practical adoption: tiny.** Sigmoid + gating are the two with momentum. Everything else stays niche.

### 6.2 Position encoding — biasing weights differently

RoPE dominates production but **NoPE** has emerged as the surprise length-generalisation winner. The 2025 ICLR/NeurIPS papers — "Round and Round We Go: What makes RoPE useful" ([arXiv 2410.06205](https://arxiv.org/pdf/2410.06205)), "RoPE to NoPE and Back Again" ([arXiv 2501.18795](https://arxiv.org/html/2501.18795v1)), "Long-Context Generalization with NoPE-hybrids" ([arXiv 2506.16640](https://arxiv.org/pdf/2506.16640)) — converge on **interleaved NoPE+RoPE layers** beating RoPE-scaling tricks. Llama-4 and Qwen3 both ship interleaved variants. NAPE (NoPE+ALiBi) is the strongest pure extrapolator.

### 6.3 Spectral / FFT / structured matrices

Already covered in §4.4. Summary: **alive in genomics, dormant in language, FFT poor-fit on Apple GPUs**. MonarchAttention ([arXiv 2505.18698](https://arxiv.org/html/2505.18698v1)) is the most pragmatic spectral retrofit. Spectral SSMs (Hazan/Agarwal, [arXiv 2312.06837](https://arxiv.org/abs/2312.06837)) have provable robustness guarantees but no scaled LM yet.

### 6.4 Kuramoto / oscillator-based networks

The "mesh of oscillators" intuition has actual papers behind it:

**AKOrN — Artificial Kuramoto Oscillatory Neurons** (Miyato et al., **ICLR 2025 Oral**, [arXiv 2410.13821](https://arxiv.org/abs/2410.13821), [project page](https://takerum.github.io/akorn_project_page/)). Replaces threshold/activation neurons with oscillators governed by the Kuramoto synchronisation model. Strong results on object discovery (binding via phase synchronisation), adversarial robustness, calibration, and Sudoku reasoning (18% → ~90% with more test-time compute). [Code at github.com/autonomousvision/akorn](https://github.com/autonomousvision/akorn).

**Continuous Thought Machines** (Sakana AI, **NeurIPS 2025 Spotlight**, [arXiv 2505.05522](https://arxiv.org/abs/2505.05522), [Sakana page](https://pub.sakana.ai/ctm/)). Each neuron has its own private weights processing a short history; representation is *neural synchronisation over time*. ImageNet 72.47% top-1 — not SOTA, but the architecture is fundamentally non-feedforward. Sakana is pushing it as a reasoning substrate.

**Language application?** Neither has been scaled to LM yet. The cost story is bad: time-stepping per neuron is expensive and hard to parallelise. The benefit story is interesting: synchrony as a binding mechanism could solve hallucination/grounding in ways attention cannot. **Track quarterly; don't bet on it for production.**

### 6.5 Geometric / hyperbolic / Lorentzian

- **HELM** (May 2025, [arXiv 2505.24722](https://arxiv.org/pdf/2505.24722)) — first hyperbolic LLM with mixture-of-curvature experts.
- **Hierarchical Mamba on Lorentz manifold** ([arXiv 2505.18973](https://arxiv.org/html/2505.18973)).

Theory appeals for hierarchical data; sinh/cosh/arcosh are expensive. Current verdict: niche, won't beat Euclidean transformers on general text.

### 6.6 JEPA / VL-JEPA / LLM-JEPA

LeCun's non-generative line. **LLM-JEPA** ([arXiv 2509.14252](https://arxiv.org/abs/2509.14252)) applies JEPA pretraining to LLMs and outperforms standard objectives, robust to overfitting. **VL-JEPA** (Dec 2025, [Meta AI](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)) hits stronger VL benchmarks than autoregressive VLMs at half the parameters. **Genuinely worth tracking** — most credible "predict embeddings, not tokens" line.

### 6.7 Test-time training / fast weights

Already covered in §4.3. **Mainstream-adjacent**. Lucidrains has working PyTorch ports. Not yet in production frontier models, but Google is investing. Watch closely — if Google ships Titans weights, this becomes a real candidate.

### 6.8 Vector Symbolic Architectures / hyperdimensional computing

**Hyperdimensional Probe** ([arXiv 2509.25045](https://arxiv.org/abs/2509.25045)) uses VSA to *interpret* LLM residual streams (83% probing@1). qFHRR (Quantised Fourier Holographic Reduced Representations) is a 2025 direction. Currently VSA is an interpretability/binding tool, not an LM substrate.

### 6.9 Neuroscience-inspired fringe

Predictive coding networks ([arXiv 2506.06332](https://arxiv.org/pdf/2506.06332)) keep reappearing as theoretical frameworks unifying sparse + predictive + divisive normalisation. **OpenAI's "sparse circuits"** (Nov 2025, [openai.com/index/understanding-neural-networks-through-sparse-circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)) on training LLMs with native sparse connectivity for interpretability is the closest to mainstream. HTM/Numenta is dead for LM. Dendrify and dendritic-computation models remain academic.

### 6.10 What to track vs what is a dead end

**Track closely.** Diffusion LMs (Mercury 2's 10× speed is the most disruptive non-AR result of 2025–2026), Titans/TTT (Google is serious), JEPA-for-LLM (LeCun's bet starting to pay off in VL), sigmoid/gated attention variants (NeurIPS 2025 Best Paper validates Apple's line), NoPE-hybrid position encodings (already in production stacks).

**Quarterly check.** AKOrN/CTM (oscillator nets — beautiful ideas, no LM scaling story yet, but Sakana and Miyato are credible), spectral SSMs (Hazan's group is patient), Hyena/Evo 2 (alive in genomics, dormant in language).

**Functionally dead for LM at scale.** FNet, hyperbolic LLMs, HTM/dendritic, pure VSA as a substrate, Mixture-of-Depths past 1.7B, CALM/early-exit decoders, Deja Vu-style activation predictors on SwiGLU.

**The unifying observation.** Every successful "alternative" to softmax attention in 2026 is a **hybrid** — Liquid LFM2 (conv+GQA), StripedHyena (Hyena+attention), Titans (attention+neural memory), Llama-4 (RoPE+NoPE interleave). Nobody wins by replacing attention; they win by interleaving it with something cheap and learning the right ratio (typically 1 attention layer per 6–8 cheap blocks).

---

## 7. Open problems

1. **Sparsity for reasoning is unsolved.** Top-k + chain-of-thought re-attention interact badly; the 2026 ForesightKV/LazyEviction line is still iterating. If you serve reasoning models, prefer Quest (top-k over a fully retained cache) over any greedy eviction.

2. **Post-hoc conversion of dense checkpoints to NSA/DSA/MLA without quality loss remains lossy.** Native pretraining is currently mandatory for the big sparse-attention wins. MoBA's per-layer fall-back to full attention is the only credible conversion path.

3. **Prefill vs decode asymmetry.** Most evictors only help decode; NSA / MoBA / DSA help both but require retraining. Quest helps both but only saves bandwidth, not FLOPs.

4. **Variable-shape KV per head/layer** (DuoAttention, SqueezeAttention) breaks naive paged-attention kernels — Metal kernels need rewriting to handle ragged caches efficiently. This is the core engineering challenge for landing DuoAttention in mlx-swift-lm.

5. **Sparsity + speculative decoding composition.** Top-k selection is per-query; drafts have different queries than verifies, so the cache-load wins partially evaporate. Open research direction. Important if mlx-swift-lm composes Quest or DuoAttention with the existing spec-decode path.

6. **Pure non-attention parity at scale.** Still unsolved in 2026 — every leaderboard win is a hybrid. The "linear beats softmax" claim from Kimi Linear is the closest thing, but fair comparison still requires the 3:1 KDA:full-attn ratio, not pure linear.

---

## 8. Concrete next-step picks for mlx-swift-lm

Given the stack as of #186 (turbo windowed eviction landed, Gated DeltaNet shipping, post-spec-006 KV hierarchy):

| Pick | Why it composes | Risk |
|---|---|---|
| **1. DuoAttention** | Calibration-pass head split, slots into windowed KV from PR #186; 2× decode + memory on long context, lossless with calibration. | Metal kernel for ragged per-head cache shapes. |
| **2. Native MTP draft heads** | DeepSeek-V3 / Qwen3 / GLM-4.5 ship them; mostly a model-loader change once spec-decode infra exists. MTPLX is a good Apple-Silicon reference. | Per-model loader work; depends on existing spec-decode path. |
| **3. Quest** | Per-page top-k over full KV, no retraining, pure bandwidth win on M-series. Composes with KV pages. | Small page-bound metadata addition. |
| **4. Granite-4-H-32B-A9B / Qwen3-Next-80B-A3B / Kimi Linear-48B-A3B** | Hybrids, MoE-active 3–9B (perfect for unified memory), chunkwise-parallel kernels you can port from FLA. | Gated DeltaNet kernel work in Metal is non-trivial; Granite is the conservative pick (Mamba2-SSD is the most studied SSM kernel). |
| **5. TEAL activation thresholding** | Training-free, targets memory-bandwidth-bound decode (M-series regime). Composes with TurboQuant. | Metal kernel work; quality margin thinner than (1)–(4). |
| **6. NoPE-hybrid** for long-context experiments | Already in production (Llama-4, Qwen3); pairs with windowed eviction. | Research project, not an inference win. |

**Suggested order:** DuoAttention first (cleanest composition with the windowed-KV plumbing), then native MTP heads (largest pure decode win), then porting one chunkwise-parallel hybrid kernel (Granite or Qwen3-Next) since the Gated DeltaNet path is already understood. That gets you to the architectures that won 2025–2026 rather than just optimising dense softmax attention.

What to skip: Mixture-of-Depths (unproven), full early-exit/CALM (KV cache contradiction unsolved), Deja Vu-style MLP predictors on SwiGLU (dead), pure RWKV-7 / pure Mamba / RetNet / Hyena / hyperbolic LLMs / FNet for production targets.

---

## 9. References

### Sparse attention (pretrained-native)

- [Native Sparse Attention (arXiv 2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek, ACL 2025 Best Paper
- [Native Sparse Attention PyTorch impl](https://github.com/lucidrains/native-sparse-attention-pytorch)
- [MoBA — Mixture of Block Attention (arXiv 2502.13189)](https://arxiv.org/abs/2502.13189)
- [MoBA GitHub](https://github.com/MoonshotAI/MoBA)
- [DeepSeek-V3 (arXiv 2412.19437)](https://arxiv.org/pdf/2412.19437) — Multi-head Latent Attention
- [DeepSeek-V3.2 — DSA (arXiv 2512.02556)](https://arxiv.org/abs/2512.02556)
- [vLLM day-0 DSA support (Red Hat)](https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference)

### Sparse attention (post-hoc)

- [DuoAttention (arXiv 2410.10819)](https://arxiv.org/abs/2410.10819) — ICLR 2025
- [DuoAttention GitHub](https://github.com/mit-han-lab/duo-attention)
- [DuoAttention project page](https://hanlab.mit.edu/projects/duo-attention)
- [Quest (arXiv 2406.10774)](https://arxiv.org/abs/2406.10774) — ICML 2024
- [Quest GitHub](https://github.com/mit-han-lab/Quest)

### KV cache eviction

- [H2O (arXiv 2306.14048)](https://arxiv.org/abs/2306.14048)
- [SnapKV (arXiv 2404.14469)](https://arxiv.org/abs/2404.14469)
- [Scissorhands (arXiv 2305.17118)](https://arxiv.org/abs/2305.17118)
- [PyramidKV (arXiv 2406.02069)](https://arxiv.org/html/2406.02069v1)
- [SqueezeAttention (arXiv 2404.04793)](https://arxiv.org/abs/2404.04793)
- [Taming KV Cache Fragility (arXiv 2510.13334)](https://arxiv.org/abs/2510.13334)
- [LazyEviction (arXiv 2506.15969)](https://arxiv.org/html/2506.15969v1)
- [NVIDIA KVPress](https://github.com/NVIDIA/kvpress)
- [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression)

### Hybrid SSM + attention models

- [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [NVIDIA Nemotron-H](https://research.nvidia.com/labs/adlr/nemotronh/)
- [Nemotron-H 56B base model](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K)
- [Falcon-H1](https://falcon-lm.github.io/blog/falcon-h1/)
- [MiniMax-01](https://www.minimax.io/news/minimax-01-series-2)
- [Jamba 1.6 (AI21)](https://www.ai21.com/blog/introducing-jamba-1-6/)
- [AI21 Hybrid LLMs essay](https://www.ai21.com/blog/rise-of-hybrid-llms/)
- [Zamba2-7B (Zyphra)](https://www.zyphra.com/post/zamba2-7b)
- [Hymba (arXiv 2411.13676, ICLR 2025)](https://arxiv.org/html/2411.13676v1)
- [RecurrentGemma](https://ai.google.dev/gemma/docs/recurrentgemma)
- [Liquid LFM2 blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- [LFM2 technical report (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404)
- [NVIDIA Mamba/Mamba2 empirical study (arXiv 2406.07887)](https://arxiv.org/html/2406.07887v1)

### Linear attention with state expansion

- [Mamba2 (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060)
- [Gated DeltaNet (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464) — ICLR 2025
- [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Kimi Linear (arXiv 2510.26692)](https://arxiv.org/pdf/2510.26692)
- [RWKV-7 Goose (arXiv 2503.14456)](https://arxiv.org/abs/2503.14456)
- [TTT layers (arXiv 2407.04620)](https://arxiv.org/abs/2407.04620)
- [Titans (arXiv 2501.00663)](https://arxiv.org/abs/2501.00663)
- [Titans / MIRAS Google blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- [Songlin Yang DeltaNet talk](https://sustcsonglin.github.io/assets/pdf/talk_250117.pdf)
- [TFLA chunkwise-parallel kernels (arXiv 2503.14376)](https://arxiv.org/pdf/2503.14376)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

### Spectral / FFT / structured matrices

- [Hyena Hierarchy (arXiv 2302.10866)](https://arxiv.org/abs/2302.10866)
- [StripedHyena 7B (Together AI)](https://www.together.ai/blog/stripedhyena-7b)
- [Convolutional Multi-Hybrid LMs at Scale (arXiv 2503.01868)](https://arxiv.org/pdf/2503.01868)
- [Spectral State Space Models (arXiv 2312.06837)](https://arxiv.org/abs/2312.06837)
- [MonarchAttention (arXiv 2505.18698)](https://arxiv.org/html/2505.18698v1)
- [StripedHyena/Evo2 in genomics (homolog.us)](https://homolog.us/blogs/bioinfo/2025/04/23/stripedhyena-evo-evo2/)

### Diffusion language models

- [LLaDA (arXiv 2502.09992)](https://arxiv.org/abs/2502.09992)
- [LLaDA-MoE (arXiv 2509.24389)](https://arxiv.org/html/2509.24389v1)
- [Mercury (arXiv 2506.17298)](https://arxiv.org/abs/2506.17298)
- [Mercury 2 announcement (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury-2)
- [Mercury (Inception Labs blog)](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [SEDD: Score Entropy Discrete Diffusion (arXiv 2310.16834)](https://arxiv.org/abs/2310.16834)

### Adaptive computation / activation sparsity

- [Deja Vu (arXiv 2310.17157)](https://arxiv.org/abs/2310.17157)
- [TEAL training-free activation sparsity (arXiv 2408.14690)](https://arxiv.org/abs/2408.14690)
- [TEAL Together AI blog](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)
- [TEAL kernel](https://github.com/FasterDecoding/TEAL)
- [CATS (Stanford)](https://scalingintelligence.stanford.edu/blogs/cats/)
- [TurboSparse / ProSparse (arXiv 2406.05955)](https://arxiv.org/pdf/2406.05955)
- [PowerInfer-2](https://powerinfer.ai/v2/)
- [SSD/MoE offload survey (arXiv 2508.06978)](https://arxiv.org/pdf/2508.06978)
- [Mixture-of-Depths (arXiv 2404.02258)](https://arxiv.org/abs/2404.02258)
- [Mixture-of-Recursions (arXiv 2507.10524)](https://arxiv.org/abs/2507.10524)

### Speculative decoding (recent)

- [EAGLE-3 / vLLM (Red Hat Developer)](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)
- [DeepSeek-V3 MTP / SGLang (LMSYS)](https://www.lmsys.org/blog/2025-07-17-mtp/)
- [LayerSkip (arXiv 2404.16710)](https://arxiv.org/abs/2404.16710)
- [DEL (COLM 2025)](https://github.com/hoenza/DEL)
- [SpecExec (Together AI)](https://www.together.ai/blog/specexec)
- [Apple Recurrent Drafter](https://machinelearning.apple.com/research/recurrent-drafter)
- [MTPLX — Apple Silicon native MTP](https://github.com/youssofal/MTPLX)
- [vLLM Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [Test-time scaling revisited (arXiv 2502.12215)](https://arxiv.org/abs/2502.12215)
- [Diminishing Returns of Early-Exit (arXiv 2603.23701)](https://arxiv.org/html/2603.23701)
- [CALM analysis (arXiv 2407.20272)](https://arxiv.org/pdf/2407.20272)

### Softmax alternatives, position encoding

- [Sigmoid Self-Attention (Apple, arXiv 2409.04431)](https://machinelearning.apple.com/research/sigmoid-self-attention)
- [Sigmoid Self-Attention sample complexity (arXiv 2502.00281)](https://arxiv.org/abs/2502.00281)
- [Qwen Attention Gating, NeurIPS 2025 Best Paper writeup](https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/)
- [Primer: Squared ReLU Attention (arXiv 2109.08668)](https://arxiv.org/pdf/2109.08668)
- [Polynomial Alternatives to Softmax (arXiv 2410.18613)](https://arxiv.org/abs/2410.18613)
- [Replacing softmax with ReLU in ViTs (arXiv 2309.08586)](https://arxiv.org/pdf/2309.08586)
- [Hopfield Networks Is All You Need (arXiv 2008.02217)](https://arxiv.org/abs/2008.02217)
- [Modern Hopfield Networks with Continuous-Time Memories (arXiv 2502.10122)](https://arxiv.org/abs/2502.10122)
- [Round and Round We Go: RoPE analysis (arXiv 2410.06205)](https://arxiv.org/pdf/2410.06205)
- [RoPE to NoPE and Back Again (arXiv 2501.18795)](https://arxiv.org/html/2501.18795v1)
- [Long-Context Generalization with NoPE hybrids (arXiv 2506.16640)](https://arxiv.org/pdf/2506.16640)

### Oscillator / dynamical / geometric

- [Artificial Kuramoto Oscillatory Neurons (arXiv 2410.13821)](https://arxiv.org/abs/2410.13821) — ICLR 2025 Oral
- [AKOrN project page](https://takerum.github.io/akorn_project_page/)
- [AKOrN code](https://github.com/autonomousvision/akorn)
- [Continuous Thought Machines (arXiv 2505.05522)](https://arxiv.org/abs/2505.05522) — NeurIPS 2025 Spotlight
- [Sakana CTM page](https://pub.sakana.ai/ctm/)
- [HELM Hyperbolic LLMs (arXiv 2505.24722)](https://arxiv.org/pdf/2505.24722)
- [Hierarchical Mamba + Hyperbolic (arXiv 2505.18973)](https://arxiv.org/html/2505.18973)
- [LLM-JEPA (arXiv 2509.14252)](https://arxiv.org/abs/2509.14252)
- [V-JEPA (Meta AI)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [Hyperdimensional Probe / VSA (arXiv 2509.25045)](https://arxiv.org/abs/2509.25045)
- [Predictive Coding Networks Intro (arXiv 2506.06332)](https://arxiv.org/pdf/2506.06332)
- [OpenAI Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)

### Frameworks and overviews

- [vLLM Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [Big LLM Architecture Comparison (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [State of LLMs 2025 (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [metal-flash-attention (Philip Turner)](https://github.com/philipturner/metal-flash-attention)
- [RetNet retrospective survey (arXiv 2506.06708)](https://arxiv.org/abs/2506.06708)

### Companion papers in this directory

- [speculative-decoding-on-apple-silicon.md](speculative-decoding-on-apple-silicon.md) — decode-throughput-focused tour, covers spec-decode, prefix caching, tape-replay
