# Beyond quadratic attention: a 2026 field survey for Apple Silicon inference

**Author:** Eric Kryski
**Hardware target:** Apple M1 / M5 Max, 16–512 GB unified memory
**Software target:** [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)
**Date:** 2026-05-09
**Status:** Working draft — research survey (revision 6)

> *Softmax attention is O(N²). Speculative decoding helps decode but not the underlying forward. What else is there — and which of it actually composes with a quantised, MoE-aware, GatedDeltaNet-capable inference stack on Apple Silicon?*

This is my working-engineer's pass at the post-2024 landscape of techniques that reduce or replace the quadratic attention cost. The field is large and noisy: most papers claim 2–10× and most of those claims do not survive an honest implementation. I'm ranking techniques by what *actually* shipped in vLLM / SGLang / TRT-LLM / llama.cpp / MLX in 2025–2026, what failed, and what's too early to call.

My prioritisation throughout is grounded in [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)'s current stack: TurboQuant Int4/Int8, windowed KV eviction (PR #186), GatedDeltaNet, and a post-spec-006 KV cache hierarchy. The Apple Silicon constraints — unified memory, ~400 GB/s bandwidth on M1 Max, GEMM-friendly Metal kernels, no tensor cores — change which 2026-era techniques are worth the engineering hours.

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
| 1 | **Native MTP / EAGLE-3 draft heads** | model-native draft heads (DeepSeek-V3, Qwen3, GLM-4.5) | Yes — model-loader + spec-decode infra | Med (model-by-model loader work) |
| 2 | **DuoAttention** | per-head split into "retrieval" (full KV) vs "streaming" (sink+window) via a calibration pass | Yes — slots into the windowed-KV hierarchy | Med (Metal kernel for ragged per-head shapes) |
| 3 | **Quest** | per-page top-k attention over full KV via min/max bounds | Yes — small page-metadata addition | Med-Low |
| 4 | **TEAL activation thresholding** | training-free magnitude sparsity in MLPs at decode | Yes — bandwidth-bound regime is the M-series sweet spot | Med-High (Metal kernels) |
| 5 | **Hybrid models** (Granite-4-H, Qwen3-Next, Kimi Linear) | mostly Mamba2/GDN with ~10–25% attention layers | Yes — extends the GDN path you already ship | High (chunkwise-parallel kernels in Metal) |
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

For mlx-swift-lm specifically, DuoAttention is the cleanest "skip work per token" win I've found that survives honest benchmarking on SwiGLU models. The streaming-head path is exactly the windowed-KV plumbing that landed in PR #186. The calibration pass is a one-time per-model cost.

### 3.3 Token eviction

Eviction trades quality for memory and bandwidth.

- **H2O** ([arXiv 2306.14048](https://arxiv.org/abs/2306.14048)) — cumulative-attention "heavy hitter" oracle. Breaks under causal masking; recent tokens get wrongly evicted. Mostly research-only in 2026.
- **SnapKV** ([arXiv 2404.14469](https://arxiv.org/abs/2404.14469)) — scores tokens via attention from a recent observation window. SOTA among uniform-budget evictors. Fixes most H2O failures.
- **Scissorhands** ([arXiv 2305.17118](https://arxiv.org/abs/2305.17118)) — assumes "importance persistence." 2025 *Taming Fragility* paper ([arXiv 2510.13334](https://arxiv.org/abs/2510.13334)) shows it fails at <10% budget.
- **PyramidKV** ([arXiv 2406.02069](https://arxiv.org/html/2406.02069v1)) — non-uniform per-layer budget (more cache in early layers). +5–10% over SnapKV.
- **SqueezeAttention** ([arXiv 2404.04793](https://arxiv.org/abs/2404.04793)) — 2D budget — clusters layers by cosine similarity, allocates accordingly. Orthogonal to per-token methods. ~70% reduction at <1.2% quality loss on Mistral-7B.
- **TriAttention** ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921), [GitHub](https://github.com/WeianMao/triattention)) — importance-ranked KV pruning that scores keys in *pre-RoPE space* using a closed-form trigonometric series derived from offline-calibrated Q/K statistics, then re-prunes every 128 tokens. Headline: on AIME25 reasoning (Qwen3-8B, 32K generation) it matches full-attention accuracy at **2.5× throughput** or **10.7× KV-memory reduction**, while SnapKV and R-KV collapse to ~half accuracy at the same budget. Wins on RULER@4K (66.1 vs SnapKV 55.6). The caveat that matters for me: the budget is fixed and the scorer is global, so it evicts aggressively. The paper's own Recursive State Query benchmark shows degradation past depth 18 at a 2048-token budget, and there's **no published NIAH at >32K** to confirm it holds at frontier-context-length scales. Treat the 2.5× as real for AIME-class reasoning workloads, but assume it shaves long-context retrieval until someone publishes the NIAH curve.
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

### 4.4 Hyena / FFT / spectral — abandoned in the open, not necessarily *dead*

The conventional read is that Hyena, StripedHyena, Monarch Mixer, Based, and M2 are all alive in **genomics** (Evo / Evo2 use StripedHyena2 for DNA) but largely abandoned for language. Together AI's StripedHyena-7B was the high-water mark; nobody scaled it for LM. The community moved on once Mamba2's SSD made SSMs equally hardware-friendly without FFT plumbing.

I want to be careful about over-reading "abandoned" as "dead." The actual reason StripedHyena lost is **kernel ergonomics**, not asymptotics. cuFFT/cuBLAS make GEMM trivial and FFT non-trivial; nobody on a CUDA target wants to write a fused long-convolution kernel when Mamba2's SSD gives them the same asymptotic with a stack of matmuls. On a different hardware target with a different cost model, that calculus could flip.

On Apple Silicon specifically:
- The Accelerate framework's `vDSP` has competitive FFT on the CPU side, but `MPSGraph` and Metal Performance Shaders do not expose a first-class batched FFT for the shapes long-convolution language models need. Anyone serious would write a custom Metal kernel (analogous to Philip Turner's metal-flash-attention).
- FFT length and cache-line alignment matter more on M-series than on H100 because of the lower memory-bandwidth ceiling. A well-tuned mixed-radix Stockham FFT with `simdgroup_async_copy` could plausibly beat naive GEMM-backed long convolutions for very long contexts, where the O(N log N) of FFT beats O(N²) attention even after the constants.
- Hyena's per-layer learned filters can be precomputed in spectral form at load time; the convolution is then a pointwise multiply + iFFT. That's a memory-bandwidth-friendly path on unified memory.

So the honest framing: **the open-source LM ecosystem walked away from FFT-based architectures because SSMs were easier to ship on CUDA, not because the math is worse**. If someone with serious Metal kernel chops invested in a fused Stockham FFT + pointwise-conv-in-spectral-domain kernel on M-series, the Hyena family could become interesting again, especially for very long context where the asymptotic gap actually shows. I'm not going to invest there before exhausting the lower-hanging fruit, but I don't think "dead" is the right verdict; it's "no-one-is-trying" and that's a different thing.

**MonarchAttention** ([arXiv 2505.18698](https://arxiv.org/html/2505.18698v1)) is the most pragmatic spectral idea in the meantime — *zero-shot conversion* of pretrained softmax attention to Monarch-structured attention with hardware-friendly cost. Not a from-scratch architecture but a retrofit; worth tracking.

Worth a separate note: the FFT-based long convolutions in Hyena are not the only sub-quadratic structured-matrix option on the menu. The **Fast Walsh-Hadamard Transform (FWHT)** is a ±1-entries butterfly network with O(N log N) cost, no complex arithmetic, and trivially Metal-friendly. We already ship a tuned FWHT Metal kernel in TurboQuant for SRHT-based outlier smoothing (QuaRot / SpinQuant lineage) — radix-2 Sylvester butterfly with `simd_shuffle_xor` for intra-SIMD stages and shared-memory cross-SIMD stages, power-of-2 head dims up to 1024. So the cost story for any butterfly- or Monarch-based architecture on Apple Silicon is much better than it is for general FFT — the primitive is already in the pipeline. Expanded in [§6.3](#63-spectral--fft--structured-matrices--wht-butterfly-and-monarch).

**Spectral SSMs** (Hazan/Agarwal, [arXiv 2312.06837](https://arxiv.org/abs/2312.06837)) have *provable* robustness guarantees independent of spectrum or dimensionality. If the theory holds at scale they could be a stable substrate for long-context. DeepMind released an implementation but nobody has shipped a scaled LM on it yet.

The 2025 "Convolutional Multi-Hybrid LMs at Scale" paper ([arXiv 2503.01868](https://arxiv.org/pdf/2503.01868)) extended the Hyena line; no LM at scale has shipped on it.

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

The architecture landscape splits into three buckets that matter for this comparison:

- **Dense softmax attention.** Gemma 4 (26B-A4B / 31B / E2B / E4B) is the cleanest 2026 reference at the 31B scale: pure attention with local sliding-window interleaved with full global softmax, no SSM/Mamba component.
- **Hybrid attention + SSM/linear.** Qwen3.5 (3:1 Gated DeltaNet : full attention, with sparse MoE on the larger tiers, 262K native / ~1M YaRN), Qwen3.6 (same 3:1 ratio plus MTP heads; Qwen3.6-27B dense beats the 397B MoE on SWE-bench Verified 77.2 vs 76.2), Qwen3-Next-80B-A3B, Kimi Linear-48B-A3B (KDA + MLA 3:1), Granite 4.0-H / 4.1, Nemotron-H, MiniMax-01, Jamba, Falcon-H1, LFM2.
- **Purely attention-free.** Pure Mamba/Mamba2, pure RWKV-7, pure LLaDA. No public checkpoint above ~8B in any of these families.

What the third-party-verifiable RULER@128K numbers I could find say:

- **Kimi Linear-48B-A3B:** RULER@128K = **84.3** ([paper](https://arxiv.org/pdf/2510.26692)), beating MLA and GDN-H baselines under fair comparison; 3.98× decode speedup over MLA at long context.
- **Granite 4.1:** RULER@128K = **73.0 (8B)** / **76.7 (30B)**. Below frontier dense scores at 128K but matches frontier on MMLU-class.
- **Qwen3-Next-80B-A3B:** RULER ≈ **91.8%** average — this is Qwen's own number, widely cited downstream but I have not found an independent re-run.
- **Nemotron-H:** Paper reports RULER 16K–128K comparable to Qwen2.5-7B at 128K ([arXiv 2504.03624](https://arxiv.org/pdf/2504.03624)).
- **MiniMax-Text-01:** Author-reported RULER 0.91–0.95 from 4K to 1M ([arXiv 2501.08313](https://arxiv.org/abs/2501.08313)); I have not found an independent reproduction.

For the three dense/hybrid models that matter most to a model-host decision today — **Qwen3.5, Qwen3.6-27B, and Gemma 4** — I have not found third-party RULER@128K or NIAH>128K reproductions outside the authors' own evaluations. Vendor numbers exist; community reproductions on r/LocalLLaMA, X, and the Vellum / ArtificialAnalysis leaderboards focus on intelligence-index composites (GPQA-D, HLE, IFBench, AA-LCR) rather than RULER. The closest third-party long-context signal is the r/LocalLLaMA observation that Gemma 4 degrades earlier than Qwen3.6 under Q8-KV — qualitative, not a benchmark.

What I can say with the public record:

1. **No purely attention-free model has matched a same-size dense softmax transformer** on the hard suite (MMLU-Pro, GPQA, AIME, RepoQA, multi-hop QA at >32K). Pure-Mamba lags ~10 MMLU points at 7B; pure-RWKV-7 is competitive only at <3B; pure-LLaDA needs AR plan-conditioning to close reasoning gaps. The 2026 "Hybrid Linear Attention Done Right" paper ([arXiv 2601.22156](https://arxiv.org/pdf/2601.22156)) and the 2026 long-context generalisation study ([arXiv 2506.16640](https://arxiv.org/pdf/2506.16640)) both reinforce this — the latter shows pure Mamba and RWKV-7 trained on 2K windows extrapolating only to ~8K–16K.
2. **Hybrids with ~25% softmax attention have closed the gap on the benchmarks where reproductions exist.** Kimi Linear-48B at 84.3 RULER@128K is the strongest third-party-verifiable data point in this bucket. The 3:1 cheap-layer-to-softmax ratio that Kimi Linear, Qwen3-Next, Qwen3.5, and Qwen3.6 converged on independently looks like the load-bearing structural fact.
3. **Whether Gemma 4 still holds a quality edge over the hybrids at >128K in independent benchmarks is something I haven't found benchmarks that confirm or deny.** It's the open question.

The recurring failure mode for the SSM/linear side is **multi-hop retrieval and multi-turn long context**: SSM hybrids degrade in multi-turn RepoQA / Math, RULER >128K still skews to full attention in the Granite 4.1 numbers. **Hybrids ≈ transformers on quality with 2–6× decode and 70%+ KV-cache savings — which is the actual win, and the picture I'd bet on continuing.**

### 4.9 What I would ship today on mlx-swift-lm

If the goal is "competitive, deployable, Metal-friendly, 30B+ class": **Granite 4.0-H-Small (32B / A9B)** or **Qwen3-Next-80B-A3B**. Both Apache 2.0, both MoE-active 3–9B (perfect for Apple unified memory), both ship with chunkwise-parallel kernels you can port from FLA.

- **Granite is the conservative pick** — Mamba2/SSD is the most studied SSM kernel on the planet, IBM publishes integration recipes, and the architecture is essentially "Llama with most attention layers swapped for SSD".
- **Qwen3-Next is the aggressive pick** — Gated DeltaNet is harder to implement (delta rule + chunkwise gating), but you already understand GDN, the codebase already has hooks for it, and you would land on a model that beats Qwen3-32B at 10% the training cost and 10× the throughput.
- **For research / experimentation: Kimi Linear-48B-A3B** — the 3:1 KDA:full-attn pattern with 75% KV-cache reduction is the cleanest bet for "can I fit a 48B model on a 64–128 GB Mac with 1 M-token context?"

A few architectures I keep watching even though I wouldn't make them my *production* target today:

- **RWKV-7 "Goose"** is underrated. Apache 2.0, O(1) recurrent decode, parallel training, the 2.9B is multilingual SOTA at its size, and the architecture is genuinely simple to port. The reason it doesn't have a 30B+ public checkpoint is that no one has thrown the compute at it, not that the architecture failed. If a serious sponsor trained a 30B+ RWKV-7 on a competitive token budget, this could be a real candidate — especially for the on-device-with-long-context use case M-series cares about.
- **Diffusion LMs** are the most credible non-AR bet on the board. Mercury 2's ~1000 tok/s at parity quality vs Claude 4.5 Haiku and GPT-5 Mini is the most disruptive 2025–2026 inference result, and the parallel-decode cost model maps well to GPU saturation on Apple Silicon. Open-weight LLaDA / LLaDA-MoE are the obvious experiment platforms. Quality at scale beyond ~10B is still unproven; that's the gating question.
- **BitNet-style ternary models** ({-1, 0, +1} weights, BitNet b1.58, Microsoft's 2024–2025 line). I'd treat ternary quantisation as a separate axis from architecture choice — it composes with whatever sequence model you use. The 2025–2026 BitNet 3B/7B/14B native checkpoints are interesting precisely because they let you skip TurboQuant's quantise-and-recover dance: the model is already in a near-integer regime, multiplications become adds, and the memory-bandwidth pressure on Apple Silicon collapses. I think this is one of the most promising bets for M-series inference specifically and want to spec it out separately once we've cleared the post-hoc work.

What I'd genuinely *skip* for a 2026 production target: pure Mamba (no scaled checkpoints), RetNet (no competitive open weights), pure Hyena for LM (no one is iterating on it openly), and hyperbolic LLMs (cost-prohibitive). Everything else stays on the watch list.

---

## 5. Skipping work per token

> *Do we need to sample every layer weight every time? Most of them are likely non-matching.*

That's the framing question I keep coming back to. The 2026 answer turns out to be pretty definitive.

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

### 5.5 Adaptive depth and early exit — early days, not dead

I originally wrote this section as a "what's dead" graveyard, but I don't think that's right. None of these are conclusively dead; they're either *underexplored* or *blocked on a specific engineering problem*. Worth being more careful:

- **CALM / SkipDecode / true confidence-based early exit** ([arXiv 2407.20272](https://arxiv.org/pdf/2407.20272)) hit a real wall: a token exited at layer 8 has no K/V at layers 9–32, so future tokens attending to it see garbage. Published wins are batch-size-1 only and the paper claims (up to 3×) haven't translated to deployed serving stacks. Not dead so much as **stuck on the KV-cache-coherence problem**. If someone figures out a clean way to back-fill K/V for early-exited tokens (e.g. periodic re-computation, or pairing with an SSM-style state that doesn't need per-layer K/V), this comes back.
- **Mixture of Depths** ([DeepMind 2024, arXiv 2404.02258](https://arxiv.org/abs/2404.02258)) — per-token routing past whole transformer blocks. The original paper tops out at ~3B and nobody at frontier scale has retried it in 18 months. That's a *lack of evidence*, not a refutation; the architecture is interesting and the cost of scaling it has just been higher than competing options like MoE.
- **Mixture of Recursions** ([arXiv 2507.10524](https://arxiv.org/abs/2507.10524)) is the 2025 successor and reports 2× throughput at iso-accuracy up to 1.7B. **MoR is genuinely underexplored.** Token-level recursive routing is one of the most natural ways to think about "spend more compute on harder tokens" and nobody has tested it at frontier scale yet. I'd watch it.
- **Deja Vu / MoEfication-style activation predictors on SwiGLU** — the published wins (2–6×) were on ReLU-era OPT-175B and don't translate to SwiGLU. The architectural move that ate this space is MoE — which is the same idea (predict which weights matter, skip the rest) baked into the training loop instead of bolted on after. So "eaten by MoE" rather than "dead."

### 5.6 Test-time compute (o1/R1) — opposing pressure

Reasoning models generate 10–100× more tokens per query and are the dominant 2025 trend. 2025 work showed longer CoT does *not* monotonically help — correct answers are often shorter than wrong ones ([arXiv 2502.12215](https://arxiv.org/abs/2502.12215)). For an inference engineer this means **decode speedups now compound massively** — a 2× spec-decode win on a 30 K-token reasoning trace saves real wall-clock. Spec decoding + reasoning is the hottest combo in vLLM/SGLang.

### 5.7 Skeptical scoreboard

Verified numbers only — anything I couldn't anchor to a published benchmark is marked **unknown**. Every "Published" cell links to the paper or vendor source the number is from. "Reproduced" is what's been confirmed in independent deployments; many techniques don't have public reproduction yet.

| Technique | Published claim | Reproduced / honest range | In vLLM/SGLang/TRT-LLM | In MLX |
|---|---|---|---|---|
| EAGLE-3 / MTP spec | 3.0–6.5× at T=0 ([EAGLE-3](https://arxiv.org/abs/2503.01840)); MTP up to 60% throughput in SGLang ([LMSYS](https://www.lmsys.org/blog/2025-07-17-mtp/)) | 1.5–2.5× real workload, batch-dependent | Yes | EAGLE: yes (mlx-community); MTP: MTPLX |
| MoE (vs dense equivalent) | Mixtral 8x7B ~6× vs Llama-2-70B ([Epoch AI](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference)) | 5–10× at iso-total-params; ~1–2× at iso-active-params | Yes | Yes |
| LayerSkip self-spec | 1.34–2.16× (1.82× coding, 2.0× semantic, 2.16× summarisation) ([arXiv 2404.16710](https://arxiv.org/abs/2404.16710)) | 1.5–2.0× with layer-dropout finetune | HF + torchtune | No |
| TEAL activation sparsity | 1.53× @ 40% sparsity, 1.8× @ 50% ([arXiv 2408.14690](https://arxiv.org/abs/2408.14690)) | 1.5–1.8× decode | No | No |
| DuoAttention | 2.18× decode / 2.55× memory (MHA); 1.50× / 1.67× (GQA) ([arXiv 2410.10819](https://arxiv.org/abs/2410.10819)) | Same as published; long-context regime | No (ref only) | No |
| TriAttention | 2.5× throughput **or** 10.7× KV-memory at matched AIME25 accuracy ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) | Unknown — no independent reproduction yet; no NIAH @>32K published | No | No |
| Deja Vu / CATS on SwiGLU | Deja Vu 2–6× on OPT-175B (ReLU); CATS ~15% latency | **Unknown** — no clean published SwiGLU wallclock | No | No |
| Mixture of Depths | ~2× at ≤3B ([arXiv 2404.02258](https://arxiv.org/abs/2404.02258)); MoDification 1.07–1.2× on Llama-2-7B ([arXiv 2410.14268](https://arxiv.org/abs/2410.14268)) | Unknown at >2B | No | No |
| Mixture of Recursions | 2× throughput at iso-accuracy ≤1.7B ([arXiv 2507.10524](https://arxiv.org/abs/2507.10524)) | Unknown at >2B | No | No |
| CALM early exit | up to 3× ([arXiv 2207.07061](https://arxiv.org/abs/2207.07061)) | Batch-size-1 only; no batched-throughput numbers published | No | No |
| PowerInfer-2 | 29.2× vs llama.cpp; 22× on Mixtral-47B; 3.84× vs LLMFlash ([arXiv 2406.06282](https://arxiv.org/abs/2406.06282)) | Same range, NAND-offload regime only | No | No |

---

## 6. Outside the box

This is the section I find the most interesting to write, even though most of what's in it won't ship in production any time soon. The questions I started with — *is softmax really the right primitive? what if model weights are more like oscillators on a mesh than gates in a feedforward stack? can we do something with FFTs, or with geometric/Cartesian structure?* — don't have clean 2026 answers, but the underlying ideas are real and worth keeping a map of. Honest field map below.

### 6.1 Softmax alternatives — bluntness of softmax

Softmax normalises every query's attention scores into a probability distribution over *all* keys, even when most of those keys carry no signal. That's the "blunt instrument" intuition: the normalisation forces every key to contribute some non-zero weight, and the exponential denominator means very large logits dominate in ways that don't always reflect actual relevance.

The 2026 state of play on alternatives:

- **Sigmoid attention** — Apple's "Theory, Analysis, and Best Practices for Sigmoid Self-Attention" ([arXiv 2409.04431](https://machinelearning.apple.com/research/sigmoid-self-attention)) proved it is a universal approximator with better regularity than softmax. **FlashSigmoid is ~17% faster than FlashAttention2** on H100, works across language/vision/speech *if* you stabilise the early-training large-norm regime. Sample-complexity follow-up at [arXiv 2502.00281](https://arxiv.org/abs/2502.00281).
- **Gated attention** — Qwen, **NeurIPS 2025 Best Paper** ([writeup](https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/)) — systematic exploration of attention gating; the most rigorous validation that softmax has alternatives that work at scale.
- **Squared-ReLU attention** — Primer (Google, 2021, [arXiv 2109.08668](https://arxiv.org/pdf/2109.08668)) showed squared-ReLU is just better for LMs at fixed compute. Wortsman et al. 2023 ([arXiv 2309.08586](https://arxiv.org/pdf/2309.08586)) showed plain ReLU divided by sequence length matches softmax in ViT.
- **Polynomial attention** ([arXiv 2410.18613](https://arxiv.org/abs/2410.18613)) — softmax's win is implicit Frobenius regularisation; well-chosen polynomials with √N scaling reproduce it.
- **Modern Hopfield** — Ramsauer & Hochreiter ([arXiv 2008.02217](https://arxiv.org/abs/2008.02217)) showed transformer attention *is* the update rule of a continuous-state Hopfield net. So "Hopfield attention" isn't really an alternative — it's a re-interpretation that explains why softmax works (energy minimisation, exponential capacity). 2025 work on continuous-time Hopfield memories ([arXiv 2502.10122](https://arxiv.org/abs/2502.10122)) revives this as a way to compress KV caches into continuous memories rather than discrete patterns. Worth tracking for KV-compression intuitions, not as a softmax replacement.

**Practical adoption: tiny.** Sigmoid + gating are the two with momentum. Everything else stays niche.

### 6.2 Position encoding — biasing weights differently

Position encoding is how the model knows where each token sits relative to its neighbours. The field has been searching for the right inductive bias for a decade — absolute embeddings, relative embeddings, ALiBi linear bias, RoPE rotary encoding, and now NoPE (no explicit position encoding at all, letting the causal mask do the work).

RoPE dominates production but **NoPE** has emerged as the surprise length-generalisation winner. The 2025 ICLR/NeurIPS papers — "Round and Round We Go: What makes RoPE useful" ([arXiv 2410.06205](https://arxiv.org/pdf/2410.06205)), "RoPE to NoPE and Back Again" ([arXiv 2501.18795](https://arxiv.org/html/2501.18795v1)), "Long-Context Generalization with NoPE-hybrids" ([arXiv 2506.16640](https://arxiv.org/pdf/2506.16640)) — converge on **interleaved NoPE+RoPE layers** beating RoPE-scaling tricks. Llama-4 and Qwen3 both ship interleaved variants. NAPE (NoPE+ALiBi) is the strongest pure extrapolator.

### 6.3 Spectral / FFT / structured matrices — WHT, butterfly, and Monarch

Spectral and structured-matrix architectures replace the dense O(N²) attention matrix (or the dense O(N·d²) weight matrices) with something cheaper to multiply: long convolutions executed via FFT (Hyena, StripedHyena), butterfly networks of ±1 entries (Walsh-Hadamard), block-diagonal butterflies (Monarch), or fixed spectral filters with provable approximation properties. The asymptotic story is compelling — O(N log N) instead of O(N²), and for weight matrices O(N√N) parameters and FLOPs instead of O(N²) — and the structure is mathematically clean.

This family is worth a closer look than §4.4's "abandoned in the open" verdict suggests, because **we already ship a butterfly primitive in production**.

**The GigaQuant connection.** Our TurboQuant-esque KV-cache codec (we call the composite GigaQuant; see [`gigaquant-a-frankenstein-compression-algorithm.md`](gigaquant-a-frankenstein-compression-algorithm.md) for the full lineage) applies a **Subsampled Randomized Hadamard Transform (SRHT)** to vectors before quantization — the standard QuaRot ([arXiv 2404.00456](https://arxiv.org/abs/2404.00456)) and SpinQuant ([arXiv 2405.16406](https://arxiv.org/abs/2405.16406)) construction. The mathematical form is `y = (1/√d) · H · D · x` where `H` is the Hadamard matrix and `D` is a fixed random ±1 diagonal — the Rademacher randomization is what gives the construction its concentration guarantee (without `D`, plain WHT can *amplify* outliers that happen to align with one of `H`'s columns; with `D`, Johnson–Lindenstrauss-style bounds guarantee the max output coordinate is `O(√(log d / d))` of the input L2 norm with high probability, flattening the distribution for int4/int8 quantization). The codec itself is target-agnostic: we ship it for KV cache today, have benched it on weights with promising early results, and activations are a credible next step on the same primitive.

What we actually ship is **two implementations**, picked by call site:

- **FWHT Metal kernel for the encode path** (K/V compression in both Path A and Path B), at `TurboQuantKernels.swift:218–338`. Radix-2 Sylvester butterfly, O(d log d). Two phases: stages 0–4 use `simd_shuffle_xor` for register-to-register intra-SIMD shuffles (no shared-memory traffic); stages 5+ use threadgroup shared memory for the cross-SIMD butterfly. Power-of-2 head dims up to 1024 — covers every head dim that modern LLMs ship (64 / 128 / 256 / 512).
- **Dense `[d, d]` Hadamard matmul** for the decode-time query rotation, at `TurboQuantKVCache.swift:546–548`. The rotation matrix `H · diag(signs) / √d` is precomputed as a bf16 tensor at codec init. Dense matmul wins over per-op MLX butterfly here because the MLX graph overhead of calling the butterfly N times per decode step beats its asymptotic advantage at d ≤ 1024.
- For **non-power-of-2 head dims** (Mistral's 80, Phi's 96), the encode path also falls back to dense matmul — Hadamard matrices of those orders either don't exist (orders not divisible by 4) or require Paley / Williamson constructions that don't map to a clean radix-2 butterfly kernel; padding to the next power of 2 would break the SRHT concentration property.

So **we don't just have the butterfly primitive abstractly — we have a Metal-tuned FWHT kernel that exploits `simd_shuffle_xor` for small radix stages and a dense-matmul fallback for the regime where graph overhead matters.** That's the cost-model nuance a real "butterfly attention on M-series" experiment would care about, and it makes the engineering cost of any architecture built on FWHT, SRHT, or general butterfly primitives much lower than the starting-cold baseline.

The family, from simplest to most general:

- **Walsh-Hadamard Transform (WHT)** and its fast variant **FWHT** (radix-2 Sylvester butterfly, O(N log N)). Multiplication by an N×N matrix of ±1 entries, no complex math, just adds and subtracts. The randomized variant **SRHT** is what's actually used in QuaRot / SpinQuant / TurboQuant for outlier smoothing — the fixed random ±1 sign diagonal is mathematically essential, not a performance choice (cost is ~zero, fuses into adjacent ops). Plain WHT can be used as a free mixing layer (attention-like cross-token shuffle with zero learned parameters); SRHT can do double duty as both mixer and distribution-flattener.
- **Butterfly matrices.** Generalisation: products of sparse permutation-and-mix factors that compose to an O(N log N) linear map. Pixelfly ([arXiv 2204.02485](https://arxiv.org/abs/2204.02485)) and Monarch ([arXiv 2204.00595](https://arxiv.org/abs/2204.00595)) are both butterfly parameterisations of *trainable* linear layers. Cheaper than dense; expressive enough to recover FFT, Hadamard, DCT, and circulant matrices as special cases.
- **Monarch matrices.** Block-diagonal butterflies. Used as drop-in replacements for the dense linear layers in attention and MLPs in **Monarch Mixer / M2** ([arXiv 2310.12109](https://arxiv.org/abs/2310.12109)), which trained competitive sub-quadratic LMs at small scale. **MonarchAttention** ([arXiv 2505.18698](https://arxiv.org/html/2505.18698v1)) is the most pragmatic 2025 application — *zero-shot conversion* of pretrained softmax attention to Monarch-structured attention, no retraining required. The fact that you can convert a softmax checkpoint to Monarch at inference time is the cleanest "we already have the primitive, we already have the weights, what's stopping us?" argument in the section.
- **Spectral SSMs** (Hazan/Agarwal, [arXiv 2312.06837](https://arxiv.org/abs/2312.06837)). Fixed convolutional filters derived from spectral filtering theory, with provable robustness guarantees independent of the spectrum or dimensionality of the underlying dynamics. DeepMind released an implementation; no scaled LM has shipped on it yet.
- **Hyena / StripedHyena / Evo 2.** Already covered in §4.4. Long convolutions via FFT, alive in genomics, dormant in open-source LM.

**What I think is actually undervalued.** The combination of (a) the butterfly primitive being trivial on Metal, (b) MonarchAttention being a zero-shot conversion from existing softmax checkpoints, and (c) our team already having the WHT kernel infrastructure means a serious "butterfly attention on Apple Silicon" experiment is much lower-cost than the asymptotic "no-one-is-trying" verdict implies. The pieces are in place; what's missing is somebody willing to run the bench. That's a Q3 or Q4 experiment after the post-hoc work in §8 ships.

The other live use case — **butterfly/Monarch parameterisations of the dense weight matrices themselves** (QKV projections, MLP up/down projections, LM head). At 70B+ scale these matrices dominate weight memory; reparameterising them as Monarch matrices is a path to 3–5× weight compression with bounded quality loss. Closer to a weight-quantisation alternative than an attention replacement, and it composes cleanly with GigaQuant (the Hadamard rotation we already do is the *first factor* of the Monarch parameterisation).

The 2025 "Convolutional Multi-Hybrid LMs at Scale" paper ([arXiv 2503.01868](https://arxiv.org/pdf/2503.01868)) extended the Hyena line; no LM at scale has shipped on it.

### 6.4 Kuramoto / oscillator-based networks

This is the section I'm most excited about, and it's also the one with the least concrete path to a 2026 inference win. The Kuramoto model describes how a network of coupled oscillators with different natural frequencies spontaneously synchronises — fireflies, neural firing patterns, power grids, pendulum clocks on the same wall. The hypothesis is that if you model neurons (or weights, or tokens) as oscillators on a coupling graph, synchronisation patterns might be a more efficient and more biologically plausible substrate for computation than the gate-and-activation stack we use today.

**AKOrN — Artificial Kuramoto Oscillatory Neurons** (Miyato et al., **ICLR 2025 Oral**, [arXiv 2410.13821](https://arxiv.org/abs/2410.13821), [project page](https://takerum.github.io/akorn_project_page/)). Replaces threshold/activation neurons with oscillators governed by the Kuramoto synchronisation model. Strong results on object discovery (binding via phase synchronisation), adversarial robustness, calibration, and Sudoku reasoning (18% → ~90% with more test-time compute). [Code at github.com/autonomousvision/akorn](https://github.com/autonomousvision/akorn).

**Continuous Thought Machines** (Sakana AI, **NeurIPS 2025 Spotlight**, [arXiv 2505.05522](https://arxiv.org/abs/2505.05522), [Sakana page](https://pub.sakana.ai/ctm/)). Each neuron has its own private weights processing a short history; representation is *neural synchronisation over time*. ImageNet 72.47% top-1 — not SOTA, but the architecture is fundamentally non-feedforward. Sakana is pushing it as a reasoning substrate.

**Why I think this deserves more thinking, not less.** Neither has been scaled to LM, and the naive cost story is bad (time-stepping per neuron is expensive and hard to parallelise). But two observations from oscillator physics make me think there are unexplored efficient algorithms here:

1. **Anchor weights and synchronisation hubs.** In a Kuramoto network, not all oscillators are equally influential — a small subset (high coupling, high in-degree) act as *anchors* that pull the rest of the network into phase. This maps onto something I think is already true of trained transformers: some weights and some attention heads do most of the structural work, and the rest are essentially decorating around them. If you can identify the anchor weights of a model and treat the rest as conditionally evaluable around them, you get something that looks structurally similar to MoE but driven by network topology rather than a learned router. Identifying these anchors offline (via spectral analysis of the weight graph or by sensitivity probes) and then doing **anchor-centred path evaluation** at inference time — rather than sweeping every weight every forward — is the path I'd want to explore.
2. **NP-complete-style path evaluation, not exhaustive evaluation.** Kuramoto synchronisation is itself related to a class of NP-complete graph-cut / clustering problems that have decent approximation schemes (semidefinite relaxations, message-passing, belief propagation). If you frame "which weights need to fire for this token" as a graph-cut problem centred on anchor weights rather than a brute-force "evaluate every weight" problem, you might get sub-linear per-token compute for the same expressive power. This is hand-wavy but I think it's the conceptually right direction. It's the version of "stop sampling every weight every time" (§5) that doesn't require pretraining-time MoE routing.
3. **Phase as a parallelisation primitive.** The cost objection to Kuramoto networks is "time-stepping per neuron is sequential" — but if you re-cast the synchronisation dynamics in a fixed-point / spectral form (analogous to how Mamba2's SSD turns scan into GEMM), the parallelisation story may not be as bad as the naive ODE-integration framing suggests. This is speculative — I haven't seen it worked out anywhere — but it's the kind of move that turned Mamba from a curiosity into a competitive substrate.

**Bottom line:** Kuramoto / oscillator-based networks are not a production target for 2026 and I'm not going to invest there before the lower-hanging fruit (Quest, DuoAttention, TEAL, hybrid model porting) is done. But this is the bucket I want to come back to once those are shipped, because the underlying physics suggests there are smarter parallelisation and filtering algorithms hiding here than the field has worked out yet. **Track closely, think about it explicitly, revisit in 6–12 months.**

### 6.5 Geometric / hyperbolic / Lorentzian

Standard transformers do all their arithmetic in Euclidean (flat) space. Hyperbolic and Lorentzian variants do it on curved manifolds — hyperbolic space has the useful property that volume grows exponentially with radius, which matches how hierarchical structures (taxonomies, syntax trees, knowledge graphs) embed naturally. The trade-off is that the manifold maps require expensive transcendental functions (sinh, cosh, arcosh).

- **HELM** (May 2025, [arXiv 2505.24722](https://arxiv.org/pdf/2505.24722)) — first hyperbolic LLM with mixture-of-curvature experts.
- **Hierarchical Mamba on Lorentz manifold** ([arXiv 2505.18973](https://arxiv.org/html/2505.18973)).

Theory appeals for hierarchical data; sinh/cosh/arcosh are expensive. Current verdict: niche, won't beat Euclidean transformers on general text.

### 6.6 JEPA / VL-JEPA / LLM-JEPA

JEPA (Joint Embedding Predictive Architecture) is Yann LeCun's bet against next-token autoregression. Instead of predicting the next token in pixel/text space, JEPA predicts the *embedding* of a future chunk from a past chunk's embedding — a non-generative objective that's supposed to be more sample-efficient and less prone to hallucination because the model never has to commit to a specific surface form.

**LLM-JEPA** ([arXiv 2509.14252](https://arxiv.org/abs/2509.14252)) applies JEPA pretraining to LLMs and outperforms standard objectives, robust to overfitting. **VL-JEPA** (Dec 2025, [Meta AI](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)) hits stronger VL benchmarks than autoregressive VLMs at half the parameters. **Genuinely worth tracking** — most credible "predict embeddings, not tokens" line.

### 6.7 Test-time training / fast weights

Test-time training (TTT) and "fast weight" architectures treat the model's hidden state as itself a small model whose weights update at *inference* time on a self-supervised loss. The forward pass becomes a meta-learning step: each token's processing produces a gradient that nudges the recurrent state, so the model effectively learns from its own context as it reads it. Titans (Google) is the most prominent recent instance, combining a fast-weight memory module with sliding-window short-term attention.

Already covered in §4.3. **Mainstream-adjacent**. Lucidrains has working PyTorch ports. Not yet in production frontier models, but Google is investing. Watch closely — if Google ships Titans weights, this becomes a real candidate.

### 6.8 Vector Symbolic Architectures / hyperdimensional computing

Vector Symbolic Architectures (VSA) and Hyperdimensional (HD) computing represent concepts as very high-dimensional random vectors and use algebraic operations (binding, bundling, permutation) to compose them. The pitch is robustness — small perturbations don't change meaning much in very high dimensions — and natural support for variable binding, which transformers handle awkwardly.

**Hyperdimensional Probe** ([arXiv 2509.25045](https://arxiv.org/abs/2509.25045)) uses VSA to *interpret* LLM residual streams (83% probing@1). qFHRR (Quantised Fourier Holographic Reduced Representations) is a 2025 direction. Currently VSA is an interpretability/binding tool, not an LM substrate.

### 6.9 Neuroscience-inspired fringe

This is the grab-bag of architectures inspired by biological neurons — predictive coding (the brain as a hierarchy of prediction-error minimisers), sparse coding (only a few neurons fire for any input), dendritic computation (treating dendrites as small nonlinear units rather than passive integrators), and Hierarchical Temporal Memory (Numenta's model of cortical columns). Most of these have theoretical appeal but have not yet produced competitive language models at scale.

Predictive coding networks ([arXiv 2506.06332](https://arxiv.org/pdf/2506.06332)) keep reappearing as theoretical frameworks unifying sparse + predictive + divisive normalisation. **OpenAI's "sparse circuits"** (Nov 2025, [openai.com/index/understanding-neural-networks-through-sparse-circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)) on training LLMs with native sparse connectivity for interpretability is the closest to mainstream. HTM/Numenta is dead for LM. Dendrify and dendritic-computation models remain academic.

### 6.10 What to track vs what is a less likely

**Track closely:**
- **Diffusion LMs.** Mercury 2's ~1000 tok/s at parity is the most disruptive 2025–2026 non-AR inference result. LLaDA / LLaDA-MoE are the open experiment platforms.
- **BitNet-style ternary models.** Composable with everything else; particularly well-suited to memory-bandwidth-bound Apple Silicon.
- **Titans / TTT / fast weights.** Google is investing; if Titans weights drop, this becomes a real candidate.
- **JEPA-for-LLM.** LLM-JEPA and VL-JEPA are starting to put up real numbers. The most credible "predict embeddings, not tokens" line.
- **Sigmoid / gated attention.** NeurIPS 2025 Best Paper momentum; FlashSigmoid already faster than FlashAttention2.
- **NoPE-hybrid position encodings.** Already in Llama-4 and Qwen3.

**Worth thinking about and experimenting with:**
- **Kuramoto / oscillator nets (AKOrN, CTM).** The anchor-weight and synchronisation-hub framing in §6.4 is the lens I want to come back to once the post-hoc work is shipped. Sakana and Miyato are credible groups; the underlying physics suggests parallelisation algorithms the field hasn't worked out yet.
- **Mixture of Recursions.** Genuinely underexplored — token-level recursive routing is one of the most natural "spend more compute on harder tokens" framings and nobody has scaled it.
- **Butterfly / FWHT / SRHT / Monarch attention on Apple Silicon.** We already ship a tuned FWHT Metal kernel in TurboQuant (SRHT for outlier smoothing) plus a dense Hadamard-matmul fallback for the decode rotation. MonarchAttention is a zero-shot conversion from existing softmax checkpoints. The pieces for a real "butterfly attention" experiment on M-series are already in the pipeline; the missing thing is the bench. See §6.3.
- **Spectral / FFT architectures with Apple-Silicon-tuned kernels.** Same "no-one-is-trying rather than doesn't work" story for the FFT side; Hyena / StripedHyena could become interesting again with a fused Stockham FFT kernel.
- **RWKV-7.** Underrated. Apache 2.0, O(1) decode, parallel training. The only reason it doesn't have a 30B+ checkpoint is funding, not architecture.

**Quarterly check (interesting but stalled):**
- Hyena / Evo 2 in genomics (alive in domain, dormant in language).
- VSA / hyperdimensional computing as an *interpretability* tool, not yet a substrate.
- Spectral SSMs (Hazan's group is patient).

**Unlikely solutions:**
- Hyperbolic LLMs at general-purpose scale (the transcendental cost is real).
- HTM / Numenta-style dendritic computation (no path to scale).
- FNet (no adaptivity).

> **The unifying observation.** Every successful "alternative" to softmax attention in 2026 is a **hybrid** — Liquid LFM2 (conv+GQA), StripedHyena (Hyena+attention), Titans (attention+neural memory), Llama-4 (RoPE+NoPE interleave), Qwen3-Next / Kimi Linear (Gated DeltaNet + softmax 3:1). Nobody wins by replacing attention outright; they win by interleaving it with something cheap and learning the right ratio (typically 1 attention layer per 6–8 cheap blocks). I expect this pattern to hold for the next 18–24 months, and the question becomes: *which cheap layer do you interleave?* That's still very open.

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
| **1. Native MTP / EAGLE-3 draft heads** | DeepSeek-V3 / Qwen3 / GLM-4.5 ship them; mostly a model-loader change once spec-decode infra exists. MTPLX is a good Apple-Silicon reference. Largest pure-decode win, lossless, composes with everything below. | Per-model loader work; depends on existing spec-decode path. Variant A on hybrid Qwen depends on spec 020 tape-replay (Tier 1). |
| **2. DuoAttention** | Calibration-pass head split, slots into windowed KV from PR #186; ~2× decode + memory on long context, lossless with calibration. Streaming-head path reuses windowed turbo cache directly. | Metal kernel for ragged per-head cache shapes. |
| **3. Quest** | Per-page top-k over full KV, no retraining, pure bandwidth win on M-series. Composes orthogonally with DuoAttention — Quest applies to the retrieval-head cache that DuoAttention identifies. | Small page-bound metadata addition; depends on paged-cache backlog (#127–#129) for the V2 fast path. |
| **4. TEAL activation thresholding** | Training-free, targets memory-bandwidth-bound MLP decode (M-series regime). Composes with TurboQuant; the FusedGateUpMLP integration site is already there. | Block-sparse Metal kernel work; honest M-series gain (1.2–1.4×) is smaller than the paper's H100 number because MLPs are a smaller share of total decode time on Apple Silicon at long context. |
| **5. Granite-4-H-32B-A9B / Qwen3-Next-80B-A3B / Kimi Linear-48B-A3B** | Hybrids, MoE-active 3–9B (perfect for unified memory), chunkwise-parallel kernels you can port from FLA. | Gated DeltaNet kernel work in Metal is non-trivial; Granite is the conservative pick (Mamba2-SSD is the most studied SSM kernel). |
| **6. NoPE-hybrid** for long-context experiments | Already in production (Llama-4, Qwen3); pairs with windowed eviction. | Research project, not an inference win. |

**Suggested order:** Native MTP / EAGLE-3 draft heads first — largest single decode win, mostly model-loader work, lossless, and the gains compound multiplicatively with everything that comes after. Then **DuoAttention** (cleanest composition with the windowed-KV plumbing from PR #186) and **Quest** (composes orthogonally — Quest applies to the retrieval-head cache DuoAttention identifies); land them in that order if effort permits, or in parallel if separate engineers are on each. Then **TEAL** for the MLP-side bandwidth saving. Finally, port one chunkwise-parallel **hybrid model** kernel (Granite or Qwen3-Next) since the Gated DeltaNet path is already understood — that's where the stack catches up with the architectures that won 2025–2026 rather than just optimising dense softmax attention.

**What I'd skip for this round** (not "dead", just not where the highest-leverage hours go right now): full confidence-based early exit/CALM (stuck on the KV-cache-coherence problem), Deja Vu-style MLP activation predictors on SwiGLU (eaten by MoE), pure-Mamba and pure-RetNet (no scaled checkpoints), hyperbolic LLMs (transcendental cost), FNet (no adaptivity). **What I'm parking for the *next* round** (worth a real investment after the post-hoc work ships): BitNet-style ternary quantisation, diffusion LMs (LLaDA-MoE experiments), Mixture of Recursions at meaningful scale, and the Kuramoto / anchor-weight framing in §6.4.

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
- [TriAttention (arXiv 2604.04921)](https://arxiv.org/abs/2604.04921)
- [TriAttention GitHub](https://github.com/WeianMao/triattention)
- [TriAttention project page](https://weianmao.github.io/tri-attention-project-page/)
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
- [Qwen3.5 architecture overview](https://medium.com/data-science-in-your-pocket/qwen-3-5-explained-architecture-upgrades-over-qwen-3-benchmarks-and-real-world-use-cases-af38b01e9888)
- [Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-27B blog (Qwen)](https://qwen.ai/blog?id=qwen3.6-27b)
- [Gemma 4 model card (Google)](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 HF blog](https://huggingface.co/blog/gemma4)
- [Qwen3-Next RULER coverage (DigitalOcean)](https://www.digitalocean.com/community/tutorials/qwen3-next-80b-a3b-instruct-long-context-ai)
- [Kimi Linear paper — RULER@128K = 84.3 (arXiv 2510.26692)](https://arxiv.org/pdf/2510.26692)
- [Kimi Linear GitHub](https://github.com/MoonshotAI/Kimi-Linear)
- [Granite 4.1 RULER@128K (8B / 30B)](https://www.creativeainews.com/articles/ibm-granite-4-1-open-llm-512k-context-coding/)
- [Nemotron-H RULER (arXiv 2504.03624)](https://arxiv.org/pdf/2504.03624)
- [MiniMax-Text-01 RULER (arXiv 2501.08313)](https://arxiv.org/abs/2501.08313)
- [Qwen 3.5 architecture commentary (Maxime Labonne)](https://medium.com/@mlabonne/qwen3-5-nobody-agrees-on-attention-anymore-4709e1bd014b)
- [Hybrid Linear Attention Done Right (arXiv 2601.22156)](https://arxiv.org/pdf/2601.22156)
- [r/LocalLLaMA post-Gemma-4 state-of-community summary](https://www.dailyneuraldigest.com/newsroom/2026-04-05-state-of-r-locallama-after-gemma4-release-/)
- [Long-context benchmarks leaderboard](https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/)

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
- [Monarch matrices (arXiv 2204.00595)](https://arxiv.org/abs/2204.00595) — block-diagonal butterfly parameterisation
- [Monarch Mixer / M2 (arXiv 2310.12109)](https://arxiv.org/abs/2310.12109) — sub-quadratic LM via Monarch matrices in attention + MLPs
- [Pixelfly butterfly sparsity (arXiv 2204.02485)](https://arxiv.org/abs/2204.02485)
- [QuaRot — Hadamard-rotated quantisation (arXiv 2404.00456)](https://arxiv.org/abs/2404.00456)
- [SpinQuant — learned rotations for quantisation (arXiv 2405.16406)](https://arxiv.org/abs/2405.16406)
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
- [PowerInfer-2 (arXiv 2406.06282)](https://arxiv.org/abs/2406.06282)
- [PowerInfer-2 project page](https://powerinfer.ai/v2/)
- [SSD/MoE offload survey (arXiv 2508.06978)](https://arxiv.org/pdf/2508.06978)
- [Mixture-of-Depths (arXiv 2404.02258)](https://arxiv.org/abs/2404.02258)
- [Mixture-of-Recursions (arXiv 2507.10524)](https://arxiv.org/abs/2507.10524)
- [MoDification (arXiv 2410.14268)](https://arxiv.org/abs/2410.14268) — MoD applied to Llama-2-7B
- [CALM original (arXiv 2207.07061)](https://arxiv.org/abs/2207.07061)
- [MoE vs dense inference (Epoch AI)](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference)

### Speculative decoding (recent)

- [EAGLE-3 paper (arXiv 2503.01840)](https://arxiv.org/abs/2503.01840)
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

### BitNet / ternary models

- [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) — the 1.58-bit ternary baseline
- [BitNet b1.58 2B4T technical report (arXiv 2504.12285)](https://arxiv.org/abs/2504.12285) — Microsoft's 2B native 1.58-bit checkpoint
- [bitnet.cpp inference framework](https://github.com/microsoft/BitNet)

### Companion papers in this directory

- [speculative-decoding-on-apple-silicon.md](speculative-decoding-on-apple-silicon.md) — decode-throughput-focused tour, covers spec-decode, prefix caching, tape-replay
