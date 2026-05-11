# GigaQuant: a frankenstein compression algorithm

**Authors:** Eric Kryski, Tom Turney
**Hardware target:** Apple M1 / M5 Max, 16–512 GB unified memory
**Software target:** [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm)
**Date:** 2026-05-09
**Status:** Working draft — implementation notes

> *Our codec is named `TurboQuant*` and it is not, in fact, an implementation of the TurboQuant paper. It is a hybrid of TurboQuant_mse, QuaRot, PolarQuant, and llama.cpp's k_quants. This paper is what I should have written when I first noticed the divergence.*

This is the honest accounting of what mlx-swift-lm's compression codec actually is, where each piece came from, what we kept, what we dropped, and what we added that none of the prior art does. The name **GigaQuant** is what I'm calling the composite from here on. The Swift files keep the `TurboQuant*` filenames for now because renaming them is mechanical churn that I'd rather defer; the headers at the top of `TurboQuantKVCache.swift` and `TurboQuantKernels.swift` already point readers here.

**Scope.** GigaQuant is a generic vector quantization codec — the same SRHT-rotation + polar-decomposition + Lloyd-Max-scalar-codebook pipeline can be applied to any high-dimensional tensor. We deployed it on **KV cache** first because that's where the bandwidth pressure hurts most on Apple Silicon, and the KV-cache path is what ships in production today. We've also **tested it on model weights** with promising early results (smaller per-matrix encode cost than QuaRot+GPTQ at comparable PPL because we skip the GPTQ Hessian step). **Activations** — the per-token tensors flowing through the network during forward — are a credible next target on the same primitive but we haven't benched them yet; the engineering question is whether the dynamic per-forward encode cost pays back via int8 or int4 GEMM kernels downstream. This paper is written KV-cache-first because that's the shipping deployment, but the math and the prior-art comparisons apply equally to the weight and activation use cases.

This is a write-up for me and anyone else who needs to reason about the codec — not a research paper claim of novelty. Most of the engineering moves are independently published elsewhere; the contribution is the specific composition and the Apple-Silicon-tuned kernels.

---

## 1. Why this paper exists

I noticed the divergence when revising [`papers/beyond-quadratic-attention-on-apple-silicon.md`](beyond-quadratic-attention-on-apple-silicon.md) §6.3 and grounding the "we already ship a butterfly primitive" claim in actual code. The §6.3 fact-check turned up the following:

- The TurboQuant paper (Zandieh et al., [arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) specifies Gaussian-QR rotation, an *analytic per-coordinate* Lloyd-Max quantizer derived from the Beta distribution of rotated coordinates, and a two-stage scheme where Algorithm 2 (QJL — Quantized Johnson-Lindenstrauss on the residual) gives the inner-product unbiasedness guarantee.
- Our codec uses a *dual rotation path* (Gaussian-QR or SRHT-via-FWHT, the latter QuaRot-style), an *empirically-derived global 1D codebook* (llama.cpp's k_quants table scaled by √(128/d)), and *no QJL second stage*.
- Our codec also makes the polar (norm + direction) decomposition a first-class operation, with explicit norm correction in the dense path, which the TurboQuant paper treats as a trivial pre-step but PolarQuant ([arXiv 2502.02617](https://arxiv.org/abs/2502.02617)) puts at the centre of its framing.
- We add several engineering moves that none of the source papers specify: pre-rotated queries reused across all keys in a layer, two-phase prefill→compress→decode architecture, asymmetric K/V bit-widths, and boundary-layer fp16 protection.

The honest summary is "TurboQuant_mse + QuaRot + PolarQuant + llama.cpp k_quants + several engineering additions." That's not what people will assume from the filename. Hence GigaQuant.

---

## 2. The actual GigaQuant pipeline

The full encode + decode story for one K or V tensor of shape `[B, H, T, D]`, at codec configuration `(bits, rotation_mode)`:

### 2.1 Encode (per cached vector x ∈ R^D)

```
1. r        = ||x||_2                                       # scalar, kept in fp16
2. u        = x / r                                          # unit-sphere direction
3. y        = Π · u                                          # random orthogonal rotation
4. (i_1..i_D) = boundaryQuantize(y, codebook)                # per-coord Lloyd-Max
5. packed   = bitpack(i_1..i_D)                              # b · D bits → uint32 lanes
6. r̂        = ||codebook[i_1..i_D]||_2                       # reconstructed norm
7. r_corr   = r / r̂                                          # norm correction (dense path only)
8. STORE (packed, r_corr)
```

Step 7's norm correction compensates for the per-coordinate quantization error in the dense-rotation path. The SRHT/FWHT path skips it because orthogonal SRHT preserves norms exactly (`r̂ = r`).

### 2.2 Rotation Π — two paths

- **Dense path:** Π is a `[D, D]` orthogonal matrix constructed by QR-decomposing a random Gaussian. Matches the TurboQuant paper. Used for non-power-of-2 head dims (e.g. Mistral's 80, Phi's 96) and for the decode-time query rotation (see §2.4).
- **SRHT / FWHT path:** Π = H · diag(s) / √D where H is the Sylvester Hadamard matrix and s is a fixed random ±1 sign vector. Implemented as a Metal kernel at [`TurboQuantKernels.swift:218–338`](../Libraries/MLXLMCommon/TurboQuantKernels.swift): radix-2 butterfly, two phases — stages 0–4 use `simd_shuffle_xor` for intra-SIMD shuffles (register-to-register, no shared memory), stages 5+ use threadgroup shared memory for the cross-SIMD butterfly. Power-of-2 head dims up to 1024.

The Rademacher randomization `diag(s)` is mathematically required for the JL concentration property that flattens activation distributions for quantization (QuaRot Sec. 4 makes the argument formally). Without it, plain WHT can *amplify* outliers aligned with H's columns. Runtime cost of SRHT vs plain WHT is essentially zero — one elementwise sign multiply that fuses into adjacent ops — so we pay nothing for the guarantee.

The rotation is fixed at codec init. Never learned. QuaRot-style, not SpinQuant-style.

### 2.3 Codebook

The codebook is a 1D table of `K = 2^bits` Lloyd-Max-optimal scalar levels. The actual values are mined from llama.cpp's `k_quants` work — they were derived empirically for unit-variance Gaussian data at d=128 and then validated on real LLM weight/activation distributions. We adapt them to arbitrary head dims by scaling: `codebook_d = codebook_128 · √(128/d)`. The scaling is a heuristic that approximately recovers the paper's analytic `1/√d` Beta-variance scaling.

The codebook is **global** — shared across all coordinates, all heads, all layers. The paper specifies *per-coordinate* quantizers; we use one shared table. For high-dimensional data where the rotated coordinates are approximately **i.i.d.** (independent and identically distributed — they don't correlate with each other and they all follow the same Beta distribution after the random rotation has whitened them) this is a reasonable approximation, but it is not what the paper specifies. The motivation is engineering simplicity and kernel register pressure — one codebook fits in registers, D per-coord codebooks don't.

### 2.4 Decode and compressed-domain attention

At inference time, the cache stores `(packed_indices, r_corr)` per token. The naive decode path is:

```
1. unpacked      = codebook[indices]                         # gather from indices
2. u_recon       = unpacked · Π^T                            # inverse rotation
3. x_recon       = r_corr · u_recon                          # rescale by norm
```

We never run that naive path during attention. Instead, the **compressed-domain attention** kernel ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift)) computes:

```
q'       = Π · q                                             # pre-rotated query, once per layer
score[t] = r_corr[t] · Σ_j q'[j] · codebook[indices[t, j]]   # per-key dot product
```

The trick is that `q · x_recon = (Π · q) · (Π · u_recon) · r_corr = q' · u_recon · r_corr`. The two rotations cancel because Π is orthogonal — `Π · Π^T = I`. So we can compute the attention score directly against the packed indices without ever materialising a dequantized `x_recon` tensor. Pre-rotating `q` once per layer and reusing it across all `T` cached keys is the bandwidth win.

Each SIMD group (32 threads) handles one `(query, key_token)` pair. The codebook (≤16 entries at 4-bit) lives in thread-local registers. Bit unpacking + codebook lookup + dot product all happen in-register. Zero global-memory dequant materialization.

This is **PolarQuant's "decoding acceleration"** idea applied with our codebook structure. The paper explicitly turns the query-key inner product into a table lookup; we do the same but lookup a scalar codebook entry rather than reconstructing a polar angle.

**Would reconstructing polar angles be faster?** Worth answering carefully, because on paper *yes* — and that's an open future direction we haven't pursued yet. The arithmetic:

PolarQuant decomposes a D-dim vector recursively into log₂(D) levels of nested polar angles plus one final radius. At D=128 that's 7 levels of angles. *Reconstructing the d-dim vector on the fly* requires recursive cos/sin terms applied to a running product — that part is genuinely expensive (transcendentals on every coordinate). So naive on-the-fly polar reconstruction is slower than our codebook lookup.

However, this isn't the regime PolarQuant actually operates in. Like us, PolarQuant **precomputes a centroid table** of quantized angle values at each polar level (e.g. 16 angles at level 1, 4 angles at levels 2–4). Decoding then becomes a sequence of table lookups + products, not transcendental evaluations. And here's the interesting bit: at the *compressed-domain scoring* level, PolarQuant only needs `log₂(D)` lookups per cached vector — not D. The score formula for a polar-encoded key against a pre-rotated query becomes a product over the 7 angle levels rather than a sum over 128 coordinates. **On paper that's ~18× cheaper at score time.**

So why don't we ship it? Three engineering reasons:

1. **The recursive decode kernel is harder to write in Metal than our single-shot codebook lookup.** Each angle level depends on the previous one's reconstructed direction, so the kernel can't flatten into one SIMD-group-per-(query, key) pair the way ours does. You either accept warp divergence at each level or restructure the kernel pipeline.
2. **The encode pass has to compute polar angles** (`atan2` per pair, recursive). Our encode is `boundaryQuantize(rotated_coord)` per element — a single binary search against the codebook. Encode is one-shot at prefill but the kernel complexity adds up.
3. **The compressed-domain scoring trick fuses with our scalar codebook for free** (one SIMD group computes the dot product against the packed indices in-register, no shared memory needed). The polar variant needs accumulation state across the angle levels, which costs either shared memory or threadgroup synchronisation.

The honest assessment is that **polar encoding is probably a small per-token decode win and a meaningful bit-rate win at the cost of a substantially more complex kernel**. Tracked as an open question in §5.7.

### 2.5 Engineering additions

Four moves not in any of the source papers:

1. **Two-phase architecture.** Prefill writes raw fp16 K and V into a buffer. At the start of decode (or at decode milestone N), the buffer is batch-encoded into compressed storage in one large dispatch. Decode then reads from compressed storage exclusively. This hides encode cost inside TTFT rather than paying it per-token.

2. **Pre-rotated queries.** Compute `q' = Π · q` once per layer at decode time; reuse across all T cached keys. The cost is one `[D, D]` matmul (or one FWHT) per layer per token; the win is that we avoid `T` inverse rotations of the cache values. At T=128K and D=128 this is a several-orders-of-magnitude reduction in rotation work per decode token.

3. **Asymmetric K/V bit-widths.** Recipes ship K and V at different precisions. The `turbo4v2` config uses 4-bit K + 2-bit V; the production `q8_0-K + turbo4-V` recipe runs 8.5-bit K + 4.25-bit V for 2.5× compression. The asymmetry is justified by the math: K precision matters because softmax exponentiates the dot products (a perturbation of ε in logit space produces an O(e^ε) change in the attention distribution); V precision matters less because the V-aggregation is a weighted *average*, where errors scale linearly through the weighted sum (`O(w_i · ε)`) and non-dominant positions have near-zero weights that absorb the noise. Catastrophic example from Tom Turney's [`asymmetric-kv-compression.md`](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md): on Qwen2.5-7B, symmetric turbo3/turbo3 produces PPL 3,556 — catastrophic — while asymmetric q8_0-K + turbo3-V achieves PPL 6.71 (+2.0% vs baseline). Independently documented in KIVI ([arXiv 2402.02750](https://arxiv.org/abs/2402.02750), "Plug-and-Play 2bit KV Cache Quantization") which uses 2-bit K + 4-bit V on the same softmax-amplification reasoning, and in KVQuant ([arXiv 2401.18079](https://arxiv.org/abs/2401.18079)) which characterises the per-channel K outlier pattern that motivates the precision asymmetry. We adopted the K-heavy recipe from this prior art rather than rediscovering it.

4. **Boundary-layer fp16 protection.** First and last N attention layers stay in fp16 because residual-stream depth is shallow at the edges — V errors in boundary layers either affect every subsequent layer's attention output (first layers) or directly distort the output distribution (last layers). Middle layers operate on abstracted representations where V precision has less marginal impact. The empirical recipe in Tom Turney's [`layer-aware-v-compression.md`](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md) ("LA-V7" / Boundary V): q8_0 V cache on layers `{0, 1, N-2, N-1}`, turbo2 V cache on all other layers, K cache at q8_0 throughout. Validated independently by `@sztlink` who confirmed boundary layers have extreme K norms (146.8 at layer 0 vs 20–40 in the middle) — those tighter Gaussian distributions are ideal for low-precision quantization elsewhere but they also mean the boundary layers are *more* sensitive to V noise because the attention is more concentrated there. Same per-layer-importance intuition as Q-Hitter ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)), SqueezeAttention ([arXiv 2404.04793](https://arxiv.org/abs/2404.04793), 2D budget allocation clusters layers by cosine similarity), and KVQuant's boundary-aware quantization. **Caveat:** the boundary-layer rule was developed on pure-attention models (phi-4, Qwen2.5); hybrid models like Qwen3.5 with Gated Delta Net (where only every fourth layer carries softmax KV state) need a different layer-counting convention or the boundary protection mis-targets the wrong layers.

---

## 3. Prior art map — what we kept, dropped, borrowed

### 3.1 TurboQuant (Zandieh, Daliri, Hadian, Mirrokni — Google Research / ICLR 2026)

**The paper:** [arXiv 2504.19874](https://arxiv.org/abs/2504.19874). Online vector quantization with near-optimal distortion rate. Two algorithms: TurboQuant_mse (rotation + analytic per-coordinate Lloyd-Max from Beta distribution) and TurboQuant_prod (TurboQuant_mse + 1-bit QJL on the residual for unbiased inner-product estimation).

**What we kept:**
- The core insight: random rotation makes coordinate distributions approximately i.i.d. Beta, which lets a fixed Lloyd-Max scalar quantizer achieve near-optimal MSE.
- The Gaussian-QR rotation path (one of our two rotation paths).
- The general bit-rate / quality story.

**What we dropped:**
- **QJL second stage entirely.** The paper's headline near-optimal inner-product distortion rate comes from QJL. We don't ship it, and dropping it was an empirical decision backed by independent benchmarking — not just an engineering convenience. Tom Turney's [turbo4 resurrection write-up (Mar 2026)](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) reports the bench that drove the decision:
  - **PPL/KLD regress with QJL on.** Three independent groups confirmed QJL eliminates the bias but "explodes variance that softmax then amplifies." On Llama-class models, QJL-on degraded from −0.28% PPL at 2K context to **+3.69% PPL at 64K** — a clear long-context degradation trend.
  - **Speed regresses with QJL on.** QJL-off decode was **79.87 tok/s vs turbo3's 76.84 tok/s** at the same effective bit-width. The extra dispatch + memory traffic for the residual projection wasn't free, and it didn't pay back in quality.
  - **NIAH retention improves with QJL off.** 31/33 vs q8_0's 30/33 at long context.
  - The fix: take the bit-budget that QJL would have used and **invest it in more centroids instead** — 16 optimal centroids vs 8-centroids + QJL bit. That's the path the turbo4 resurrection took, and it's the path we ship.

  Mathematically the failure mode is straightforward: QJL is an *unbiased* estimator of the inner product but with higher variance than the MSE-only estimator. Variance compounds across decode steps in autoregressive generation, and softmax exponentiates the error — a small variance increase on logits becomes a large probability shift on the sampled token. The bias QJL removes is the 2/π term from Sec. 3.2 of the paper, which diminishes with bit-width and is acceptable at 4-bit. Trading a small bounded bias for unbounded variance amplification was the wrong call for attention, even if it gives optimal-rate guarantees for offline vector search where the estimator's variance averages out over many queries.

- **Per-coordinate codebooks.** The paper specifies one Lloyd-Max quantizer per dimension. We use one global 1D codebook for all coordinates. The approximation is justified by the high-dim i.i.d.-Beta assumption but it is a real divergence.
- **Analytic codebook derivation.** Paper gives closed-form quantizer values per bit-width from the Beta distribution variance. We use llama.cpp's empirically-derived k_quants table.

**What we added:**
- Dual rotation paths (paper has Gaussian-QR only; we have Gaussian-QR + SRHT/FWHT).
- Explicit polar decomposition with norm correction (paper has norm rescaling as a pre-step, not first-class).
- Pre-rotated queries + compressed-domain attention kernel (paper is generic VQ; this is an attention-specific optimization).
- Two-phase prefill→compress→decode architecture.
- Asymmetric K/V, boundary-layer protection.

**Verdict:** what we ship is the **TurboQuant_mse half** of the paper, structurally faithful to the rotation + Lloyd-Max scalar quant insight, but with the per-coordinate codebook approximated as a global codebook and the QJL guarantee given up.

### 3.2 TurboQuant_plus (community fork — TheTom)

**The repo:** [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus). Experimental llama.cpp-targeted fork of TurboQuant. Not a paper. The canonical write-up of the QJL removal experiment is [`turbo4-resurrection.md`](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) (Tom Turney, March 2026) — the same bench results cited in §3.1 above.

**Notes that match our codec:**
- Replaces Gaussian-QR with SRHT via WHT.
- Drops QJL on the empirical bench reasoning detailed in §3.1 (long-context PPL/KLD regression with QJL on, plus a small decode-tok/s regression from the extra residual-projection dispatch). The resurrection-paper conclusion that "more centroids > error correction" matches the call we made.
- Adopts PolarQuant's norm-extraction framing.
- Discusses asymmetric K/V, boundary protection, sparse-V.

**Verdict:** our codec is *closer in spirit to turboquant_plus than to the ICLR paper*, by direct collaboration — we worked alongside Tom Turney to port several of his insights (the QJL-off decision, the asymmetric K/V recipes, the boundary-layer protection scheme) into mlx-swift-lm's Metal kernel pipeline. He's listed as co-author on this paper for that reason.

### 3.3 QuaRot (Ashkboos et al. — [arXiv 2404.00456](https://arxiv.org/abs/2404.00456))

**The paper:** establishes Hadamard rotations as the practical fast alternative to dense Gaussian rotations for LLM quantization. Random Hadamard pre-rotation flattens activation distributions so int4/int8 grids cover the dynamic range without clipping. Rotations are fixed at calibration time.

**What we borrowed:**
- The SRHT / FWHT rotation path. Our `whtSigns` + radix-2 Sylvester butterfly is literally the QuaRot construction.
- The justification for *why* the rotation is mathematically necessary (JL concentration, not just an outlier-smoothing heuristic).

**What we didn't borrow:**
- QuaRot quantizes weights and activations *and* KV cache. We only do KV cache. Weight quantization is handled separately (TurboQuant is KV-specific in our stack).
- QuaRot uses uniform-grid int4 after rotation. We use Lloyd-Max non-uniform centroid grid (from llama.cpp k_quants).

### 3.4 SpinQuant (Liu et al. — [arXiv 2405.16406](https://arxiv.org/abs/2405.16406))

**The paper:** *learned* rotations. Instead of random fixed rotations, fine-tune the rotation matrix on a calibration corpus to minimize quantization error. Small training cost, better quality than random.

**What we borrowed:** nothing directly.

**Why not:** our rotations are random and fixed. Adopting SpinQuant-style learned rotations would mean adding a per-model calibration pass that updates Π. The engineering cost is real (per-model training infrastructure, dependency on calibration data, sanitize() complexity) and the quality lift over random is in the 1–3% PPL range — useful but not blocking. **Tracked as a future direction** (§5 below).

### 3.5 PolarQuant (Han, Kacham, Mirrokni, Zandieh, Karbasi — [arXiv 2502.02617](https://arxiv.org/abs/2502.02617); NeurIPS 2025 variant at [arXiv 2502.00527](https://arxiv.org/abs/2502.00527))

**The paper:** quantizes the KV cache by (1) applying a random orthogonal preconditioning rotation, (2) decomposing each vector recursively into nested polar angles + a final radius, (3) quantizing the angles only, keeping the radius in fp16. ~3.875 effective bits per coord with near-lossless quality and ~14% decode speedup.

**What we borrowed:**
- The **norm + direction** decomposition as a first-class operation. Our `r = ||x||` + `u = x/r` split is the PolarQuant framing made explicit.
- The decode-time **query-key inner product as a table-lookup** framing. PolarQuant's polar-table lookup is structurally analogous to our compressed-domain scoring kernel, just with a scalar codebook instead of a polar angle table.

**What we didn't borrow:**
- The actual polar-angles encoding. We quantize the rotated *Cartesian* coordinates against a 1D scalar codebook; PolarQuant quantizes nested polar *angles*. The angles encoding gives PolarQuant slightly lower bit rate at comparable quality, but the decode kernel is recursive (cos/sin reconstruction at each polar level) which is non-trivial on Metal.

**Verdict:** PolarQuant influences our *framing* (norm/direction as a first-class split) and our *decode-time optimization* (pre-rotated queries → compressed-domain scoring), but not our *encoding* (we use Cartesian-coord scalar quant, not polar-angles quant).

### 3.6 RaBitQ (Gao & Long — [SIGMOD 2024, arXiv 2405.12497](https://arxiv.org/abs/2405.12497); Extended SIGMOD 2025, [arXiv 2409.09913](https://arxiv.org/abs/2409.09913))

**The paper:** *randomized binary vector quantization* for nearest-neighbor search in high-dim vector databases. Random orthogonal rotation, then encode each vector by the *signs* of its rotated coordinates — 1 bit per dimension. Theoretical guarantee: inner-product estimation error is O(1/√D), provably matching the FOCS 2017 lower bound for D-bit codes.

**What we borrowed:** nothing yet, but the conceptual relationship is tight enough to call out.

**Why it's relevant:** RaBitQ is the **extreme low-bit endpoint** of the same family as TurboQuant and our codec. RaBitQ at 1-bit = "sign of rotated coordinate"; our codec at 4-bit = "nearest of 16 Lloyd-Max levels of rotated coordinate." Same rotation, same Cartesian-coord-after-rotation framing, much smaller codebook. The author has publicly contested TurboQuant's novelty claim relative to RaBitQ — the 1-bit special case is structurally identical, and RaBitQ pre-dates TurboQuant. See [author's medium post](https://dev.to/gaoj0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-1i00) and the response paper [arXiv 2604.19528](https://arxiv.org/abs/2604.19528).

**Why we don't ship 1-bit RaBitQ today:** 1-bit per coord is too aggressive for KV cache — quality regresses sharply on retrieval-heavy workloads. The Extended RaBitQ (2–9 bits/coord) is closer to our operating point, and its 4-bit configuration is essentially equivalent to our 4-bit GigaQuant minus the Lloyd-Max codebook (RaBitQ uses uniform-grid nested codes). **Worth a bench if/when we want a cleaner mathematical analysis** of our codec's quality vs RaBitQ's theoretical bound.

### 3.7 QuIP# (Tseng, Chee, Sun, Kuleshov, De Sa — Cornell RelaxML — [arXiv 2402.04396](https://arxiv.org/abs/2402.04396); ICML 2024)

**The paper:** "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." The conceptual ancestor of HIGGS — the same Hadamard-incoherence + structured-codebook framing that HIGGS later applies in a data-free regime. Original QuIP ([arXiv 2307.13304](https://arxiv.org/abs/2307.13304), NeurIPS 2023) introduced the "incoherence" argument: a random orthogonal pre-rotation makes the rotated weights have rows/columns of approximately even magnitude and rounding directions that aren't axis-aligned, enabling provable 2-bit quantization bounds. QuIP# upgrades both the rotation (Kronecker random orthogonal → Randomized Hadamard Transform, O(n log n) instead of O(n√n)) and the codebook (scalar adaptive rounding → **vector quantization on the E8 lattice**, the optimal 8-dim sphere-packing).

**What we borrowed:** nothing directly, but the framing is the conceptual root of half this paper's prior art. QuIP# established that **(a)** a Hadamard rotation makes the rotated distribution provably sub-Gaussian and ball-shaped, and **(b)** a codebook co-designed with the post-rotation distribution (E8 lattice, optimal for 8-dim Gaussian balls) is the principled choice. HIGGS later generalised the codebook to multivariate Gaussian-MSE-optimal grids at arbitrary block size p ∈ {1, …, 5}. QuaRot and SpinQuant carried the Hadamard rotation idea over from weights to activations and KV cache.

**Why it's not in the head-to-head:**
- **Weight-only.** QuIP# quantizes pretrained weights once; it does not handle activations, KV cache, or any streaming/online quantization regime. KV cache adaptation would be non-trivial: (a) you can't run QuIP#'s inter-layer fine-tuning at inference, (b) E8 lattice VQ packs 8-dim chunks of the weight matrix together, which conflicts with per-token per-head granularity in a KV cache, (c) the codebook needs to live in threadgroup memory at decode time (~1 KB — workable on Metal, but a real kernel engineering item).
- **Vector quantization on lattices**, not polar / scalar / per-coord. Conceptually closer to HIGGS's p-dim lattice than to our scalar codebook.
- **No polar decomposition.** Norm and direction are quantized together via the E8 lattice, not separated.
- **No Metal port.** CUDA-only repo; an Apple-Silicon adaptation would mean rewriting the lattice-lookup kernel from scratch.

**Verdict:** adjacent prior art — the originator of the Hadamard-incoherence + lattice-codebook framing that HIGGS inherits, and a sibling-in-spirit to QuaRot/SpinQuant for the rotation argument. Not a direct competitor to GigaQuant since it's weight-only with no streaming variant. Listed in the summary table (§4) for completeness but the comparison is structurally apples-to-oranges.

### 3.8 HIGGS (Malinovskii et al. — Yandex Research / ISTA — [arXiv 2411.17525](https://arxiv.org/abs/2411.17525); NAACL 2025)

**The paper:** "Pushing the Limits of LLM Quantization via the Linearity Theorem." Proves that under mild assumptions, end-to-end PPL is approximately linear in layer-wise relative Frobenius error — so minimizing layer-wise reconstruction error is approximately optimal for end-to-end quality. Uses this to allocate bits unevenly across layers in a data-free regime. Implementation: random Hadamard rotation + *Gaussian-MSE-optimal lattice grid* on small blocks (p ∈ {1, 2, 3, 4, 5}); p=1 reduces to scalar quant similar to NF4, p≥2 is genuine multivariate VQ.

**What we borrowed:** nothing directly.

**What's relevant:**
- The **linearity theorem** is the formal justification for our boundary-layer-protection heuristic. HIGGS proves you can rationally allocate fewer bits to "easy" layers and more to "hard" ones; we approximate that with a coarse fp16-at-edges rule, but a calibration-driven per-layer bit budget is the principled version.
- The **multivariate Gaussian-MSE grid** (p≥2) is the natural upgrade from our 1D scalar codebook. p=2 codes pairs of coordinates jointly with a 2D Voronoi grid; p=3 codes triples. The compression and quality wins are real (Llama-3.1-8B at 3.25 bits: p=4 hits 6.64 PPL on WikiText vs scalar's 7.13).

**Why we don't ship HIGGS-style lattice VQ today:** the encode kernel needs a small-LUT indexed gather per p-dim block, which is feasible on Metal but not yet implemented in MLX mainline. **Tracked as a future direction** — multivariate Gaussian grids are probably the single most credible upgrade to our codebook structure.

---

## 4. Summary table

| Property | GigaQuant (us) | TurboQuant | QuaRot | SpinQuant | PolarQuant | RaBitQ | QuIP# | HIGGS |
|---|---|---|---|---|---|---|---|---|
| Target | KV (shipping) + weights (tested) + acts (theoretical) | KV / activations | weights + acts + KV | weights + acts | KV cache | vector search (KV-adj.) | weights | weights |
| Rotation | Gaussian-QR + SRHT/FWHT | Gaussian-QR | SRHT (Hadamard) | Learned Hadamard | Random ortho | Random ortho | RHT (Hadamard) | Random Hadamard |
| Rotation learned? | No | No | No | **Yes** | No | No | No | No |
| Polar split | **Yes (explicit)** | Pre-step | No | No | **Yes (recursive)** | No | No | No |
| Codebook | Lloyd-Max 1D global | Lloyd-Max per-coord (analytic) | Uniform int4 | Uniform int4 | Polar angles | Sign (1-bit) or nested (multi-bit) | **E8 lattice VQ (8-dim)** | Gaussian-MSE p-dim lattice (p∈{1..5}) |
| Codebook source | llama.cpp k_quants + √(128/d) | Closed-form Beta | Uniform | Uniform | Polar quantization | Theory-optimal | E8 sphere-packing | CLVQ-trained |
| Bit rate | 2 / 4 / 8 (asymm K/V) | 2.5 (with QJL) / 4 | 4 | 4 | ~3.875 | 1–9 | 2 / 3 / 4 | 2 / 3 / 4 |
| QJL residual? | **No** | Yes (Algo 2) | No | No | No | No | No | No |
| Decode optimization | Pre-rotated Q + compressed scoring | Generic VQ | Standard dequant | Standard dequant | Polar table lookup | XOR + popcount | Lattice LUT + finetune | LUT gather |
| Apple Silicon kernel | **Tuned FWHT + compressed-domain scorer** | None published | None published | None published | None published | NEON-friendly | None published (CUDA-only) | FLUTE (CUDA-only) |

---

## 5. Open questions and future directions

The questions worth specing or research-tracking as follow-on work:

1. **Per-coordinate codebooks.** The paper specifies per-dim Lloyd-Max quantizers; we use one global codebook. Worth benchmarking whether per-dim recovers measurable PPL on real models. Kernel cost is modest — D codebooks instead of 1 — but register pressure goes up.

2. **HIGGS-style multivariate Gaussian-MSE lattice.** Replace our 1D scalar codebook with a 2D or 3D Voronoi grid trained via CLVQ on Gaussian samples. Probably the single most credible quality upgrade to GigaQuant. Tracked separately; needs a small LUT-gather kernel on Metal.

3. **SpinQuant-style learned rotations.** Fine-tune Π per model on a calibration corpus. Quality lift is ~1–3% PPL — useful, but real engineering cost (per-model training infrastructure, sanitize() complexity, calibration data dependency).

4. **Centroid-routed sparse decode** (= GigaQuant + spec 034 fused). Today our codebook is per-coordinate scalar; if we upgrade to a *vector* or *product-quantization* codebook (HIGGS p≥2 style), then the codebook itself becomes the per-query top-k selector. Precompute `Q · centroids` once per layer (small table), then per-cached-vector scoring becomes one index lookup + one scalar multiply — `O(1)` per key instead of `O(D)`. This composes cleanly with spec 034 and is one of the most exciting research directions for the KV codec. See `papers/beyond-quadratic-attention-on-apple-silicon.md` §6.3 for context.

5. **RaBitQ-style 1-bit storage tier.** For workloads that can tolerate it (long-context summarization, low-entropy retrieval), a 1-bit "sign of rotated coord" tier would give 32× compression over fp16. Probably too aggressive for general use but could be a third tier alongside 4-bit and 8-bit.

6. **Fix the codebook scaling.** The √(128/d) heuristic is a hack to adapt the d=128 Lloyd-Max table to other head dims. The principled fix is to re-derive the codebook analytically per head dim from the Beta distribution variance, as the TurboQuant paper specifies. Low engineering cost; might recover small quality on non-128 head dims (Mistral, Phi, anything with d ∈ {64, 80, 96, 256}).

7. **PolarQuant-style polar angle encoding** as an alternative codebook. Detailed in §2.4 — on paper the compressed-domain scoring is ~18× cheaper per cached key (log₂(D) angle-table lookups instead of D scalar-codebook lookups) and the bit rate is ~3% smaller at comparable quality. The cost is a more complex Metal kernel (recursive accumulation across angle levels can't flatten into one SIMD-group-per-(query,key) the way our scalar lookup does, and the encode pass needs `atan2` + recursive polar reduction). Worth a real bench against turbo4-V to settle whether the score-time win materialises into end-to-end decode tok/s.

8. **The codec rename.** Should we actually rename `TurboQuant*` → `GigaQuant*` in the source tree? Arguments for: honest naming, fewer surprised readers. Arguments against: mechanical churn, breaks downstream search/grep, the existing header comments already point readers to this paper. Deferred.

9. **Scope expansion — weights and activations.** The pipeline is target-agnostic, so the codec applies to any high-dim tensor. KV is what ships; weights have been tested with promising early results; activations remain untested. The interesting engineering questions for each target are different:
   - **Weights:** the codec already works on weight matrices. The remaining question is whether it competes against QuaRot+GPTQ at the same bit rate without the GPTQ Hessian pass — preliminary numbers say yes for matrices small enough that the per-tensor SRHT rotation amortises (most of the FFN, all of QKV). Worth a full sweep against existing 4-bit weight schemes on Llama-class and Qwen-class checkpoints.
   - **Activations:** dynamic quantization per-forward changes the engineering profile substantially. Encode cost matters because it runs N times per token instead of once per cache write. Only worth doing if the downstream GEMM uses the int8/int4 path natively (avoiding a quantize-then-dequantize round-trip). Composes naturally with the planned BitNet-style ternary work (`papers/beyond-quadratic-attention-on-apple-silicon.md` §4.9) — quantized activations × ternary weights is the cleanest version of this story. **Untested as of this writing**; the question is whether SRHT + per-coord Lloyd-Max recovers enough quality for int4 activation × int4 weight matmuls on Llama/Qwen-class models, vs the QuaRot+W4A8 baseline.
   - **Composition with the shipping KV path.** If activations are quantized in the rotated/Lloyd-Max domain *before* they get written to KV cache, the cache write becomes a no-op (just a copy). That's a real bandwidth win at long context, and it's the unified codec story that makes scope expansion strategically attractive.

---

## 6. References

### Primary algorithms cited

- [TurboQuant (Zandieh et al., ICLR 2026; arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)
- [TurboQuant — OpenReview](https://openreview.net/forum?id=tO3ASKZlok)
- [TurboQuant — Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant_plus fork (TheTom)](https://github.com/TheTom/turboquant_plus) — community fork closer to our codec
- [turbo4-resurrection.md (Tom Turney, March 2026)](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) — canonical write-up of the QJL-removal bench. PPL/KLD/decode-tok-s numbers cited in §3.1.
- [asymmetric-kv-compression.md (Tom Turney, March–April 2026)](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) — bench-driven justification for the 4-bit-K + 2-bit-V recipe, cross-backend (Metal / CUDA / HIP / Vulkan), 7 models.
- [layer-aware-v-compression.md (Tom Turney, March 2026)](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md) — boundary-V recipe (q8_0 V on layers {0,1,N-2,N-1}, turbo2 V elsewhere); the source of GigaQuant's boundary-layer protection.
- [QuIP (Chee et al., NeurIPS 2023; arXiv 2307.13304)](https://arxiv.org/abs/2307.13304) — incoherence framing
- [QuIP# (Tseng et al., ICML 2024; arXiv 2402.04396)](https://arxiv.org/abs/2402.04396) — Hadamard + E8 lattice; the conceptual ancestor of HIGGS
- [QuIP# code (Cornell RelaxML)](https://github.com/Cornell-RelaxML/quip-sharp)
- [QuaRot (Ashkboos et al., 2024; arXiv 2404.00456)](https://arxiv.org/abs/2404.00456)
- [SpinQuant (Liu et al., 2024; arXiv 2405.16406)](https://arxiv.org/abs/2405.16406)
- [PolarQuant (Han, Kacham, Mirrokni, Zandieh, Karbasi, 2025; arXiv 2502.02617)](https://arxiv.org/abs/2502.02617)
- [PolarQuant — NeurIPS 2025 variant (arXiv 2502.00527)](https://arxiv.org/abs/2502.00527)
- [PolarQuant — NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118745)
- [RaBitQ (Gao & Long, SIGMOD 2024; arXiv 2405.12497)](https://arxiv.org/abs/2405.12497)
- [Extended RaBitQ (Gao et al., SIGMOD 2025; arXiv 2409.09913)](https://arxiv.org/abs/2409.09913)
- [RaBitQ-Library (official impl)](https://github.com/VectorDB-NTU/RaBitQ-Library)
- [Revisiting RaBitQ and TurboQuant (arXiv 2604.19528)](https://arxiv.org/abs/2604.19528) — attribution rebuttal
- [HIGGS / Linearity Theorem (Malinovskii et al., NAACL 2025; arXiv 2411.17525)](https://arxiv.org/abs/2411.17525)
- [HIGGS — Hugging Face Transformers docs](https://huggingface.co/docs/transformers/quantization/higgs)
- [HIGGS — ISTA-DASLab model collection](https://huggingface.co/collections/ISTA-DASLab/higgs-675308e432fd56b7f6dab94e)

### Adjacent / context

- [QJL (Zandieh et al., 2024; arXiv 2406.03482)](https://arxiv.org/abs/2406.03482) — the residual-quantization second stage we omit
- [KIVI: Plug-and-Play 2bit KV Cache Quantization (arXiv 2402.02750)](https://arxiv.org/abs/2402.02750) — canonical asymmetric K/V precedent (2-bit K + 4-bit V on softmax-amplification reasoning)
- [KVQuant (arXiv 2401.18079)](https://arxiv.org/abs/2401.18079) — boundary protection + per-channel K outlier characterisation
- [Q-Hitter (arXiv 2402.14905)](https://arxiv.org/abs/2402.14905) — per-layer importance / heavy-hitter retention
- [SqueezeAttention (arXiv 2404.04793)](https://arxiv.org/abs/2404.04793) — 2D per-layer budget allocation via cosine-similarity clustering
- [Atom (arXiv 2310.19102)](https://arxiv.org/abs/2310.19102) — earlier asymmetric weight/activation/KV quantization
- [llama.cpp k_quants](https://github.com/ggml-org/llama.cpp) — source of the empirical Lloyd-Max codebook
- [metal-flash-attention (Philip Turner)](https://github.com/philipturner/metal-flash-attention) — Apple Silicon kernel reference

### Implementation files

- [`Libraries/MLXLMCommon/TurboQuantKVCache.swift`](../Libraries/MLXLMCommon/TurboQuantKVCache.swift) — codec + cache
- [`Libraries/MLXLMCommon/TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift) — Metal kernels (FWHT + compressed-domain scoring)

### Companion papers

- [`speculative-decoding-on-apple-silicon.md`](speculative-decoding-on-apple-silicon.md) — decode-throughput tour
- [`beyond-quadratic-attention-on-apple-silicon.md`](beyond-quadratic-attention-on-apple-silicon.md) — sparse / sub-quadratic / adaptive compute survey; §6.3 covers the WHT/butterfly/Monarch family that this codec sits inside
