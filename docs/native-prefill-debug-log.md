# Native Prefill Bridge Debug Log

## Current bug read

The C++ native prefill bridge (`generic_prefill.cpp`) produces zeroed embedding output for certain models. The embedding tensor has correct shape (e.g., `[1,5,4096]`) but `sum=0.0000` — metadata is intact, data is dead.

This means one of:
- Embedding weight/scales/biases data is zeroed (Metal buffer reclaimed)
- `dequantize()` produces zero because `scales * weight + biases = 0`
- Quantization parameters (bits, group_size) are wrong

The correct output shape rules out "bridge treats quantized as unquantized" — packed uint32 would have wrong column count. So the dequantize path IS entered, but output is zero.

## Known working models

| Model | Architecture | Path in bridge |
|-------|-------------|----------------|
| MiniMax M2 | MoE | `build_minimax_layer` |
| Qwen3-Coder | MoE | `build_qwen3_moe_layer` |
| Phi3 | Dense | generic else branch |

## Known failing models

| Model | Architecture | Path in bridge |
|-------|-------------|----------------|
| Qwen2 / Qwen2.5 dense | Dense | `build_qwen_dense_layer` |
| Qwen3 dense | Dense | `build_qwen_dense_layer` |
| Mistral dense | Dense | generic else branch |

## Established facts

- Without the bridge, failing models produce correct output.
- With bridge, first request produces garbage or incoherent output.
- Second request sometimes behaves differently (not trustworthy).
- Warmup does not fix it.
- Removing `clear_cache()` does not fix it.
- Dedicated Qwen dense builder did not fix it.
- Aggregate-init mismatch was found and fixed — not root cause.
- Qwen2 attention bias was suspected — but Mistral also fails and has no bias.
- Bridge initialization succeeds for all models.
- **Phi3 works through the same generic dense path that Mistral fails on.** This is the key comparison.
- Failing models show `embed chunk sum=0.0000` inside bridge `forward()`.

## Ranked hypotheses

| # | Hypothesis | Strength | Key evidence |
|---|-----------|----------|-------------|
| ~~H1~~ | ~~Working models have unquantized embeddings; failing models have quantized embeddings.~~ | **ELIMINATED** | All models have quantized embeddings. Checked safetensors directly. |
| H2 | Metal buffer reclamation zeros embedding weight/scales/biases between build and forward. | **Strong** (now top) | Code comments warn about this. Correct shape + zero data = metadata alive, data dead. |
| H3 | `quant_group_size`/`quant_bits` config mismatch — hardcoded 64/4 but model uses different values. | Weak | Would produce garbage, not specifically sum=0. |
| H4 | Struct default `array(0.0f)` shadows real data via move semantics. | Weak | Aggregate init was already fixed. Explicit member assignment should overwrite. |
| H5 | Singleton bridge reuse — two `modelType:"generic"` models cached. | Sanity check | Quick to rule out. |

## Experiments

| ID | Experiment | Purpose | Code location | Status | Result | Conclusion |
|----|-----------|---------|---------------|--------|--------|------------|
| E1 | Embedding key presence and quantization status | Determine if Phi3 vs Mistral differ in quantization | `make_embedding()`, `gp_set_weight()` | **done** | Both quantized. Phi3 bridge always fails init (fused qkv_proj). | H1 eliminated. Phi3 was never a valid comparison. |
| E2 | Embedding data liveness at build time and pre-forward | Determine if data is valid after construction and still valid at first forward | `build_model()`, `gp_run()` | **done** | CHECKPOINT-A already zero for Mistral. Data dies during `build_model()`, before first forward. | Corruption window = between `gp_set_weight` and post-build inspection. Allocator reclamation is leading hypothesis, not yet proven cause. |
| E3 | 3-point discriminator: set vs get vs make_embedding | Narrow corruption to g_weights storage, get_w retrieval, or post-assignment | `gp_set_weight`, `get_w`, `make_embedding` | **done** | scales/biases nonzero at POINT-1 (set), zero at POINT-2 (get). Data dies inside g_weights map. | **g_weights storage/ownership bug.** Not allocator reclamation during build — data is already gone when retrieved from the map. |
| E4 | Hard shape/assert validation for quantized path | Fail loudly if quant layout assumptions are wrong | `make_embedding()`, `Embedding::operator()` | planned | — | — |
| E5 | Pre-dequantized embedding proof test | If tensors present but dequant broken, test build-time dequant | `make_embedding()` | planned | — | — |

## Results

### E1 — Embedding key presence and quantization status

**Status:** Running

**Instrumentation added:**
- `make_embedding()` now logs: quantized flag, weight/scales/biases shapes, dtypes, sizes, detected bits, group_size
- `gp_set_weight()` already logs `embed_tokens.scales` key

**Expected discriminator:**
- If Phi3 shows `quantized=0` and Mistral shows `quantized=1` → H1 confirmed
- If both same → H1 eliminated, move to E2

**Pre-experiment: Safetensor key inspection (bypasses bridge entirely)**

Checked actual safetensor files on disk:

```
=== Phi-4-mini-instruct-4bit ===
  model.embed_tokens.biases: shape=(200064, 48) dtype=float16
  model.embed_tokens.scales: shape=(200064, 48) dtype=float16
  model.embed_tokens.weight: shape=(200064, 384) dtype=uint32

=== Mistral-7B-Instruct-v0.3-4bit ===
  model.embed_tokens.biases: shape=(32768, 64) dtype=float16
  model.embed_tokens.scales: shape=(32768, 64) dtype=float16
  model.embed_tokens.weight: shape=(32768, 512) dtype=uint32

=== Qwen3-Coder-30B-A3B-Instruct-MLX-6bit (working MoE) ===
  model.embed_tokens.biases: shape=[151936, 32] dtype=bfloat16
  model.embed_tokens.scales: shape=[151936, 32] dtype=bfloat16
  model.embed_tokens.weight: shape=[151936, 384] dtype=uint32

=== Qwen2.5-3B-Instruct-4bit (failing) ===
  model.embed_tokens.biases: shape=[151936, 32] dtype=float16
  model.embed_tokens.scales: shape=[151936, 32] dtype=float16
  model.embed_tokens.weight: shape=[151936, 256] dtype=uint32

=== Qwen3-4B-4bit (failing) ===
  model.embed_tokens.biases: shape=[151936, 40] dtype=bfloat16
  model.embed_tokens.scales: shape=[151936, 40] dtype=bfloat16
  model.embed_tokens.weight: shape=[151936, 320] dtype=uint32
```

**Result: H1 ELIMINATED.** All models — working AND failing — have quantized embeddings with weight/scales/biases keys present in safetensors. The quantized vs unquantized embedding hypothesis is dead.

**Observed (bridge logs):**

Mistral-7B-Instruct-v0.3-4bit:
```
[gp] set_weight 'model.embed_tokens.biases': dtype=9 shape=[32768,64] size=2097152 available=1
[gp] set_weight 'model.embed_tokens.scales': dtype=9 shape=[32768,64] size=2097152 available=1
[gp] set_weight 'model.embed_tokens.weight': dtype=3 shape=[32768,512] size=16777216 available=1
[gp] make_embedding 'model.embed_tokens': quantized=1
   weight: shape=[32768,512] dtype=3 size=16777216
   scales: shape=[32768,64] dtype=9 size=2097152
   biases: shape=[32768,64] dtype=9 size=2097152
   detected bits=4 group_size=64
[gp] CHECKPOINT-A embed weight abssum=0.000000 absmax=nan
[gp] CHECKPOINT-A embed scales abssum=0.000000 absmax=0.000000
[gp] CHECKPOINT-A embed biases abssum=0.000000 absmax=0.000000
```

Phi-4-mini-instruct-4bit:
```
[gp] set_weight 'model.embed_tokens.weight': dtype=3 shape=[200064,384] available=1
[gp] Finalizing with 452 weights
[gp] finalize error: Missing weight: model.layers.0.self_attn.q_proj.weight
[GenericPrefill] finalize failed
```
Phi3 bridge **always fails init** because Phi3 uses fused `qkv_proj`, not separate `q_proj/k_proj/v_proj`. Falls through to Swift prefill every time. **Phi3 was never actually using the bridge.**

## Conclusions

1. **H1 eliminated** — All tested models have quantized embeddings. The quantization status is not the discriminator.
2. **Phi3 "works" is a red herring** — Phi3 bridge always fails init (fused qkv_proj key mismatch). It falls through to Swift prefill. There is NO known dense model that works through the bridge.
3. **Corruption localized to `g_weights` map** — 3-point discriminator (E3) proves:
   - POINT-1 (`gp_set_weight`): scales abssum=1.003, biases abssum=1.004 → **data is valid at ingress**
   - POINT-2 (`get_w` during `make_embedding`): scales abssum=0, biases abssum=0 → **data is dead when retrieved from map**
   - The corruption happens between `g_weights.insert_or_assign(k, arr)` and `g_weights.find(key)` retrieval
   - This rules out post-assignment reclamation — the data is already gone inside the map itself
   - Leading candidates: (a) the 32 layer construction loop calls `get_w` hundreds of times, and those retrievals + the `array(0.0f)` defaults in Layer structs cause the underlying buffers of OTHER map entries to be reclaimed; (b) `insert_or_assign` with later keys overwrites/invalidates earlier entries' shared buffer references; (c) `extract_array` copy doesn't properly retain the underlying data through map operations
   - Weight (uint32) POINT-1 shows abssum=0/absmax=nan — inconclusive due to uint32 dtype behavior with abs/sum/max, not necessarily zero data

## Next actions

1. ~~E1 done~~ — Both models quantized. Phi3 bridge always fails (fused qkv_proj).
2. ~~E2 done~~ — CHECKPOINT-A already zero.
3. ~~E3 done~~ — **set nonzero, get zero.** Data dies inside `g_weights` map between ingress and retrieval.
4. **POINT-1.5 result: data survives loading, dies during `build_model()`.**
   - scales/biases are alive at finalize-start (POINT-1.5: abssum ~1.0)
   - scales/biases are dead at first get_w retrieval during make_embedding (POINT-2: abssum=0)
   - This rules out the weight-loading/insert phase as the problem
   - The corruption happens during `build_model()` — the 32-layer construction loop calls `get_w` hundreds of times + creates many `array(0.0f)` defaults, and somewhere in that process the embed_tokens entries' underlying data gets invalidated while the map entries themselves remain present
5. **Proof test A result: building embedding first does NOT protect it.**
   - scales/biases alive at PROOF-A-after-create (abssum ~1.0)
   - scales/biases dead at PROOF-A-after-layers (abssum=0.0)
   - `get_w` retrieval worked perfectly when called early — data was valid through POINT-2, POINT-3, and into the Embedding struct
   - But the 32-layer construction loop destroys the underlying data of arrays **already held by `m.embed_tokens`**
   - This is NOT a late-retrieval bug. This is array-lifetime/ownership after assignment into the model struct. The layer loop's mass creation of arrays (via `get_w` + `make_qlinear` + `array(0.0f)` defaults) invalidates buffers of previously-assigned arrays in the same model struct.
6. **Ownership test: independently-allocated buffers also die.**
   - Created new buffers via `x + zeros_like(x)` → `eval()` for scales, biases, and weight
   - These are NOT shared with g_weights — they're fresh Metal allocations with independent ownership
   - PROOF-A-after-create: alive (abssum ~1.0)
   - PROOF-A-after-layers: dead (abssum=0.0)
   - **This is NOT a shared-buffer bug from g_weights.** The layer loop destroys independently-owned, evaluated Metal buffers held by the model struct. This is broader MLX array lifetime breakage under construction pressure — likely the Metal allocator reclaiming buffers it considers available despite live C++ array references.
7. **Binary search + toggles narrowed to `push_back(Layer{})`:**
   - Dead after layer 1 — immediate, not gradual
   - `resize(32)` alone kills it (creates 32 default-constructed Layers)
   - `reserve(32)` does NOT kill it (no construction)
   - `push_back(Layer{})` kills it BEFORE any `make_qlinear`/`make_norm` is called
   - Per-layer `eval(layer_weights)` does NOT help
   - A single default-constructed `Layer{}` — ~20 `array(0.0f)` scalar members — is sufficient to destroy independently-owned, eval'd Metal buffers on `m.embed_tokens`
   - This strongly implicates `array(0.0f)` default construction as the trigger for MLX Metal buffer reclamation/invalidation of previously allocated arrays
8. **Standalone `array(0.0f)` locals do NOT kill embedding.** 20 locals created and destroyed — embedding survives. But `push_back(Layer{})` still kills it immediately. The culprit is NOT `array(0.0f)` construction per se. It's something specific to default-constructing a Layer inside `vector::push_back` — likely the move/copy of the Layer struct or vector reallocation side effects. Key difference from standalone locals: `push_back` creates a temporary `Layer{}`, move-constructs it into the vector, then destroys the temporary. The move constructor of `mlx::core::array` may have destructive side effects on the global allocator state, or vector operations on a struct containing many arrays trigger different allocator behavior than standalone locals.
9. **Local `Layer{}` on the stack kills it — not a vector issue.**
   - A plain `Layer layer{};` local variable kills embedding before any vector operation
   - `emplace_back()` also kills it (equivalent — both default-construct a Layer)
   - But 20 standalone `array(0.0f)` locals do NOT kill it
   - The difference: Layer has ~20 array members as **default member initializers** (`array weight = array(0.0f)`), initialized as part of aggregate/default struct construction. Something about constructing many arrays as struct members (vs standalone locals) triggers the allocator to reclaim other live buffers.
   - Possible mechanisms: (a) the compiler-generated default constructor initializes members in a different order or context than individual locals; (b) there's a destructor interaction where a moved-from or temporary array within struct initialization releases a buffer that happens to be the same backing as the embedding; (c) `array(0.0f)` shares a global singleton/pooled buffer that gets recycled during struct construction
10. **Threshold found: ~17-23 `array(0.0f)` members in a single struct triggers corruption.**
    - S16 (16 arrays) isolated: safe
    - S24 (24 arrays) isolated: kills scales, corrupts biases (nan)
    - S32 (32 arrays) isolated: kills both
    - Layer (~51 arrays via nested QuantizedLinear/Norm/KVCache): kills both
    - The corruption is NOT cumulative across separate struct lifecycles — S16 alone is safe, S24 alone kills
    - 20 standalone `array(0.0f)` locals are safe — the trigger requires struct member initialization context
    - This is a threshold-based MLX allocator bug: constructing+destructing ~20+ `array(0.0f)` objects as struct members in a single lifecycle causes the Metal allocator to reclaim live buffers belonging to unrelated evaluated arrays
11. **std::optional refactor: embedding survives construction, dies during forward.**
    - Replaced all `array X = array(0.0f)` default member initializers with `std::optional<array> X` across QuantizedLinear, Norm, Embedding, KVCache, and Layer
    - Moved MoE and PLE fields to separate `std::optional<MoEWeights>` / `std::optional<PLEWeights>` sub-structs
    - POST-BUILD scales abssum=1.003135 — **embedding survives layer construction**
    - Warmup 4-token forward: abssum=1.0022 — **first forward works**
    - Second forward (real 127-token run): abssum=0.0000 — **embedding dies during/after forward pass**
    - The forward pass creates many intermediate arrays (attention, MLP, chunked eval) which triggers the same allocator reclamation
    - RUN-START check confirms: scales=1.003 on first run, scales=0.0 on second run
    - The first `forward()` (warmup, 4 tokens, 32 layers) kills the embedding buffers
    - `eval()` won't help — arrays are already evaluated, the allocator reclaims their Metal backing anyway
    - This is the same fundamental MLX allocator issue: it treats evaluated buffers as reclaimable regardless of live C++ array references
    - The `std::optional` refactor removed ~59 `array(0.0f)` default member initializers from Layer, which eliminated the construction-time trigger. But the forward pass creates hundreds of intermediate arrays (attention, MLP, residuals) across 32 layers, which is a much larger trigger than 24 struct members.
12. **Forward-pass kill point: between layers 8 and 16 (first occurrence).**
    - Probes during first forward show oscillation: alive at 1/8, dead at 16, alive at 24, dead at 32
    - The probes themselves call `eval()` which rematerializes the buffer — so each probe is a mini-recovery
    - Without probes, death would happen earlier and be permanent
    - Pattern indicates cumulative lazy-graph pressure: ~8 layers of uneval'd computation (~120 pending ops) is enough to trigger reclamation
    - This explains why `build_qwen_dense_layer`'s per-layer eval works for that path — it keeps the lazy graph small
    - Practical fix direction: either add per-layer eval barriers in the generic dense forward, or eval the embedding arrays specifically at regular intervals during forward
13. **Periodic eval barriers (EVAL_CADENCE=8) with embedding in eval set: does NOT help.**
    - eval() on already-evaluated arrays is a no-op — doesn't pin their Metal buffers
    - Including embedding weight/scales/biases in the eval set has no effect
    - Keeping g_weights alive (not clearing after finalize) also has no effect
    - Multiple live C++ `array` references (model struct + g_weights map) do not prevent reclamation
    - The MLX Metal allocator does not respect C++ reference counts for buffer retention
    - The probe "oscillation" (dead at 16, alive at 24) was likely caused by `sum(abs())` creating a new computation that transiently read from the buffer before it was fully reclaimed, not actual rematerialization
    - This is a confirmed MLX framework bug: the Metal memory allocator reclaims buffers of live, evaluated arrays when sufficient new allocations occur during forward-pass computation
14. **STANDALONE REPRO CONFIRMED — MLX Metal allocator bug proven in 30 lines.**
    - `mlx_allocator_repro.cpp`: create one eval'd `[32768,64]` float16 array, run 32 layers of matmul pressure, buffer is corrupted after first layer
    - Baseline abssum=4311220224, corrupted to abssum=1.003784 (drift=100%)
    - No bridge code, no model code, no Swift, no safetensors — pure MLX C++ API
    - **CRITICAL UPDATE: Buffer is NOT corrupted.** Direct CPU float16 readback shows stable data (abssum=1,672,880 on every iteration). But MLX `sum(abs())` returns wildly different values (4.3B baseline, 1.0 after pressure). Even the BASELINE MLX value was wrong — 4.3B vs CPU 1.67M for the same data.
    - The bug is NOT in the allocator or buffer lifecycle. The raw Metal buffer memory is fine.
    - The bug is in MLX's GPU computation path: the Metal kernel for `sum(abs())` reads from a DIFFERENT buffer than the array's actual data buffer.
    - This is a compute-graph or buffer-binding bug, not an allocation bug.
    - The `array` object correctly points to its buffer (CPU reads are fine), but when MLX dispatches a GPU kernel that uses this array as input, the kernel binds to a wrong buffer.
    - **FINAL DIAGNOSIS: float16 sum reduction bug, NOT allocator/binding.**
    - `sum(abs(long_lived))` on float16 returns 4.3B. CPU truth is 1.67M. Even BEFORE any pressure.
    - `astype(float32)` then `sum(abs())` returns correct 1.67M.
    - `abs()` alone produces correct output (CPU-verified).
    - `sum([1,2,3,4,5])` on float32 returns correct 15.0.
    - The float16 `sum` Metal kernel has a numerical overflow/precision bug for large reductions.
    - ALL "corruption" observations were our observer (sum(abs())) lying to us. The buffer was never corrupted.
    - **REVALIDATION WITH SAFE OBSERVER: Embedding was NEVER corrupted.**
    - Replaced all sum(abs()) probes with safe_abssum (astype f32 first) and cpu_hash
    - Every checkpoint (POINT-1 through RUN-START across 5 runs) shows identical values: safe_abssum=1718.90 cpu_hash=f046884a49e1f461
    - The embedding data is perfectly stable from ingress through construction, through build, through warmup, through repeated forward passes
    - The `std::optional` refactor was unnecessary for correctness — the original `array(0.0f)` default initializers were not causing buffer corruption
    - **FORWARD COMPUTATION IS CORRECT.** Side-by-side comparison of bridge vs Swift hidden states:
      - Embedding output: identical first4 values
      - Layer 0 post_mlp: matches to 3-4 significant digits
      - Layer 31 post_mlp: ~3% abssum drift (expected fp accumulation)
      - Final norm output: first4 values match to 3 sig figs
    - The bridge forward pass produces numerically correct hidden states
    - **ROOT CAUSE FOUND: KV dtype mismatch.** Bridge stores KV in bfloat16, Swift model expects float16.
      - Bridge `generic_attention`: `k = astype(k, bfloat16); v = astype(v, bfloat16);`
      - Swift native cache: dtype=float16
      - Bridge injected cache: dtype=bfloat16
      - When bfloat16 KV is injected into Swift's cache and subsequent decode attention reads it, the bit layout is wrong (bfloat16 ≠ float16)
      - Fix: change bridge to store KV as float16, or cast before injection
    - All previous "corruption" findings (CHECKPOINT-A zero, PROOF-A-after-layers zero, BSEARCH dead-after-layer-1, etc.) were artifacts of the broken float16 sum reduction kernel
15. **Remaining question: why do MoE models survive this?** MiniMax M2 has 62 layers (even more intermediates) and works. Either: (a) MoE forward creates fewer per-layer intermediates due to `gather_qmm` being a single op vs many separate matmuls, (b) the per-layer eval in MoE builders somehow protects against forward-time reclamation, or (c) MoE models have different memory allocation patterns that avoid the threshold. This is worth investigating if the forward-time corruption isn't solvable with a simple workaround.

## Anti-patterns / do not do yet

- Do NOT change Swift wrappers until C++ bridge evidence points there.
- Do NOT chase RoPE, KV cache, attention bias, or masking until embedding is proven healthy.
- Do NOT add new model-specific builders unless an experiment proves construction path matters.
- Do NOT trust output text quality as the main signal — use `sum(abs(x))` and `max(abs(x))`.
- Do NOT branch into multiple failing models at once — use Phi3 vs Mistral first.
- Do NOT use plain `sum(x)` — it can hide sign-cancellation. Use `sum(abs(x))`.
- Do NOT do broad refactors.
- Do NOT silently fix unrelated things.
