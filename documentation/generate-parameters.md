# `GenerateParameters` Reference

Every knob that changes runtime inference behaviour lives on the
`GenerateParameters` struct. You pass it to `generate(...)`,
`generateTask(...)`, `generateBatched(...)`, or to the high-level
`ChatSession`'s initialiser. Defaults are tuned for chat / assistant
workloads; override per call site as needed.

For the high-level ChatSession surface (system prompt, streaming) see
[llm/using.md](llm/using.md). For KV-cache compression specifics see
[kv-cache.md](kv-cache.md). For wired-memory + concurrency coordination
see [memory-management.md](memory-management.md).

## Sampling

| Field | Default | Notes |
|---|---|---|
| `temperature` | `0.6` | Sampling temperature. `0` selects greedy. |
| `topP` | `1.0` | Nucleus sampling threshold. |
| `topK` | `0` | Top-k cutoff (0 = disabled). |
| `minP` | `0.0` | Minimum probability filter. |
| `repetitionPenalty` | `nil` | DRY-style penalty for recent tokens. |
| `repetitionContextSize` | `20` | Window applied to `repetitionPenalty`. |
| `presencePenalty` | `nil` | OpenAI-style additive penalty for tokens present in recent context. |
| `presenceContextSize` | `20` | Window applied to `presencePenalty`. |
| `frequencyPenalty` | `nil` | OpenAI-style additive penalty that scales with token frequency in recent context. |
| `frequencyContextSize` | `20` | Window applied to `frequencyPenalty`. |
| `additionalProcessors` | `[]` | Custom `LogitProcessor`s applied after the built-in penalty processors. Use to inject EOS suppression for thinking models, grammar-constrained decoding, etc. |
| `maxTokens` | `nil` | Upper bound on generated tokens. |

## KV cache

| Field | Default | Notes |
|---|---|---|
| `maxKVSize` | `nil` | User-budget cap on KV cache tokens. When set on a layer that's **not** architecturally sliding, engages rotating eviction at this cap uniformly across **all three** cache classes (`StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`) — closed the affine UX wart in spec 041 phase 1.2 / [#215](https://github.com/ekryski/mlx-swift-lm/pull/215). Architectural `slidingWindow` (per-layer, model-driven) takes precedence when both are set. |
| `compressionAlgorithm` | `nil` | KV-cache compression (`.affine(bits:groupSize:)` / `.turbo(keyBits:valueBits:)`). `nil` = no compression (raw FP16/BF16). Parsed via `KVCache.CompressionAlgorithm.init?(_:)` from a string like `"turbo4v2"`. See [kv-cache.md](kv-cache.md). |
| `draftCompressionAlgorithm` | `nil` | Optional draft-model KV cache compression for speculative decoding. When `nil`, the draft falls back to `compressionAlgorithm` (matches the main model). Use to run a smaller draft with looser compression — e.g. `.turbo` on main, `.affine(bits: 4)` on draft. Each model uses the cache class best suited to its size and supported kernels (spec 006 PR 4). |
| `turboBoundarySkip` | `2` | TurboQuant codebook boundary skip; lower raises PPL slightly but speeds up encode. |

## Prefill / throughput

| Field | Default | Notes |
|---|---|---|
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill — lower = lower peak GPU at the cost of prefill throughput. Falls back to the model's `defaultPrefillStepSize` (Qwen3.5 dense `1024` / Qwen3.5 MoE `4096` / Gemma 4 `4096` / GPT-OSS `2048` / Nemotron `1024`). Post-#215, the resolved value (`prefillStepSize ?? defaultPrefillStepSize ?? 256`) is threaded uniformly into the `step:` constructor parameter of all three cache classes — `concatenated(...)` growth events collapse 1:1 with prefill chunks for every scheme. M1 Max sweep on Qwen 2B / ctx=16k / `--kv none`: 256 → 2.26 GB / 1106 tok/s · 512 → 2.27 GB / 1132 · 1024 → 2.38 GB / 1148 · 2048 → 2.51 GB / 1182. |

## Speculative decoding (n-gram prompt-lookup)

| Field | Default | Notes |
|---|---|---|
| `ngramSize` | `0` | N-gram length for prompt-lookup speculation. `0` disables. Net win only on repetitive output (code, templates). |
| `maxNgramDraftTokens` | `0` | Max draft tokens per speculation round. Pair with `ngramSize`. |
| `ngramDraftMin` | `1` | Minimum draft tokens to issue a verify batch. Mirrors llama.cpp `--draft-min`. |
| `ngramMinHits` | `1` | Minimum times an n-gram must appear before it's drafted. Mirrors llama.cpp `--spec-ngram-min-hits`. |
| `minNgramSize` | `2` | Floor of the multi-size fallback ladder. Set equal to `ngramSize` to disable fallback. |

**N-gram speculative + hybrid models (Qwen 3.5 / 3.6 GDN+attention)** is supported as of spec 020 (shipped 2026-05-11). The router auto-engages on hybrid stacks when `canRollbackPromptCache(cache) == true` — see [speculative-decoding.md](speculative-decoding.md#hybrid-model-coverage-spec-020). Mamba-using models (Nemotron-H, Jamba) opt out via per-cache `canStateReplay = false` and fall back to vanilla `TokenIterator` at parity.

## Cross-request prefix KV cache (spec 017)

Caches the target KV state at the end of one request and hydrates it at
the start of the next when the prompt shares a stable prefix. **2-10× TTFT
improvement** on multi-turn chat workloads (measured: ~4.3× on Qwen3.5-35B-A3B,
~2.5× on Gemma 4 E2B / Qwen3.5-0.8B).

**Opt-in for v1.** `GenerateParameters()` with no arguments leaves the
cache off; enable per-call via `prefixCacheEnabled: true` or process-wide
via `MLX_PREFIX_CACHE=1`. Default-on was attempted 2026-05-12 and reverted
the same day after bench validation surfaced two `--kv turbo4v2` interaction
issues ([#196](https://github.com/ekryski/mlx-swift-lm/issues/196),
[#197](https://github.com/ekryski/mlx-swift-lm/issues/197)). The
auto-resolved policy + auto-resolved modelID logic remains in place — once
the two follow-up issues close, flipping the default is a one-line change.

| Field | Default | Notes |
|---|---|---|
| `prefixCacheEnabled` | `false` | Master toggle. Env overrides: `MLX_PREFIX_CACHE=1` (force on) / `=0` (force off). When true, `generate(...)` consults `PrefixKVCache.shared` for a longest-prefix hit, hydrates if found, and on stream completion snapshots the cache at the stable-prefix boundary. |
| `prefixCachePolicy` | auto-resolved (when enabled) | When nil **and** prefix cache is enabled, the runtime calls `AssistantOpener.detect(forModelID:)` against the resolved model ID and constructs a `LastAssistantOpenerPolicy` for known families: **Qwen** (1.x – 3.6, QwQ → ChatML opener), **Gemma** (1/2/3/4 → `<start_of_turn>model\n`), **GPT-OSS** (harmony). Unknown families fall back to `IdentityPolicy` — completion still caches; chat on unknown families just doesn't get the chat-cache speedup (no regression). Override explicitly when needed. |
| `prefixCacheModelID` | `nil` (auto-resolved when enabled) | Stable model identifier used to scope snapshots. When nil **and** prefix cache is enabled, the runtime auto-resolves it from `ModelContext.configuration.name`, so single-model apps need zero setup once the flag is on. Apps that share `PrefixKVCache.shared` across multiple variants of the same architecture (e.g. quantization swaps) SHOULD set this to disambiguate. |
| `prefixCacheDiskEnabled` | `false` | Promote / persist snapshots to disk at `~/.cache/mlx-swift-lm/prefix/`. **Strictly opt-in** — never bloats disk unless explicitly enabled. Env override: `MLX_PREFIX_CACHE_DISK=1`. On L1 miss, falls through to disk; on hit, promotes back into L1. |

### Minimal opt-in example

```swift
import MLXLMCommon

var params = GenerateParameters()
params.prefixCacheEnabled = true   // policy + modelID auto-resolve from context

let stream = try await container.generate(
    input: lmInput, parameters: params)
```

### Customising

```swift
var params = GenerateParameters()
params.prefixCacheEnabled = true
// Tokenizer-aware chat trimming for a specific family:
params.prefixCachePolicy = LastAssistantOpenerPolicy(
    opener: .qwenChatML, tokenizer: ctx.tokenizer)
// Opt into L2 disk persistence:
params.prefixCacheDiskEnabled = true
```

### Process-wide on/off via env

```bash
export MLX_PREFIX_CACHE=1   # force on for every generate() call in this process
export MLX_PREFIX_CACHE=0   # force off (overrides any prefixCacheEnabled: true)
```

For diagnostics, read `PrefixKVCache.shared.stats` after a request to see hits / misses / saved prefill tokens. `PrefixKVCache.shared.resetStats()` zeros counters without clearing entries; `clear()` empties entries. See [llm/using.md](llm/using.md) for the `ChatSession`-level surface.

### Known limitations

Full list in `specs/017-prefix-kv-cache.md`; the table below summarises the user-visible ones with follow-up issue links.

| # | Limitation | Affected | Follow-up |
|---|---|---|---|
| 1 | TurboQuant compressed-mode snapshot refused — `cache_bytes=0` throughout multi-turn runs (cache silently never engages) | Qwen 3.5 / 3.6 / NemotronH under `--kv turbo4v2` | [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) |
| 2 | Lookup misses despite successful inserts — cache uses memory but no TTFT benefit | Gemma 4 26B-A4B / 31B under `--kv turbo4v2` | [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) |
| 3 | Sliding-window=128 wraps mid-generation → no cache benefit (silent skip, no error) | GPT-OSS-20B | Spec 017 phase 5 |
| 4 | Hybrid SSM-layer snapshot includes a few generation steps past the stable prefix; attention layers are exact (dominant prefill cost) | Qwen 3.5 / 3.6 GDN | Spec 017 phase 5 |
| 5 | `PrefixKey.kvHeadDim` is a placeholder in auto-key derivation; cross-model gating still works via modelID + kvBits | All models (cosmetic) | Spec 017 phase 5 |

Issues 1+2 are the gate for re-enabling default-on. **Practical guidance**: opt in safely for any `--kv none` workload and for `--kv affine4`. Avoid combining with `--kv turbo4v2` until [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) closes — no crash, but no benefit either, just wasted memory on Gemma 4 26B/31B.

## Thinking / reasoning

| Field | Default | Notes |
|---|---|---|
| `reasoningEffort` | `nil` | Hint passed to chat templates that support it (`"low"` / `"medium"` / `"high"`). |
| `thinkStartTokenId` / `thinkEndTokenId` | `nil` | Token IDs for thinking-phase boundaries; enables phase-separated logprob tracking when set. |
| `thinkingPhasePrefilled` | `false` | Set when the prompt already opens with `<think>`. |
| `harmonyChannelMarkerTokenId` / `harmonyThinking…` / `harmonyGeneration…` | `nil` / `[]` / `[]` | GPT-OSS harmony-format phase machine. |

## Logprobs / quality tracking

| Field | Default | Notes |
|---|---|---|
| `collectPerTokenData` | `false` | Store per-token logprobs / IDs / phase labels for downstream KLD. |
| `trackPerplexity` | `false` | Accumulate logprobs for end-of-run PPL. |

## Environment-variable overrides

These take precedence over the `GenerateParameters` defaults. They exist
for **diagnostics, A/B testing, and tuning** — not as the primary
user-facing API. Set in the shell before launching an inference process;
read once at first use and cached.

For wired-memory env vars (`MLX_MEMORY_LIMIT`, `MLX_SMART_MEMORY`) see
[memory-management.md](memory-management.md). For TurboQuant /
attention-path env vars (`TURBO_*`, `MLX_AFFINE_SDPA`) see
[kv-cache.md](kv-cache.md).

### Model-specific perf knobs

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. May be removed in future. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. The architecture default is on for some configurations and off where the wrapper costs ~10 % decode (e.g. 26B-A4B MoE). |
| `GDN_EVAL_INTERVAL=N` | GatedDeltaNet (Qwen 3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

### N-gram speculative-decoding env knobs (spec 016 / spec 020)

Most users opt in via `ngramSize` + `maxNgramDraftTokens` on `GenerateParameters`. The env vars below cover the env-only opt-in path and the diagnostic / A-B knobs on the n-gram routing logic. See `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` for the per-knob reference and defaults — listed here so they're discoverable, not exhaustively detailed.

| Variable | Effect |
|---|---|
| `MLX_NGRAM_ENABLED=1` | Opt-in path #2 (env-only). Convenient for benchmarks: enables n-gram routing without changing call sites that use a bare `GenerateParameters()`. Pairs with sensible defaults (`ngramSize=3`, `maxNgramDraftTokens=4`). |
| `MLX_NGRAM_DEBUG=1` | Verbose tracing — prints per-round draft/accept stats for diagnostic A/B work. |
| `MLX_NGRAM_LEVIATHAN=0` | Disable the multi-candidate verify-batch optimisation (default ON). Use to A/B against the simpler single-candidate verify path. |
| `MLX_NGRAM_ADAPTIVE=0` | Disable adaptive draft-size sizing (default ON). Adaptive starts at `maxNgramDraftTokens` and floats based on rolling acceptance — disable to pin draft size. |
| `MLX_NGRAM_FORCE_AR=1` | Force the autoregressive fallback on every round (skips speculation entirely). Diagnostic only. |
| Tuning knobs | `MLX_NGRAM_AR_BATCH`, `MLX_NGRAM_ADAPTIVE_HI`, `MLX_NGRAM_ADAPTIVE_LO`, `MLX_NGRAM_ADAPTIVE_WINDOW`, `MLX_NGRAM_MULTI_CANDIDATE`, `MLX_NGRAM_MULTI_LOOKAHEAD`, `MLX_NGRAM_DOMINANCE`, `MLX_NGRAM_STRICT_GREEDY`, `MLX_NGRAM_STRICT_EPSILON` — fine-grained knobs on the routing / acceptance / fallback logic. Keep defaults unless A/B-tuning. See `NgramSpeculativeDecoding.swift` for per-knob defaults and rationale. |

### Prefix-cache env knobs (spec 017)

| Variable | Effect |
|---|---|
| `MLX_PREFIX_CACHE=1` | Force the cross-request prefix KV cache on for any request (overrides `prefixCacheEnabled`). |
| `MLX_PREFIX_CACHE=0` | Force off. |
| `MLX_PREFIX_CACHE_DISK=1` | Enable the opt-in L2 disk cache at `~/.cache/mlx-swift-lm/prefix/` (overrides `prefixCacheDiskEnabled`). Off by default. |
| `MLX_PREFIX_CACHE_DEBUG=1` | Verbose snapshotter trace lines (`[PREFIX-CACHE-DEBUG]`). Diagnostic only. |

### Bench-only env vars

The bench harness (`./scripts/benchmark.sh`) reads a number of
`MLX_BENCH_*` and `MLX_METAL_PROFILE` env vars that aren't part of the
public library API. They're documented in
[`benchmarks/README.md`](../benchmarks/README.md) — don't treat them as
inference-tuning knobs.

## See also

- [llm/using.md](llm/using.md) — high-level `ChatSession` surface.
- [kv-cache.md](kv-cache.md) — KV-cache compression algorithms, the
  full TurboQuant env-var set, and how to pass a custom cache.
- [memory-management.md](memory-management.md) — wired-memory budgets +
  `MLX_MEMORY_LIMIT` / `MLX_SMART_MEMORY`.
- [batched-decoding.md](batched-decoding.md) — `generateBatched(...)`
  and how batch size interacts with the parameters above.
