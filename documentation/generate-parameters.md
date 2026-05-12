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
| `presencePenalty` / `frequencyPenalty` | `nil` | OpenAI-style penalties. |
| `maxTokens` | `nil` | Upper bound on generated tokens. |

## KV cache

| Field | Default | Notes |
|---|---|---|
| `maxKVSize` | `nil` | Hard cap on KV cache tokens; backs `StandardKVCache` in `.window` eviction mode. |
| `compressionAlgorithm` | `.none` | KV-cache compression (`.affine(bits:groupSize:)` / `.turbo(keyBits:valueBits:)` / `.none`). Parsed via `KVCache.CompressionAlgorithm.init?(_:)` from a string like `"turbo4v2"`. See [kv-cache.md](kv-cache.md). |
| `turboBoundarySkip` | `2` | TurboQuant codebook boundary skip; lower raises PPL slightly but speeds up encode. |

## Prefill / throughput

| Field | Default | Notes |
|---|---|---|
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill — lower = lower peak GPU at the cost of prefill throughput. Falls back to the model's `defaultPrefillStepSize` (Qwen3.5 dense `1024` / Qwen3.5 MoE `4096` / Gemma 4 `4096` / GPT-OSS `2048` / Nemotron `1024`). M1 Max sweep on Qwen 2B / ctx=16k / `--kv none`: 256 → 2.26 GB / 1106 tok/s · 512 → 2.27 GB / 1132 · 1024 → 2.38 GB / 1148 · 2048 → 2.51 GB / 1182. |

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

All four fields default to off — opt in per call site or via env var.

| Field | Default | Notes |
|---|---|---|
| `prefixCacheEnabled` | `false` | Master toggle. Env override: `MLX_PREFIX_CACHE=1`. When true, `generate(...)` consults `PrefixKVCache.shared` for a longest-prefix hit, hydrates if found, and on stream completion snapshots the cache at the stable-prefix boundary. |
| `prefixCachePolicy` | `IdentityPolicy()` | Decides where the stable prefix ends. Pass `FixedTrimPolicy(trimSuffix: N)` for chat workloads where the trailing N tokens are template boilerplate, or `LastAssistantOpenerPolicy(opener: …)` to scan for the chat-template assistant opener directly (use the convenience `init?(opener:tokenizer:)` with one of `AssistantOpener.qwenChatML` / `.gemma4` / `.gptOSSHarmony` / `.custom(_)`). |
| `prefixCacheModelID` | `nil` | Stable model identifier used to scope snapshots in the shared cache. Apps that share `PrefixKVCache.shared` across multiple models **must** set this to avoid cross-model snapshot reuse; single-model apps can leave it nil. Typically `ModelConfiguration.name` or the HuggingFace repo ID. |
| `prefixCacheDiskEnabled` | `false` | Promote / persist snapshots to disk at `~/.cache/mlx-swift-lm/prefix/`. **Off by default** — won't bloat disk unless explicitly enabled. Env override: `MLX_PREFIX_CACHE_DISK=1`. On L1 miss, falls through to disk; on hit, promotes back into L1. |

### Quick example

```swift
import MLXLMCommon

// Single-model app — minimal opt-in:
var params = GenerateParameters()
params.prefixCacheEnabled = true
params.prefixCachePolicy = FixedTrimPolicy(trimSuffix: 4)  // trim assistant opener
params.prefixCacheModelID = "Qwen/Qwen3.5-9B-Instruct"

let stream = try await container.generate(input: lmInput, parameters: params)
```

For diagnostics, read `PrefixKVCache.shared.stats` after a request to see hits / misses / saved prefill tokens. `PrefixKVCache.shared.resetStats()` zeros counters without clearing entries; `clear()` empties entries. See [llm/using.md](llm/using.md) for the `ChatSession`-level surface.

**Limitations** (documented in detail in `specs/017-prefix-kv-cache.md`):
- GPT-OSS-20B's sliding-window=128 layers wrap mid-generation; no cache benefit (skip is silent).
- Hybrid Qwen 3.5 SSM-layer snapshots are slightly stale; attention layers are exact and carry the dominant prefill cost.

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
attention-path env vars (`TURBO_*`) see [kv-cache.md](kv-cache.md).

### Model-specific perf knobs

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. May be removed in future. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. The architecture default is on for some configurations and off where the wrapper costs ~10 % decode (e.g. 26B-A4B MoE). |
| `GDN_EVAL_INTERVAL=N` | GatedDeltaNet (Qwen 3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

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
