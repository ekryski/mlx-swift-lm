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
