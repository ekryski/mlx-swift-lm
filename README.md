# MLX Swift LM

A Swift package for running large language models (LLMs), vision-language
models (VLMs), and embedding models on Apple Silicon, built on
[MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key capabilities:

- ~50 LLM and ~20 VLM reference architectures, plus ~3 embedder families
- High-level `ChatSession` API for chat-shaped use cases (text + images +
  video, multi-turn, streaming)
- Lower-level `ModelFactory` + `generate(...)` API for batched decode,
  speculative decoding, custom sampling, and direct cache management
- Multiple KV-cache compression algorithms (Affine, TurboQuant — symmetric
  and asymmetric K/V) plus a typed `compressionAlgorithm` parameter
- Wired-memory coordination so multiple inference tasks can share one GPU
  budget without stepping on each other
- Tool-call parsers for the major chat-template families (Qwen, Llama 3,
  Pythonic, Harmony, Hermes)

For example apps and tools that consume this package, see [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Everything lives under [`documentation/`](documentation/).

Start here:

- **[Installation](documentation/installation.md)** — SwiftPM / Xcode setup,
  picking integration packages
- **[Quick start](documentation/quickstart.md)** — generate text in 5 lines
  (LLM and VLM)
- **[Architecture](documentation/architecture.md)** — module layout and the
  LLM ↔ VLM consolidation map
- **[Models](documentation/models.md)** — supported architectures, registries,
  per-model known gaps

Then drill in:

| LLM | VLM | Embeddings |
|---|---|---|
| [Overview](documentation/llm/overview.md) | [Overview](documentation/vlm/overview.md) | [Overview](documentation/embeddings/overview.md) |
| [Using an LLM](documentation/llm/using.md) | [Using a VLM](documentation/vlm/using.md) | |
| [Evaluation](documentation/llm/evaluation.md) | | |
| [Adding an LLM](documentation/llm/adding-a-model.md) | [Adding a VLM](documentation/vlm/adding-a-model.md) | |

Cross-cutting topics:

- [KV cache + compression](documentation/kv-cache.md)
- [Wired memory coordination](documentation/wired-memory.md)
- [Speculative decoding](documentation/speculative-decoding.md)
- [Migrating to v3](documentation/migrations/v2-to-v3.md) /
  [Migrating to v4](documentation/migrations/v3-to-v4.md)
- [Publishing a release](documentation/publishing-a-release.md)

For local development:

- [Developing in mlx-swift-lm](documentation/developing/developing.md)
- [Porting models from Python](documentation/developing/porting.md)
- [Benchmarking](documentation/developing/benchmarking.md)

## Quick start (5 lines)

```swift
import MLXLLM
import MLXLMCommon

let model = try await loadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

For a VLM, replace `LLMRegistry.gemma3_1B_qat_4bit` with
`VLMRegistry.qwen2_5VL3BInstruct4Bit` and pass `image:` into `respond`.
Full walkthrough in [`documentation/quickstart.md`](documentation/quickstart.md).

## Installation

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.32.0-alpha")),
```

You also need a downloader and a tokenizer. Pick one of the three integration
paths — see [`documentation/llm/using.md § Picking an integration`](documentation/llm/using.md#picking-an-integration).

Full setup in [`documentation/installation.md`](documentation/installation.md).

## Upgrading

- **From 3.x → 4.x** — KV-cache architecture rewrite under spec 006: class
  renames, typed `KVCache.CompressionAlgorithm`, `maybeQuantizeKVCache`
  removed in favour of `makeAttentionCache(...)`. See
  [`documentation/migrations/v3-to-v4.md`](documentation/migrations/v3-to-v4.md).
- **From 2.x → 3.x** — decoupled tokenizer + downloader, new imports,
  loading API changes. See
  [`documentation/migrations/v2-to-v3.md`](documentation/migrations/v2-to-v3.md).

## Choosing a deployment shape (Apple Silicon)

The 416-run batched sweep in [`benchmarks/batched-sweep-2026-04-29.md`](benchmarks/batched-sweep-2026-04-29.md)
tells a clear story for inference workloads on Apple Silicon. Two patterns
are worth promoting as defaults:

**For batched serving (B>1) — pick MoE.** MoE models with active-parameter
sparsity (only a fraction of weights run per token) leave plenty of GeMM
headroom for batching to amortize. Best observed speedups at B=2 ctx=1024:

| Model | B=1 tok/s | B=2 agg tok/s | Speedup |
|---|---:|---:|---:|
| `gemma4-26b-a4b` (active 4B) | 28.0 | 39.2 | **1.40×** |
| `qwen35-35b-a3b` (active 3B) | 64.6 | 85.8 | **1.32×** |
| `nemotron-30b-a3b` (active 3B) | 75.4 | 80.5 | 1.07× |

Dense models at the same parameter count peak around 1.05–1.20× and lose
ground above ctx=4k. `gemma4-26b-a4b` is the standout — it holds **1.30× at
ctx=8k**, the only model in the registry to do so.

**For single-stream (B=1) on memory-constrained hardware —
`gemma4-e2b` / `qwen35-2b` / `qwen35-4b`.** Dense decode tok/s stays high
through 16k context with peak GPU usage well under what a 16 GB Mac can
wire (under 5 GB at ctx=16k for 9B 4-bit B=1). Pair with `--kv turbo4v2` if
you want to fit a larger model class at the same memory budget.

**TurboQuant note.** At B=1 turbo4v2 closely matches no-quant on every
model (within 5%). At B>1 long-context the gap widens significantly (0.60×
on 9B at ctx=32k B=2) — this is a known regression filed in the follow-up
issue list. Use turbo4v2 when memory matters; skip it when speed matters
and you have RAM.

## `GenerateParameters` reference

Knobs that change runtime inference behaviour. Set on the `GenerateParameters`
struct passed to `generate(...)`. Defaults shown in parentheses.

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
| `maxKVSize` | `nil` | Hard cap on KV cache tokens; backs `StandardKVCache` in `.window` eviction mode. |
| `compressionAlgorithm` | `.none` | KV-cache compression (`.affine(bits:groupSize:)` / `.turbo(keyBits:valueBits:)` / `.none`). Parsed via `KVCache.CompressionAlgorithm.init?(_:)` from a string like `"turbo4v2"`. See [kv-cache.md](documentation/kv-cache.md). |
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill. Lower = lower peak GPU at the cost of prefill throughput. Falls back to the model's `defaultPrefillStepSize` (Qwen35 dense 1024 / Qwen35 MoE 4096 / Gemma 4 4096 / GPT-OSS 2048 / Nemotron 1024). M1 Max sweep on Qwen 2B / ctx=16k / `--kv none`: 256 → 2.26 GB / 1106 tok/s · 1024 → 2.38 GB / 1148 · 2048 → 2.51 GB / 1182. |
| `turboBoundarySkip` | `2` | TurboQuant codebook boundary skip; lowers raise PPL slightly but speed up encode. |
| `ngramSize` | `0` | Prompt-lookup n-gram speculative decoding (n-gram length). `0` disables. Net win only on repetitive output (code, templates). |
| `maxNgramDraftTokens` | `0` | Max draft tokens per speculation round. Pair with `ngramSize`. |
| `reasoningEffort` | `nil` | Hint passed to chat templates that support it (`"low"` / `"medium"` / `"high"`). |
| `thinkStartTokenId` / `thinkEndTokenId` | `nil` | Token IDs for thinking-phase boundaries; enables phase-separated logprob tracking when set. |
| `thinkingPhasePrefilled` | `false` | Set when the prompt already opens with `<think>`. |
| `harmonyChannelMarkerTokenId` / `harmonyThinking…` / `harmonyGeneration…` | `nil` / `[]` / `[]` | GPT-OSS harmony-format phase machine. |
| `collectPerTokenData` | `false` | Store per-token logprobs / IDs / phase labels for downstream KLD. |
| `trackPerplexity` | `false` | Accumulate logprobs for end-of-run PPL. |

### KV cache toggles

| Flag | Default | Notes |
|---|---|---|
| `TurboQuantizedKVCache.useCompressedAttention` | `true` | **B path is the default as of [`c5ca7a3`](https://github.com/ekryski/mlx-swift-lm/commit/c5ca7a3).** Decode runs the fused bulk-dequant Metal kernel + `MLXFast.scaledDotProductAttention` on the compressed cache — within ~3% of `--kv none` on Qwen 9B and faster than `--kv none` at short context. Set the constructor flag to `false` (or use `TURBO_COMPRESSED_ATTENTION=0`) for the historic A path that keeps a raw fp16 cache. |
| `BatchedKVCache.maxBatch` | constructor arg | Max simultaneous decode streams sharing one cache. Must match the request shape. |
| `StandardKVCache.reserve(_:)` | not called | **Opt-in workload-size hint** for windowed eviction. Pre-allocates the rotating buffer to a known size up front (typically `prompt_length + maxTokens`) instead of growing in `step`-sized chunks (default `step=256`). Idempotent: only takes effect before the first write. Clamped to `maxCacheSize`, floored at `step`. No-op when eviction is `.unbounded`. <br/>`let cache = StandardKVCache(maxSize: 4096); cache.reserve(promptLen + maxTokens)`. <br/><br/>**Programmatic construction:** `makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2), eviction: .unbounded)` — single source of truth for `compressionAlgorithm` string parsing. |

### Environment-variable overrides

These take precedence over the constructor / `GenerateParameters` defaults.
They exist for **diagnostics, A/B testing, and tuning** — not as the
primary user-facing API. Set in the shell before launching an inference
process; read once at first use and cached.

#### Cache / attention path

| Variable | Effect |
|---|---|
| `TURBO_COMPRESSED_ATTENTION=0` | Force the raw-fp16 working buffer "A" path globally. `=1` forces the compressed "B" path. Unset honours the constructor (B by default). The A path is faster but bloats memory; the B path is more true to the Turbo compression algorithm because the compressed KV cache is accessed by either a custom fused Dequant + SDPA Metal kernel (default) or the TurboFlash kernel. **Trade-off: memory savings vs. speed.** |
| `TURBO_DEQUANT_SDPA=0` | Disable the fused-dequant + matrix-engine SDPA path; fall back to TurboFlash. Useful when sweeping very long contexts where TurboFlash's per-token bit-unpack still wins (≥ 24k on Qwen 9B / Nemotron-class). |
| `TURBO_DEQUANT_JIT=1` | Force the JIT'd `MLXFast.metalKernel` bulk-dequant path instead of the precompiled `MLXFast.turboBulkDequantRotated`. Use for A/B comparison when iterating on the dequant kernel itself. |
| `TURBO_FLASH_BLOCK_SIZE=N` | Pin TurboFlash pass1's kernel block size (override the adaptive `tokenCount/32` heuristic). Powers of two only. |
| `TURBO_FLASH_NR0=N` | Number of query rows handled per SIMD group in the first pass of TurboFlash decode. Default `2`; `1` falls back to single-row first pass. |
| `TURBO_SPARSE_V_THRESHOLD=N` | Skip-V threshold for the separated `mseWeightedSum` kernel. Default `1e-6`. `0.0` disables; `1e-4` is too aggressive and clips long-context attention. |
| `TURBO_DEBUG=1` | Verbose logging from `compressedAttention` (offsets, shapes, key-norm sanity). Only enable for short debugging — impacts speed. |

#### Model-specific

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. Architecture default is on for some configurations and off where the wrapper costs ~10 % decode. |
| `GDN_EVAL_INTERVAL=N` | GatedDelta (Qwen3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

#### Wired memory

`WiredMemoryUtils.resolveTicket(...)` honours these env vars when sizing a
wired-memory ticket. Bench harness uses this directly; library callers can
opt in via the same API.

| Variable | Effect |
|---|---|
| `MLX_MEMORY_LIMIT` | Explicit wired-memory limit. Accepts plain bytes or human-friendly units (`32g`, `32GB`, `512m`, `4k`, `1.5g`), case-insensitive. Bypasses the smart estimator and `MLX_SMART_MEMORY`. Clamped to `GPU.maxRecommendedWorkingSetBytes()` when available. |
| `MLX_SMART_MEMORY` | `0` disables the model-aware estimator (then ticket falls back to `GPU.maxRecommendedWorkingSetBytes()`). Anything else, including unset, leaves the smart estimator on (the default). The estimator computes `weights + kv(maxTokens × batchSize, compressionAlgorithm) + workspace` from the loaded model — accurate when callers pass `kvHeadsOverride`/`headDimOverride`, conservative heuristic otherwise. |

Bench-only env vars (`MLX_BENCH_*`, `MLX_METAL_PROFILE`) live in
[`benchmarks/README.md`](benchmarks/README.md).

## Building locally

After cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

For the full build pipeline reference (why `make` instead of `swift build`,
incremental rebuilds, `make doctor`, dep-chain diagnostics) see
[`documentation/architecture.md § Build pipeline`](documentation/architecture.md#build-pipeline)
and [`documentation/developing/developing.md`](documentation/developing/developing.md).

## Testing

`mlx-swift-lm` tests run in **release** config (the Metal library is built
into the release path; debug-config `swift test` will fail with "Failed to
load the default metallib"):

```bash
swift test -c release
```

Or via Xcode: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'`.

Benchmarks are gated behind a Swift Testing filter and run via
`./scripts/benchmark.sh`. See
[`documentation/developing/benchmarking.md`](documentation/developing/benchmarking.md).
