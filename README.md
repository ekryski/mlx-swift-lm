# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/v2-to-v3-migration) for detailed instructions on upgrading.
>
> If that page shows a 404 you can view the source:
> [upgrading](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/v2-to-v3-migration.md) 
> and [using](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md)

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.
- Multiple KV compression algorithms at different bit sizes (ie. Affine, Turbo - both symmetric and asymmetric K/V)

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).

## Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose an [integration package for downloaders and tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


## Quick Start

After installing the package you can use LLMs to generate content with only a few lines
of code.  (Note: the exact line to load the model depends on the [integration package](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages)).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

// customize this line per the integration package
let model = try await loadModelContainer(
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

## Upgrading

### Migrating to Version 4

See [v3-to-v4-migration.md](Libraries/MLXLMCommon/Documentation.docc/v3-to-v4-migration.md) for the full upgrade guide (KV-cache architecture rewrite under spec 006: class renames, typed `KVCache.CompressionAlgorithm`, `maybeQuantizeKVCache` removal, new `makeAttentionCache(...)` factory).

### Migrating to Version 3

See [v2-to-v3-migration.md](Libraries/MLXLMCommon/Documentation.docc/v2-to-v3-migration.md) for the v2→v3 upgrade (decoupled tokenizer + downloader, new imports, loading API changes, renamed methods).

## Development Setup

Run once after cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

### Why `make` instead of `swift build`

Swift Package Manager (SPM) does not handle several parts of this repo's build pipeline:

- **Metal shaders** — SPM cannot compile `.metal` files. The MLX Metal kernels must be compiled separately into `mlx.metallib` and copied into the test bundle.
- **Submodule staleness** — When you modify C/C++ files deep in git submodules (`mlx-swift` → `mlx` → `mlx-c`), SPM's build cache may not detect the change. It tracks content signatures keyed by the dependency's git revision, so edits within a submodule can go stale.
- **Test bundle regeneration** — `swift build --build-tests` can regenerate the `.xctest` bundle, wiping previously-copied Metal shaders.
- **Stale repository cache** — SPM keeps a global bare-repo cache at `~/Library/Caches/org.swift.swiftpm/repositories/` that survives `swift package reset`. When a tracked branch advances (e.g. our `ekryski/mlx-swift` alpha picks up new submodule pins), that cache can serve a stale revision on the next resolve, producing build errors against missing symbols. `make clean-all` clears the cached entries for this project's three forked deps (`mlx-swift`, `mlx`, `mlx-c`) by inspecting each entry's origin URL — unrelated packages like `mlx-audio-swift` or `mlx-vlm` are left alone.

The project [Makefile](Makefile) wraps SPM and fills these gaps using file-timestamp dependency tracking. It only rebuilds what actually changed:

| What changed | What rebuilds | What stays cached |
|---|---|---|
| A `.metal` or kernel `.h` file | Metal shaders only | SPM targets |
| A `.cpp`/`.c`/`.h` in `mlx` or `mlx-c` | SPM's Cmlx target only | Swift targets, Metal |
| Swift sources | SPM incremental rebuild | Metal |
| Nothing | Artifact copy only (~instant) | Everything |

After every build, the metallib is copied to the release directory and test bundle automatically.

You do not need to use `make` directly for typical workflows — `setup-dev.sh` and `benchmark.sh` both call it internally. For manual builds or targeted rebuilds, see `make help`.

#### Manual builds

For targeted rebuilds when working on specific parts of the stack:

```bash
make                # Full incremental build (only rebuilds what changed)
make metal          # Recompile Metal shaders only
make spm            # Swift build only (with Cmlx cache invalidation)
make status         # Show what's built and what's stale
make doctor         # Verify resolved deps have required symbols + submodule pin consistency
make clean-cmlx     # Force SPM to recompile C/C++ on next build
make help           # Full reference
```

`make doctor` is a fast offline diagnostic that catches the two common ways the dep chain goes silently stale before you sit through a minute-long build that fails at link time:

1. **Stale SPM pin** — the resolved `mlx-swift` checkout is too old to have a symbol our Swift code calls (e.g. `MLXFast.turboBulkDequantRotated`). Manifests at compile time as `type 'MLXFast' has no member 'X'`.
2. **Submodule drift** — `mlx-swift`'s `Source/Cmlx/mlx` or `Source/Cmlx/mlx-c` checkouts have advanced past the SHAs the gitlink expects (e.g. someone manually `git pull`-ed inside a submodule).

For each it prints either OK or a one-line remediation hint. Add new `MLXFast` symbols our code requires to `DOCTOR_REQUIRED_SYMBOLS` in the Makefile so they are checked too.

## Testing

Tests require Metal and must be run via Xcode's build system so that the MLX Metal shaders (`default.metallib`) are built and available. Running `swift test` directly will fail with "Failed to load the default metallib" because SwiftPM does not build Metal shaders.

In Xcode: open the package and run tests (Cmd-U), or from the command line:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

Benchmarks are an exception — they're gated behind a Swift Testing filter (`swift test --filter benchmark`) and invoked via `./scripts/benchmark.sh`, which handles the Metal shader build via `make`. See the next section.

## Benchmarking

Inference benchmarks measure prefill throughput, decode tok/s, TTFT, perplexity, KL divergence, and GPU memory across models, quantization levels, and KV cache configurations. Benchmarks run in **release mode** and write markdown reports to `benchmarks/`.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete CLI reference, methodology details, and environment variable API.

This resolves Swift packages, compiles Metal shaders, does an initial release build, and copies all artifacts into the test bundle. After setup, all benchmark commands work immediately.

### Basic benchmark

Benchmark any registered model family or HuggingFace repo directly:

```bash
# Known model family (downloads automatically on first run)
./scripts/benchmark.sh --model qwen35-0.8b --context 128

# Any HuggingFace model by repo ID
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128

# With perplexity tracking
./scripts/benchmark.sh --model mlx-community/Qwen3-4B-4bit --context 128 --ppl

# Multi-model sweep — all rows land in one hardware-dated report file
./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2 --quick
```

Results are saved as hardware-dated markdown files in `benchmarks/`, one file per sweep per day (e.g. `benchmarks/m1-max-64gb-2026-04-16.md`). All runs for the same model across the sweep are grouped together in that file.

For more advanced benchmark combinations and options see [`benchmarks/README.md`](benchmarks/README.md).

### Choosing a deployment shape (Apple Silicon)

The 416-run batched sweep in [`benchmarks/batched-sweep-2026-04-29.md`](benchmarks/batched-sweep-2026-04-29.md) tells a clear story for inference workloads on Apple Silicon. Two patterns are worth promoting as defaults:

**For batched serving (B>1) — pick MoE.** MoE models with active-parameter sparsity (only a fraction of weights run per token) leave plenty of GeMM headroom for batching to amortize. Best observed speedups at B=2 ctx=1024:

| Model | B=1 tok/s | B=2 agg tok/s | Speedup |
|---|---:|---:|---:|
| `gemma4-26b-a4b` (active 4B) | 28.0 | 39.2 | **1.40×** |
| `qwen35-35b-a3b` (active 3B) | 64.6 | 85.8 | **1.32×** |
| `nemotron-30b-a3b` (active 3B) | 75.4 | 80.5 | 1.07× |

Dense models at the same parameter count peak around 1.05–1.20× and lose ground above ctx=4k. `gemma4-26b-a4b` is the standout — it holds **1.30× at ctx=8k**, the only model in the registry to do so.

**For single-stream (B=1) on memory-constrained hardware — `gemma4-e2b` / `qwen35-2b` / `qwen35-4b`.** Dense decode tok/s stays high through 16k context with peak GPU usage well under what a 16 GB Mac can wire (under 5 GB at ctx=16k for 9B 4-bit B=1). Pair with `--kv turbo4v2` if you want to fit a larger model class at the same memory budget.

**TurboQuant note.** At B=1 turbo4v2 closely matches no-quant on every model (within 5%). At B>1 long-context the gap widens significantly (0.60× on 9B at ctx=32k B=2) — this is a known regression filed in the follow-up issue list. Use turbo4v2 when memory matters; skip it when speed matters and you have RAM.

## Configuration

Knobs that change runtime behavior at the inference level — not the bench harness. Bench/profiling-only flags (`MLX_BENCH_*`, `MLX_METAL_PROFILE`) live in [`benchmarks/README.md`](benchmarks/README.md).

### `GenerateParameters` (programmatic API)

Set on the `GenerateParameters` struct passed to `generate(...)`. Defaults shown in parentheses.

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
| `kvScheme` | `nil` | KV-cache compression scheme (`"turbo4"`, `"turbo4v2"`, `"affine4"`, etc.). When unset, models load `StandardKVCache` or their architecture default. Parsed via `KVCache.CompressionAlgorithm.init?(_:)`. |
| `kvBits` / `kvGroupSize` | `nil` / `64` | Generic quantized KV cache (independent of `kvScheme`). |
| `quantizedKVStart` | `0` | First token at which generic-quant KV kicks in. |
| `prefillStepSize` | `nil` | Chunk size for long-prompt prefill — lower = lower peak GPU at the cost of prefill throughput. Falls back to the model's `defaultPrefillStepSize` (Qwen35 dense 1024 / Qwen35 MoE 4096 / Gemma 4 4096 / GPT-OSS 2048 / Nemotron 1024). M1 Max sweep on Qwen 2B / ctx=16k / `--kv none` (peak / prefill tok/s): 256 → 2.26 GB / 1106 · 512 → 2.27 GB / 1132 · 1024 → 2.38 GB / 1148 · 2048 → 2.51 GB / 1182. |
| `turboBoundarySkip` | `2` | TurboQuant codebook boundary skip; lowers raise PPL slightly but speed up encode. |
| `ngramSize` | `0` | Prompt-lookup n-gram speculative decoding (n-gram length). `0` disables. Net win only on repetitive output (code, templates). |
| `maxNgramDraftTokens` | `0` | Max draft tokens per speculation round. Pair with `ngramSize`. |
| `reasoningEffort` | `nil` | Hint passed to chat templates that support it (`"low"`, `"medium"`, `"high"`). |
| `thinkStartTokenId` / `thinkEndTokenId` | `nil` | Token IDs for thinking-phase boundaries; enables phase-separated logprob tracking when set. |
| `thinkingPhasePrefilled` | `false` | Set when the prompt already opens with `<think>`. |
| `harmonyChannelMarkerTokenId` / `harmonyThinking…` / `harmonyGeneration…` | `nil` / `[]` / `[]` | GPT-OSS harmony-format phase machine. |
| `collectPerTokenData` | `false` | Store per-token logprobs / IDs / phase labels for downstream KLD. |
| `trackPerplexity` | `false` | Accumulate logprobs for end-of-run PPL. |

### KV cache toggles

| Flag | Default | Notes |
|---|---|---|
| `TurboQuantizedKVCache.useCompressedAttention` | `true` | **B path is the default as of [`c5ca7a3`](https://github.com/ekryski/mlx-swift-lm/commit/c5ca7a3).** Decode runs the fused bulk-dequant Metal kernel + `MLXFast.scaledDotProductAttention` on the compressed cache — within ~3% of `--kv none` on Qwen 9B and faster than `--kv none` at short context. Set the constructor flag to `false` (or use the env override below) for the historic A path that keeps a raw fp16 cache. |
| `BatchedKVCache.maxBatch` | constructor arg | Max simultaneous decode streams sharing one cache. Must match the request shape. |
| `StandardKVCache.reserve(_:)` | not called | **Opt-in workload-size hint** for windowed eviction. Pre-allocates the rotating buffer to a known size up front (typically `prompt_length + maxTokens`) instead of growing in `step`-sized chunks (default `step=256`). Idempotent: only takes effect before the first write. Clamped to `maxCacheSize`, floored at `step`. No-op when eviction is `.unbounded`. Most useful when `maxKVSize` is generous (e.g. a model's full context window) but the actual workload uses only a fraction — the hint sizes the buffer to the workload instead of growing incrementally or pre-allocating the full window. Behaviour is unchanged when never called. <br/>`let cache = StandardKVCache(maxSize: 4096)` <br/>`cache.reserve(promptLen + maxTokens)`<br/><br/>**Programmatic construction:** `makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2), eviction: .unbounded)` — single source of truth for `kvScheme` string parsing. |

### Environment variable overrides

These env vars take precedence over the constructor / `GenerateParameters` defaults. They exist for **diagnostics, A/B testing, and tuning** — not as the primary user-facing API. Set them in the shell before launching an inference process; they are read once at first use and cached.

#### Cache / attention path

| Variable | Effect |
|---|---|
| `TURBO_COMPRESSED_ATTENTION=0` | Force the raw-fp16 working buffer "A" path globally — overrides the constructor's `useCompressedAttention=true` default. `TURBO_COMPRESSED_ATTENTION=1` forces the compressed "B" path. If unset honors the constructor value ("B" path compressed attention by default). From profiling the "A" path is faster but bloats memory because we have our Turbo compressed KV cache and a constant working fp16 buffer sent to the default SDPA metal kernel. With compressed attention it is more true to the Turbo compression algorithm because the compressed KV cache is accessed by either a custom fused Dequant + SDPA metal kernel (default due to speed) or the TurboFlash metal kernel (can be enabled by setting `TURBO_DEQUANT_SDPA=1`), both of which are currently slower than the default MLX SDPA. **The trade off is memory savings vs. speed.** |
| `TURBO_DEQUANT_SDPA=0` | Disable the fused-dequant + matrix-engine SDPA path; falls back to TurboFlash. Useful when sweeping over very long contexts where TurboFlash's per-token bit-unpack still wins (ie. ≥ 24k on Qwen 9B / Nemotron-class). |
| `TURBO_DEQUANT_JIT=1` | Force the JIT'd `MLXFast.metalKernel` bulk-dequant path instead of the precompiled `MLXFast.turboBulkDequantRotated`. Use for A/B comparison when iterating on the dequant kernel itself; the precompiled C code kernel path is the shipping default since it avoids the first-dispatch PSO compile in TTFT. |
| `TURBO_FLASH_BLOCK_SIZE=N` | Pin TurboFlash pass1's kernel block size (override the adaptive `tokenCount/32` heuristic). Powers of two only. A performance knob to tune for particular models and context sizes. |
| `TURBO_FLASH_NR0=N` | Set the number of query Rows handled per SIMD group in the first pass (Index/Pass `0`) of the TurboFlash decode kernel. Default `2` ; `1` falls back to single-row in first pass. The default `2` register cost (dim=128, NR0=2): ~24 extra floats per thread vs NR0=1 fits comfortably inside Apple's 96-register/thread budget. NR0=4/8 might pay off on bigger register files/future architecture but aren't instantiated in the metallib yet. Not really something to mess with unless profiling new hardware tuning command buffers. |
| `TURBO_SPARSE_V_THRESHOLD=N` | Skip-V threshold for the separated `mseWeightedSum` kernel. Default `1e-6`. `0.0` disables skip; `1e-4` is too aggressive and clips long-context attention. |
| `TURBO_DEBUG=1` | Verbose logging from `compressedAttention` (offsets, shapes, key-norm sanity). Only enable for short debugging because it will impact speed. |

#### Model-specific

| Variable | Effect |
|---|---|
| `GEMMA4_FUSED_NORM_ROPE=0` | Disable the fused norm + RoPE Metal kernel on Gemma 4 (default on). For A/B testing. May be removed in future. |
| `MLX_COMPILE_SHARED_MLP=1` / `=0` | Force the Gemma 4 shared-MLP `compile(shapeless:)` wrapper on / off. The architecture default is on for some configurations and off where the wrapper costs ~10 % decode. |
| `GDN_EVAL_INTERVAL=N` | GatedDelta (Qwen3.5 / Nemotron-H) prefill eval cadence. Default `128`. Lower values sync the GPU pipeline more aggressively; higher values reduce sync overhead at the cost of less granular timing. |

#### Wired memory

`WiredMemoryUtils.resolveTicket(...)` honours these env vars when sizing a wired-memory ticket. Bench harness uses this directly; library callers can opt in via the same API.

| Variable | Effect |
|---|---|
| `MLX_MEMORY_LIMIT` | Explicit wired-memory limit. Accepts plain bytes or human-friendly units (`32g`, `32GB`, `512m`, `4k`, `1.5g`), case-insensitive. Bypasses the smart estimator and `MLX_SMART_MEMORY`. Clamped to `GPU.maxRecommendedWorkingSetBytes()` when available. |
| `MLX_SMART_MEMORY` | `0` disables the model-aware estimator (then ticket falls back to `GPU.maxRecommendedWorkingSetBytes()`). Anything else, including unset, leaves the smart estimator on (the default). The estimator computes `weights + kv(maxTokens × batchSize, kvScheme) + workspace` from the loaded model — accurate when callers pass `kvHeadsOverride`/`headDimOverride`, conservative heuristic otherwise. |

## Publishing a Release

See [publishing-a-release.md](Libraries/MLXLMCommon/Documentation.docc/publishing-a-release.md) for the manual-trigger release pipeline (workflow inputs, semver guidance, hotfix branching, cross-repo coordination across the `mlx-c → mlx → mlx-swift → mlx-swift-lm` chain).