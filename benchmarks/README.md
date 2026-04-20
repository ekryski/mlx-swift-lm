# Inference Benchmarks

Automated benchmarks for MLX Swift LM inference across model families, weight quantizations, and KV cache compression strategies running on Apple Silicon.

The CLI (`benchmark.sh`) is designed to be language-agnostic — all configuration is passed via environment variables, making it straightforward to add backends in other languages (Python, Java) for cross-platform benchmarking.

Benchmark reports in this directory also serve as **baselines** — periodic full-matrix snapshots of prefill / decode tokens-per-second across the supported model range on a specific piece of hardware at a specific point in time. Use them when:

- **Diagnosing a perf regression** — compare current numbers against the most recent baseline on matching hardware.
- **Landing a kernel or framework change** — re-run the affected rows and update the baseline if the delta is material.
- **Picking a model for a target device** — the TL;DR table shows prefill/decode at 1k context and whether 8k coherency holds.

## Setup

Run once after cloning (or after fetching new `mlx-swift` changes):

```bash
./scripts/setup-dev.sh
```

This resolves Swift packages, compiles Metal shaders, builds the prefill bridge dylib, and does an initial release build. After setup, all benchmark commands work immediately.

Internally, `setup-dev.sh` and `benchmark.sh` both call `make build-tests`, which handles the full build pipeline incrementally — only rebuilding what actually changed. See the [main README](../README.md#why-make-instead-of-swift-build) for details on why `make` is used.

If you are iterating on C/C++ code in the `mlx` or `mlx-c` submodules and benchmarks are using stale artifacts, run:

```bash
make clean-cmlx     # Invalidate SPM's C/C++ cache
make status         # Verify what's built
```

Then re-run your benchmark — it will recompile only the C/C++ target.

## Quick Start

```bash
# Simple chat evaluation (default method)
./scripts/benchmark.sh --model qwen35-0.8b

# Simple eval with perplexity tracking
./scripts/benchmark.sh --model qwen35-0.8b --ppl

# Context-scaling summarization (3 quick context sizes)
./scripts/benchmark.sh --model qwen35-9b --method summarization --quick

# WikiText-2 perplexity at a specific context
./scripts/benchmark.sh --model qwen35-0.8b --method wikitext2 --context 1024

# Needle-in-a-haystack
./scripts/benchmark.sh --model qwen35-9b --method niah --context 4096

# With KLD quality metrics
./scripts/benchmark.sh --model qwen35-0.8b --method summarization --kv affine4 --kld

# Full matrix: all quants × all KV configs
./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all --quick

# Multi-model sweep: two models, two KV configs, quick contexts — one output file
./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2 --quick

# Two methods against the same model
./scripts/benchmark.sh --model qwen35-0.8b --method simple,summarization

# GPT-OSS with high-effort reasoning, thinking + PPL tracking
./scripts/benchmark.sh --model gpt-oss-20b --reasoning high --think --ppl
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--model MODELS` | **(required)** Model family / HF repo ID. Comma-separated for multi-model sweeps. | — |
| `--method METHODS` | Benchmark method(s), comma-separated (see [Methods](#methods)) | `simple` |
| `--quant QUANTS` | Weight quantization(s): `bf16`, `8bit`, `4bit`, or `all`. Comma-separated for multiple. | `4bit` |
| `--kv CONFIGS` | KV cache config(s) (see [KV Cache Configurations](#kv-cache-configurations)). Comma-separated or `all`. | `none` |
| `--context SIZES` | Comma-separated context sizes (e.g., `128,1024,4096`) | All 11 sizes |
| `--quick` | Quick mode: 128 + 1024 + 4096 + 32768 tokens only | Off |
| `--ppl` | Track per-token perplexity during generation | Off |
| `--kld` | Compute KL divergence vs bf16/8bit baseline | Off |
| `--baseline` | Auto-select highest-fidelity variant that fits in GPU memory | Off |
| `--batch N` | Run N concurrent generations | `1` |
| `--think` | Enable thinking mode for thinking-capable models | Off |
| `--reasoning EFFORT` | Reasoning effort for models that support it (e.g. GPT-OSS). Values: `low`, `medium`, `high`. Ignored by models without a reasoning-effort setting. | `medium` |
| `--ngram SIZE` | N-gram speculative decoding size. `0` disables speculation entirely (pure autoregressive decode). `3` matches the library's typical-use default (trigram matching, 3-token drafts). Higher values require longer repeated sequences in generated text to hit. Disabled by default so benchmarks measure deterministic decode without accept-rate variance. | `0` |
| `-h`, `--help` | Show usage | — |

**Comma-separated lists** on `--model`, `--method`, `--quant`, and `--kv` produce the full Cartesian product of permutations. Every permutation runs in sequence, and every row lands in the **same** hardware-dated output file (see [Output](#output)), grouped by model. A sweep like:

```bash
./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b \
  --method simple,summarization --quant 4bit --kv none,turbo4v2 --quick
```

runs 2 × 2 × 1 × 2 = 8 permutations and produces one report file covering both models.

`--quant all` expands to `bf16,8bit,4bit`. `--kv all` expands to `none,affine8,affine4,turbo4,turbo4v2,turbo3`. These can be combined with comma lists on the other dimensions.

> **Max speed tip:** For pure throughput measurements, omit `--ppl` and `--kld`. Both flags add significant compute overhead — `--ppl` tracks per-token log-probabilities during generation, and `--kld` loads a second baseline model and runs a full forced-decode pass after generation completes. Leave them off when you only care about tok/s and TTFT.

## Methods

| Method | Description | Context Scaling | Generation | Pass/Fail |
|--------|-------------|:---:|:---:|:---:|
| `simple` | Basic chat prompt — generation speed + PPL | No | Yes | No |
| `summarization` | Pre-sized prompts across context sizes | Yes | Yes | No |
| `wikitext2` | Standard LM perplexity via forced decode on WikiText-2 | Yes | No | No |
| `niah` | Needle-in-a-haystack retrieval at multiple depths | Yes | Yes | Yes |
| `multi-turn` | Multi-turn conversation with name recall | No | Yes | Yes |
| `tool-calling` | Tool call generation and validation | No | Yes | Yes |

## Model Families

All Qwen3.5 models use a hybrid **GatedDeltaNet** architecture: 75% linear attention layers (MambaCache) + 25% standard attention layers (KVCacheSimple), with full attention every 4th layer.

| Family | Short Name | Quantizations | Architecture |
|--------|------------|---------------|--------------|
| Qwen3.5 0.8B | `qwen35-0.8b` | bf16, 8bit, 4bit | GatedDeltaNet |
| Qwen3.5 2B | `qwen35-2b` | bf16, 8bit, 4bit | GatedDeltaNet |
| Qwen3.5 4B | `qwen35-4b` | bf16, 8bit, 4bit | GatedDeltaNet |
| Qwen3.5 9B | `qwen35-9b` | bf16, 8bit, 4bit | GatedDeltaNet |
| Qwen3.5 27B | `qwen35-27b` | bf16, 8bit, 4bit | GatedDeltaNet |
| Qwen3.5 35B A3B | `qwen35-35b-a3b` | bf16, 8bit, 4bit | GatedDeltaNet MoE |
| GPT-OSS 20B | `gpt-oss-20b` | bf16, 4bit | Transformer |
| Nemotron Cascade 2 30B A3B | `nemotron-30b-a3b` (aliases: `nemotron-cascade-2`, `nemotron-cascade-2-30b-a3b`, …) | 8bit, 4bit, nvfp4, mxfp4 | Nemotron H (hybrid Mamba / attention / MoE) |
| Gemma 4 E2B | `gemma4-e2b` | bf16, 8bit, 4bit, mxfp4 | Dense + PLE |
| Gemma 4 E4B | `gemma4-e4b` | bf16, 8bit, 4bit, mxfp4 | Dense + PLE |
| Gemma 4 26B A4B | `gemma4-26b-a4b` | bf16, 8bit, 4bit, mxfp4 | Transformer MoE |
| Gemma 4 31B | `gemma4-31b` | bf16, 8bit, 4bit, mxfp4 | Dense |

Any HuggingFace repo ID can also be passed directly as `--model org/repo-id`.

## Model cache and downloads

Models are downloaded on first use via [`HubClient`](https://github.com/huggingface/swift-huggingface) (through the `#hubDownloader()` macro in [`Tests/Benchmarks/InferenceBenchmark.swift`](../Tests/Benchmarks/InferenceBenchmark.swift)) and cached in the **Python-compatible Hugging Face cache structure**. That means snapshots downloaded by Python's `huggingface-cli` or `huggingface_hub` are read directly by the benchmark harness without re-downloading, and vice versa.

### Cache resolution order

`HubClient` picks the cache directory from the first of these that is set:

1. **`HF_HUB_CACHE`** — absolute path to the cache directory. Highest priority, use this to point at an external drive, a shared team volume, etc.
2. **`HF_HOME`** — HF home directory; the cache is placed at `$HF_HOME/hub`.
3. **Default**: `~/.cache/huggingface/hub` on non-sandboxed macOS. (Sandboxed Apple apps use `Library/Caches/huggingface/hub`; that path does not apply to the benchmark harness, which runs unsandboxed as a test target.)

### Examples

```bash
# Use an external SSD as the cache root (takes precedence over ~/.cache)
export HF_HUB_CACHE="/Volumes/FastSSD/hf-cache"
./scripts/benchmark.sh --model qwen35-0.8b

# Or organise HF state under one parent directory
export HF_HOME="$HOME/work/hf"              # cache ends up at $HF_HOME/hub
./scripts/benchmark.sh --model qwen35-0.8b

# Fall back to the default (~/.cache/huggingface/hub)
unset HF_HUB_CACHE HF_HOME
./scripts/benchmark.sh --model qwen35-0.8b
```

If you set `HF_HUB_CACHE` / `HF_HOME` in your shell rc file, every subsequent `benchmark.sh` invocation and every `swift test` subprocess it spawns will inherit the setting.

### Cache-first behaviour

Every `download(...)` call routes through `HubClient.downloadSnapshot`, which checks for a complete cached snapshot (`{cache}/models--{org}--{name}/snapshots/{rev}/…`) before making any HTTP calls. If the exact `(repo, revision, file globs)` is already on disk, the call returns the local directory URL immediately — progress jumps to 100%, no network. A missing file triggers a native Swift download via `URLSession`; no Python `huggingface-cli` or other external tooling is required.

If you want to guarantee offline behaviour (fail rather than download), that's an upstream feature of `HubClient` via `localFilesOnly: true`. The current harness doesn't expose it as a flag — if you need it, open a tracking issue.

### Private / gated repositories

Authentication is handled by the same environment conventions as the Python clients: `HF_TOKEN` in your environment (or `$HF_HOME/token`) gets picked up automatically by `HubClient.default`. You do not need to touch benchmark code.

### What is cached

Every snapshot download pulls the file globs the model factory asks for — typically `*.safetensors`, `*.json`, and the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `*.jinja`). Non-matching files in the repo (training artifacts, README images, etc.) are not downloaded. Compressed caches are **not** supported — each snapshot is stored as symlinked blobs under `blobs/` plus a `snapshots/{rev}/` tree, matching the Python client layout.

## Context Sizes

Context-scaling methods (`summarization`, `wikitext2`, `niah`) run across 11 sizes by default:

**128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K** tokens

Use `--context` to specify a subset, or `--quick` for 128 + 1024 + 4096.

For non-scaling methods (`simple`, `multi-turn`, `tool-calling`), the context limit is fixed at 4096 tokens, enforced via `RotatingKVCache` (`maxKVSize`) to simulate a realistic chat deployment.

## KV Cache Configurations

| Config | Compression | Description |
|--------|-------------|-------------|
| `none` | — | Unquantized FP16 KV cache (baseline) |
| `affine8` | K 8-bit, V 8-bit | MLX affine 8-bit quantization (quantized start at offset 512) |
| `affine4` | K 4-bit, V 4-bit | MLX affine 4-bit quantization (quantized start at offset 512) |
| `turbo8` | K 8-bit, V 8-bit (symmetric) | TurboQuant MSE 8-bit compression (starts at offset 0) |
| `turbo8v4` | K 8-bit, V 4-bit (asymmetric) | TurboQuant asymmetric: 8-bit keys, 4-bit values |
| `turbo8v2` | K 8-bit, V 2-bit (asymmetric) | TurboQuant asymmetric: 8-bit keys, 2-bit values |
| `turbo4` | K 4-bit, V 4-bit (symmetric) | TurboQuant MSE 4-bit compression (starts at offset 0) |
| `turbo4v3` | K 4-bit, V 3-bit (asymmetric) | TurboQuant asymmetric: 4-bit keys, 3-bit values |
| `turbo4v2` | K 4-bit, V 2-bit (asymmetric) | TurboQuant asymmetric: 4-bit keys, 2-bit values |
| `turbo3` | K 3-bit, V 3-bit (symmetric) | TurboQuant MSE 3-bit compression |
| `turbo3v2` | K 3-bit, V 2-bit (asymmetric) | TurboQuant asymmetric: 3-bit keys, 2-bit values |
| `all` | — | Shortcut: expands to `none,affine8,affine4,turbo4,turbo4v2,turbo3` |

**Affine** configs use MLX's built-in quantized-cache path (per-group scale + zero point) and defer quantization until offset 512 — the first 512 KV slots stay full precision to avoid stomping on the short-context path that matters most for TTFT. **Turbo** configs use the TurboQuant MSE-optimal codebook starting at offset 0. **Asymmetric** turbo variants (`turbo{K}v{V}`) compress values more aggressively than keys, exploiting the fact that value projections tolerate more quantization noise than keys in attention.

## Methodology

### System prompts

Several methods use the **same** short assistant system message (defined in the benchmark suite as `minimalSystemPrompt`):

> You are a helpful assistant. Keep responses concise.

That string applies to **`simple`**, **`multi-turn`**, **`tool-calling`**, and **`niah`**. Individual benchmark markdown files link here instead of repeating long user prompts.

**`summarization`** (including the warmup pass) uses **no** system role — only user messages built from the bundled prompt files (see Summarization below). **`wikitext2`** has no chat template system role; it evaluates raw WikiText-2 continuation.

### Simple

Sends a basic chat prompt ("Hello! What is your name and what can you help me with?") to the model with a 4096-token context limit. Measures generation speed, TTFT, and perplexity. Quick single evaluation without context scaling. No pass/fail validation.

### Summarization

Loads pre-generated prompt files (excerpts from The Great Gatsby) sized to each target token count. The context limit (`maxKVSize`) matches the target size, so the KV cache is bounded. The model generates a summary response (up to 200 tokens). Measures prefill throughput, generation throughput, TTFT, perplexity, and GPU memory at each context size. No pass/fail validation.

### WikiText-2

Standard LM perplexity evaluation via **forced decode** on the WikiText-2 test dataset. No generation occurs.

**How it works:**
1. The WikiText-2 test text is tokenized using the model's tokenizer and sliced to the target context size.
2. Tokens are fed through the model in 2048-token chunks, capturing logits at every position.
3. At each position, the log-probability of the actual next token is computed via `log(softmax(logits, axis: -1))`.
4. KV cache quantization (if configured) is applied after each chunk, measuring its impact on prediction quality.
5. **Word-level PPL** (the standard metric per EleutherAI) is computed: `exp(total_NLL / word_count)`, where word count is from the decoded text. This normalizes across different tokenizers, making results comparable across model families.

Token-level PPL is also computed and logged but the word-level metric is reported in the results table. Expected ranges: ~15-30 for instruct-tuned 0.8B models on raw Wikipedia text (instruct models score higher than base models since they're fine-tuned for conversations, not raw text continuation).

### Needle-in-a-Haystack (NIAH)

Tests long-context retrieval by inserting a known fact into filler text at multiple depth positions and asking the model to retrieve it.

**How it works:**
1. For each context size, filler text (Great Gatsby excerpts from the pre-sized prompt files) is loaded.
2. The needle — `"The special magic verification code is BLUE TIGER 42."` — is inserted at **5 depth positions**: 10%, 25%, 50%, 75%, and 90% of the document.
3. The retrieval question — `"What is the special magic verification code mentioned in the text above? Reply with only the code, nothing else."` — is appended after the filler.
4. Each depth produces a separate benchmark run (5 rows per context size).
5. The KV cache is **unbounded** (no `maxKVSize`) so the model can see the entire prompt.
6. **Pass/fail**: output is checked for the substring "BLUE TIGER 42" (case-insensitive). Results are prefixed `PASS(@50%)` or `FAIL(@50%)` in the Output column.

This produces a depth × context matrix showing where the model succeeds or fails at retrieval. Small models (0.8B) typically fail at larger contexts and certain depth positions, while larger models (9B+) should pass consistently.

### Multi-Turn

Tests context recall across a multi-turn conversation. Two names ("Bob" and "Alice") are introduced early in the conversation, and the model is asked to recall each one.

**Conversation structure:**
1. User: "Hello, what is your name?" → Assistant responds
2. User: "My name is Bob and my partner's name is Alice." → Assistant acknowledges (without repeating names)
3. Test 1: "What is my name?" → **PASS** if output contains "Bob"
4. Test 2: "What is my partner's name?" → **PASS** if output contains "Alice"

Each recall test produces a separate row in the results table. The assistant's acknowledgment is kept neutral ("Nice to meet you! What can I help you with?") to avoid leaking answers.

### Tool Calling

Tests whether the model correctly generates a tool call when given a tool-use prompt and a tool specification.

**Setup:**
- Prompt: "What is the current date and time?"
- Tool: `execute_shell` — a mock shell execution tool with a `command` string parameter
- The model is expected to generate a tool call to `execute_shell` with a command containing "date"

**Pass/fail** (strict validation):
1. The generation stream must produce a `.toolCall` event → otherwise `FAIL(no tool call)`
2. The tool call function name must be `execute_shell` → otherwise `FAIL(wrong tool: <name>)`
3. The command argument must contain "date" → otherwise `FAIL(wrong command: <cmd>)`
4. All three pass → `PASS`

### Perplexity (Think PPL / Gen PPL)

Perplexity is computed as `exp(mean negative log-probability)` over generated tokens. It is tracked separately for the **thinking phase** and the **generation phase**. Lower values indicate higher model confidence in its predictions.

Thinking is **disabled by default** for maximum speed. Use `--think` to enable it:

```bash
# Speed benchmark (no thinking overhead)
./scripts/benchmark.sh --model gemma4-e2b --quant 4bit --method summarization --quick --ppl

# Quality benchmark with thinking separation
./scripts/benchmark.sh --model gemma4-e2b --quant 4bit --method summarization --quick --ppl --think
```

When `--think` is enabled for thinking-capable models:
- **Qwen3.5**: Prefills `<think>\n` in the assistant turn; tracks tokens between `<think>` and `</think>`
- **Gemma 4**: Passes `enable_thinking=true` to the chat template; tracks tokens between `<|channel>` and `<channel|>`
- A thinking budget processor forces the end-think token after 200 thinking tokens and suppresses EOS during thinking to ensure both phases are measured

For `wikitext2`, the Gen PPL column reports word-level perplexity from the forced-decode evaluation (no thinking phase applies).

### KL Divergence (Think KLD / Gen KLD)

When `--kld` is enabled, KL divergence measures how much a deployment configuration (weight quantization + KV cache compression) degrades the model's output distribution compared to the highest-fidelity baseline available for that model family.

**How it works:**

1. The target model generates tokens normally with per-token log-probability tracking enabled.
2. After generation completes, the highest-fidelity baseline model (bf16 preferred, 8-bit fallback if bf16 exceeds GPU memory) is loaded without KV cache compression.
3. The target's generated tokens are **forced-decoded** through the baseline model — each token is fed sequentially, and the baseline's log-probability for that token is recorded.
4. KLD is computed per phase as: `mean(target_logprob - baseline_logprob)` (always >= 0).

Values near **0** indicate negligible quality loss. Higher values indicate greater divergence from the gold standard.

**KLD decision matrix:**

| Target Quant | KV Config | Baseline Selected | KLD Runs? | What It Measures |
|--------------|-----------|-------------------|-----------|------------------|
| bf16 | none | — | No | Target IS the baseline |
| bf16 | affine4/turbo | bf16 | Yes | KV compression cost |
| 8bit | none | bf16 | Yes | Weight quantization cost |
| 8bit | affine4/turbo | bf16 | Yes | Weight quant + KV compression |
| 4bit | none | bf16 | Yes | Weight quantization cost |
| 4bit | affine4/turbo | bf16 | Yes | Weight quant + KV compression |

When bf16 exceeds GPU memory (e.g., 27B models on 48GB):

| Target Quant | KV Config | Baseline Selected | KLD Runs? | What It Measures |
|--------------|-----------|-------------------|-----------|------------------|
| 8bit | none | 8bit | No | Same config, skipped |
| 8bit | affine4/turbo | 8bit | Yes | KV compression cost |
| 4bit | none | 8bit | Yes | Weight quantization cost |
| 4bit | affine4/turbo | 8bit | Yes | Weight quant + KV compression |

### GPU Memory (GPU Baseline / GPU Peak / KV Delta)

Three memory metrics are reported for each benchmark run:

- **GPU Baseline**: GPU memory after the model weights are loaded but before generation starts. This is the static cost of holding the model in memory.
- **GPU Peak**: High-water mark of GPU memory during the entire run, including transient allocations. Captured via `MLX.Memory.peakMemory`.
- **KV Delta**: The increase in active GPU memory from the KV cache, measured as `activeGPU - baselineGPU` after generation completes. For KV-quantized runs (affine4, turbo4, turbo3), this reflects the compressed cache size. Comparing KV Delta between `none` and a quantized config at the same context shows how much memory the compression saves.

**Why GPU Peak is much higher than GPU Baseline + KV Delta:**

The gap is primarily intermediate computation tensors allocated during the forward pass — attention scores, QKV projections, FFN activations, softmax buffers, Conv1d state (for GatedDeltaNet), and recurrent state updates. These are allocated during each forward step and freed afterward, but they contribute to the peak memory high-water mark.

Key factors:
- **Prefill dominates peak**: Prefill processes the full prompt at once (e.g., 1024 tokens), creating much larger intermediate tensors than single-token generation. The peak is usually hit during prefill.
- **MLX memory pool**: MLX does not immediately return freed memory to the OS — it caches freed allocations for reuse. `peakMemory` reflects the cumulative high-water mark, not just what is actively held.
- **GatedDeltaNet overhead**: The hybrid GatedDeltaNet architecture (used by all Qwen3.5 models) has higher intermediate memory than standard transformers due to simultaneous QKV projections, conv state concatenation, and gated delta updates per layer.

## Environment Variables

All configuration is passed via environment variables, enabling any backend to implement the benchmark interface.

| Env Var | Values | Default | Description |
|---------|--------|---------|-------------|
| `MLX_BENCH_MODEL` | registry name or HF repo (single value — CLI iterates over multiple) | *required* | Model to benchmark |
| `MLX_BENCH_METHOD` | `simple`, `summarization`, `wikitext2`, `niah`, `multi-turn`, `tool-calling` | `simple` | Test method |
| `MLX_BENCH_QUANT` | `bf16`, `8bit`, `4bit` | `4bit` | Weight quantization |
| `MLX_BENCH_KV` | `none`, `affine8`, `affine4`, `turbo8`, `turbo8v4`, `turbo8v2`, `turbo4`, `turbo4v3`, `turbo4v2`, `turbo3`, `turbo3v2` | `none` | KV cache config |
| `MLX_BENCH_CONTEXT` | comma-separated ints (e.g., `128,1024,4096`) | all 11 sizes | Context sizes to test |
| `MLX_BENCH_PPL` | `1` | unset | Enable perplexity tracking |
| `MLX_BENCH_KLD` | `1` | unset | Enable KLD computation |
| `MLX_BENCH_BASELINE` | `1` | unset | Auto-select highest-fidelity variant that fits in GPU memory |
| `MLX_BENCH_BATCH` | integer | `1` | Number of concurrent generations |
| `MLX_BENCH_THINK` | `1` | unset | Enable thinking mode |
| `MLX_BENCH_REASONING` | `low`, `medium`, `high`, or passthrough | unset (falls back to the model family's registered default) | Reasoning effort for models that honour it (GPT-OSS). Plumbed into `GenerateParameters.reasoningEffort`; ignored by models whose chat templates don't consume it. |
| `MLX_BENCH_NGRAM` | non-negative integer | `0` | N-gram speculative-decoding size. Plumbed into both `GenerateParameters.ngramSize` and `maxNgramDraftTokens`. `0` disables speculation entirely; any positive value enables trigram-style drafting with N tokens of history matched and N draft tokens proposed per round. Benchmark default is `0` so measurements are deterministic; the library itself defaults to `3`. |
| `MLX_BENCH_PROMPT` | string | built-in | Override the `simple`-method user prompt |
| `MLX_BENCH_PROFILE` | `1`, `2` | unset | Lifecycle profiling (see [Profiling](#profiling)). `1` = inline `[PROFILE]` breakdown at end of run. `2` = everything in `1` **plus** `os_signpost` intervals at every phase boundary (captured by Instruments / `xctrace`). Zero-overhead when unset; level 2 adds ~50 µs over a 200-token run when no tracer is attached. |
| `MLX_MAX_OPS_PER_BUFFER` | integer | hardware default (200 on Max/Ultra) | MLX Metal command-buffer commit threshold. Captured into every Parameters block so report readers can see what was in effect. |

The underlying test binary (`InferenceBenchmark.swift`) reads a **single** model / method / quant / KV permutation per process — one row of the sweep. `benchmark.sh` does the enumeration and re-invokes `swift test` once per permutation. All processes in a single sweep write into the same hardware-dated report file via the JSON state sidecar described in [Output](#output), so the grouping is preserved even though each row lives in its own process.

## Profiling

Two opt-in profile levels are controlled via `MLX_BENCH_PROFILE`. Both are off by default — a normal benchmark run is unaffected.

### Level 1 — inline lifecycle breakdown

Sets a few timestamps at phase boundaries and prints a `[PROFILE]` block after the standard `[BENCH]` report. Useful when you want a single-run, eyeballable split of where wall-clock time went without any external tools. Zero impact on inference timing.

```bash
MLX_BENCH_PROFILE=1 MLX_BENCH_MODEL=gpt-oss-20b MLX_BENCH_METHOD=simple \
  MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark
```

Produces:

```
[PROFILE] ── Lifecycle breakdown ───────────────────────────────
[PROFILE] model_load                :  2745.9 ms  (cold)
[PROFILE] prompt_prep               :    44.5 ms  (tokenize + template)
[PROFILE] prefill                   :   470.2 ms  (101 tokens @ 214.8 tok/s)
[PROFILE] first_token_overhead     :     1.7 ms  (TTFT − prefill: kernel JIT + first decode)
[PROFILE] ttft                      :   471.9 ms
[PROFILE] decode_warmup_per_token  :   21.49 ms  (tokens 2..4 avg)
[PROFILE] decode_steady_per_token  :   21.60 ms  (tokens 11..end avg) = 46.3 tok/s
[PROFILE] generation_total         :  4296.3 ms  (199 tokens @ 46.3 tok/s)
[PROFILE] benchmark_total           :  7565.7 ms
[PROFILE] ──────────────────────────────────────────────────────
```

Columns:

- **model_load** — wall-clock of `loadOrCacheModel`. `(cache hit)` when the model was already loaded by an earlier row in the same process. MLX uses mmap-based lazy weight loading, so this typically reflects page-cache warmth rather than true upload cost.
- **prompt_prep** — tokenization + chat template rendering (CPU-only).
- **prefill** — taken from `GenerateCompletionInfo.promptTime`; the GPU-side prompt processing.
- **first_token_overhead** — `TTFT − prefill`. Whatever happens between prefill ending and the first decoded token arriving in Swift. Usually dominated by kernel JIT / pipeline creation on cold runs, negligible once Metal's pipeline cache is warm.
- **ttft** — `firstTokenTime`. Matches `[BENCH] TTFT`.
- **decode_warmup_per_token** — average of tokens 2..4. Isolates the first-few-tokens slowdown that JIT, buffer pool fill, and expert routing caches produce.
- **decode_steady_per_token** — average of tokens 11..end. The steady-state hot loop — this is what matters for long generations.
- **generation_total** — `(total − ttft)` divided by `(tokenCount − 1)`. Matches `[BENCH] Generation` tok/s; may underestimate the steady rate when the output is short because warmup dominates the average.
- **benchmark_total** — entry of `runGenerationBenchmark` to its return; includes model load + prompt prep + generation + any trailing bookkeeping.

### Level 2 — `os_signpost` tracing (CPU + GPU timelines)

Everything at level 1, **plus** `os_signpost` intervals emitted at every phase boundary. Instruments and `xctrace` capture these and render them as a timeline track you can overlay on CPU samples (Time Profiler) and Metal kernel executions (Metal System Trace). This is the right tool when level 1 tells you *which phase* is slow and you need to know *which CPU function* or *which Metal kernel* inside that phase is responsible.

**Runtime cost**: `os_signpost` begin/end is ~40 ns per event. A 200-token decode adds ~50 µs total overhead with no tracer attached — well under 0.01% of wall clock. When a tracer *is* attached, the overhead is dominated by the kernel-to-user buffer copies Instruments does on its own, not by the app.

**What gets traced** (subsystem `ai.mlx.bench`, category `PointsOfInterest`):

| Signpost | Type | Spans | Metadata |
|---|---|---|---|
| `model_load` | interval | `loadOrCacheModel` entry → return | `cold:<repoId>` or `cache_hit` |
| `prompt_prep` | interval | `container.prepare` start → end | end: `prompt_tokens=N` |
| `prefill` | interval | generation start → first chunk yielded | begin: `prompt_tokens=N`, end: `first_token=true` |
| `first_token` | point event | emitted at first chunk | `ttft_ms=N` |
| `decode_step` | interval | per token (new chunk → next new chunk) | begin: `token_idx=N`, end: `tokens_so_far=N+1` |

All intervals nest cleanly, so Instruments renders them as a hierarchical timeline.

#### Recording with `xctrace` (headless)

Saves to a `.trace` file you can open in Instruments after the fact.

```bash
xcrun xctrace record \
  --template 'Time Profiler' \
  --output /tmp/mlx-profile.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 \
    MLX_BENCH_MODEL=gpt-oss-20b \
    MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
    MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
    /usr/bin/swift test --skip-build -c release \
      --package-path "$(pwd)" \
      --filter benchmark

open /tmp/mlx-profile.trace
```

Swap `Time Profiler` for `Metal System Trace` to get the GPU kernel timeline instead. Or record twice to the same trace via `--instrument` flags if you want both.

#### Recording interactively with Instruments

1. Launch Instruments (`open -a Instruments` or from Spotlight).
2. Pick one of:
   - **Time Profiler** — CPU samples + signposts + thread state.
   - **Metal System Trace** — GPU kernel timeline + command buffer timings + signposts.
   - **Blank** → add `os_signpost`, `Time Profiler`, and `Metal System Trace` instruments manually for the full picture.
3. Target: `Choose target...` → browse to the test binary at
   `.build/arm64-apple-macosx/release/mlx-swift-lmPackageTests.xctest/Contents/MacOS/mlx-swift-lmPackageTests`
   and set the env vars via the target settings panel.
4. Click record, kick off the bench.

For models where generation ends quickly (< 1 s), increase `MLX_BENCH_MAX_TOKENS` so Instruments has time to attach and capture the hot loop cleanly.

#### Filtering & reading the trace

Once the trace is open in Instruments:

- In the `os_signpost` track, filter by **Subsystem == `ai.mlx.bench`** to isolate benchmark signposts from system noise.
- Click a `decode_step` interval to see the CPU samples and Metal kernels that fired inside that token's budget. Match the `token_idx=N` metadata against the token sequence in the run to correlate specific tokens with backend behaviour.
- Expand `prefill` and `model_load` to see where cold-start CPU time goes (Metal pipeline creation, Safetensor parsing, tokenizer init).

#### Exporting signpost rows for scripted comparison

The trace's signpost table is exportable to XML, which makes diffing between branches straightforward:

```bash
xcrun xctrace export \
  --input /tmp/mlx-profile.trace \
  --xpath '//trace-toc/run/data/table[@schema="os-signpost"]' \
  > signposts.xml
```

Each row carries `name`, `event-type` (Begin / End / Event), `time` (ns since trace start), `subsystem`, `category`, and the metadata string (`token_idx=7`, `ttft_ms=136`, etc.). Parse with `xmllint --xpath`, a small Python script, or any XML tool to compute per-phase deltas across runs.

#### When to reach for each level

- Level 1: you want a one-line regression signal, you're writing a baseline report, or you don't have Instruments available.
- Level 2: you've narrowed the problem to a specific phase (say, decode is slow) and need to know which kernel or Swift function is eating the time. Instruments' overlay of signposts + CPU samples + GPU traces is the right tool for this, and the overhead is negligible so you can leave it on during the run you care about.

## Output

Benchmark reports are saved as Markdown files in this `benchmarks/` directory, **one file per run, scoped to the hardware and date it was produced on** — not one file per model. A full-matrix sweep across 14 models lives in a single file, making cross-model comparisons and regression diffs easy to eyeball.

### File naming

```
{hardware}-{ram}-{YYYY-MM-DD}.md
```

Examples:
- `m1-max-64gb-2026-04-16.md`
- `m5-max-128gb-2026-04-16.md`
- `m3-ultra-192gb-2026-04-30.md`

If multiple runs happen on the same hardware on the same day, append a run index or a short tag: `m5-max-128gb-2026-04-16-pr45.md`, `m1-max-64gb-2026-04-16-run2.md`.

### File structure

Each benchmark file follows this layout, top to bottom:

1. **Title** — `# Benchmark: {chip} — {YYYY-MM-DD}`.
2. **Environment block** — hardware (chip + unified memory + GPU limit), OS, branch, commit, NAX state, creation timestamp.
3. **`## Models`** — wrapper heading.
4. **`### {Model name}`** per model, each containing:
   - `#### Results` — a single table with one row per `(quant / kv / method)` config × context size, in column order:
     `Config | Ctx | Prompt | Prefill tok/s | Decode tok/s | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Base | GPU Peak`
   - `#### Output samples` — fenced code block per config showing the first ~400 characters of the generated output. Proves the run didn't produce garbage.
   - `#### Parameters` — one block per config (below the results as per the owner's preference), with the full parameter table: KV strategy, max KV size, KV bits/scheme/group/start, prefill step size, max tokens, temperature, top_p, top_k, min_p, repetition / presence / frequency penalties, reasoning effort, thinking config, per-token data tracking, n-gram speculative settings, PPL / KLD / batch / speculative / `MLX_MAX_OPS_PER_BUFFER`.
5. **`## Methodology`** — single-line pointer back to this README.

### How rows are written

Each `(model, quant, kv, method, context)` run appends exactly one row to the results table for its model. Configs within the same model share a single Results table (rows include the Config column so you can eyeball quant/kv/method deltas side-by-side). Configs remain in insertion order — the order `benchmark.sh` visited them during the sweep.

State of truth lives in a JSON sidecar next to the markdown file: `benchmarks/.{chip}-{ram}-{date}.state.json`. The markdown is re-rendered from the sidecar on every append. This means:

- Multi-process sweeps (every `swift test` invocation is its own process) accumulate into the same report.
- Editing the markdown by hand is fine for annotation but the sidecar is the authoritative input — next write overrides the markdown from the sidecar.
- If you need a clean slate on a given day, delete both the `.md` and the matching `.state.json`.

### Content rules

- **Parameters live beneath their own config's results row (not in a single top-level block).** This keeps parameter deltas visible when a sweep varied quant or KV config across rows of the same model.
- **Output samples stay short.** ~400 characters is enough to verify coherence. Longer outputs bloat diffs and obscure the signal.
- **Don't repeat the methodology in every file.** The Methodology link at the bottom is the single source of truth; individual files should only add notes when they deviate from the defaults documented here.
- **Record hardware-default `MLX_MAX_OPS_PER_BUFFER`** — on M1 Max/Ultra and M5 Max the committed default is 200; the effective value is captured in every Parameters block regardless of whether it was set via env.

### When to update baselines

- A material perf delta lands (≥ 5% on prefill or decode, or any peak-memory change > 10%).
- A kernel or framework change affects the row (even if the net delta is small — the new snapshot anchors the "it worked here" state).
- New hardware is added.

## Directory layout

```
benchmarks/
├── README.md                           # This file
├── m1-max-64gb-2026-04-16.md           # Full-matrix baseline on M1 Max
├── m5-max-128gb-2026-04-16.md          # Full-matrix baseline on M5 Max
├── m3-ultra-192gb-2026-04-30.md        # Full-matrix baseline on M3 Ultra
```

