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

### Decode tok/s vs Steady tok/s

Two decode-throughput metrics are reported side-by-side for every generation run:

- **Decode tok/s** — the user-facing average: `(genTokens − 1) / (totalTime − TTFT)`. Includes the first ~10 tokens where the Metal pipeline cache, JIT-compiled shaders, and threadgroup-shape caches are still warming up. Matches what a user experiences on a single short turn (a few hundred tokens).
- **Steady tok/s** — the hot-loop rate: mean of per-token intervals starting at token 11, which excludes warmup. Computed from the same per-token arrival timestamps used by Phase 1 profiling (`decode_steady_per_token`). `—` when the run generated ≤ 10 tokens.

**Why both?** A kernel or framework change that's neutral at steady-state but adds a few ms to the first few tokens' first-dispatch cost will show up as a 5–15% Decode-tok/s regression on a 200-token benchmark, while Steady tok/s stays flat. Reporting them together separates the two failure modes:

- Decode down, Steady down → real throughput regression in the hot loop.
- Decode down, Steady flat → warmup regression (first-dispatch cost, pipeline creation, Custom-primitive init).
- Decode up, Steady flat → warmup *improvement* (e.g. pre-warming a kernel).
- Decode flat, Steady up → real hot-loop win that will surface on longer generations but is masked here by warmup.

For long-running workloads (chat sessions, streaming completions >1000 tokens), Steady tok/s is the better predictor. For short interactive turns, Decode tok/s is closer to what users feel.

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
| `MLX_BENCH_PROFILE` | `1`, `2` | unset | Lifecycle profiling (see [Profiling](#profiling)). `1` = Phase 1 inline `[PROFILE]` breakdown at end of run. `2` = Phase 2: everything in `1` **plus** `os_signpost` intervals at every benchmark phase boundary (subsystem `ai.mlx.bench`, captured by Instruments / `xctrace`). Zero-overhead when unset. |
| `MLX_METAL_PROFILE` | `1` | unset | Enables Phase 3 kernel-level tracing — per Metal dispatch `os_signpost` intervals plus `end_encoding` / `commit` / `synchronize` lifecycle spans, under subsystem `ai.mlx.metal`. Typically combined with `MLX_BENCH_PROFILE=2`. ~60 µs/token overhead (1500 dispatches × 2 signposts × 20 ns) when no tracer is attached; zero when unset. |
| `MLX_MAX_OPS_PER_BUFFER` | integer | hardware default (200 on Max/Ultra) | MLX Metal command-buffer commit threshold. Captured into every Parameters block so report readers can see what was in effect. |

The underlying test binary (`InferenceBenchmark.swift`) reads a **single** model / method / quant / KV permutation per process — one row of the sweep. `benchmark.sh` does the enumeration and re-invokes `swift test` once per permutation. All processes in a single sweep write into the same hardware-dated report file via the JSON state sidecar described in [Output](#output), so the grouping is preserved even though each row lives in its own process.

## Profiling

Three opt-in profile phases, each a superset of the previous. All off by default — a normal benchmark run is unaffected.

| Phase | Env | What it captures | Cost when off | Cost when on | Needs Instruments? |
|---|---|---|---|---|---|
| **1** | `MLX_BENCH_PROFILE=1` | Inline `[PROFILE]` breakdown: model load, prompt prep, prefill, warmup-vs-steady decode | 0 | ~40 µs/run | No |
| **2** | `MLX_BENCH_PROFILE=2` | Phase 1 + `os_signpost` intervals at every benchmark phase boundary (`model_load`, `prefill`, per-token `decode_step`). Subsystem `ai.mlx.bench`. | 0 | ~50 µs / 200 tokens | Yes (capture) |
| **3** | `MLX_BENCH_PROFILE=2` + `MLX_METAL_PROFILE=1` | Phase 2 + per-kernel-dispatch `os_signpost` intervals for every Metal dispatch, plus `end_encoding` / `commit` / `synchronize` lifecycle signposts. Subsystem `ai.mlx.metal`. Kernel names from the compute pipeline's label (e.g. `sdpa_vector_bfloat16_t_64_64`, `affine_qmv_float_gs_64_b_4`). | 0 | ~60 µs/token (1500 dispatches × 2 × 20 ns) | Yes (capture) |

**Rule of thumb:**
- **Phase 1** → regression signal. Run it when you changed something and want to know if total wall-clock moved.
- **Phase 2** → narrow to a lifecycle phase. Run it when Phase 1 shows something shifted and you need to know *which phase* (e.g. decode slowed but prefill didn't).
- **Phase 3** → narrow inside a phase to specific kernels or CommandEncoder lifecycle calls. Run it when Phase 2 shows the decode loop is slow and you need to know *which Metal kernels* or *which command-buffer operations* are dominating.

### Phase 1 — inline lifecycle breakdown

Captures wall-clock timestamps at phase boundaries and prints a `[PROFILE]` block at the end. No external tools.

```bash
# Phase 1, simplest possible invocation
MLX_BENCH_PROFILE=1 \
  MLX_BENCH_MODEL=gpt-oss-20b \
  MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
  MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark
```

Output (tail):

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

**How to read the columns:**

| Field | Meaning |
|---|---|
| `model_load` | `loadOrCacheModel` wall-clock. `(cache hit)` when a prior row in the same process loaded it. MLX uses mmap-based lazy weight loading, so cold runs mostly reflect page-cache warmth rather than true upload cost. |
| `prompt_prep` | Tokenization + chat template rendering (CPU-only). |
| `prefill` | GPU-side prompt processing, from `GenerateCompletionInfo.promptTime`. |
| `first_token_overhead` | `TTFT − prefill`. Usually dominated by kernel JIT / pipeline creation on cold runs, negligible once Metal's pipeline cache is warm. |
| `ttft` | `firstTokenTime`. Matches `[BENCH] TTFT`. |
| `decode_warmup_per_token` | Average of tokens 2..4. Isolates the first-few-tokens slowdown that JIT, buffer pool fill, and expert routing caches produce. |
| `decode_steady_per_token` | Average of tokens 11..end. The steady-state hot loop — what matters for long generations. |
| `generation_total` | `(total − ttft) / (tokenCount − 1)`. Matches `[BENCH] Generation` tok/s; may underestimate the steady rate when the output is short because warmup dominates the average. |
| `benchmark_total` | `runGenerationBenchmark` entry → return. Includes load + prompt prep + generation + bookkeeping. |

**Phase 1 examples:**

```bash
# Short-output model (Gemma 4 E2B) — warmup dominates the reported
# generation tok/s; steady-state is more representative
MLX_BENCH_PROFILE=1 MLX_BENCH_MODEL=gemma4-e2b \
  MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
  MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark

# MoE model (Qwen3.5-35B-A3B) — check prefill tok/s (often the
# weak spot for GatedDeltaNet prefill)
MLX_BENCH_PROFILE=1 MLX_BENCH_MODEL=qwen35-35b-a3b \
  MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
  MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark

# Custom HuggingFace repo by ID (first run downloads)
MLX_BENCH_PROFILE=1 MLX_BENCH_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit \
  MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
  MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark

# Multi-run average (ModelCache is per-process so each run reloads;
# first numbers are representative of cold-start; use decode_steady_per_token
# for apples-to-apples)
for i in 1 2 3; do
  echo "=== Run $i ==="
  MLX_BENCH_PROFILE=1 MLX_BENCH_MODEL=gpt-oss-20b \
    MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
    MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
    swift test --skip-build -c release --filter benchmark 2>&1 \
    | grep -E "PROFILE\]|Generation"
done
```

### Phase 2 — phase-level `os_signpost` tracing

Everything in Phase 1 plus `os_signpost` intervals at every lifecycle boundary. Instruments and `xctrace` capture these and render them as a timeline track you can overlay on CPU samples (Time Profiler) and Metal kernel executions (Metal System Trace).

**Runtime cost:** `os_signpost` begin/end is ~40 ns per event. A 200-token decode adds ~50 µs total overhead when no tracer is attached. When Instruments *is* recording, overhead is dominated by Instruments' own kernel-to-user buffer copies, not by the app.

**Subsystem `ai.mlx.bench`, category `PointsOfInterest`:**

Lifecycle phases (one each per benchmark cell):

| Signpost | Type | Span | Metadata |
|---|---|---|---|
| `model_load` | interval | `loadOrCacheModel` entry → return | `cold:<repoId>` or `cache_hit` |
| `prompt_prep` | interval | `container.prepare` start → end | end: `prompt_tokens=N` |
| `prefill` | interval | generation start → first chunk yielded | begin: `prompt_tokens=N`, end: `first_token=true` |
| `first_token` | point event | at first chunk | `ttft_ms=N` |
| `decode_step` | interval | one per token | begin: `token_idx=N`, end: `tokens_so_far=N+1` |

Attention sub-phases (one each per attention layer per token, all paths):

| Signpost | Type | Path | Span |
|---|---|---|---|
| `kv_update` | interval | all | `cache.update` / `updateAndDequant` / `updateQuantized` — KV append step |
| `sdpa` | interval | TurboQuant A, default cache | `MLXFast.scaledDotProductAttention` |
| `qsdpa` | interval | affine quantized | `quantizedScaledDotProductAttention` |

TurboQuant B path sub-phases (`useCompressedAttention=true`, fires inside `compressedAttention`):

| Signpost | Type | Span |
|---|---|---|
| `tq_encode` | interval | `encodeNewToken` — quantize new K/V into packed buffer |
| `tq_score` | interval | Q*K (matmul or compressed `mseScore`) |
| `tq_softmax` | interval | softmax over scores (separated path only) |
| `tq_value` | interval | Attn*V (TurboFlash kernel or `mseWeightedSum`) |
| `tq_rotate` | interval | Q rotation + inverse value rotation |

For `useCompressedAttention=false` (the default), `kv_update` + `sdpa` fire instead — no `tq_*` signposts.

#### Quick CPU-side breakdown without Instruments (`[MLX-PROFILE]`)

Phase 2 also dumps an in-process per-label CPU wall-clock aggregator to stdout at the end of every cell — no Instruments / xctrace required, no extra config:

```text
[MLX-PROFILE] CPU wall-clock aggregator (in-process; not GPU time):
[MLX-PROFILE]   label               count    total(ms)       %    avg(µs)
[MLX-PROFILE]   decode_step           400      3436.65   51.6%    8591.62
[MLX-PROFILE]   model_load              1      1592.70   23.9% 1592702.98
[MLX-PROFILE]   prefill                 1      1404.18   21.1% 1404181.00
[MLX-PROFILE]   tq_encode            2406       101.43    1.5%      42.16
[MLX-PROFILE]   prompt_prep             1        80.00    1.2%   79997.90
[MLX-PROFILE]   tq_value             2406        25.91    0.4%      10.77
[MLX-PROFILE]   tq_rotate            2406        16.93    0.3%       7.04
```

This measures **CPU time between `begin`/`end`** — i.e. dispatch + any synchronous CPU-side prep — not the GPU's actual kernel execution time. For decode-phase signposts most of the budget runs on the GPU asynchronously, so the CPU totals are a *lower bound* on per-phase work, useful for comparing dispatch overhead between configurations or for ranking phases by activity. For accurate GPU attribution, use the xctrace path below.

#### Recording with `xctrace` (headless)

```bash
# Phase 2 — save a .trace file you can open in Instruments later
xcrun xctrace record \
  --template 'Time Profiler' \
  --instrument 'Points of Interest' \
  --output /tmp/mlx-phase2.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 \
    MLX_BENCH_MODEL=gpt-oss-20b \
    MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
    MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=200 \
    /usr/bin/swift test --skip-build -c release \
      --package-path "$(pwd)" \
      --filter benchmark

open /tmp/mlx-phase2.trace
```

Pick an alternative template based on what you're chasing:
- `'Time Profiler'` — CPU samples + signposts + thread state (best for "what CPU function is eating decode time?")
- `'Metal System Trace'` — GPU kernel timeline + command buffer timings. **Note: this template drops os_signpost rows by default**; see below for the combined setup.

> **Gotcha.** `ai.mlx.bench` and `ai.mlx.metal` use category `.pointsOfInterest`, which is a "live" category — events are only persisted while a matching instrument is recording. You must add `--instrument 'Points of Interest'` (capital P) to xctrace, OR use a template that already includes it (`Time Profiler` does; `Metal System Trace` does NOT, hence the warning above). Specifying `--instrument 'os_signpost'` instead is *not* sufficient — the generic `os_signpost` instrument captures system signposts but not user `.pointsOfInterest` subsystems. Symptoms of getting it wrong: the trace exports cleanly with an `os-signpost` table present, but `xpath`-filtering for `subsystem="ai.mlx.bench"` returns zero rows.

#### Useful instruments to combine with `Time Profiler`

Stack as many `--instrument <Name>` flags as you need. Recommended for MLX inference work:

| Instrument | What it captures | Why useful for MLX | SIP required |
|---|---|---|---|
| `Points of Interest` | `ai.mlx.bench` and `ai.mlx.metal` signposts | Phase boundaries (`decode_step`, `tq_*`, etc.) + per-kernel dispatch labels | no |
| `Time Profiler` | CPU sampling + callstacks | What Swift/MLX/C++ functions are eating decode time | no |
| `Metal Application` | Command buffer timeline + GPU kernel execution windows | True GPU per-kernel time, correlates with `kernel_dispatch` signposts | no |
| `Metal GPU Counters` | Hardware counters (ALU active, memory bandwidth, occupancy) | Diagnose GPU underutilization vs memory-bound kernels | no |
| `Metal Resource Events` | MTLBuffer/Texture creation, residency-set membership | Catch transient buffer churn (e.g. compressRawCache transition peaks) | no |
| `Metal Performance Overview` | High-level GPU utilization | Quick check whether GPU is the bottleneck | no |
| `Hangs` | Main-thread blocks > 100 ms | Catch unintended synchronous `eval()` barriers | no |
| `VM Tracker` | Virtual memory regions (`VM_OBJECT`, dirty/swapped pages) | Unified-memory visibility — KV cache region, weights region, etc. | no |
| `Virtual Memory Trace` | Page faults | Measure first-touch cost of large KV reallocations | no |
| `Thread State Trace` | Thread blocking + runqueue | Find unintended sync points between dispatch threads | no |
| `os_log` | All `os_log` messages from any subsystem | See model-load logs, codec init, MLX warnings | no |
| `Allocations` | Heap allocations (Swift / ObjC / malloc) | Track per-phase memory growth, find unexpected retentions | **yes** |
| `Leaks` | Leaked Swift / ObjC objects | Catch cache instances or kernel handles that aren't released | **yes** |

The two with SIP requirements need either disabling SIP (`csrutil disable` in Recovery — a system-wide change, not recommended) **or** launching the test bundle binary directly instead of going through `/usr/bin/swift`:

```bash
# Test bundle path (after `make build-tests`):
TEST_BIN=.build/arm64-apple-macosx/release/mlx-swift-lmPackageTests.xctest/Contents/MacOS/mlx-swift-lmPackageTests

xcrun xctrace record --template 'Time Profiler' \
  --instrument 'Points of Interest' \
  --instrument 'Allocations' \
  --instrument 'Leaks' \
  --output /tmp/mlx-mem.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 MLX_BENCH_MAX_TOKENS=60 \
    MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_METHOD=summarization \
    MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
    MLX_BENCH_NGRAM=0 \
    "$TEST_BIN" --testing-library swift-testing
```

The test bundle is signed locally (not Apple-signed), so SIP doesn't restrict tracing.

#### Full capture template: kitchen-sink performance + memory profile

The kitchen-sink command for an end-to-end perf + memory snapshot of one bench cell. Doesn't include `Allocations` / `Leaks` (those need the test-bundle-direct invocation above):

```bash
xcrun xctrace record --template 'Time Profiler' \
  --instrument 'Points of Interest' \
  --instrument 'Metal Application' \
  --instrument 'Metal GPU Counters' \
  --instrument 'Metal Resource Events' \
  --instrument 'Hangs' \
  --instrument 'VM Tracker' \
  --instrument 'Virtual Memory Trace' \
  --instrument 'Thread State Trace' \
  --output /tmp/mlx-full.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 MLX_BENCH_MAX_TOKENS=60 \
    MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_METHOD=summarization \
    MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
    MLX_BENCH_NGRAM=0 \
    /usr/bin/swift test --skip-build -c release \
      --package-path "$(pwd)" --filter benchmark
```

Trace files balloon quickly with `Metal GPU Counters` + `Virtual Memory Trace` enabled — keep `MLX_BENCH_MAX_TOKENS` ≤ 60 for these, otherwise expect 1+ GB traces.

#### Saving as a reusable `.tracetemplate`

xctrace's `--instrument` stacking gets repetitive. Save the configuration once via Instruments UI, then reference the saved template by name afterwards:

1. Launch Instruments (`open -a Instruments`).
2. **File → New…** → pick **Blank**.
3. Add the instruments you want from the library (⌘L) — e.g. `Points of Interest`, `Time Profiler`, `Metal Application`, `Hangs`, `VM Tracker`.
4. **File → Save as Template…** → name it (e.g. `MLX Decode Profile`).
5. Re-record by name:
   ```bash
   xcrun xctrace record --template "MLX Decode Profile" \
     --output /tmp/mlx-decode.trace --launch -- …
   ```

The saved template persists across Xcode updates and lives in `~/Library/Application Support/Instruments/Templates/`. There isn't a clean way to commit a `.tracetemplate` file to the repo (the bundle format is tied to the installed Xcode version), so the recommendation is: save your own once locally and document the instrument list in code review.

#### Interactive Instruments capture

1. Open Instruments.
2. Choose a template (Time Profiler, Metal System Trace, or your custom).
3. `Choose target…` → browse to the test binary at:
   ```
   .build/arm64-apple-macosx/release/mlx-swift-lmPackageTests.xctest/Contents/MacOS/mlx-swift-lmPackageTests
   ```
4. Set the env vars via the target settings panel: `MLX_BENCH_PROFILE=2`, `MLX_BENCH_MODEL=...`, etc.
5. Click record and kick off the bench.

For models that finish quickly (< 1 s), bump `MLX_BENCH_MAX_TOKENS` so Instruments has time to attach and capture the hot loop cleanly.

#### Filtering & reading Phase 2 traces

In Instruments' `os_signpost` track:

- Filter by **Subsystem == `ai.mlx.bench`** to isolate benchmark signposts from system noise.
- Click a `decode_step` interval to see the CPU samples and Metal kernels that fired inside that token's budget. The `token_idx=N` metadata lets you correlate specific tokens with the output sequence.
- Expand `model_load` to see Safetensor parsing, tokenizer init, and Metal library load.
- Expand `prefill` to see the first forward pass — kernel JIT + pipeline cache population happens here.

#### Comparing TurboQuant A vs B paths

Single common recipe — just flip `useCompressedAttention` (e.g. via a benching-only env var or per-layer cache config). Both runs print the `[MLX-PROFILE]` aggregator at the end:

```bash
# A path (default — bypass-rotation; raw FP16 cache + standard SDPA)
MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_METHOD=summarization \
  MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 \
  MLX_BENCH_CONTEXT=4096 MLX_BENCH_MAX_TOKENS=200 MLX_BENCH_NGRAM=0 \
  swift test --skip-build -c release --filter benchmark

# B path (compressed-domain Metal kernels — opt in by setting
# `useCompressedAttention=true` on the cache; not env-gated by default)
```

Reading the diff:
- **`tq_encode` per-call vs `sdpa` per-call**: ratio reveals B's per-step compression overhead vs A's single fused SDPA. Typical 9:1 (e.g. 41 µs vs 2 µs) on Qwen 0.8B.
- **`decode_step` total / `tq_*` totals**: difference is the GPU async work that the CPU dispatch spans don't capture. Big gap → GPU-bound.
- **`tq_value` count (B) vs `sdpa` count (A)**: should be equal — `n_attention_layers × n_decode_tokens + prefill_chunks`. Mismatched counts mean some layers are routing through a different path.

### Phase 3 — per-kernel-dispatch tracing

Adds `MLX_METAL_PROFILE=1` on top of Phase 2. Every Metal kernel dispatch gets its own `os_signpost` interval labelled with the pipeline name. Plus the CommandEncoder lifecycle calls (`end_encoding`, `commit`, `synchronize`) emit their own intervals.

**Runtime cost:** ~60 µs/token when no tracer is attached (1500 dispatches × 2 signposts × 20 ns). Decode throughput is unchanged to within measurement noise. When recording, Instruments buffer overhead starts to matter — for long runs (>500 tokens) the recording itself can slow things down by a few percent.

**Subsystem `ai.mlx.metal`, category `PointsOfInterest`:**

| Signpost | Type | Span | Metadata |
|---|---|---|---|
| `kernel_dispatch` | interval | `set_compute_pipeline_state` → `dispatch_*` | kernel name (e.g. `sdpa_vector_bfloat16_t_64_64_nomask_qnt_nc_nosinks`) |
| `end_encoding` | interval | `CommandEncoder::end_encoding` — compute encoder closes, fences resolve | — |
| `commit` | interval | `MTL::CommandBuffer::commit` + fresh buffer allocation | — |
| `synchronize` | interval | `end_encoding` + `commit` + `waitUntilCompleted` — a full stream drain | — |

#### Phase 3 recording

```bash
# Phase 3 — includes everything Phase 2 does PLUS per-kernel labels
# and CommandEncoder lifecycle spans
xcrun xctrace record \
  --template 'Time Profiler' \
  --instrument 'Points of Interest' \
  --output /tmp/mlx-phase3.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
    MLX_BENCH_MODEL=gpt-oss-20b \
    MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
    MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=60 \
    /usr/bin/swift test --skip-build -c release \
      --package-path "$(pwd)" \
      --filter benchmark

open /tmp/mlx-phase3.trace
```

Keep `MLX_BENCH_MAX_TOKENS` modest (30–60) for Phase 3 — the trace file grows with dispatch count (roughly 3 MB per 10 tokens on GPT-OSS 20B).

#### Phase 3 examples

```bash
# Attribute decode-loop time to specific kernels
MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
  MLX_BENCH_MODEL=gpt-oss-20b \
  MLX_BENCH_MAX_TOKENS=60 \
  … (under xctrace record)

# Find the hot kernel for a new MoE model
MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
  MLX_BENCH_MODEL=qwen35-35b-a3b \
  MLX_BENCH_MAX_TOKENS=30 \
  … (under xctrace record)

# Measure per-token overhead of lifecycle calls (commit, end_encoding,
# synchronize) on a small model — useful before vs after a
# CommandEncoder refactor
MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
  MLX_BENCH_MODEL=gemma4-e2b \
  MLX_BENCH_MAX_TOKENS=100 \
  … (under xctrace record)
```

#### Filtering & reading Phase 3 traces

- In the `os_signpost` track, filter by **Subsystem == `ai.mlx.metal`** to isolate kernel and lifecycle signposts from the phase-level signposts (`ai.mlx.bench`) and system noise.
- **Group by Name** → see roll-up by kernel name with count, sum, mean. The top entries by sum are the kernels consuming the most CPU encoding time across the trace.
- **Expand a single `decode_step`** (in the `ai.mlx.bench` subsystem) → see the exact kernel sequence for that token. Useful when one token is an outlier.
- **Jump to Time Profiler samples** during a specific `kernel_dispatch` interval → see what CPU function was executing inside MLX (typically `mlx::core::metal::CommandEncoder::set_input_array`, `set_bytes`, etc.).
- **Overlay Metal System Trace** (if in the combined template) → the GPU-side kernel execution track aligns with our CPU encoding track. You'll see the encoding window (CPU) to the left of the GPU execution window for each dispatch.

Representative findings from GPT-OSS 20B 4-bit decode (30 tokens, steady-state token 15):

```
step_span=27.20 ms   1523 dispatches   kernel_encoding_CPU=2.25 ms (8.3% of step)

Top kernels by CPU encoding time across this step:
  v_copybfloat16float32              ×395  mean= 1.2µs  sum=0.46 ms
  col_reduce_small_1_reduce_sum      × 24  mean=44.6µs  sum=1.07 ms   ← outlier
  vv_Addfloat32                      ×232  mean= 1.1µs  sum=0.26 ms
  mxfp4_gather_qmv_float_gs_32_b_4   × 70  mean= 2.8µs  sum=0.20 ms
  affine_qmv_float_gs_64_b_4_batch_0 × 70  mean= 2.3µs  sum=0.16 ms
  gather_frontbfloat16_uint32_int_2  × 74  mean= 1.5µs  sum=0.11 ms
  rmsfloat32                         × 47  mean= 2.1µs  sum=0.10 ms

CommandEncoder lifecycle (over 31 decode steps):
  synchronize       ×3       957 µs mean    ~3 ms total      (explicit bench syncs)
  commit            ×7170    6.2 µs mean    ~45 ms total     (~230 commits/token)
  end_encoding      ×51552   3.2 µs mean    ~166 ms total    (~1660 end_encodes/token)
```

Observations this exposes:
- **end_encoding dominates CPU overhead**: ~5.4 ms/token (20% of decode time) spent in encoder close + fence bookkeeping. Every dispatch effectively triggers one — the command-buffer rotation granularity is "per dispatch" via `maybeInsertBarrier` logic, not "per commit".
- **1524 dispatches/token** is dominated by `v_copy*` and `vv_Add*` kernels (copies + elementwise adds) — ~627 of the 1524. Fusing these at the MLX graph layer would reduce both CPU encoding and GPU launch cost.
- **One-off outlier: `col_reduce_small_1_reduce_sum`** at 44 µs/dispatch despite small work — likely a slow-path shape hitting a non-optimal kernel variant. Worth a focused investigation.
- **Warmup tax is a single kernel**: during tokens 2-4 only, `vn_copybfloat16float32` costs ~20× its steady-state time — consistent with Metal residency-set fill for a specific buffer group.

### Exporting signpost data for scripted analysis

For comparing two runs / two branches, export signposts to XML:

```bash
xcrun xctrace export \
  --input /tmp/mlx-phase3.trace \
  --xpath '//trace-toc/run/data/table[@schema="os-signpost"]' \
  > signposts.xml
```

Each row carries `name`, `event-type` (`Begin` / `End` / `Event`), `time` (ns since trace start), `subsystem`, `category`, `os-signpost-identifier`, and any metadata. xctrace's XML uses dictionary-encoded refs — the first occurrence of a value defines `id="N"`, subsequent references use `ref="N"`. Any XML parser works; `xmllint --xpath` for ad-hoc filters, a short Python script for aggregation.

Example — top 10 kernels by sum CPU encoding time over a Phase 3 trace, in ~30 lines of Python:

```python
from xml.etree import ElementTree as ET
from collections import defaultdict
import statistics

tree = ET.parse("signposts.xml")
rows = tree.getroot().findall(".//row")

# xctrace dictionary-encodes repeat values as ref="N"; build a registry.
registry = {}
for r in rows:
    for child in r.iter():
        if "id" in child.attrib and "fmt" in child.attrib:
            registry[(child.tag, child.attrib["id"])] = child.attrib["fmt"]

def field(row, tag):
    for c in row:
        if c.tag == tag:
            if "fmt" in c.attrib: return c.attrib["fmt"]
            if "ref" in c.attrib: return registry.get((tag, c.attrib["ref"]))
            return c.text
    return None

# Match begin/end pairs, accumulate by kernel name.
durations = defaultdict(list)
open_spans = {}
for r in rows:
    if field(r, "subsystem") != "ai.mlx.metal":
        continue
    if field(r, "signpost-name") != "kernel_dispatch":
        continue
    sid = field(r, "os-signpost-identifier")
    t_ns = int(r.find("event-time").text)
    ev = field(r, "event-type")
    if ev == "Begin":
        name = None
        md = r.find("os-log-metadata")
        if md is not None:
            s = md.find("string")
            if s is not None and "fmt" in s.attrib:
                name = s.attrib["fmt"]
        open_spans[sid] = (t_ns, name)
    elif ev == "End" and sid in open_spans:
        t0, name = open_spans.pop(sid)
        if name:
            durations[name].append((t_ns - t0) / 1000.0)  # µs

ranked = sorted(
    [(n, len(d), sum(d), statistics.mean(d)) for n, d in durations.items()],
    key=lambda x: x[2], reverse=True)
for name, count, total_us, mean_us in ranked[:10]:
    print(f"{name[:60]:60}  {count:7d}  sum={total_us/1000:7.1f} ms  mean={mean_us:5.2f} µs")
```

### Combining phases with benchmark matrix runs

The profile levels compose with the regular `--model` / `--method` / `--quant` / `--kv` matrix. Set the env vars before `./scripts/benchmark.sh`:

```bash
# Phase 1 across a model sweep — one report file with [PROFILE]
# breakdowns inline for every permutation
MLX_BENCH_PROFILE=1 ./scripts/benchmark.sh \
  --model qwen35-0.8b,qwen35-2b,qwen35-9b \
  --method simple --quant 4bit --kv none

# Phase 3 for a single A/B comparison between branches — capture
# two traces on the same machine, same greedy seed, then diff
# the kernel roll-ups
for branch in alpha ek/my-optimization; do
  git checkout "$branch"
  make clean-cmlx && make
  xcrun xctrace record --template 'Time Profiler' \
    --instrument 'Points of Interest' \
    --output "/tmp/trace-${branch//\//-}.trace" \
    --launch -- /usr/bin/env \
      MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
      MLX_BENCH_MODEL=gpt-oss-20b \
      MLX_BENCH_METHOD=simple MLX_BENCH_QUANT=4bit MLX_BENCH_KV=none \
      MLX_BENCH_TEMP=0 MLX_BENCH_MAX_TOKENS=30 \
      /usr/bin/swift test --skip-build -c release \
        --package-path "$(pwd)" --filter benchmark
done
```

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
     `Config | Ctx | Prompt | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Base | GPU Peak`
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

