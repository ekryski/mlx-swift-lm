# Inference Benchmarks

Automated benchmarks for MLX Swift LM inference across model families, weight quantizations, and KV cache compression strategies running on Apple Silicon.

The CLI (`benchmark.sh`) is designed to be language-agnostic — all configuration is passed via environment variables, making it straightforward to add backends in other languages (Python, Java) for cross-platform benchmarking.

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
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--model MODEL` | **(required)** Model family or HuggingFace repo ID | — |
| `--method METHOD` | Benchmark method (see Methods below) | `simple` |
| `--quant QUANT` | Weight quantization: `bf16`, `8bit`, `4bit`, `all` | `4bit` |
| `--kv CONFIG` | KV cache config: `none`, `affine4`, `turbo4`, `turbo3`, `all` | `none` |
| `--context SIZES` | Comma-separated context sizes (e.g., `128,1024,4096`) | All 11 sizes |
| `--quick` | Quick mode: 128 + 1024 + 4096 tokens only | Off |
| `--ppl` | Track per-token perplexity during generation | Off |
| `--kld` | Compute KL divergence vs bf16/8bit baseline | Off |
| `--baseline` | Auto-select highest-fidelity variant that fits in GPU memory | Off |
| `--batch N` | Run N concurrent generations (default: 1) | `1` |
| `--think` | Enable thinking mode for thinking-capable models | Off |

> **Max speed tip:** For pure throughput measurements, omit `--ppl` and `--kld`. Both flags add significant compute overhead — `--ppl` tracks per-token log-probabilities during generation, and `--kld` loads a second baseline model and runs a full forced-decode pass after generation completes. Leave them off when you only care about tok/s and TTFT.

When `--quant all` is specified, the CLI loops over bf16, 8bit, and 4bit sequentially. When `--kv all`, it loops over none, affine4, turbo4, and turbo3. These can be combined for a full matrix.

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

## Context Sizes

Context-scaling methods (`summarization`, `wikitext2`, `niah`) run across 11 sizes by default:

**128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K** tokens

Use `--context` to specify a subset, or `--quick` for 128 + 1024 + 4096.

For non-scaling methods (`simple`, `multi-turn`, `tool-calling`), the context limit is fixed at 4096 tokens, enforced via `RotatingKVCache` (`maxKVSize`) to simulate a realistic chat deployment.

## KV Cache Configurations

| Config | Description |
|--------|-------------|
| `none` | Standard unquantized KV cache |
| `affine4` | MLX affine 4-bit quantization (starts at offset 512) |
| `turbo4` | TurboQuant MSE 4-bit compression (starts at offset 0) |
| `turbo3` | TurboQuant MSE 3-bit compression (starts at offset 0) |

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
| `MLX_BENCH_MODEL` | registry name or HF repo | *required* | Model to benchmark |
| `MLX_BENCH_METHOD` | simple, summarization, wikitext2, niah, multi-turn, tool-calling | simple | Test method |
| `MLX_BENCH_QUANT` | bf16, 8bit, 4bit | 4bit | Weight quantization |
| `MLX_BENCH_KV` | none, affine4, turbo4, turbo3 | none | KV cache config |
| `MLX_BENCH_CONTEXT` | comma-separated ints | all 11 sizes | Context sizes to test |
| `MLX_BENCH_PPL` | 1 | unset | Enable perplexity tracking |
| `MLX_BENCH_KLD` | 1 | unset | Enable KLD computation |
| `MLX_BENCH_BASELINE` | 1 | unset | Auto-select best quant |
| `MLX_BENCH_BATCH` | integer | 1 | Number of concurrent generations |
| `MLX_BENCH_THINK` | 1 | unset | Enable thinking mode |

## Output

Benchmark reports are saved as Markdown files organized by model family:

```
benchmarks/
├── qwen3.5-0.8b/
│   ├── qwen3.5-0.8b-4bit-no-quant-simple-benchmark-2026-04-01-1120.md
│   ├── qwen3.5-0.8b-4bit-affine-4-summarization-benchmark-2026-04-01-0957.md
│   └── ...
├── qwen3.5-9b/
│   └── ...
├── gpt-oss-20b/
│   └── ...
└── ...
```

Filenames follow the pattern: `{model}-{quant}-{kv}-{method}-benchmark-{date}.md`

Each report contains hardware info, generation parameters, methodology notes, and a results table with: method, context limit, prompt tokens, KV config, prefill/generation throughput, TTFT, perplexity (think/gen), KLD (think/gen), GPU memory metrics, and output previews with pass/fail status.
