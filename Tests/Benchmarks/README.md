# Inference Benchmarks

Automated benchmarks for MLX Swift LM inference across model families, weight quantizations, and KV cache compression strategies running on Apple Silicon.

## Quick Start

```bash
# Simple chat evaluation (default)
./scripts/benchmark.sh --model qwen35-0.8b

# Context-scaling summarization
./scripts/benchmark.sh --method summarization --model qwen35-9b --quick

# WikiText-2 perplexity at a specific context
./scripts/benchmark.sh --method wikitext2 --model qwen35-0.8b --context 1024

# Needle-in-a-haystack
./scripts/benchmark.sh --method niah --model qwen35-9b --context 4096

# With KLD quality metrics and specific KV config
./scripts/benchmark.sh --method summarization --model qwen35-0.8b --kv affine4 --kld

# Run all methods
./scripts/benchmark.sh --method all --model qwen35-0.8b --quick
```

## Methods

| Method | Description | Context Scaling | Generation |
|--------|-------------|:---:|:---:|
| `simple` | Basic chat prompt — generation speed + PPL | No | Yes |
| `summarization` | Pre-sized prompts across 11 context sizes | Yes | Yes |
| `wikitext2` | Standard LM perplexity via forced decode on WikiText-2 | Yes | No |
| `niah` | Needle-in-a-haystack retrieval at various depths | Yes | Yes |
| `multi-turn` | 3-message multi-turn conversation | No | Yes |
| `tool-calling` | Tool call generation | No | Yes |

Default is `simple` when no `--method` is specified.

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--method METHOD` | Benchmark method (see above) | `simple` |
| `--model MODEL` | Model family (e.g., `qwen35-9b`) or `hf:org/repo` | All models |
| `--quant QUANT` | Weight quantization: `bf16`, `8bit`, `4bit` | `4bit` |
| `--kv CONFIG` | KV cache config: `none`, `affine4`, `turbo4`, `turbo3`, `all` | `all` |
| `--context SIZE` | Single context size (e.g., `1024`) | All sizes |
| `--quick` | Quick mode: 128 + 1024 + 4096 tokens only | Off |
| `--kld` | Compute KL divergence quality metrics against baseline | Off |
| `--baseline` | Auto-select highest-fidelity variant that fits in GPU memory | Off |
| `--speed` | Alias for `--method simple` | — |
| `--tool` | Alias for `--method tool-calling` | — |
| `--all` | Alias for `--method all` | — |

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
| Nemotron 30B A3B | `nemotron-30b-a3b` | 8bit, 4bit | Transformer MoE |

## Context Sizes

Context-scaling methods (`summarization`, `wikitext2`, `niah`) run across 11 sizes:

**128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K** tokens

## KV Cache Configurations

| Config | Description |
|--------|-------------|
| `none` | Standard unquantized KV cache |
| `affine4` | MLX affine 4-bit quantization (starts at offset 512) |
| `turbo4` | TurboQuant MSE 4-bit compression (starts at offset 0) |
| `turbo3` | TurboQuant MSE 3-bit compression (starts at offset 0) |

## Methodology

### Simple

Sends a basic chat prompt ("Hello! What is your name and what can you help me with?") and measures generation speed, TTFT, and perplexity. Quick single evaluation without context scaling.

### Summarization

Loads pre-generated prompt files (excerpts from The Great Gatsby) sized to each target token count. The model generates a summary response. Measures prefill throughput, generation throughput, TTFT, perplexity, and GPU memory at each context size.

### WikiText-2

Standard LM perplexity evaluation via **forced decode**. Tokenizes WikiText-2 test data, feeds it through the model, and computes the log-probability of predicting each next token. No generation — pure evaluation of the model's predictive ability.

PPL = exp(mean negative log-probability over all positions)

This is the industry-standard perplexity metric. KV cache quantization applies during the forced decode, measuring its impact on model quality.

### Needle-in-a-Haystack (NIAH)

Tests long-context retrieval by inserting a known fact ("The secret verification code is BLUE TIGER 42.") at approximately 50% depth in filler text, then asking the model to retrieve it. Measures whether the model can find and return the needle, along with standard throughput and perplexity metrics.

### Perplexity (Think PPL / Gen PPL)

Perplexity is computed as `exp(mean negative log-probability)` over generated tokens. It is tracked separately for the **thinking phase** (tokens between `<think>` and `</think>`) and the **generation phase** (tokens after `</think>`). Lower values indicate higher model confidence.

For models that support thinking mode, a thinking budget processor forces `</think>` after 200 thinking tokens and suppresses EOS during thinking to ensure both phases are measured.

### KL Divergence (Think KLD / Gen KLD)

When `--kld` is enabled, KL divergence measures how much a deployment configuration (weight quantization + KV cache compression) degrades the model's output distribution compared to the highest-fidelity baseline available for that model family.

**How it works:**

1. The target model generates tokens normally with per-token log-probability tracking.
2. The highest-fidelity baseline model (bf16 preferred, 8-bit fallback if bf16 exceeds GPU memory) is loaded without KV cache compression.
3. The target's generated tokens are **forced-decoded** through the baseline model — each token is fed sequentially, and the baseline's log-probability for that token is recorded.
4. KLD is computed per phase as: `mean(target_logprob - baseline_logprob)`.

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
- **KV Delta**: The increase in active GPU memory from the KV cache, measured as `activeGPU - baselineGPU` after generation completes. For KV-quantized runs (affine4, turbo4, turbo3), this reflects the compressed cache size. Comparing KV Delta between `no-quant` and a quantized config at the same context shows how much memory the compression saves.

**Why GPU Peak is much higher than GPU Baseline + KV Delta:**

The gap is primarily intermediate computation tensors allocated during the forward pass — attention scores, QKV projections, FFN activations, softmax buffers, Conv1d state (for GatedDeltaNet), and recurrent state updates. These are allocated during each forward step and freed afterward, but they contribute to the peak memory high-water mark.

Key factors:
- **Prefill dominates peak**: Prefill processes the full prompt at once (e.g., 1024 tokens), creating much larger intermediate tensors than single-token generation. The peak is usually hit during prefill.
- **MLX memory pool**: MLX does not immediately return freed memory to the OS — it caches freed allocations for reuse. `peakMemory` reflects the cumulative high-water mark, not just what is actively held.
- **GatedDeltaNet overhead**: The hybrid GatedDeltaNet architecture (used by all Qwen3.5 models) has higher intermediate memory than standard transformers due to simultaneous QKV projections, conv state concatenation, and gated delta updates per layer.

## Method × Flag Interaction

| Method | `--context` | `--kv` | `--kld` | `--quick` |
|--------|:---:|:---:|:---:|:---:|
| simple | No effect | Yes | Yes | No effect |
| summarization | Yes | Yes | Yes | Yes |
| wikitext2 | Yes | Yes | N/A (IS perplexity) | Yes |
| niah | Yes | Yes | Yes | Yes |
| multi-turn | No effect | Yes | Yes | No effect |
| tool-calling | No effect | Yes | Yes | No effect |

## Output

Benchmark reports are saved as Markdown files organized by model family:

```
benchmarks/
├── qwen3.5-0.8b/
│   ├── qwen3.5-0.8b-4bit-affine-4-2026-04-01-0957-benchmark.md
│   ├── qwen3.5-0.8b-bf16-no-quant-2026-03-30-1210-benchmark.md
│   └── ...
├── qwen3.5-9b/
│   └── ...
├── gpt-oss-20b/
│   └── ...
└── ...
```

Each report contains hardware info, generation parameters, methodology notes, and a results table with: scenario, prefill/generation throughput, TTFT, perplexity (think/gen), KLD (think/gen), GPU memory usage, and output previews.
