# Benchmarking

`mlx-swift-lm` ships a release-mode bench harness that measures prefill /
decode tok/s, TTFT, perplexity, KL divergence vs a baseline, and GPU memory
across model × quantization × KV-cache configurations. It writes
hardware-dated markdown reports under `benchmarks/`.

The full CLI reference, methodology, env-var API, and bench-result format
live alongside the harness at [`benchmarks/README.md`](../../benchmarks/README.md).
This page covers when to use it and the most common shapes.

## When to bench

- **Verifying a model port produces coherent output**
  → simplest smoke. Pass `--method simple` for text, `--method vision` plus
    `--vision-prompt` / `--vision-expect` for VLMs.
- **Diagnosing a perf regression**
  → run the affected rows against the most recent baseline in `benchmarks/`
    and look at the prefill / decode delta.
- **Landing a kernel or framework change**
  → re-run affected rows; if the delta is material, update the baseline.
- **Picking a deployment shape for target hardware**
  → see the README's [§ Choosing a deployment shape](../../README.md#choosing-a-deployment-shape-apple-silicon)
    and the most recent baseline for that hardware in `benchmarks/`.

## Setup

```bash
./scripts/setup-dev.sh   # one-time: SPM resolve + Metal + initial build
```

`benchmark.sh` calls `make build-tests` internally, so subsequent runs are
incremental.

## Common shapes

```bash
# Smoke: simple chat, default 4-bit, no KV compression
./scripts/benchmark.sh --model qwen35-0.8b

# VLM smoke
./scripts/benchmark.sh --model qwen2vl-2b --method vision \
    --vision-prompt "Describe what you see in this image." --vision-expect dog

# Perplexity tracking on
./scripts/benchmark.sh --model qwen35-0.8b --ppl

# Context-scaling summarization (4 contexts: 128 / 1024 / 4096 / 32768)
./scripts/benchmark.sh --model qwen35-9b --method summarization --quick

# WikiText-2 perplexity at a fixed context
./scripts/benchmark.sh --model qwen35-0.8b --method wikitext2 --context 1024

# Needle-in-a-haystack
./scripts/benchmark.sh --model qwen35-9b --method niah --context 4096

# KL divergence vs bf16 baseline
./scripts/benchmark.sh --model qwen35-0.8b --method summarization \
    --kv affine4 --kld

# Full matrix: all quants × all KV configs (long)
./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all --quick

# Multi-model, multi-KV sweep — one output file
./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2 --quick

# GPT-OSS with high-effort reasoning + thinking + PPL
./scripts/benchmark.sh --model gpt-oss-20b --reasoning high --think --ppl
```

## Reports

- One file per chip × RAM × day, e.g. `benchmarks/m1-max-64gb-2026-05-07.md`.
- All runs in a sweep land in the same file, grouped by model.
- Each row records prefill / decode (aggregated and sequential) tok/s, TTFT,
  PPL / KLD if requested, GPU baseline / peak memory, KV cache size, and a
  truncated output sample for sanity-checking.

## Bench-only env vars

The bench harness reads a number of `MLX_BENCH_*` and `MLX_METAL_*` env vars
that aren't part of the public library API. These are documented in
[benchmarks/README.md § Environment variables](../../benchmarks/README.md). Don't
treat them as inference-tuning knobs — they exist for diagnostics and A/B
testing.

For runtime knobs that **do** belong in the library API
(`compressionAlgorithm`, `prefillStepSize`, `turboBoundarySkip`, …) see the
[`GenerateParameters` table in the main README](../../README.md#generateparameters-programmatic-api).

## See also

- [`benchmarks/README.md`](../../benchmarks/README.md) — full CLI reference,
  methodology details, env-var API, report format.
- [Architecture § Build pipeline](../architecture.md#build-pipeline) — why
  bench uses `make` instead of `swift build`.
- [KV cache](../kv-cache.md) — what `--kv none / affine4 / turbo4v2` actually
  selects.
