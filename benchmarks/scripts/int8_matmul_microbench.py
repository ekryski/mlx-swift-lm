#!/usr/bin/env python3
"""
Phase 1d: Int8 matmul microbenchmark.

Compares throughput of FP16 vs Int8 matmul on Apple Silicon GPU.
Tests whether Int8 quantization provides a speed advantage for MoE expert projections.

Typical MoE dimensions:
  - Qwen3.5-35B: input=2560, hidden=1536, numExperts=64
  - Gemma4-26B: input=2816, hidden=704, numExperts=160
  - GPT-OSS-20B: input=4096, hidden=2048, numExperts=128
"""

import mlx.core as mx
import time
import sys


def benchmark_matmul(M, K, N, dtype, warmup=5, iters=50):
    """Benchmark a single matmul: [M, K] @ [K, N] -> [M, N]"""
    a = mx.random.normal([M, K]).astype(dtype)
    b = mx.random.normal([K, N]).astype(dtype)
    mx.eval(a, b)

    # Warmup
    for _ in range(warmup):
        c = a @ b
        mx.eval(c)

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
        mx.eval(c)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000
    flops = 2 * M * K * N  # multiply-accumulate = 2 ops
    tflops = (flops * iters / elapsed) / 1e12
    return avg_ms, tflops


def benchmark_quantized_matmul(M, K, N, bits=8, group_size=64, warmup=5, iters=50):
    """Benchmark quantized matmul: [M, K] @ quantized[K, N] -> [M, N]"""
    a = mx.random.normal([M, K]).astype(mx.float16)
    b_fp = mx.random.normal([N, K]).astype(mx.float16)  # [N, K] for quantize (row-major)

    b_q, scales, biases = mx.quantize(b_fp, group_size=group_size, bits=bits)
    mx.eval(a, b_q, scales, biases)

    # Warmup
    for _ in range(warmup):
        c = mx.quantized_matmul(a, b_q, scales, biases, transpose=True,
                                 group_size=group_size, bits=bits)
        mx.eval(c)

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iters):
        c = mx.quantized_matmul(a, b_q, scales, biases, transpose=True,
                                 group_size=group_size, bits=bits)
        mx.eval(c)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000
    effective_flops = 2 * M * K * N
    tflops = (effective_flops * iters / elapsed) / 1e12
    return avg_ms, tflops


def main():
    print("=" * 80)
    print("Int8 vs FP16 vs Int4 Matmul Microbenchmark")
    print("=" * 80)
    print(f"Device: {mx.default_device()}")
    print()

    # Test configurations: (M, K, N, description)
    # M = batch/sequence tokens, K = input_dim, N = output_dim
    configs = [
        # Decode: single token through expert projection
        (1, 2560, 1536, "Qwen3.5 decode (1 token, expert)"),
        (1, 2816, 704, "Gemma4-26B decode (1 token, expert)"),
        (1, 4096, 2048, "GPT-OSS decode (1 token, expert)"),

        # Small prefill batch through expert
        (32, 2560, 1536, "Qwen3.5 small batch (32 tokens)"),
        (32, 2816, 704, "Gemma4-26B small batch (32 tokens)"),

        # Larger prefill — test where quantized loses its advantage
        (128, 2560, 1536, "Qwen3.5 medium batch (128 tokens)"),
        (512, 2560, 1536, "Qwen3.5 large batch (512 tokens)"),
        (1024, 2560, 1536, "Qwen3.5 prefill (1024 tokens)"),
        (2048, 2560, 1536, "Qwen3.5 prefill (2048 tokens)"),
        (4096, 2560, 1536, "Qwen3.5 prefill (4096 tokens)"),

        # LM head projection (large N) at various batch sizes
        (1, 2560, 151936, "Qwen3.5 LM head decode"),
        (1, 2816, 262144, "Gemma4 LM head decode"),
        (32, 2816, 262144, "Gemma4 LM head 32 tokens"),
        (128, 2816, 262144, "Gemma4 LM head 128 tokens"),
    ]

    print(f"{'Config':<42} {'FP16 ms':>9} {'Int8 ms':>9} {'Int4 ms':>9} {'8/16 ratio':>11} {'4/16 ratio':>11}")
    print("-" * 120)

    for M, K, N, desc in configs:
        # FP16 baseline
        fp16_ms, fp16_tflops = benchmark_matmul(M, K, N, mx.float16)

        # Int8 quantized
        int8_ms, int8_tflops = benchmark_quantized_matmul(M, K, N, bits=8, group_size=64)

        # Int4 quantized (our current default)
        int4_ms, int4_tflops = benchmark_quantized_matmul(M, K, N, bits=4, group_size=64)

        ratio_8_16 = int8_ms / fp16_ms
        ratio_4_16 = int4_ms / fp16_ms

        print(f"{desc:<42} {fp16_ms:>8.3f}  {int8_ms:>8.3f}  {int4_ms:>8.3f}  {ratio_8_16:>10.3f}x  {ratio_4_16:>10.3f}x")

    print()
    print("Ratios < 1.0 mean quantized is faster than FP16.")
    print("If Int8 is significantly faster than FP16 but close to Int4,")
    print("it may offer a better quality/speed tradeoff for KV cache or activations.")


if __name__ == "__main__":
    main()
