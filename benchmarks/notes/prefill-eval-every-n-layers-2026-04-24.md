# Prefill eval every N layers — A/B sweep (2026-04-24)

A/B of `MLX_PREFILL_EVAL_INTERVAL=8` (spec 008) vs alpha's per-layer
`asyncEval` across the Qwen3.5 / Qwen3.6 / NemotronH lineup. Also
included: GPT-OSS-20B and Gemma 4 E4B as controls (not in the modified
code path; expected delta = 0).

Hardware: M1 Max 64 GB.
Method: `summarization`, 4-bit, `--kv none`, contexts 1024 → 32768
(doubled). All rows are a single run, so ±2% is within run-to-run
noise on this machine. Generation tok/s (Decode) column includes
warmup; for a cleaner steady-state signal see the new `Steady tok/s`
column introduced by [PR #72](https://github.com/ekryski/mlx-swift-lm/pull/72).

## Results — ∆ at N=8 vs N=1

### Affected code path (Qwen35 / NemotronH)

| Model | Ctx | Prefill Δ | Decode Δ | Peak Δ |
|---|---:|---:|---:|---:|
| **Qwen3.5 0.8B** | 1024  | +4.1%  | +4.0%  | −27.1% |
| **Qwen3.5 0.8B** | 2048  | +0.3%  | +3.2%  | −19.4% |
| **Qwen3.5 0.8B** | 4096  | +4.0%  | +5.3%  | −18.7% |
| **Qwen3.5 0.8B** | 8192  | +2.4%  | −4.1%  | −22.6% |
| **Qwen3.5 0.8B** | 16384 | +2.0%  | +11.7% | −21.0% |
| **Qwen3.5 0.8B** | 32768 | **+7.8%** | **+15.5%** | −18.1% |
| **Qwen3.5 2B**   | 1024  | +4.6%  | +12.6% | −12.6% |
| **Qwen3.5 2B**   | 2048  | **+10.9%** | **+10.7%** | −3.3%  |
| **Qwen3.5 2B**   | 4096  | **+10.5%** | **+13.2%** | −3.3%  |
| Qwen3.5 2B       | 8192  | −0.6%  | +0.2%  | −3.6%  |
| Qwen3.5 2B       | 16384 | −0.2%  | +0.3%  | −1.6%  |
| Qwen3.5 2B       | 32768 | −0.7%  | −0.5%  | +1.5%  |
| **Qwen3.5 4B**   | 1024  | −0.2%  | +5.2%  | −9.2%  |
| **Qwen3.5 4B**   | 4096  | +0.2%  | +5.1%  | −7.8%  |
| Qwen3.5 4B       | 8192+ | ±1%    | ±1%    | −3–5%  |
| **Qwen3.5 9B**   | 8192  | +2.4%  | +2.6%  | +3.0%  |
| **Qwen3.5 9B**   | 16384 | +2.9%  | +4.3%  | +2.2%  |
| Qwen3.5 9B       | other | ±1%    | ±1%    | ~flat  |
| Qwen3.5 27B      | all   | ±1%    | ±1%    | ~flat  |
| Qwen3.6 27B      | all   | ±1%    | ±2–3%  | ~flat  |
| Qwen3.5 35B A3B  | all   | −0–2%  | **−2–6%** | ~flat  |
| Nemotron 30B A3B | all   | −1–4%  | ±2%    | **+3–5%** |

Bold = meaningful win. 35B A3B ctx=4096 decode −5.7% and Nemotron
uniform +3–5% peak regression were the two regressions that drove
gating the optimization.

### Controls — not in modified code path

| Model | Prefill Δ range | Decode Δ range | Peak Δ |
|---|---:|---:|---:|
| GPT-OSS 20B | ±1% | ±1% | 0.0% (identical) |
| Gemma 4 E4B | ±2% | ±1% | 0.0% (identical) |

Controls confirm the env flag doesn't leak through unrelated code
paths.

## Gating decision

Empirically the optimization wins on **small dense hybrids**, is
neutral on **mid/large dense hybrids**, and regresses on **MoE
hybrids** and on **NemotronH**. Shipped as a gated opt-in based on a
simple static check in `Qwen35TextModelInner.init`:

```swift
self.batchedPrefillEvalEligible = (args.numExperts == 0) && (args.hiddenSize <= 4096)
```

| Model | `numExperts` | `hiddenSize` | Eligible? |
|---|---:|---:|:---:|
| Qwen3.5 0.8B    | 0   | 1024 | ✓ |
| Qwen3.5 2B      | 0   | 2048 | ✓ |
| Qwen3.5 4B      | 0   | 2560 | ✓ |
| Qwen3.5 9B      | 0   | 4096 | ✓ |
| Qwen3.5 27B     | 0   | 5120 | ✗ |
| Qwen3.6 27B     | 0   | 5120 | ✗ |
| Qwen3.5 35B A3B | 256 | 2048 | ✗ (MoE) |

NemotronH reverted to alpha's per-layer `asyncEval` (the +3–5% peak
regression across all contexts wasn't offset by any throughput gain).
GPT-OSS and Gemma 4 never entered the modified code path.

## Env flag

- `MLX_PREFILL_EVAL_INTERVAL=N` — global override.
  - `N=8` (default on eligible models): batched eval every 8 layers.
  - `N=1`: force per-layer `asyncEval` everywhere, including on
    eligible models. Primary rollback lever if a new Qwen3.5 / 3.6
    variant regresses.
  - `N` between 2 and 7: exposed for re-tuning on new hardware.

## Gate verification

On the gated code with `MLX_PREFILL_EVAL_INTERVAL=8`, Qwen3.5-27B
(ineligible) produced 78.7 / 81.2 tok/s prefill at ctx=1024 / 4096,
matching N=1 on the same thermal state (77.4 / 80.1) within ±2%.
Confirms the gate correctly bypasses the batched path on ineligible
models.

## How to reproduce

```bash
# N=1 baseline (alpha-equivalent path)
MLX_PREFILL_EVAL_INTERVAL=1 ./scripts/benchmark.sh \
  --model qwen35-0.8b,qwen35-2b,qwen35-4b,qwen35-9b,qwen35-27b,qwen36-27b,qwen35-35b-a3b,nemotron-30b-a3b,gpt-oss-20b,gemma4-e4b \
  --method summarization --quant 4bit --kv none \
  --context 1024,2048,4096,8192,16384,32768

# N=8 (default on eligible models)
./scripts/benchmark.sh --model … (same flags, drop the env var)
```

Total wall-clock for both sweeps on this machine: ~3 hours.
