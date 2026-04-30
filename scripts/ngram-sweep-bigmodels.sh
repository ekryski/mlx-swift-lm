#!/bin/bash
# Run narrow ngram-sweep on the big models, sequentially.
# Each model uses the top ~10 cells from the Qwen 0.8B + Gemma E2B small-model
# data. Output goes to /tmp/ngram-sweep-<model>.log and is also persisted to
# benchmarks/notes/ngram-sweep/<model>.log on completion.

set -e
cd "$(dirname "$0")/.."

# Top 10 cells from combined Qwen 0.8B + Gemma 4 E2B small-model data
# (median speedup ≥ 1.089×, 32-36/36 wins):
#   1. n=4 D=4  H=1   (most robust — 35/36 wins, min 0.963×, median 1.110×)
#   2. n=5 D=12 H=1   (median 1.117×, 34/36)
#   3. n=5 D=12 H=2   (median 1.120×)
#   4. n=4 D=4  H=2   (1.109×)
#   5. n=4 D=8  H=1   (1.109×)
#   6. n=3 D=16 H=1   (1.114×)
#   7. n=3 D=16 H=2   (1.103×)
#   8. n=4 D=12 H=2   (1.090×)
#   9. n=4 D=16 H=1   (1.089×)
#  10. n=5 D=16 H=2   (1.093×)
NARROW_CELLS="4:4:1,5:12:1,5:12:2,4:4:2,4:8:1,3:16:1,3:16:2,4:12:2,4:16:1,5:16:2"

# Models in run order — fastest first so we get partial data even if a later
# run dies. KV is `none` for all (per spec — eliminates KV-quant interactions).
#
# IMPORTANT: spec decode requires a trimmable KV cache. Hybrid models with
# `MambaCache` layers (cumulative recurrent state) cannot be rolled back when
# a draft token is rejected and silently fall through to plain `TokenIterator`
# inside `MLXLMCommon.generate()`. Models in that list:
#   - Qwen 3.5 family (0.8B, 2B, 4B, 9B, 27B, 35B-A3B MoE) — all use GDN
#   - Qwen 3.6 family (Qwen3Next architecture)
#   - NemotronH, Jamba, GraniteMoeHybrid, BaichuanM1, FalconH1, LFM2 / LFM2MoE
# We omit them from this sweep since they'd produce identical baseline numbers
# whether ngram is "on" or "off".
MODELS=(
    "gpt-oss-20b"
    "gemma4-26b-a4b"
)

for model in "${MODELS[@]}"; do
    log="/tmp/ngram-sweep-${model}.log"
    persist="benchmarks/notes/ngram-sweep/${model}.log"

    if [ -s "$log" ]; then
        echo "[chain] $model already has a log at $log — skipping. Delete to rerun."
        continue
    fi

    echo
    echo "=================================================================="
    echo "[chain] Running narrow sweep on $model — $(date)"
    echo "[chain] Log: $log"
    echo "=================================================================="

    if MLX_BENCH_MAX_TOKENS=100 \
       MLX_BENCH_NGRAM_SWEEP_CELLS="$NARROW_CELLS" \
       ./scripts/benchmark.sh --model "$model" --method ngram-sweep --kv none \
       2>&1 | tee "$log"; then
        echo "[chain] $model done — persisting log to $persist"
        cp "$log" "$persist"
        # Run analysis and persist alongside.
        python3 scripts/ngram-sweep-analyze.py "$log" 2>/dev/null \
            > "${persist%.log}-analysis.md" || true
    else
        echo "[chain] $model FAILED — continuing with next model"
    fi
done

echo
echo "[chain] All big-model sweeps complete. See benchmarks/notes/ngram-sweep/"
