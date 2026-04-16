#!/bin/bash
# Inference Benchmarks — Release Mode
#
# Language-agnostic CLI that drives benchmark backends via environment variables.
# Currently supports the Swift backend (mlx-swift-lm).
#
# Usage:
#   ./scripts/benchmark.sh --model qwen35-0.8b                     # Simple eval (default)
#   ./scripts/benchmark.sh --model qwen35-0.8b --method summarization --quick
#   ./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all # Full matrix
#   ./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2
#
# --model, --method, --quant, and --kv all accept comma-separated lists.
# All permutations (model × quant × kv × method) run in sequence; every row
# lands in the same hardware-dated file at benchmarks/{chip}-{ram}-{date}.md,
# grouped by model.

set -e

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
METHOD="simple"
MODEL=""
QUANT="4bit"
KV="none"
CONTEXT=""
QUICK=false
KLD=false
PPL=false
BASELINE=false
BATCH=1
THINK=false
REASONING="medium"
QUICK_CONTEXTS="128,1024,4096,32768"

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────
show_help() {
    cat <<'HELP'
Usage: ./scripts/benchmark.sh --model MODEL [OPTIONS]

Inference benchmarks in RELEASE mode. Measures throughput, perplexity, TTFT,
KL divergence, and GPU memory.

Options:
  --model MODELS     (required) One or more model families / HuggingFace repo IDs
                       Comma-separated for multiple: qwen35-0.8b,qwen35-2b
  --method METHODS   Benchmark method(s), comma-separated (default: simple)
                       simple         Basic chat prompt — generation speed + PPL
                       summarization  Context-scaling with pre-sized prompts
                       wikitext2      Standard LM perplexity via forced decode
                       niah           Needle-in-a-haystack retrieval
                       multi-turn     Multi-turn conversation
                       tool-calling   Tool call generation
  --quant QUANTS     Weight quantization(s): bf16, 8bit, 4bit, all (default: 4bit)
                       Comma-separated for multiple: bf16,4bit
  --kv CONFIGS       KV cache config(s): none, affine8, affine4, turbo4, turbo4v2,
                     turbo4v3, turbo3, turbo3v2, turbo8, turbo8v2, turbo8v4, all
                     (default: none). Comma-separated for multiple: none,turbo4v2
  --context SIZE     Comma-separated context sizes (default: all 11 sizes for scaling methods)
  --quick            Quick mode: 128 + 1024 + 4096 + 32768 tokens only
  --ppl              Track per-token perplexity during generation
  --kld              Compute KL divergence vs bf16/8bit baseline
  --baseline         Auto-select highest-fidelity variant (bf16 → 8bit → 4bit)
  --batch N          Run N concurrent generations (default: 1)
  --think            Enable thinking mode for thinking-capable models
  --reasoning EFFORT Reasoning effort for models that support it (GPT-OSS).
                     Values: low, medium, high (default: medium).
                     Maps to MLX_BENCH_REASONING; ignored by models without
                     a reasoning_effort setting.
  -h, --help         Show this help

Model families:
  qwen35-0.8b      Qwen3.5 0.8B (GatedDeltaNet)
  qwen35-2b        Qwen3.5 2B (GatedDeltaNet)
  qwen35-4b        Qwen3.5 4B (GatedDeltaNet)
  qwen35-9b        Qwen3.5 9B (GatedDeltaNet)
  qwen35-27b       Qwen3.5 27B (GatedDeltaNet)
  qwen35-35b-a3b   Qwen3.5 35B A3B (GatedDeltaNet MoE)
  gpt-oss-20b      GPT-OSS 20B
  nemotron-30b-a3b Nemotron Cascade 2 30B A3B (nemotron_h; also: nemotron-cascade-2,
                   nemotron-cascade2, nemotron-cascade-2-30b-a3b, …)
  gemma4-e2b       Gemma 4 E2B (Dense, ~2B)
  gemma4-e4b       Gemma 4 E4B (Dense, ~4B)
  gemma4-26b-a4b   Gemma 4 26B A4B (MoE, 128 experts)
  gemma4-31b       Gemma 4 31B (Dense)
  <org/repo-id>    Custom HuggingFace model

Examples:
  ./scripts/benchmark.sh --model qwen35-0.8b                                      # Simple eval
  ./scripts/benchmark.sh --model qwen35-9b --method summarization --quick         # Fast summarization
  ./scripts/benchmark.sh --model qwen35-0.8b --method wikitext2 --context 1024
  ./scripts/benchmark.sh --model qwen35-0.8b --quant bf16 --kv none               # bf16 baseline
  ./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all --quick         # Full matrix
  ./scripts/benchmark.sh --model qwen35-9b --kv affine4 --kld                     # With KLD
  ./scripts/benchmark.sh --model qwen35-0.8b,qwen35-2b --kv none,turbo4v2 --quick # Multi-model sweep
  ./scripts/benchmark.sh --model qwen35-0.8b --method simple,summarization        # Two methods
  ./scripts/benchmark.sh --model nemotron-cascade-2 --quant 4bit                  # Nemotron (alias)
  ./scripts/benchmark.sh --model gpt-oss-20b --reasoning high --think --ppl       # Harmony reasoning at 'high'

HELP
}

# ─────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --method)   METHOD="$2"; shift 2 ;;
        --quant)    QUANT="$2"; shift 2 ;;
        --kv)       KV="$2"; shift 2 ;;
        --context)  CONTEXT="$2"; shift 2 ;;
        --quick)    QUICK=true; shift ;;
        --kld)      KLD=true; shift ;;
        --ppl)      PPL=true; shift ;;
        --baseline) BASELINE=true; shift ;;
        --batch)    BATCH="$2"; shift 2 ;;
        --think)    THINK=true; shift ;;
        --reasoning) REASONING="$2"; shift 2 ;;
        -h|--help)  show_help; exit 0 ;;
        *) log_error "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

# Validate required args
if [ -z "$MODEL" ]; then
    log_error "--model is required"
    show_help
    exit 1
fi

# Build model list (comma-separated for multi-model sweeps)
MODELS=()
IFS=',' read -ra MODELS <<< "$MODEL"

# Build method list (comma-separated for multi-method sweeps)
METHODS=()
IFS=',' read -ra METHODS <<< "$METHOD"
for m in "${METHODS[@]}"; do
    case "$m" in
        simple|summarization|wikitext2|niah|multi-turn|tool-calling) ;;
        *) log_error "Unknown method: $m"; exit 1 ;;
    esac
done

# Validate reasoning effort — warn rather than error so new model vocabularies
# pass through without requiring a harness change.
case "$REASONING" in
    low|medium|high) ;;
    *) log_warn "Unusual reasoning effort '$REASONING' (expected low/medium/high); passing through" ;;
esac

# Build quant list
QUANTS=()
if $BASELINE; then
    QUANTS=("baseline")
elif [ "$QUANT" = "all" ]; then
    QUANTS=("bf16" "8bit" "4bit")
else
    IFS=',' read -ra QUANTS <<< "$QUANT"
fi

# Build KV list
KVS=()
case "$KV" in
    all)  KVS=("none" "affine8" "affine4" "turbo4" "turbo3" "turbo4v2") ;;
    *)    IFS=',' read -ra KVS <<< "$KV" ;;
esac

# Context sizes
CONTEXTS="$CONTEXT"
if $QUICK; then
    CONTEXTS="$QUICK_CONTEXTS"
fi

# ─────────────────────────────────────────────
# Print configuration
# ─────────────────────────────────────────────
echo ""
log_info "╔══════════════════════════════════════════╗"
log_info "║   Inference Benchmarks                   ║"
log_info "║   RELEASE MODE                           ║"
log_info "╚══════════════════════════════════════════╝"
log_info ""
log_info "Models:  $(IFS=,; echo "${MODELS[*]}")"
log_info "Methods: $(IFS=,; echo "${METHODS[*]}")"
log_info "Quants:  $(IFS=,; echo "${QUANTS[*]}")"
log_info "KVs:     $(IFS=,; echo "${KVS[*]}")"
$KLD && log_info "KLD:     yes"
$PPL && log_info "PPL:     yes"
$THINK && log_info "Think:   yes (reasoning=$REASONING)"
[ "$BATCH" -gt 1 ] && log_info "Batch:   $BATCH"
[ -n "$CONTEXTS" ] && log_info "Context: $CONTEXTS"
log_info ""

# ─────────────────────────────────────────────
# Build in release mode
# ─────────────────────────────────────────────
cd "$PROJECT_ROOT"

log_info "Building (make build-tests)..."
if ! make -C "$PROJECT_ROOT" build-tests; then
    log_error "make build-tests failed. Re-run: make build-tests"
    exit 1
fi

# ─────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────
log_info "Running benchmarks..."
echo ""

# Common env vars (stable across all permutations).
if [ -n "$CONTEXTS" ]; then export MLX_BENCH_CONTEXT="$CONTEXTS"; else unset MLX_BENCH_CONTEXT; fi
if $KLD; then export MLX_BENCH_KLD=1; else unset MLX_BENCH_KLD; fi
if ${PPL:-false}; then export MLX_BENCH_PPL=1; else unset MLX_BENCH_PPL; fi
if [ "$BATCH" -gt 1 ]; then export MLX_BENCH_BATCH="$BATCH"; else unset MLX_BENCH_BATCH; fi
if $THINK; then export MLX_BENCH_THINK=1; else unset MLX_BENCH_THINK; fi
# Reasoning effort — only set when non-default so unset tells the harness to
# fall back to the model family's registered value.
if [ "$REASONING" != "medium" ]; then export MLX_BENCH_REASONING="$REASONING"; else unset MLX_BENCH_REASONING; fi

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#METHODS[@]} * ${#QUANTS[@]} * ${#KVS[@]} ))
RUN_INDEX=0
FAILED_RUNS=()

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for q in "${QUANTS[@]}"; do
            for kv in "${KVS[@]}"; do
                RUN_INDEX=$(( RUN_INDEX + 1 ))

                export MLX_BENCH_MODEL="$model"
                export MLX_BENCH_METHOD="$method"
                export MLX_BENCH_KV="$kv"

                if [ "$q" = "baseline" ]; then
                    export MLX_BENCH_BASELINE=1
                    unset MLX_BENCH_QUANT
                else
                    unset MLX_BENCH_BASELINE
                    export MLX_BENCH_QUANT="$q"
                fi

                log_info "[$RUN_INDEX/$TOTAL_RUNS] model=$model method=$method quant=$q kv=$kv"

                # Stream filtered output in real-time via a PTY (script -q) so Swift Testing
                # flushes print() calls mid-test. Full output captured for post-mortem.
                TMPOUT=$(mktemp)
                script -q /dev/null swift test --skip-build -c release --filter "benchmark" 2>&1 \
                    | tee "$TMPOUT" \
                    | grep -E --line-buffered "\[ENV\]|\[WARMUP\]|\[BENCH\]|\[MEM\]|\[KLD\]|\[RESULT\]|\[KV-QUANT\]|\[TURBO\]|\[PROGRESS\]|Test.*passed|Test.*failed|[Ee]rror|[Ff]atal|BenchmarkError|threw|[Ee]xception|issue at"
                EXIT_CODE=${PIPESTATUS[0]}

                if [ "$EXIT_CODE" -ne 0 ]; then
                    log_error "Run failed (exit=$EXIT_CODE model=$model method=$method quant=$q kv=$kv):"
                    grep -iE "error|fatal|BenchmarkError|threw|exception|exceeds|issue" "$TMPOUT" | tail -20
                    FAILED_RUNS+=("$model/$method/$q/$kv")
                fi

                rm -f "$TMPOUT"
                echo ""
            done
        done
    done
done

log_info "Benchmark complete! $((TOTAL_RUNS - ${#FAILED_RUNS[@]}))/$TOTAL_RUNS runs succeeded."
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log_warn "Failed runs:"
    for r in "${FAILED_RUNS[@]}"; do log_warn "  - $r"; done
fi
log_info "Results: benchmarks/{chip}-{ram}-$(date +%Y-%m-%d).md"
