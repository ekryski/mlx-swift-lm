#!/bin/bash
# Inference Speed Benchmarks — Release Mode
#
# Benchmarks inference speed, perplexity, and memory across model families,
# quantization levels, and KV cache configurations.
#
# IMPORTANT: Builds and runs in RELEASE mode for accurate benchmarks.
#
# Usage:
#   ./scripts/benchmark.sh                              # Run full matrix
#   ./scripts/benchmark.sh --model qwen35-9b            # Single model family
#   ./scripts/benchmark.sh --quant bf16                  # Use bf16 quantization
#   ./scripts/benchmark.sh --baseline --model qwen35-9b  # Run bf16 baseline
#   ./scripts/benchmark.sh --model hf:org/repo-id       # Custom HuggingFace model
#   ./scripts/benchmark.sh --quick                       # 128+1024+4096 only
#   ./scripts/benchmark.sh --context 1024                # Single context size
#   ./scripts/benchmark.sh --kv turbo4                   # Single KV config
#   ./scripts/benchmark.sh --speed                       # Tool call + multi-turn only
#
# Results are printed with [BENCH] prefix and saved as markdown in benchmarks/.

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
MODE="context"  # context | speed | tool | all
MODEL_FILTER=""
KV_FILTER=""
CONTEXT_FILTER=""
QUANT=""
BASELINE=false
QUICK=false
CUSTOM_MODEL=""
QUICK_CONTEXTS="128,1024,4096"

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────
show_help() {
    cat <<'HELP'
Usage: ./scripts/benchmark.sh [OPTIONS]

Inference speed benchmarks in RELEASE mode. Measures prefill/generation throughput,
perplexity, TTFT, and GPU memory.

Options:
  --model MODEL      Filter by model family or custom HuggingFace repo
  --quant QUANT      Model quantization: bf16, 8bit, 4bit, nvfp4, mxfp4 (default: 4bit)
  --baseline         Run with highest-fidelity variant that fits in memory (bf16 → 8bit → 4bit)
  --kv CONFIG        KV cache config: none, affine4, turbo4, turbo3, all (default: all)
  --context SIZE     Single context size (e.g., 1024)
  --quick            Quick comparison: 128 + 1024 + 4096 tokens only
  --speed            Simple query speed tests (no context scaling)
  --tool             Tool calling tests
  --all              Run everything (context + speed + tool)
  -h, --help         Show this help

Model families:
  qwen35-0.8b      Qwen3.5 0.8B
  qwen35-2b        Qwen3.5 2B
  qwen35-4b        Qwen3.5 4B
  qwen35-9b        Qwen3.5 9B
  qwen35-27b       Qwen3.5 27B
  qwen35-35b-a3b   Qwen3.5 35B A3B (MoE)
  gpt-oss-20b      GPT-OSS 20B
  nemotron-30b-a3b Nemotron Cascade 2 30B A3B
  hf:org/repo-id   Custom HuggingFace model

Examples:
  ./scripts/benchmark.sh --quick                              # Fast: 3 contexts × all
  ./scripts/benchmark.sh --model qwen35-27b --kv turbo4       # Single model+KV config
  ./scripts/benchmark.sh --model qwen35-9b --quant bf16       # 9B in bf16
  ./scripts/benchmark.sh --baseline --model qwen35-9b         # bf16 baseline (auto memory check)
  ./scripts/benchmark.sh --model hf:mlx-community/Qwen3.5-9B-4bit --context 128
  ./scripts/benchmark.sh --context 1024                       # All models at 1024 tokens
  ./scripts/benchmark.sh --speed --model gpt-oss-20b          # Speed test
HELP
}

# ─────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL_FILTER="$2"; shift 2 ;;
        --quant)    QUANT="$2"; shift 2 ;;
        --baseline) BASELINE=true; shift ;;
        --kv)       KV_FILTER="$2"; shift 2 ;;
        --context)  CONTEXT_FILTER="$2"; shift 2 ;;
        --quick)    QUICK=true; shift ;;
        --speed)    MODE="speed"; shift ;;
        --tool)     MODE="tool"; shift ;;
        --all)      MODE="all"; shift ;;
        -h|--help)  show_help; exit 0 ;;
        *) log_error "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────
# Handle custom HuggingFace model
# ─────────────────────────────────────────────
if [[ "$MODEL_FILTER" == hf:* ]]; then
    CUSTOM_MODEL="${MODEL_FILTER#hf:}"
    MODEL_FILTER="custom"
fi

# ─────────────────────────────────────────────
# Map model shortnames to test function prefixes
# ─────────────────────────────────────────────
model_to_test_prefix() {
    case "$1" in
        qwen35-0.8b)     echo "qwen35_08B" ;;
        qwen35-2b)       echo "qwen35_2B" ;;
        qwen35-4b)       echo "qwen35_4B" ;;
        qwen35-9b)       echo "qwen35_9B" ;;
        qwen35-27b)      echo "qwen35_27B" ;;
        qwen35-35b-a3b)  echo "qwen35_35B_A3B" ;;
        gpt-oss-20b)     echo "gptOSS20B" ;;
        nemotron-30b-a3b) echo "nemotron30B" ;;
        custom)          echo "custom" ;;
        *) log_error "Unknown model: $1"; exit 1 ;;
    esac
}

# ─────────────────────────────────────────────
# Build test filter
# ─────────────────────────────────────────────
build_filter() {
    local filters=()

    local models=()
    case "$MODEL_FILTER" in
        custom)
            models=("custom")
            ;;
        ""|all)
            models=(
                "qwen35_08B" "qwen35_2B" "qwen35_4B" "qwen35_9B"
                "qwen35_27B" "qwen35_35B_A3B" "gptOSS20B" "nemotron30B"
            )
            ;;
        *)
            models=("$(model_to_test_prefix "$MODEL_FILTER")")
            ;;
    esac

    # KV config suffixes
    local kvs=()
    case "$KV_FILTER" in
        none)     kvs=("noQuant") ;;
        affine4)  kvs=("affine4") ;;
        turbo4)   kvs=("turbo4") ;;
        turbo3)   kvs=("turbo3") ;;
        all|"")   kvs=("noQuant" "affine4" "turbo4" "turbo3") ;;
        *) log_error "Unknown KV config: $KV_FILTER"; exit 1 ;;
    esac

    for model in "${models[@]}"; do
        for kv in "${kvs[@]}"; do
            local test_name="${model}_${kv}"

            # Skip invalid combinations
            if [[ "$model" == "gptOSS20B" && "$kv" == "affine4" ]]; then
                continue
            fi

            case "$MODE" in
                context)
                    filters+=("$test_name")
                    ;;
                speed)
                    if [[ "$kv" == "noQuant" || "$kv" == "turbo4" ]]; then
                        filters+=("${model}_tool_${kv}")
                        filters+=("${model}_multiTurn_${kv}")
                    fi
                    ;;
                tool)
                    if [[ "$kv" == "noQuant" || "$kv" == "turbo4" ]]; then
                        filters+=("${model}_tool_${kv}")
                    fi
                    ;;
                all)
                    filters+=("$test_name")
                    if [[ "$kv" == "noQuant" || "$kv" == "turbo4" ]]; then
                        filters+=("${model}_tool_${kv}")
                        filters+=("${model}_multiTurn_${kv}")
                    fi
                    ;;
            esac
        done
    done

    # Join with | for regex alternation
    local IFS='|'
    echo "${filters[*]}"
}

FILTER=$(build_filter)
if [ -z "$FILTER" ]; then
    log_error "No test cases match the given filters"
    exit 1
fi

# ─────────────────────────────────────────────
# Print configuration
# ─────────────────────────────────────────────
echo ""
log_info "╔══════════════════════════════════════════╗"
log_info "║   Inference Speed Benchmarks             ║"
log_info "║   RELEASE MODE                           ║"
log_info "╚══════════════════════════════════════════╝"
log_info ""
log_info "Mode:    $MODE"
log_info "Models:  ${MODEL_FILTER:-all}"
[ -n "$QUANT" ] && log_info "Quant:   $QUANT"
$BASELINE && log_info "Baseline: yes (bf16 → 8bit → 4bit auto-select)"
log_info "KV:      ${KV_FILTER:-all}"
$QUICK && log_info "Quick:   yes (128+1024+4096 only)"
[ -n "$CONTEXT_FILTER" ] && log_info "Context: $CONTEXT_FILTER tokens"
[ -n "$CUSTOM_MODEL" ] && log_info "Custom:  $CUSTOM_MODEL"
log_info "Filter:  $FILTER"
log_info ""

# ─────────────────────────────────────────────
# Build in release mode
# ─────────────────────────────────────────────
cd "$PROJECT_ROOT"

log_info "Building test target in RELEASE mode..."
swift build --build-tests -c release -Xswiftc -enable-testing 2>&1 | tail -3

# ─────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────
log_info "Running benchmarks..."
echo ""

# Set environment variables
export MLX_CONTEXT_BENCH=1
export MLX_SPEED_BENCH=1

# Quantization selection
if $BASELINE; then
    export MLX_BENCH_BASELINE=1
elif [ -n "$QUANT" ]; then
    export MLX_BENCH_QUANT="$QUANT"
fi

# Custom model
if [ -n "$CUSTOM_MODEL" ]; then
    export MLX_BENCH_MODEL="$CUSTOM_MODEL"
fi

# Quick mode limits context sizes via environment
if $QUICK; then
    export MLX_BENCH_CONTEXT="$QUICK_CONTEXTS"
fi
if [ -n "$CONTEXT_FILTER" ]; then
    export MLX_BENCH_CONTEXT="$CONTEXT_FILTER"
fi

# Run with release configuration
swift test --skip-build -c release --filter "$FILTER" 2>&1 | grep -E "\[BENCH\]|\[KV-QUANT\]|\[TURBO\]|Test.*passed|Test.*failed|error:|Fatal"

echo ""
log_info "Benchmark complete!"
log_info "Results saved to benchmarks/"
log_info "Parse results with: grep '\\[BENCH\\]' to extract data"
