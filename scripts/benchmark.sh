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
#
# Results saved as markdown in benchmarks/<model-family>/.

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
BASELINE=false
QUICK_CONTEXTS="128,1024,4096"

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────
show_help() {
    cat <<'HELP'
Usage: ./scripts/benchmark.sh --model MODEL [OPTIONS]

Inference benchmarks in RELEASE mode. Measures throughput, perplexity, TTFT,
KL divergence, and GPU memory.

Options:
  --model MODEL      (required) Model family or HuggingFace repo ID
  --method METHOD    Benchmark method (default: simple)
                       simple         Basic chat prompt — generation speed + PPL
                       summarization  Context-scaling with pre-sized prompts
                       wikitext2      Standard LM perplexity via forced decode
                       niah           Needle-in-a-haystack retrieval
                       multi-turn     Multi-turn conversation
                       tool-calling   Tool call generation
  --quant QUANT      Weight quantization: bf16, 8bit, 4bit, all (default: 4bit)
  --kv CONFIG        KV cache config: none, affine4, turbo4, turbo3, all (default: none)
  --context SIZE     Comma-separated context sizes (default: all 11 sizes for scaling methods)
  --quick            Quick mode: 128 + 1024 + 4096 tokens only
  --kld              Compute KL divergence vs bf16/8bit baseline
  --baseline         Auto-select highest-fidelity variant (bf16 → 8bit → 4bit)
  -h, --help         Show this help

Model families:
  qwen35-0.8b      Qwen3.5 0.8B (GatedDeltaNet)
  qwen35-2b        Qwen3.5 2B (GatedDeltaNet)
  qwen35-4b        Qwen3.5 4B (GatedDeltaNet)
  qwen35-9b        Qwen3.5 9B (GatedDeltaNet)
  qwen35-27b       Qwen3.5 27B (GatedDeltaNet)
  qwen35-35b-a3b   Qwen3.5 35B A3B (GatedDeltaNet MoE)
  gpt-oss-20b      GPT-OSS 20B
  nemotron-30b-a3b Nemotron Cascade 2 30B A3B
  <org/repo-id>    Custom HuggingFace model

Examples:
  ./scripts/benchmark.sh --model qwen35-0.8b                                  # Simple eval
  ./scripts/benchmark.sh --model qwen35-9b --method summarization --quick     # Fast summarization
  ./scripts/benchmark.sh --model qwen35-0.8b --method wikitext2 --context 1024
  ./scripts/benchmark.sh --model qwen35-0.8b --quant bf16 --kv none          # bf16 baseline
  ./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all --quick    # Full matrix
  ./scripts/benchmark.sh --model qwen35-9b --kv affine4 --kld                # With KLD
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
        --baseline) BASELINE=true; shift ;;
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

# Validate method
case "$METHOD" in
    simple|summarization|wikitext2|niah|multi-turn|tool-calling) ;;
    *) log_error "Unknown method: $METHOD"; exit 1 ;;
esac

# Build quant list
QUANTS=()
if $BASELINE; then
    QUANTS=("baseline")
elif [ "$QUANT" = "all" ]; then
    QUANTS=("bf16" "8bit" "4bit")
else
    QUANTS=("$QUANT")
fi

# Build KV list
KVS=()
case "$KV" in
    all)      KVS=("none" "affine4" "turbo4" "turbo3") ;;
    *)        KVS=("$KV") ;;
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
log_info "Model:   $MODEL"
log_info "Method:  $METHOD"
log_info "Quant:   $(IFS=,; echo "${QUANTS[*]}")"
log_info "KV:      $(IFS=,; echo "${KVS[*]}")"
$KLD && log_info "KLD:     yes"
[ -n "$CONTEXTS" ] && log_info "Context: $CONTEXTS"
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

for q in "${QUANTS[@]}"; do
    for kv in "${KVS[@]}"; do
        # Set env vars for the backend
        export MLX_BENCH_MODEL="$MODEL"
        export MLX_BENCH_METHOD="$METHOD"
        export MLX_BENCH_KV="$kv"

        if [ "$q" = "baseline" ]; then
            export MLX_BENCH_BASELINE=1
            unset MLX_BENCH_QUANT
        else
            unset MLX_BENCH_BASELINE
            export MLX_BENCH_QUANT="$q"
        fi

        if [ -n "$CONTEXTS" ]; then
            export MLX_BENCH_CONTEXT="$CONTEXTS"
        else
            unset MLX_BENCH_CONTEXT
        fi

        if $KLD; then
            export MLX_BENCH_KLD=1
        else
            unset MLX_BENCH_KLD
        fi

        log_info "Running: quant=$q kv=$kv method=$METHOD"
        swift test --skip-build -c release --filter "benchmark" 2>&1 | grep -E "\[BENCH\]|\[KLD\]|\[KV-QUANT\]|\[TURBO\]|Test.*passed|Test.*failed|error:|Fatal"
        echo ""
    done
done

log_info "Benchmark complete!"
log_info "Results saved to benchmarks/"
