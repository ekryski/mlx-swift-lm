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
PPL=false
BASELINE=false
BATCH=1
THINK=false
BRIDGE=false
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
  --model MODEL      (required) Model family or HuggingFace repo ID
  --method METHOD    Benchmark method (default: simple)
                       simple         Basic chat prompt — generation speed + PPL
                       summarization  Context-scaling with pre-sized prompts
                       wikitext2      Standard LM perplexity via forced decode
                       niah           Needle-in-a-haystack retrieval
                       multi-turn     Multi-turn conversation
                       tool-calling   Tool call generation
  --quant QUANT      Weight quantization: bf16, 8bit, 4bit, all (default: 4bit)
  --kv CONFIG        KV cache config: none, affine8, affine4, turbo4, turbo3, all (default: none)
                       Comma-separated for multiple: none,turbo4,turbo4v2
  --context SIZE     Comma-separated context sizes (default: all 11 sizes for scaling methods)
  --quick            Quick mode: 128 + 1024 + 4096 tokens only
  --kld              Compute KL divergence vs bf16/8bit baseline
  --baseline         Auto-select highest-fidelity variant (bf16 → 8bit → 4bit)
  --batch N          Run N concurrent generations (default: 1)
  --think            Enable thinking mode for thinking-capable models
  --bridge           Native C++ prefill (NATIVE_PREFILL=1); builds dylibs via
                       scripts/build-prefill-bridge.sh after the Swift build
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
  ./scripts/benchmark.sh --model qwen35-0.8b                                  # Simple eval
  ./scripts/benchmark.sh --model qwen35-9b --method summarization --quick     # Fast summarization
  ./scripts/benchmark.sh --model qwen35-0.8b --method wikitext2 --context 1024
  ./scripts/benchmark.sh --model qwen35-0.8b --quant bf16 --kv none          # bf16 baseline
  ./scripts/benchmark.sh --model qwen35-0.8b --quant all --kv all --quick    # Full matrix
  ./scripts/benchmark.sh --model qwen35-9b --kv affine4 --kld                # With KLD
  ./scripts/benchmark.sh --model nemotron-cascade-2 --quant 4bit             # Nemotron Cascade 2 (alias)

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
        --bridge)   BRIDGE=true; shift ;;
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
log_info "Model:   $MODEL"
log_info "Method:  $METHOD"
log_info "Quant:   $(IFS=,; echo "${QUANTS[*]}")"
log_info "KV:      $(IFS=,; echo "${KVS[*]}")"
$KLD && log_info "KLD:     yes"
$THINK && log_info "Think:   yes"
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

# Optional native prefill dylibs are not produced by SPM — clang++ builds them.
# swift build --build-tests can recreate the .xctest bundle, so run this after make.
if $BRIDGE; then
    log_info "Building native prefill bridge dylibs (--bridge)..."
    if ! bash "$PROJECT_ROOT/scripts/build-prefill-bridge.sh"; then
        log_error "build-prefill-bridge.sh failed. Fix toolchain / MLX paths, then retry."
        exit 1
    fi
fi

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

        if ${PPL:-false}; then
            export MLX_BENCH_PPL=1
        else
            unset MLX_BENCH_PPL
        fi

        if [ "$BATCH" -gt 1 ]; then
            export MLX_BENCH_BATCH="$BATCH"
        else
            unset MLX_BENCH_BATCH
        fi

        if $THINK; then
            export MLX_BENCH_THINK=1
        else
            unset MLX_BENCH_THINK
        fi

        if $BRIDGE; then
            export NATIVE_PREFILL=1
        else
            unset NATIVE_PREFILL
        fi

        log_info "Running: quant=$q kv=$kv method=$METHOD"

        # Stream filtered output in real-time.
        #
        # Swift Testing (the `@Test` framework) captures stdout from tests and
        # only emits it after each test completes — so `print()` inside the
        # test body never streams when `swift test` is piped (non-TTY).
        # Wrapping with `script -q /dev/null` allocates a PTY for swift test,
        # which bypasses Swift Testing's capture and lets prints flush live.
        # The Swift test also calls setlinebuf(stdout) as a belt-and-suspenders.
        # grep --line-buffered ensures matched lines appear without delay.
        # Full output captured to $TMPOUT for post-mortem on failure.
        TMPOUT=$(mktemp)
        script -q /dev/null swift test --skip-build -c release --filter "benchmark" 2>&1 \
            | tee "$TMPOUT" \
            | grep -E --line-buffered "\[ENV\]|\[WARMUP\]|\[BENCH\]|\[MEM\]|\[KLD\]|\[RESULT\]|\[KV-QUANT\]|\[TURBO\]|\[PROGRESS\]|Test.*passed|Test.*failed|[Ee]rror|[Ff]atal|BenchmarkError|threw|[Ee]xception|issue at"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ "$EXIT_CODE" -ne 0 ]; then
            log_error "Run failed (exit code $EXIT_CODE, quant=$q kv=$kv):"
            # Show additional context from the full output
            grep -iE "error|fatal|BenchmarkError|threw|exception|exceeds|issue" "$TMPOUT" | tail -20
        fi

        rm -f "$TMPOUT"
        echo ""
    done
done

log_info "Benchmark complete!"
log_info "Results saved to benchmarks/"
