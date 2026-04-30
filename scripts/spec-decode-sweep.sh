#!/bin/bash
# spec-decode-sweep.sh — canonical eval harness for speculative-decoding PRs
#
# Runs a (model × prompt × cell-config) sweep with N trials per cell, writing
# a parser-friendly log. Use with `scripts/spec-decode-compare.py` to compute
# median + speedup-ratio between two configurations.
#
# Why this exists: every speculative-decoding spec PR (013, 015, 016, 017,
# 020, 021, 022, 023, ...) wants the same answer to "did this change improve
# speeds?". Before this script that question was a fresh shell loop in /tmp/
# every time. Now it's reproducible from one place.
#
# USAGE:
#   spec-decode-sweep.sh \
#     --model <short-name> \
#     --quant <bf16|4bit|mxfp4> \
#     --prompts <file-or-dir> \
#     --cells <cell-spec>[,<cell-spec>,...] \
#     [--trials N] \
#     [--max-tokens N] \
#     [--output FILE]
#
# CELL-SPEC: colon-separated 4-tuple
#   LABEL:NGRAM:TEMP:EXTRA_ENV
# where
#   LABEL       = human-friendly cell name (e.g. "TI@0", "NGlev@0.6")
#   NGRAM       = MLX_BENCH_NGRAM value (0 disables n-gram, 3 enables)
#   TEMP        = MLX_BENCH_TEMPERATURE value (0 = greedy, 0.6 = sampling)
#   EXTRA_ENV   = additional env vars as KEY=VALUE pairs separated by ';'
#                 (empty string for none). Example:
#                 "MLX_NGRAM_LEVIATHAN=1;MLX_BENCH_REPETITION_PENALTY=1.1"
#
# Comma-separated multiple cells run sequentially. Each cell × every prompt
# × N trials.
#
# EXAMPLES:
#
# 1. Canonical Leviathan A/B on Gemma 4 26B A4B + recipe + lighthouse:
#
#    spec-decode-sweep.sh \
#      --model gemma4-26b-a4b --quant 4bit \
#      --prompts Tests/Benchmarks/Resources/ngram-sweep-prompts/recipe-bulk \
#      --cells "TI@0:0:0:,NGgreedy@0:3:0:,TI@0.6:0:0.6:,NGlev@0.6:3:0.6:MLX_NGRAM_LEVIATHAN=1" \
#      --trials 5
#
# 2. Quick smoke (1 prompt, 3 trials) for a new spec PR:
#
#    spec-decode-sweep.sh \
#      --model gemma4-26b-a4b --quant 4bit \
#      --prompts Tests/Benchmarks/Resources/ngram-sweep-prompts/recipe-bulk/01-five-soups.txt \
#      --cells "baseline:0:0:,treatment:3:0:" \
#      --trials 3
#
# OUTPUT FORMAT:
#   ============================================================
#   MODEL: <name>    PROMPT: <prompt-name>
#   ============================================================
#   --- Cell <LABEL>: ngram=<N> temp=<T> extra=<env> ---
#     trial 1: gen=<X> tok/s accept=<a/b>
#     ...
#
# Compatible with `scripts/spec-decode-compare.py` which parses this layout.

set -u

MODEL=""
QUANT="4bit"
PROMPTS=""
CELLS=""
TRIALS=5
MAX_TOKENS=200
OUTPUT="/dev/stdout"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --quant)      QUANT="$2"; shift 2 ;;
    --prompts)    PROMPTS="$2"; shift 2 ;;
    --cells)      CELLS="$2"; shift 2 ;;
    --trials)     TRIALS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --output)     OUTPUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '3,/^#$/p' "$0" | sed 's/^# //; s/^#$//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$MODEL"   ]] && { echo "--model required" >&2; exit 2; }
[[ -z "$PROMPTS" ]] && { echo "--prompts required" >&2; exit 2; }
[[ -z "$CELLS"   ]] && { echo "--cells required" >&2; exit 2; }

# Collect prompt files: a directory expands to *.txt inside it; a single file
# is used directly. Recursive directory walk is intentional — many of the
# ngram-sweep-prompts subdirs have only 1-3 prompts.
declare -a PROMPT_FILES=()
if [[ -d "$PROMPTS" ]]; then
  while IFS= read -r f; do PROMPT_FILES+=("$f"); done < <(find "$PROMPTS" -name '*.txt' -type f | sort)
elif [[ -f "$PROMPTS" ]]; then
  PROMPT_FILES+=("$PROMPTS")
else
  echo "--prompts must be a file or directory: $PROMPTS" >&2
  exit 2
fi
[[ ${#PROMPT_FILES[@]} -eq 0 ]] && { echo "no prompts found in $PROMPTS" >&2; exit 2; }

# Parse cell specs into parallel arrays.
IFS=',' read -ra CELL_SPECS <<< "$CELLS"

run_cell() {
  local LABEL=$1; local NGRAM=$2; local TEMP=$3; local EXTRA=$4; local PROMPT="$5"

  # Build the env set: clean baseline + bench knobs + per-cell overrides.
  local env_args=(
    HOME="$HOME" PATH="$PATH"
    MLX_BENCH_MODEL="$MODEL"
    MLX_BENCH_QUANT="$QUANT"
    MLX_BENCH_METHOD=simple
    MLX_BENCH_MAX_TOKENS="$MAX_TOKENS"
    MLX_BENCH_PROMPT="$PROMPT"
    MLX_BENCH_NGRAM="$NGRAM"
    MLX_BENCH_TEMPERATURE="$TEMP"
  )
  if [[ -n "$EXTRA" ]]; then
    IFS=';' read -ra EXTRA_PAIRS <<< "$EXTRA"
    for kv in "${EXTRA_PAIRS[@]}"; do
      [[ -n "$kv" ]] && env_args+=("$kv")
    done
  fi

  local raw
  raw=$(env -i "${env_args[@]}" \
        swift test -c release --filter "benchmark" 2>&1 \
        | grep -E "Generation:|Spec decode:" | head -2)
  local GEN ACC
  GEN=$(echo "$raw" | grep "Generation:" | awk '{print $3}')
  ACC=$(echo "$raw" | grep "Spec decode:" | grep -oE '[0-9]+/[0-9]+' | head -1)
  printf "  trial %s: gen=%s tok/s accept=%s\n" "$1_$2" "$GEN" "${ACC:-—}"
}

{
  for prompt_file in "${PROMPT_FILES[@]}"; do
    PROMPT=$(cat "$prompt_file")
    PROMPT_NAME=$(basename "$prompt_file" .txt)
    PARENT=$(basename "$(dirname "$prompt_file")")
    DISPLAY="$PARENT/$PROMPT_NAME"

    echo
    echo "============================================================"
    echo "MODEL: $MODEL    PROMPT: $DISPLAY"
    echo "============================================================"

    for spec in "${CELL_SPECS[@]}"; do
      IFS=':' read CELL NGRAM TEMP EXTRA <<< "$spec"
      echo "--- Cell $CELL: ngram=$NGRAM temp=$TEMP extra=${EXTRA:-—} ---"
      for trial in $(seq 1 "$TRIALS"); do
        # Reuse the same run helper but with trial-distinguishing label.
        env_args=(
          HOME="$HOME" PATH="$PATH"
          MLX_BENCH_MODEL="$MODEL"
          MLX_BENCH_QUANT="$QUANT"
          MLX_BENCH_METHOD=simple
          MLX_BENCH_MAX_TOKENS="$MAX_TOKENS"
          MLX_BENCH_PROMPT="$PROMPT"
          MLX_BENCH_NGRAM="$NGRAM"
          MLX_BENCH_TEMPERATURE="$TEMP"
        )
        if [[ -n "$EXTRA" ]]; then
          IFS=';' read -ra EXTRA_PAIRS <<< "$EXTRA"
          for kv in "${EXTRA_PAIRS[@]}"; do
            [[ -n "$kv" ]] && env_args+=("$kv")
          done
        fi
        raw=$(env -i "${env_args[@]}" \
              swift test -c release --filter "benchmark" 2>&1 \
              | grep -E "Generation:|Spec decode:" | head -2)
        GEN=$(echo "$raw" | grep "Generation:" | awk '{print $3}')
        ACC=$(echo "$raw" | grep "Spec decode:" | grep -oE '[0-9]+/[0-9]+' | head -1)
        printf "  trial %s: gen=%s tok/s accept=%s\n" "$trial" "$GEN" "${ACC:-—}"
      done
    done
  done
} > "$OUTPUT"
