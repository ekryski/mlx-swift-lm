#!/bin/bash
# Build the native C++ prefill bridge dylib.
# Uses MLX headers from mlx-swift checkout (preferred) or Python mlx package.
# Usage: ./scripts/build-prefill-bridge.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$PROJECT_ROOT/Sources/NativePrefillBridge/libprefill_bridge_v2.dylib"
SRC="$PROJECT_ROOT/Sources/NativePrefillBridge/prefill_bridge_v2.cpp"

if [ ! -f "$SRC" ]; then
    echo "Skipping prefill bridge dylib: $SRC not found (Gemma/Qwen/generic prefill is built via SPM target NativePrefillBridge)."
    exit 0
fi

# ─── Locate MLX headers and library ──────────────────────────────────────────
# Priority: env vars > mlx-swift checkout (sibling or SPM) > Python mlx package

# Find mlx-swift root
MLX_SWIFT_ROOT="${MLX_SWIFT_PATH:-}"
if [ -z "$MLX_SWIFT_ROOT" ]; then
    for candidate in \
        "$PROJECT_ROOT/../mlx-swift" \
        "$PROJECT_ROOT/.build/checkouts/mlx-swift"; do
        if [ -d "$candidate/Source/Cmlx/mlx" ]; then
            MLX_SWIFT_ROOT="$candidate"
            break
        fi
    done
fi

# MLX C++ include path
if [ -n "${MLX_INCLUDE_PATH:-}" ]; then
    MLX_INC="$MLX_INCLUDE_PATH"
elif [ -n "$MLX_SWIFT_ROOT" ] && [ -d "$MLX_SWIFT_ROOT/Source/Cmlx/mlx/mlx" ]; then
    # mlx/mlx.h lives at Source/Cmlx/mlx/mlx/mlx.h — include parent so #include "mlx/mlx.h" resolves
    MLX_INC="$MLX_SWIFT_ROOT/Source/Cmlx/mlx"
else
    MLX_INC="$(python3 -c 'import os, mlx; print(os.path.dirname(mlx.__path__[0]))')"
fi

# MLX library path (for linking libmlx)
if [ -n "${MLX_LIB_PATH:-}" ]; then
    MLX_LIB="$MLX_LIB_PATH"
elif [ -f "$PROJECT_ROOT/.build/arm64-apple-macosx/release/libCmlx.a" ]; then
    # SPM builds Cmlx as a static library — link against that
    MLX_LIB="$PROJECT_ROOT/.build/arm64-apple-macosx/release"
else
    MLX_LIB="$(python3 -c 'import mlx; print(mlx.__path__[0] + "/lib")')"
fi

# mlx-c headers
MLX_C="${MLX_C_PATH:-${MLX_SWIFT_ROOT:-$PROJECT_ROOT/.build/checkouts/mlx-swift}/Source/Cmlx/mlx-c}"

echo "Building prefill bridge..."
echo "  MLX include: $MLX_INC"
echo "  MLX lib:     $MLX_LIB"
echo "  MLX-C:       $MLX_C"

clang++ -std=c++20 -O3 -shared -fPIC \
  -I"$MLX_INC" -I"$MLX_C" -I"$(dirname "$SRC")" \
  -L"$MLX_LIB" -lmlx \
  -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_LIB" \
  -o "$OUT" "$SRC"
echo "Built: $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
