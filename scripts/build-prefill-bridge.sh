#!/bin/bash
# Build the native C++ prefill bridge dylib.
# Requires MLX Python package (pip install mlx).
# Usage: ./scripts/build-prefill-bridge.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$PROJECT_ROOT/Sources/NativePrefillBridge/libprefill_bridge_v2.dylib"
SRC="$PROJECT_ROOT/Sources/NativePrefillBridge/prefill_bridge_v2.cpp"
MLX_INC="${MLX_INCLUDE_PATH:-$(python3 -c 'import os, mlx; print(os.path.dirname(mlx.__path__[0]))')}"
MLX_LIB="${MLX_LIB_PATH:-$(python3 -c 'import mlx; print(mlx.__path__[0] + "/lib")')}"
MLX_C="${MLX_C_PATH:-${MLX_SWIFT_PATH:-$PROJECT_ROOT/.build/checkouts/mlx-swift}/Source/Cmlx/mlx-c}"
echo "Building prefill bridge..."
clang++ -std=c++20 -O3 -shared -fPIC \
  -I"$MLX_INC" -I"$MLX_C" -I"$(dirname "$SRC")" \
  -L"$MLX_LIB" -lmlx \
  -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_LIB" \
  -o "$OUT" "$SRC"
echo "Built: $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
