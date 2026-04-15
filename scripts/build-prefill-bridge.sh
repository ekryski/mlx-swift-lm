#!/bin/bash
# Build the native C++ prefill bridges as standalone dylibs.
#
# Two modes:
#   --standalone: Link Cmlx objects INTO the dylib (own MLX copy, max perf)
#   (default):    Use -undefined dynamic_lookup (share host MLX, safe allocator)
#
# Usage: ./scripts/build-prefill-bridge.sh [--standalone]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/Sources/NativePrefillBridge"
MLX_SWIFT="${MLX_SWIFT_PATH:-$PROJECT_ROOT/.build/checkouts/mlx-swift}"
MLX_INC="$MLX_SWIFT/Source/Cmlx/mlx"
MLX_C_INC="$MLX_SWIFT/Source/Cmlx/mlx-c"
RELEASE_DIR="$PROJECT_ROOT/.build/arm64-apple-macosx/release"

STANDALONE=false
[ "${1:-}" = "--standalone" ] && STANDALONE=true

COMMON="-std=c++20 -O3 -DNDEBUG -fPIC -I$MLX_INC -I$MLX_C_INC -I$SRC_DIR -target arm64-apple-macosx14.0"

if $STANDALONE; then
    CMLX_OBJS=$(find "$RELEASE_DIR/Cmlx.build" -name "*.o" 2>/dev/null)
    if [ -z "$CMLX_OBJS" ]; then
        echo "ERROR: No Cmlx objects. Run 'swift build -c release' first."
        exit 1
    fi
    LINK_FLAGS="$CMLX_OBJS -framework Metal -framework Foundation -framework Accelerate"
    echo "Building STANDALONE dylibs (embedded MLX)..."
else
    LINK_FLAGS="-Wl,-undefined,dynamic_lookup"
    echo "Building DYNAMIC dylibs (shared host MLX)..."
fi

for src in prefill_bridge_gemma prefill_bridge_qwen; do
    dylib="$SRC_DIR/lib${src}.dylib"
    echo "  ${src}.cpp -> $dylib"
    clang++ $COMMON -shared $LINK_FLAGS -o "$dylib" "$SRC_DIR/${src}.cpp"
    cp "$dylib" "$RELEASE_DIR/" 2>/dev/null || true
    TEST_BUNDLE="$RELEASE_DIR/mlx-swift-lmPackageTests.xctest/Contents/MacOS/"
    cp "$dylib" "$TEST_BUNDLE" 2>/dev/null || true
done

echo "Done."
ls -lh "$SRC_DIR"/libprefill_bridge_*.dylib
