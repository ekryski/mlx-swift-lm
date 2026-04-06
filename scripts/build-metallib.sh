#!/bin/bash
# build-metallib.sh
# Compiles MLX Metal shaders into mlx.metallib for SPM builds.
# SPM doesn't compile .metal files automatically, so this must be run once
# before benchmarks or tests can execute.
#
# Usage: ./scripts/build-metallib.sh [release|debug|both]
#
# Produces: .build/arm64-apple-macosx/{config}/mlx.metallib
#           .build/arm64-apple-macosx/{config}/{package}Tests.xctest/Contents/MacOS/mlx.metallib

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-release}"

# Metal source dir is populated by `swift package resolve`
MLX_CHECKOUT="$PROJECT_ROOT/.build/checkouts/mlx-swift"
METAL_SRC_DIR="$MLX_CHECKOUT/Source/Cmlx/mlx-generated/metal"

if [ ! -d "$METAL_SRC_DIR" ]; then
    echo "Error: Metal source not found at $METAL_SRC_DIR"
    echo "Run 'swift package resolve' first to fetch dependencies."
    exit 1
fi

build_metallib_for_config() {
    local cfg="$1"
    local BUILD_DIR="$PROJECT_ROOT/.build/arm64-apple-macosx/$cfg"

    echo ""
    echo "Building mlx.metallib ($cfg)..."

    # Create temp directory for .air files
    local AIR_DIR
    AIR_DIR=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf $AIR_DIR" EXIT

    local METAL_FILES
    METAL_FILES=$(find "$METAL_SRC_DIR" -name "*.metal" -type f | sort)
    local AIR_FILES=""
    local count=0

    while IFS= read -r metal_file; do
        local rel_path="${metal_file#$METAL_SRC_DIR/}"
        local air_name
        air_name=$(echo "$rel_path" | sed 's|/|_|g; s|\.metal$|.air|')
        local air_file="$AIR_DIR/$air_name"

        xcrun metal \
            -std=metal3.1 \
            -O2 \
            -I "$METAL_SRC_DIR" \
            -c "$metal_file" \
            -o "$air_file" 2>&1

        AIR_FILES="$AIR_FILES $air_file"
        count=$((count + 1))
    done <<< "$METAL_FILES"

    mkdir -p "$BUILD_DIR"

    local OUTPUT="$BUILD_DIR/mlx.metallib"
    echo "Linking $count shaders -> $OUTPUT"
    # shellcheck disable=SC2086
    xcrun metallib $AIR_FILES -o "$OUTPUT"
    echo "  Created: $OUTPUT"

    # Also copy into the .xctest bundle if it exists
    local XCTEST_MACOS="$BUILD_DIR/mlx-swift-lmPackageTests.xctest/Contents/MacOS"
    if [ -d "$XCTEST_MACOS" ]; then
        cp "$OUTPUT" "$XCTEST_MACOS/mlx.metallib"
        echo "  Copied to test bundle: $XCTEST_MACOS/mlx.metallib"
    fi
}

case "$CONFIG" in
    both)
        build_metallib_for_config "release"
        build_metallib_for_config "debug"
        ;;
    release|debug)
        build_metallib_for_config "$CONFIG"
        ;;
    *)
        echo "Error: Unknown config '$CONFIG'. Use: release, debug, or both"
        exit 1
        ;;
esac

echo ""
echo "Metal shaders compiled. Run benchmarks with:"
echo "  ./scripts/benchmark.sh --model <model>"
