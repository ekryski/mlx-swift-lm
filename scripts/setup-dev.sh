#!/bin/bash
# setup-dev.sh
# Development environment setup for mlx-swift-lm.
# Compiles MLX Metal shaders and builds test targets for benchmarking.
#
# Run once after cloning or after fetching new mlx-swift changes:
#   ./scripts/setup-dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  ⚠${NC}  $1"; }
fail() { echo -e "${RED}  ✗${NC} $1"; exit 1; }

echo ""
echo "Setting up mlx-swift-lm development environment..."
echo ""

# ─────────────────────────────────────────────
# Prerequisites
# ─────────────────────────────────────────────
echo "Checking prerequisites..."

# Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    fail "Xcode Command Line Tools not found. Install with: xcode-select --install"
fi
ok "Xcode CLI tools: $(xcode-select -p)"

# xcrun (Metal compiler)
if ! xcrun --find metal &>/dev/null 2>&1; then
    fail "xcrun metal not found. Install Xcode or Xcode Command Line Tools."
fi
ok "xcrun metal: available"

# Swift
if ! command -v swift &>/dev/null; then
    fail "Swift not found. Install Xcode or Swift toolchain."
fi
SWIFT_VERSION=$(swift --version 2>&1 | grep -oE 'Swift version [0-9]+\.[0-9]+' | head -1)
ok "Swift: $SWIFT_VERSION"

# ─────────────────────────────────────────────
# Resolve packages
# ─────────────────────────────────────────────
echo ""
echo "Resolving Swift package dependencies..."
cd "$PROJECT_ROOT"
swift package resolve
ok "Packages resolved"

# ─────────────────────────────────────────────
# Build (SPM + Metal shaders + native dylibs)
# ─────────────────────────────────────────────
echo ""
echo "Building (make build-tests)..."
make -C "$PROJECT_ROOT" build-tests
ok "Build complete — all artifacts in place"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Run a quick benchmark:"
echo "  ./scripts/benchmark.sh --model qwen35-0.8b --context 128"
echo ""
echo "Run a comprehensive benchmark sweep:"
echo "  ./scripts/benchmark.sh --model qwen35-2b --quant all --kv all --quick"
echo ""
echo "See benchmarks/README.md for the full CLI reference."
echo ""
echo "Note: For unit tests (non-benchmark), use xcodebuild:"
echo "  xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'"
echo ""
