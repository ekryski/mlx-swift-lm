#!/usr/bin/env bash
# Run tests for mlx-swift-lm. Uses xcodebuild so Metal shaders (default.metallib)
# are built and available; "swift test" cannot build them and will fail at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  "$@"
