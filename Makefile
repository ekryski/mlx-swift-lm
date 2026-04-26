# Makefile — Smart build orchestrator for mlx-swift-lm
#
# Swift Package Manager doesn't track Metal shaders, native dylibs, or changes
# deep in git submodules (mlx, mlx-c). This Makefile fills the gaps using
# file-timestamp dependencies so only stale targets rebuild.
#
# Quick reference:
#   make              — full incremental build for tests/benchmarks
#   make metal        — recompile Metal shaders only
#   make spm          — Swift build only (with Cmlx cache invalidation)
#   make status       — show what's built and what's stale
#   make clean        — remove build artifacts (keep checkouts)
#   make clean-all    — full reset (re-fetches dependencies)
#   make clean-cmlx   — force SPM to recompile C/C++ on next build

SHELL := /bin/bash
.DEFAULT_GOAL := build-tests

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT := $(CURDIR)
BUILD_DIR    := $(PROJECT_ROOT)/.build
STAMP_DIR    := $(BUILD_DIR)/.stamps
RELEASE_DIR  := $(BUILD_DIR)/arm64-apple-macosx/release
XCTEST_DIR   := $(RELEASE_DIR)/mlx-swift-lmPackageTests.xctest/Contents/MacOS

CONFIG       ?= release
SWIFT_FLAGS  ?= -Xswiftc -enable-testing

# ─── Locate mlx-swift (local override > sibling > SPM checkout) ───────────────

MLX_SWIFT_DIR := $(or \
    $(wildcard $(MLX_SWIFT_PATH)), \
    $(wildcard $(PROJECT_ROOT)/../mlx-swift), \
    $(BUILD_DIR)/checkouts/mlx-swift)

METAL_SRC_DIR := $(MLX_SWIFT_DIR)/Source/Cmlx/mlx-generated/metal
CMLX_SRC_DIR  := $(MLX_SWIFT_DIR)/Source/Cmlx

# ─── Source file discovery (evaluated each invocation) ────────────────────────
# Make re-expands $(shell ...) every time you run `make`, so new/deleted files
# are picked up automatically.

METAL_SOURCES := $(shell find "$(METAL_SRC_DIR)" \( -name '*.metal' -o -name '*.h' \) -type f 2>/dev/null)
CMLX_SOURCES  := $(shell find "$(CMLX_SRC_DIR)/mlx" "$(CMLX_SRC_DIR)/mlx-c" \
                    \( -name '*.cpp' -o -name '*.c' -o -name '*.h' \) -type f 2>/dev/null)
SWIFT_SOURCES := $(shell find "$(PROJECT_ROOT)/Libraries" "$(PROJECT_ROOT)/Sources" "$(PROJECT_ROOT)/Tests" \
                    -name '*.swift' -type f 2>/dev/null)

# Output artifacts
METALLIB        := $(RELEASE_DIR)/mlx.metallib

# Stamp files — touch-files that record when each step last succeeded
STAMP_METAL := $(STAMP_DIR)/metallib
STAMP_SPM   := $(STAMP_DIR)/spm-build
STAMP_CMLX  := $(STAMP_DIR)/cmlx-check

# ─── Directory creation ──────────────────────────────────────────────────────

$(STAMP_DIR):
	@mkdir -p $@

# ─── Metal shader compilation ────────────────────────────────────────────────
# SPM cannot compile .metal files. We compile them into mlx.metallib ourselves.
# Only recompiles when a .metal or kernel .h file changes.

.PHONY: metal
metal: $(STAMP_METAL)

$(STAMP_METAL): $(METAL_SOURCES) | $(STAMP_DIR)
	@echo "==> Metal sources changed — recompiling shaders..."
	@bash "$(PROJECT_ROOT)/scripts/build-metallib.sh" $(CONFIG)
	@touch $@

# ─── Cmlx cache invalidation ────────────────────────────────────────────────
# When C/C++ files inside the mlx or mlx-c submodules change, SPM's build.db
# may still consider Cmlx "up to date" (it caches content signatures keyed by
# the dependency revision in workspace-state.json).
#
# Fix: delete just .build/.../Cmlx.build/ — this forces SPM to recompile the
# C target on the next `swift build` while leaving all Swift targets cached.

$(STAMP_CMLX): $(CMLX_SOURCES) | $(STAMP_DIR)
	@if [ -f "$@" ] && [ -d "$(RELEASE_DIR)/Cmlx.build" ]; then \
		echo "==> Cmlx sources changed since last build — invalidating SPM Cmlx cache..."; \
		rm -rf "$(RELEASE_DIR)/Cmlx.build"; \
	elif [ ! -f "$@" ]; then \
		echo "==> Seeding Cmlx stamp (first run — not invalidating existing cache)"; \
	fi
	@touch $@

# ─── SPM build ───────────────────────────────────────────────────────────────
# Runs `swift build --build-tests` only if Swift sources changed OR the Cmlx
# cache was invalidated (which updates the STAMP_CMLX sentinel).

.PHONY: spm
spm: $(STAMP_SPM)

$(STAMP_SPM): $(SWIFT_SOURCES) $(STAMP_CMLX) | $(STAMP_DIR)
	@echo "==> Building Swift targets ($(CONFIG))..."
	swift build --build-tests -c $(CONFIG) $(SWIFT_FLAGS)
	@touch $@

# ─── Artifact installation ──────────────────────────────────────────────────
# Copies metallib into .build/release/ and the .xctest bundle.
# Must run AFTER spm because `swift build --build-tests` can regenerate the
# test bundle directory, wiping previously-copied files.

.PHONY: install-artifacts
install-artifacts: $(STAMP_SPM) $(STAMP_METAL)
	@mkdir -p "$(RELEASE_DIR)"
	@# --- metallib ---
	@if [ -f "$(METALLIB)" ] && [ -d "$(XCTEST_DIR)" ]; then \
		cp "$(METALLIB)" "$(XCTEST_DIR)/mlx.metallib"; \
		echo "  copied mlx.metallib -> test bundle"; \
	fi

# ─── Main targets ───────────────────────────────────────────────────────────

.PHONY: build
build: spm metal

.PHONY: build-tests
build-tests: build install-artifacts
	@echo ""
	@echo "Build complete — all artifacts in place."

# ─── Clean targets ───────────────────────────────────────────────────────────

.PHONY: clean
clean:
	swift package clean
	rm -rf "$(STAMP_DIR)"
	@echo "Cleaned build artifacts. Dependency checkouts preserved."

.PHONY: clean-all
clean-all:
	swift package reset
	rm -rf "$(STAMP_DIR)"
	@echo "Full reset. Run 'swift package resolve' or 'make' to re-fetch."

# Surgical: force SPM to recompile just the C/C++ target on next build
.PHONY: clean-cmlx
clean-cmlx:
	rm -rf "$(RELEASE_DIR)/Cmlx.build"
	rm -f "$(STAMP_CMLX)" "$(STAMP_SPM)"
	@echo "Cmlx cache invalidated. Next 'make' will recompile C/C++ sources."

.PHONY: clean-metal
clean-metal:
	rm -f "$(METALLIB)"
	rm -f "$(STAMP_METAL)"
	@echo "Metal cache cleared. Next 'make' will recompile shaders."

# ─── Status ──────────────────────────────────────────────────────────────────

.PHONY: status
status:
	@echo "mlx-swift-lm build status"
	@echo "========================="
	@echo ""
	@echo "mlx-swift location: $(MLX_SWIFT_DIR)"
	@echo ""
	@echo "Stamps (last successful build):"
	@if [ -d "$(STAMP_DIR)" ]; then \
		for f in $(STAMP_DIR)/*; do \
			[ -f "$$f" ] && echo "  $$(basename $$f): $$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' $$f)"; \
		done; \
	else \
		echo "  (none — run 'make' to build)"; \
	fi
	@echo ""
	@echo "Artifacts:"
	@[ -f "$(METALLIB)" ] \
		&& echo "  mlx.metallib (release):             OK" \
		|| echo "  mlx.metallib (release):             MISSING"
	@[ -f "$(XCTEST_DIR)/mlx.metallib" ] 2>/dev/null \
		&& echo "  mlx.metallib (test bundle):          OK" \
		|| echo "  mlx.metallib (test bundle):          MISSING"
	@echo ""
	@echo "SPM Cmlx cache:"
	@[ -d "$(RELEASE_DIR)/Cmlx.build" ] \
		&& echo "  Cmlx.build: present ($$(du -sh "$(RELEASE_DIR)/Cmlx.build" | cut -f1))" \
		|| echo "  Cmlx.build: empty (will recompile on next build)"

# ─── Help ────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "mlx-swift-lm build targets:"
	@echo ""
	@echo "  make              Full incremental build for tests/benchmarks"
	@echo "  make spm          Swift build (Swift, Cmlx invalidation)"
	@echo "  make metal        Recompile Metal shaders only"
	@echo "  make status       Show build state and artifact locations"
	@echo ""
	@echo "  make clean        Remove build artifacts (keep dependency checkouts)"
	@echo "  make clean-all    Full reset (re-fetches everything)"
	@echo "  make clean-cmlx   Force recompile of C/C++ sources (mlx, mlx-c)"
	@echo "  make clean-metal  Force recompile of Metal shaders"
	@echo ""
	@echo "Environment:"
	@echo "  MLX_SWIFT_PATH    Override mlx-swift dependency location"
	@echo "  CONFIG            release (default) or debug"
	@echo "  SWIFT_FLAGS       Extra flags for swift build"
