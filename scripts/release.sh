#!/bin/bash
# Compute the next release version from the most recent reachable tag,
# then create the release branch + tag and push them.
#
# Usage (env-driven):
#   BUMP_TYPE=major|minor|patch         # default: minor
#   PRERELEASE_TAG=alpha|beta|rc|none   # default: none
#   OVERRIDE_VERSION=                   # optional; bypasses auto-bump
#   TAG_PREFIX=auto|v|                  # default: auto (inherit from last tag)
#   PUSH=1                              # default: 1 (set 0 for dry-run)
#   ./scripts/release.sh
#
# Outputs (printed to stdout, also written to $GITHUB_OUTPUT if present):
#   tag=<tag>             # e.g. v3.1.0-alpha
#   version=<version>     # e.g. 3.1.0-alpha
#   release_branch=<name> # e.g. release/v3.1.0-alpha

set -euo pipefail

BUMP_TYPE="${BUMP_TYPE:-minor}"
PRERELEASE_TAG="${PRERELEASE_TAG:-none}"
OVERRIDE_VERSION="${OVERRIDE_VERSION:-}"
TAG_PREFIX_INPUT="${TAG_PREFIX:-auto}"
PUSH="${PUSH:-1}"

# ─── Detect tag prefix ─────────────────────────────────────────────
# Some repos prefix tags with "v" (mlx, mlx-c upstream); others don't
# (mlx-swift, mlx-swift-lm). Auto-detect from the most recent reachable
# tag so the new tag matches the existing convention.
detect_prefix() {
  local last
  last=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
  if [[ "$last" == v* ]]; then
    echo "v"
  else
    echo ""
  fi
}

if [ "$TAG_PREFIX_INPUT" = "auto" ]; then
  TAG_PREFIX=$(detect_prefix)
else
  TAG_PREFIX="$TAG_PREFIX_INPUT"
fi

# ─── Compute next version ──────────────────────────────────────────
if [ -n "$OVERRIDE_VERSION" ]; then
  # User-supplied; strip any leading "v" so we can re-attach the
  # detected prefix consistently.
  NEW_VERSION="${OVERRIDE_VERSION#v}"
else
  CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "${TAG_PREFIX}0.0.0")
  CLEAN="${CURRENT_TAG#v}"             # strip leading v
  CLEAN="${CLEAN%%-*}"                 # strip any -alpha/-beta suffix

  IFS='.' read -r MAJOR MINOR PATCH <<< "$CLEAN"
  MAJOR="${MAJOR:-0}"; MINOR="${MINOR:-0}"; PATCH="${PATCH:-0}"

  case "$BUMP_TYPE" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
    *) echo "Invalid BUMP_TYPE: $BUMP_TYPE" >&2; exit 1 ;;
  esac

  NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
fi

if [ "$PRERELEASE_TAG" != "none" ] && [[ "$NEW_VERSION" != *-* ]]; then
  NEW_VERSION="${NEW_VERSION}-${PRERELEASE_TAG}"
fi

NEW_TAG="${TAG_PREFIX}${NEW_VERSION}"
RELEASE_BRANCH="release/${NEW_TAG}"

# ─── Guard: don't clobber an existing tag ──────────────────────────
if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
  echo "Tag $NEW_TAG already exists. Refusing to overwrite." >&2
  exit 1
fi

# ─── Create branch + tag ───────────────────────────────────────────
COMMIT=$(git rev-parse HEAD)

# Release branch: kept open for hotfixes against this version line.
git branch "$RELEASE_BRANCH" "$COMMIT" 2>/dev/null || \
  echo "Release branch $RELEASE_BRANCH already exists; reusing it."

# Annotated tag on the release commit.
git tag -a "$NEW_TAG" "$COMMIT" -m "Release $NEW_TAG"

# ─── Push ──────────────────────────────────────────────────────────
if [ "$PUSH" = "1" ]; then
  git push origin "$RELEASE_BRANCH"
  git push origin "$NEW_TAG"
fi

# ─── Outputs ───────────────────────────────────────────────────────
echo "tag=${NEW_TAG}"
echo "version=${NEW_VERSION}"
echo "release_branch=${RELEASE_BRANCH}"
echo "commit=${COMMIT}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "tag=${NEW_TAG}"
    echo "version=${NEW_VERSION}"
    echo "release_branch=${RELEASE_BRANCH}"
    echo "commit=${COMMIT}"
  } >> "$GITHUB_OUTPUT"
fi
