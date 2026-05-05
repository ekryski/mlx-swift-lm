# Publishing a Release

How releases are tagged, branched, and published across the fork.

## Overview

The fork uses a manual-trigger release pipeline that:

1. Runs the full test suite on the release commit.
2. Computes the next version (auto-bump or override).
3. Creates a `release/<tag>` branch from the release commit, **kept open
   indefinitely** so hotfixes can be cherry-picked / merged into it later.
4. Tags the release commit with an annotated tag (`v<X.Y.Z>` or `v<X.Y.Z>-<prerelease>`).
5. Publishes a GitHub Release with auto-generated notes.

The same pipeline is mirrored in each fork repo (`mlx-swift-lm`,
`mlx-swift`, `mlx`, `mlx-c`) — the workflow shape is identical; only
the per-repo test step differs.

## When to bump major / minor / patch

The fork follows semver, with a pre-release suffix (`-alpha`, `-beta`,
`-rc`) for any release that hasn't been declared stable.

| Bump type | When |
|---|---|
| **major** | Public-API breaking changes that callers must adapt to. Example: removing or renaming a `KVCache` class, changing the signature of `loadModelContainer(...)`, removing a `GenerateParameters` field that downstream code reads. |
| **minor** | New public API surface, new model support, performance work that doesn't break existing call sites. Example: adding `RotatingKVCache.reserve(_:)`, adding a new `KVCache.CompressionAlgorithm` case, shipping a new model family. |
| **patch** | Bug fixes, internal refactors, doc changes, dependency bumps. No public-API change. Example: fixing an off-by-one in a kernel wrapper, restoring a regressed default. |

Pre-release suffix (`-alpha` etc.) sticks until the work is declared
stable. A pre-release tag does **not** prevent another pre-release at
the same version line — `v3.32.0-alpha` followed by `v3.32.0-alpha.2`
is fine, as is `v3.32.0-alpha` → `v3.32.0-beta` → `v3.32.0`.

## How to publish

### Option A — GitHub UI

1. Open the repo on github.com → **Actions** tab → **Release** workflow → **Run workflow**.
2. Pick the inputs:
   - `bump_type`: `major` / `minor` / `patch`.
   - `prerelease_tag`: `alpha` / `beta` / `rc` / `none`.
   - `override_version` (optional): e.g. `3.32.0-alpha` — bypasses auto-bump. Useful when the auto-detected previous tag isn't the right baseline (e.g., the `mlx-swift-lm` fork's `alpha` branched off `2.31.3`, so auto-bump can't reach `3.x` without help).
   - `notes_baseline` (optional): generate notes since this ref (e.g. `main`). Default = since the previous reachable tag.
   - `tag_prefix`: `v` (default), `auto` (inherit from last tag), or empty.
3. Confirm. The workflow runs tests, then tags + branches + publishes.

### Option B — Local CLI (faster, lets you preview notes before publishing)

```bash
# 1. From the commit you want to release:
PUSH=0 \
OVERRIDE_VERSION=3.32.0-alpha \
PRERELEASE_TAG=alpha \
TAG_PREFIX=v \
./scripts/release.sh
# Inspect the printed tag + branch — abort here if wrong.

# 2. Re-run with PUSH=1 to actually create the branch + tag and push them.
OVERRIDE_VERSION=3.32.0-alpha \
PRERELEASE_TAG=alpha \
TAG_PREFIX=v \
./scripts/release.sh

# 3. Publish the GitHub Release manually with a preview of the notes:
gh release create v3.32.0-alpha \
  --target release/v3.32.0-alpha \
  --title v3.32.0-alpha \
  --generate-notes \
  --notes-start-tag main \
  --prerelease
```

The script auto-detects the tag prefix from the most recent reachable
tag if `TAG_PREFIX` isn't set explicitly. Override when the convention
is changing (the fork is moving toward `v` prefix on every repo).

## What the workflow does, step by step

1. **Test gate** (per-repo). On `mlx-swift-lm` this is `make doctor && make && swift test --skip-build -c release --skip Benchmarks`. The other repos use their own test commands — see each repo's `.github/workflows/release.yml`.
2. **Compute version**. `scripts/release.sh` reads `OVERRIDE_VERSION` if set; otherwise reads `git describe --tags --abbrev=0`, strips any `v` and pre-release suffix, bumps according to `BUMP_TYPE`, then re-attaches the pre-release suffix and tag prefix.
3. **Create release branch**. `git branch release/<tag> HEAD`. The branch is kept open for the lifetime of the version line — hotfixes targeting `v3.32.x` land on `release/v3.32.0-alpha` (or a fresh `release/v3.32.x` cut at the time of the hotfix, depending on the fix's scope).
4. **Tag**. `git tag -a <tag> HEAD -m "Release <tag>"`. Annotated tags carry the message and the tagger identity.
5. **Push**. Both branch and tag go up to `origin`.
6. **Publish GitHub Release**. `gh release create --generate-notes` produces the changelog from merged PRs, categorized by label (config in `.github/release.yml`).

## Hotfixes

When a hotfix is needed for a previously released version line:

1. Cut a branch from the release branch, not from `alpha` or `main`:
   ```bash
   git checkout release/v3.32.0-alpha
   git checkout -b hotfix/<short-description>
   ```
2. Land the fix on the hotfix branch via PR (target = the release branch).
3. After merge, run the Release workflow against the release branch with `bump_type: patch`. Output: `v3.32.1-alpha` on a new `release/v3.32.1-alpha` branch.
4. Cherry-pick or merge the hotfix back into `alpha` if the bug also applies to ongoing development.

## Cross-repo coordination

The fork chain (mlx-c → mlx → mlx-swift → mlx-swift-lm) pins by version,
not commit SHA, wherever Swift Package Manager allows it:

- `mlx-swift-lm`'s `Package.swift` pins `mlx-swift` by `.upToNextMajor(from: "<version>")`. After tagging a new `mlx-swift` release, bump this pin in `mlx-swift-lm` and re-run its release workflow.
- `mlx-swift` pins `mlx` and `mlx-c` via git submodules (SHA-based). Submodule pins still get updated to the release commit, but the tag is the durable cross-reference.
- `mlx-c` pins `mlx` via CMake `FetchContent`, currently overridden to a local checkout in CI; outside CI the production reference is the tagged version of the upstream/fork.

Tag in dependency order: **mlx-c → mlx → mlx-swift → mlx-swift-lm**.
Each downstream repo's release notes can then reference the upstream
fork tag it pinned against.

## Release-notes config

PR-label-based categorization is configured in `.github/release.yml`:

| Section | Labels |
|---|---|
| 💥 Breaking Changes | `breaking` |
| ✨ Features | `feature`, `enhancement` |
| 🚀 Performance | `performance`, `perf` |
| 🐛 Bug Fixes | `bug`, `fix` |
| 📚 Documentation | `documentation`, `docs` |
| 🧪 Tests | `test`, `tests` |
| 🔧 Other Changes | `*` (catch-all) |

Add the relevant label to a PR before merging to slot it into a section.
PRs labeled `ignore-for-release` or `dependencies` are excluded.

## Reference

- Workflow file: `.github/workflows/release.yml`
- Version-bump script: `scripts/release.sh`
- Notes config: `.github/release.yml`
