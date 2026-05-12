# Contributing to MLX Swift Examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications IntegrationTesting
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests run without any special hardware and do not download models.
Note: `swift test` [does not work yet](https://github.com/ml-explore/mlx-swift?tab=readme-ov-file#xcodebuild) — use `xcodebuild` instead:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

Integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. These
tests do not run in CI.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS'

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

## Cross-repo pull requests (ekryski fork)

Some changes need to land across the full chain — `mlx` → `mlx-c` →
`mlx-swift` → `mlx-swift-lm`. Native Metal kernels, new C ABI bridges, and
new `MLXFast` Swift wrappers all touch every link. This section describes
how to keep CI green throughout the lifecycle of those changes.

### Conventions

1. **Identical branch names across every repo in the chain.** Pick one
   name (e.g. `ek/spec-NNN-feature`) and create that branch on every fork
   you'll touch. CI in `mlx-c` and `mlx-swift-lm` auto-detects matching
   branches on the upstream dependency and uses them; falls back to
   `alpha` when no match is found. Misaligned names mean CI silently
   falls back to `alpha`, which won't have your new symbols, and the
   build breaks at link time.

2. **One PR per repo, all open in parallel.** The dependency chain only
   matters at merge time — for development the four PRs build
   independently against each other's branches.

3. **Submodule pins on `mlx-swift` are explicit.** `Source/Cmlx/mlx` and
   `Source/Cmlx/mlx-c` are git submodules pinned by SHA. Each cross-repo
   PR includes one commit per dep bumping the submodule to the current
   tip of the matching branch. The bump is part of the PR diff (useful
   provenance — reviewers see exactly which dep SHA was built against).
   Auto-detection does NOT silently move submodule HEADs.

4. **`mlx-swift-lm` Package.swift uses `.package(path: "../mlx-swift")`
   only during cross-repo dev.** The CI workflow checks out
   `ekryski/mlx-swift` as a sibling at `../mlx-swift`. On `alpha`,
   Package.swift uses `.package(url:, exact: "<tag>")` against a real
   release tag. The flip from URL → path is part of the PR diff and
   gets reverted at merge time (see checklist below).

### Pre-merge checklist

Merge in dependency order. After each step verifies CI green on the
target repo's `alpha`:

1. **`mlx`** (no internal deps). Merge to `alpha`. No special steps.
2. **`mlx-c`**. CI auto-resolves against `ekryski/mlx@<your-branch>`
   while the PR is open. Before merge: nothing to revert — the
   workflow's auto-detect step will naturally fall back to `alpha`
   once the matching branch is gone. Merge.
3. **`mlx-swift`**.
   - Bump `Source/Cmlx/mlx` submodule to the `alpha` merge commit from
     step 1.
   - Bump `Source/Cmlx/mlx-c` submodule to the `alpha` merge commit
     from step 2.
   - Push, verify CI green, merge.
   - Cut a new pre-release tag on `alpha` (e.g. `v0.32.2-alpha`) — the
     downstream `mlx-swift-lm` Package.swift will pin to this.
4. **`mlx-swift-lm`**.
   - Revert `Package.swift` from `.package(path: "../mlx-swift")` back
     to `.package(url: "https://github.com/ekryski/mlx-swift", exact:
     "<new-tag>")` using the tag from step 3.
   - The CI workflow's auto-detect step naturally falls back to
     `alpha` once the matching branch is gone — no manual revert
     needed there.
   - Run `swift package resolve` to update `Package.resolved`.
   - Verify CI green, merge.

### What the auto-resolve does

Both `mlx-c` and `mlx-swift-lm` `alpha-ci.yml` workflows include a
"Resolve dependency ref" step that runs before checking out the
sibling repo. It looks up `${{ github.head_ref }}` (the PR's branch
name) on the dependency fork and uses that ref if it exists, else
falls back to `alpha`. No editing the workflow per PR; no flipping
hardcoded `ref:` values; the same workflow file works for both
cross-repo PRs and regular single-repo PRs.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift Examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
