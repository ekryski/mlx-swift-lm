# Testing

`mlx-swift-lm` ships an XCTest + swift-testing test suite covering the core
inference infrastructure (KV cache, generation loop, tool-call parsers,
chat templates, configuration decoding, sampling, ngram speculative
decoding, batched decode, wired memory). Tests run in **release
configuration** because debug builds skip the Metal library that several
suites need.

## Running tests

### Full suite

```bash
swift test -c release
```

`make build-tests` runs first if needed, copies `mlx.metallib` into the
test bundle, and then drives `swift test -c release` underneath. From a
fresh checkout, do:

```bash
./scripts/setup-dev.sh    # one-time SPM resolve + Metal + initial build
swift test -c release
```

You can also run tests through Xcode (`Cmd-U`) — same suites, same
release-config requirement. From the command line that's:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

### Filter to a specific suite or test

```bash
# Single XCTest class
swift test -c release --filter Gemma4ConfigurationTests

# A specific swift-testing suite
swift test -c release --filter NGramSpeculativeTests

# Single test by name
swift test -c release --filter 'NGramLookup minHits filters single-occurrence patterns'

# All KV-cache tests across both frameworks
swift test -c release --filter KVCache

# Exclude long-running benchmark target (the bench harness lives behind
# a Swift Testing filter; without --filter benchmark it never runs)
swift test -c release   # benchmarks are gated, won't run without an MLX_BENCH_MODEL
```

### Test layout

```
Tests/
├── MLXLMTests/            Core LLM unit tests (model configs, eval, tool tests, KV cache, samplers, …)
├── MLXVLMTests/           VLM-specific unit tests (UserInput modality ordering, etc.)
├── MLXEmbeddersTests/     Embedder model tests
└── Benchmarks/            Bench harness (XCTest-shaped but gated behind Swift Testing filters
                           and an MLX_BENCH_MODEL env var; not part of the default test pass)
```

The default `swift test -c release` run executes everything **except** the
benchmark target (which is a real-model harness intended to be invoked via
`./scripts/benchmark.sh`).

## Best practices for writing tests

### 1. Every new public API gets unit-test coverage

If you add a public function, type, or protocol, ship a test in the same
PR. New behaviour without coverage is one of the few things that gets a
review block — pre-merge. The bar is "would you trust this to keep
working in six months when nobody remembers writing it?"

For the harder cases:

- **Pure logic** (parsers, math, sanitizers, config decoders, attention
  masks) — covered with XCTest assertions, no MLX kernels needed. See
  `Tests/MLXLMTests/Gemma4ConfigurationTests.swift` or
  `Tests/MLXLMTests/ToolTests.swift` for the shape.
- **Cache shape / KV semantics** — assertions on `cache.offset`,
  `cache.state.count`, `cache.memoryBytes`, `cache.storageKind`. See
  `Tests/MLXLMTests/Cache*Tests.swift` for the conventions.
- **Generation loop / sampling** — exercise via `EvalTests` with random
  weights + a mock tokenizer. The loop is the unit; the model isn't.
- **Hybrid SSM / GDN dispatch** — `Tests/MLXLMTests/Qwen35BatchedTests`
  patterns: build a cache, drive add/remove slot, assert lockstep.
- **Real-model behaviour** — gate behind the bench harness or
  `IntegrationTesting.xcodeproj`, not the default test run. The default
  pass should not require model downloads.

### 2. Tests must be deterministic

Seed any random sources (`MLXRandom.seed(...)`, `numpy`-equivalent shape
arguments). Don't assert on tok/s, GPU memory, or wall-clock timings in
unit tests — those belong in the bench harness.

### 3. Don't break or change existing tests

Existing tests encode behaviours we've decided to maintain. **Break or
modify an existing test only when**:

- The previous behaviour was wrong (and you can demonstrate that). Cite
  the bug or PR in your commit message.
- The public API has been renamed or removed and the test reflects only
  the old name. Update the test to the new name; keep the assertion intact.
- The test was checking an implementation detail that's no longer
  reachable from the new architecture (rare). Replace it with a test of
  the equivalent observable behaviour.

If you find yourself deleting an assertion to make your change pass, stop
and ask. The assertion almost always represents a real invariant.

### 4. Prefer swift-testing for new suites

New suites should use the swift-testing framework (`@Suite`, `@Test`,
`#expect`, etc.) over XCTest. Existing XCTest classes can stay XCTest —
no need for mass conversion. Examples in tree:

- swift-testing: `Tests/MLXLMTests/NGramSpeculativeTests.swift`,
  `Tests/MLXLMTests/SpeculativeDecodingTests.swift`,
  `Tests/MLXLMTests/BatchTokenIteratorTests.swift`.
- XCTest: `Tests/MLXLMTests/EvalTests.swift`,
  `Tests/MLXLMTests/Gemma4ConfigurationTests.swift`.

### 5. Test the contract, not the implementation

Cache classes have a small public contract (`offset`, `state`, `update`,
`peek`, `trim`, `makeMask`, `copy`, `storageKind`, `isDonor`,
`memoryBytes`). Your tests should drive that contract. Internal storage
shapes (`keys`, `values` arrays inside `StandardKVCache`) are
implementation details — don't pin them in tests unless you're testing
the implementation specifically.

### 6. Smoke tests aren't a substitute for unit tests

A `./scripts/benchmark.sh` smoke run shows that a model produces
*coherent output*. It doesn't show that:

- the cache trims / copies correctly,
- the configuration decodes from edge-case JSON,
- the chat template handles every Chat.Message.Role permutation,
- the tool-call processor recovers from malformed partial output,
- the wired-memory ticket budget arithmetic is right.

Cover those with unit tests. Smoke runs are a final gate on top of
clean unit tests, not a replacement.

### 7. Per-model regressions go in the bench harness

If you add a model port and want to lock in "this model produces correct
output and stays at decode tok/s X", add a row to the bench registry at
`Tests/Benchmarks/Utils/ModelRegistry.swift` and call it out in
[`benchmarks/README.md`](../../benchmarks/README.md). Don't try to encode
real-model behaviour as a swift-testing assertion in the default pass.

### 8. Failing tests are red

If a test fails after your change, the answer is one of:

- Your change has a bug — fix it.
- The test was wrong (rare; see #3) — fix the test, document why.
- The test depended on an implementation detail you legitimately changed
  — replace with a contract-level assertion.

"Skip the test and revisit later" is not on the list. Skipped tests rot.

## Common gotchas

- **Metal library not loading** — you're on debug config. Switch to
  `swift test -c release` or use Xcode's release scheme. The error
  manifests as `Failed to load the default metallib`.
- **`MLX_BENCH_MODEL not set` recorded as an issue** — that's the
  benchmark harness's env-var check, not a test failure. The default
  test run still passes.
- **Concurrency warnings under strict checking** — `MLXArray` is not
  `Sendable`; tests must keep arrays inside one isolation domain. Use
  `SerialAccessContainer` patterns from
  `Tests/MLXLMTests/SerialAccessTests.swift` for any test that crosses
  task boundaries.

## See also

- [Developing in mlx-swift-lm](developing.md) — local dev
  workflow and `make` reference.
- [Benchmarking](benchmarking.md) — bench harness, methodology,
  hardware-dated reports.
- [Architecture](../architecture.md) — module layout (relevant when picking
  where a new test lives).
