# Developing in mlx-swift-lm

What's where, why, and how to develop locally.

## Layout

```
mlx-swift-lm/
├── Libraries/
│   ├── MLXLMCommon/        Cross-cutting infra (KV cache, generate(), wired memory, downloader/tokenizer protocols)
│   │   └── Models/         Bit-identical layers shared between LLM and VLM (Gemma 3, Gemma 4, GLM 4, LFM 2, Mistral 3, Qwen 2/3/3.5)
│   ├── MLXLLM/             Text-only LLMs — one Swift file per architecture under Models/, plus LLMModelFactory
│   ├── MLXVLM/             Vision-language models — same shape as MLXLLM plus a vision encoder + processor + chat template per file
│   ├── MLXEmbedders/       Encoder / embedding models (BERT, NomicBERT, ModernBERT)
│   └── MLXHuggingFace/     Optional macros for the official HF downloader/tokenizer stack
├── Tests/                  XCTest + swift-testing suites + the bench harness
├── Tools/                  CLI tools (mlx-swift-lm bench harness lives under Tests/Benchmarks)
├── benchmarks/             Hardware-dated bench result reports (markdown, ~1 file/day)
├── documentation/          ← you are here
├── scripts/                build-metallib.sh, benchmark.sh, release.sh, setup-dev.sh
├── skills/                 In-repo Skill manifest (Claude Code)
├── specs/                  Implementation specs by tier (KV cache, ANE LM head, chunkwise GDN, …)
└── Makefile                Smart build orchestrator — see [architecture.md § Build pipeline](../architecture.md#build-pipeline)
```

## What kind of change are you making

Pick the section that matches your work — each has a different friction
profile.

### Infrastructure (KV cache, samplers, generate loop, tool-call parser)

Forks easiest of all. Most code in `Libraries/MLXLMCommon/` has thorough unit
tests (`Tests/MLXLMTests/`) that don't require downloading model weights.
There's also a small set of model-shaped unit tests in
`Tests/MLXLMTests/EvalTests.swift` that exercise the generate loop with
random weights and a mock tokenizer — useful for testing the loop itself
rather than any particular model.

```bash
swift test -c release                      # full suite
swift test -c release --filter KVCache     # narrow to a topic
swift test -c release --filter ToolTests
```

### Porting / modifying a model

Three options, roughly in order of friction:

1. **`IntegrationTesting.xcodeproj`** (in this repo). Integrates with the
   HuggingFace downloader / tokenizer via the `MLXHuggingFace` macros (see
   [llm/using.md § Picking an integration](../llm/using.md#picking-an-integration)).
   Code lives in `Libraries/IntegrationTestHelpers/`. **Not run in CI** but
   the most direct way to bring up a real model end-to-end while iterating.
2. **The bench harness** (`./scripts/benchmark.sh`). Fastest for verifying
   coherent output and decode tok/s on a known checkpoint. See
   [benchmarking.md](benchmarking.md).
3. **`llm-tool` from [`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples/blob/main/Tools/llm-tool/README.md)
   or your own app.** Same as option 2 but with a real CLI / app harness.
   To point `mlx-swift-examples` at your local mlx-swift-lm fork:

   - Drag the `mlx-swift-lm` directory onto the project in Xcode's navigator
     and choose _Reference files in place_, **or**
   - [Edit a package dependency as a local package](https://developer.apple.com/documentation/xcode/editing-a-package-dependency-as-a-local-package).

   Both approaches let you edit `mlx-swift-lm` and the consumer at the same
   time. Same trick works for any consumer that depends on this package.

For the actual porting steps see [Adding an LLM](../llm/adding-a-model.md) /
[Adding a VLM](../vlm/adding-a-model.md), with the deeper Python → Swift map
in [porting.md](porting.md).

### Build pipeline / Metal kernels

Anything touching `Source/Cmlx/`, `.metal` files, or the dep chain
(`mlx-swift` → `mlx` → `mlx-c`):

- Use the `Makefile`. Plain `swift build` doesn't compile Metal shaders,
  doesn't catch C/C++ submodule changes, and doesn't refresh the test
  bundle. See [architecture.md § Build pipeline](../architecture.md#build-pipeline).
- Run `make doctor` before assuming a stale-cache failure is real — it
  diagnoses the two common silent-staleness modes (SPM pin too old; submodule
  drift) in seconds.
- For cross-repo work (mlx-swift-lm + mlx-swift simultaneously), check
  out `mlx-swift` as a sibling directory; the Makefile picks up
  `../mlx-swift` automatically.

## Local dev workflow

Run once after cloning, or after fetching new `mlx-swift` changes:

```bash
./scripts/setup-dev.sh
```

This resolves the SwiftPM graph, builds the Metal library, runs `make
build-tests`, and verifies `make doctor`.

For day-to-day work:

```bash
make                        # incremental build (only rebuilds what changed)
make build-tests            # build + copy artifacts into the .xctest bundle
swift test -c release       # full suite (release config; debug fails on metallib)
make metal                  # rebuild Metal shaders only
make spm                    # SPM only (with Cmlx cache invalidation)
make status                 # show what's built and what's stale
make doctor                 # offline dep-chain sanity check
make clean-cmlx             # force C/C++ rebuild
make clean-all              # nuke SPM + Metal caches (preserves unrelated package caches)
make help                   # full reference
```

You usually don't need to call `make` directly — `setup-dev.sh` and
`benchmark.sh` both invoke it internally.

## Submitting a change

- One PR per concern. The WS-C consolidation sprint (PRs #172–#180) used
  one PR per model family for clean review boundaries.
- Include a smoke run in the PR description for any model-affecting change
  (`./scripts/benchmark.sh --model <model> --quant 4bit --kv none --method
  simple`). For VLM changes also include `--method vision`.
- For perf-sensitive paths, keep the perf delta within ±2% of the pre-PR
  numbers on M1 Max 64GB unless you're explicitly opting for a regression.
- Existing alpha-branch optimisations (compiled QKV, `MLXFast.rmsNormRoPE`,
  `FusedGateUpSwitchGLU`, conditional MLP-gate compile) are documented in
  the file headers and commit history. If you're touching one, read why
  it's there before changing the shape.

## See also

- [Porting guide](porting.md) — Python → Swift mapping, dtype quirks, KV-cache
  adapters.
- [Benchmarking](benchmarking.md) — full bench-harness reference.
- [Publishing a release](../publishing-a-release.md) — release pipeline,
  workflow inputs, cross-repo coordination.
- [Architecture](../architecture.md) — module layout and the LLM ↔ VLM
  consolidation map.
