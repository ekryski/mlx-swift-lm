# Memory Management

`mlx-swift-lm` runs on Apple Silicon's **unified memory** — there's a
single physical pool the CPU and GPU share, and Metal exposes a
"wired" budget the GPU is guaranteed to keep resident. By default,
the OS reserves about **75 % of unified memory** for the wired budget
(so ~48 GB on a 64 GB Mac). On machines with more headroom, you can
raise that with `sudo sysctl iogpu.wired_limit_mb=<value>` — the
change resets on reboot.

> ⚠️ Be careful raising the wired limit too far. Anything you give the
> GPU comes out of what the OS can use to keep the rest of the system
> responsive.

Inside that budget, **wired-memory tickets** are how `mlx-swift-lm`
coordinates concurrent inference. A ticket is a hint that says _"this
work will use up to X bytes of wired memory until I finish"_. Policies
look at the active set of tickets and compute the wired limit the
process should ask Metal for. When several tasks request inference at
once, the policy decides admission and limit; tickets gracefully
release on completion.

For the upstream MLX wired-memory primitives (manager / policy /
ticket / hysteresis), see
<https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Documentation.docc/Articles/wired-memory.md>.
The rest of this page covers the LLM-specific helpers `mlx-swift-lm`
adds on top: smart memory (the default), policy / ticket selection,
and weight reservations.

## Smart memory (the default)

You don't have to do anything for memory management to work — the
library ships with a smart estimator on by default. `WiredMemoryUtils`
sizes a ticket from the loaded model in three precedence-ordered modes:

1. **Explicit limit (`MLX_MEMORY_LIMIT`).** Bypass the estimator
   entirely. Accepts plain bytes and human-friendly suffixes (`32g`,
   `32GB`, `512m`, `4k`, `1.5g`), case-insensitive. Clamped to
   `GPU.maxRecommendedWorkingSetBytes()`.
2. **Smart estimate (`MLX_SMART_MEMORY != "0"`, the default).** Computes
   `weights + kv(maxTokens × batchSize, compressionAlgorithm) + workspace`
   from the loaded model. The KV term is precise when callers pass
   `kvHeadsOverride` and `headDimOverride` (most production paths do
   via `KVCacheDimensionProvider`); otherwise falls back to a
   conservative heuristic (`kvHeads=8`, `headDim=128`, fp16).
3. **Fallback.** When smart memory is disabled and no explicit limit
   is set, the ticket sizes itself at
   `GPU.maxRecommendedWorkingSetBytes()` (the OS-decided wired
   budget).

```swift
// One-shot ticket from the smart estimate.
let ticket = WiredMemoryUtils.estimatedTicket(
    model: ctx.model,
    maxTokens: contextSize + maxNewTokens,
    parameters: generateParameters
)

let stream = try generate(
    input: lmInput,
    parameters: generateParameters,
    context: ctx,
    wiredMemoryTicket: ticket
)
```

For more knobs (override KV dims, mix in batch size, hand-tune
workspace) use `WiredMemoryUtils.resolveTicket(...)`:

```swift
let ticket = WiredMemoryUtils.resolveTicket(
    model: ctx.model,
    maxTokens: contextSize + maxNewTokens,
    parameters: generateParameters,
    batchSize: 4,                        // KV scales linearly with B
    kvHeadsOverride: ctx.model.kvHeads,  // when KVCacheDimensionProvider conformer
    headDimOverride: 128
)
```

## Wired-memory ticket lifecycle

When you control the lifecycle yourself (multi-tenant scheduler, test
harness, custom admission), reach for `WiredMemoryManager` and a
policy directly:

```swift
let policy = WiredSumPolicy(cap: 12 * 1024 * 1024 * 1024)  // 12 GB cap
let ticket = policy.ticket(size: estimatedBytes, kind: .active)

let stream = try generate(
    input: lmInput,
    parameters: GenerateParameters(),
    context: ctx,
    wiredMemoryTicket: ticket
)
```

The ticket auto-releases when the stream finishes. For deterministic
start/end pairing under task cancellation, use
`WiredMemoryTicket.withWiredLimit(_:_:)`.

### Active vs reservation tickets

- **`.active`** — contributes to the wired limit while inference is
  running. Use for the per-request KV + workspace cost.
- **`.reservation`** — tracks long-lived budgets (typically model
  weights) without keeping the wired limit elevated when no active
  tickets exist. Useful when you keep a model loaded across many
  short-lived requests.

```swift
let weightsReservation = policy.ticket(size: weightBytes, kind: .reservation)
let inferenceTicket   = policy.ticket(size: kvAndWorkspaceBytes, kind: .active)
```

### Policy selection

| Policy | When to use |
|---|---|
| `WiredSumPolicy(cap:)` | Common default. Limit = baseline + sum(active sizes), capped at `cap`. |
| `WiredMaxPolicy()` | Limit = max(baseline, largest active request). Lower memory pressure under bursty loads. |
| `WiredFixedPolicy(limit:)` | Constant limit when any work is active. Predictable but inflexible. |
| `WiredBudgetPolicy(baseBytes:cap:)` | Limit = `baseBytes` (e.g. measured weights+workspace) + sum(active). Use after measurement. |

### Measurement-driven budgeting

For long-running services, measure once with
`WiredMemoryUtils.tune(...)` and reuse:

```swift
let measurement = try await WiredMemoryUtils.tune(
    context: ctx,
    tokenCount: 2048,
    parameters: parameters
)

let baseBytes = measurement.weightBytes + measurement.workspaceBytes
let policy   = WiredBudgetPolicy(baseBytes: baseBytes)
let ticket   = policy.ticket(size: measurement.kvBytes, kind: .active)
```

For VLMs, pass the real `userInput` so image / video tensors are
counted:

```swift
let measurement = try await WiredMemoryUtils.tune(
    userInput: userInput,
    context: ctx,
    parameters: parameters
)
```

## CPU and unsupported backends

When wired-limit control is unavailable (CPU-only, simulator), keep
policy math + admission control active so the same code path works on
both:

```swift
await WiredMemoryManager.shared.updateConfiguration { config in
    config.policyOnlyWhenUnsupported = true
}
```

In DEBUG builds you can stream manager events for policy stacking and
limit changes. In release builds the stream is a no-op.

```swift
Task {
    for await event in WiredMemoryManager.shared.events() {
        print(event)
    }
}
```

## Environment-variable overrides

| Variable | Effect |
|---|---|
| `MLX_MEMORY_LIMIT` | Explicit wired-memory limit. Accepts plain bytes or human-friendly units (`32g`, `32GB`, `512m`, `4k`, `1.5g`), case-insensitive. Bypasses the smart estimator and `MLX_SMART_MEMORY`. Clamped to `GPU.maxRecommendedWorkingSetBytes()`. |
| `MLX_SMART_MEMORY` | `0` disables the model-aware estimator (then ticket falls back to `GPU.maxRecommendedWorkingSetBytes()`). Anything else, including unset, leaves the smart estimator on (the default). |

## Estimating model weight bytes

The sections below are the canonical reference for **weight-reservation
sizing** — when you want to budget the long-lived weights component
independently from per-request inference cost.

### Measuring weight bytes at runtime

If you can afford to load the model, the most accurate approach is to sum `nbytes` across all
parameter arrays. Loading a model already materializes weights, so you get a reliable number
without running any extra inference.

```swift
let context = try await LLMModelFactory.shared.load(
    from: downloader,
    using: tokenizerLoader,
    configuration: config
)
let weightBytes = context.model
    .parameters()
    .flattened()
    .reduce(0) { $0 + $1.1.nbytes }
```

You can optionally sanity check with `Memory.snapshot()` before/after load. In practice, the
difference between the sum of `nbytes` and MLX active memory has been very small in our tests.

### Avoiding load: estimate from tensor files

If you want a **no-load estimate**, sum the tensor file sizes on disk (for example, all
`.safetensors` shards in the model directory). This is fast and avoids allocating the model,
but it includes file metadata and may slightly exceed the in-memory representation.

```swift
let tensorExtensions: Set<String> = ["safetensors", "bin", "gguf"]
let sizes = fileSizes(in: modelURL, tensorExtensions: tensorExtensions)
let estimatedBytes = sizes.tensorTotalBytes
```

### What affects the difference?

- **File metadata/headers**: safetensors includes a JSON header; shard totals usually exceed
  `nbytes` by a small amount.
- **Allocator alignment/overhead**: MLX active memory may be a tiny bit larger than the logical
  `nbytes` sum.
- **Format differences**: compressed or container formats can cause larger gaps between on-disk
  size and in-memory representation.

### Observed deltas (local measurements)

These measurements were taken on two local models using the runtime `nbytes` sum, tensor file
sizes, and MLX `Memory.snapshot()` right after load (no inference):

| Model | Sum of `nbytes` | Tensor file total | Active memory after load | Notes |
| --- | ---: | ---: | ---: | --- |
| Qwen3-4B-Sky-High-Hermes-4bit | 2,262,535,712 | 2,262,637,937 | 2,264,337,376 | +102,225 bytes vs files; +1,801,664 bytes vs active |
| Qwen3-Next-80B-A3B-Instruct-MLX-4bit | 44,844,060,160 | 44,844,286,608 | 44,844,101,616 | +226,448 bytes vs files; +41,456 bytes vs active |

These examples suggest that **`nbytes` is a reliable basis** for a reservation ticket when you
can load the model, and file-size estimates are a close approximation when you cannot.

### Diagnostic utilities

MLXLMCommon includes lightweight helpers to measure real memory usage so you can
model tickets based on observed behavior rather than only static estimates.
The utilities are policy-agnostic; use the measurements to size tickets or
validate a policy's budget assumptions.

Use `WiredMemoryUtils.tune(...)` to capture:

- `weightBytes` from `nbytes` (stable)
- `kvBytes` from actual cache arrays after prefill
- `workspaceBytes` from the prefill peak (transient)

The returned `WiredMemoryMeasurement` can be used to build a budget policy or to
validate manual calculations. For multimodal models, prefer the overload that
accepts a prepared `LMInput` or a `UserInput` so the measurement includes image
or video tensors.

### Practical guidance for tickets

- If you **can load**: compute `nbytes` once at load time and reuse it for the model's lifetime.
- If you **cannot load**: sum tensor file sizes as a proxy.
- Add a **small fixed margin** (e.g., 16-64 MB) to cover allocator overhead and minor variance.

For inference workloads, keep **weights**, **KV cache**, and **activation** budgets separate so
policies can scale the wired limit based on what is actually active.

### Policy-only budgeting on CPU

If wired memory control is unavailable (CPU-only execution), you can still use
policies for admission gating and budgeting by enabling policy-only mode on the
manager. This keeps ticket tracking and limit math active without attempting to
change the wired limit. Policy-only mode defaults to `true` on unsupported
backends.

```swift
await WiredMemoryManager.shared.updateConfiguration { configuration in
    configuration.policyOnlyWhenUnsupported = true
}
```

You can also provide `baselineOverride` (a fixed budget), or rely on
`GPU.maxRecommendedWorkingSetBytes()` when running on Apple Silicon with unified
memory.

## Estimating KV cache and attention workspace

Inference tickets are typically driven by **KV cache** (persistent) plus **prefill workspace**
(transient). Dense models are straightforward; MoE and hybrid models (like Qwen3-Next with
full-attention + linear/SSM layers) need a layer-by-layer sum using config values.

### Dense full-attention KV cache

For standard attention layers:

```
elements per token per layer = 2 * kvHeads * headDim
layer elements = tokens * elements per token per layer
layer bytes = layer elements * bytesPerElement
total KV bytes = layer bytes * numAttentionLayers
```

Where `bytesPerElement` is 2 for FP16/BF16, 1 for INT8, and 0.5 for INT4.

### Hybrid / MoE models with SSM (example: Qwen3-Next)

Qwen3-Next alternates full-attention layers with linear/SSM layers. Use the same KV math above
for **full-attention layers**, then add the SSM cache sizes for the linear layers.

For the SSM cache per linear layer, one workable approximation is:

```
convState elements = B * (convKernelSize - 1) * convDim
convDim = (keyHeadDim * numKeyHeads) * 2 + (valueHeadDim * numValueHeads)

state elements = B * numValueHeads * valueHeadDim * keyHeadDim

linear layer bytes = (convState elements + state elements) * bytesPerElement
total linear bytes = linear layer bytes * numLinearLayers
```

This yields a small but non-zero persistent cache budget for the linear/SSM layers.

### Prefill attention workspace (transient)

Prefill can allocate large temporary buffers proportional to the **prefill chunk size** `L`. A
simple upper bound for a single attention layer in FP16/BF16 is:

```
Q = B * H * L * D
K = B * Hkv * L * D
V = B * Hkv * L * D
Scores = B * H * L * L
Output = B * H * L * D
```

Multiply each by `bytesPerElement`, then sum to estimate peak transient workspace. If the model
uses an additional gating tensor, include it as `B * L * (H * D)`.

### Practical guidance

In MLXLMCommon, most callers will **create a single ticket** and run `generate()` inside the
ticket scope. In that case, budget the ticket for the **peak** expected usage
(weights + KV cache + prefill workspace). If you already created a **separate reservation
ticket** for weights, then the inference ticket should cover **KV cache + prefill workspace**
only.

If you need tighter control, you can split budgets by phase (e.g., a transient add-on for
prefill), but the common path is a single ticket.

- Compute **KV cache** separately from **weights**; KV persists for the duration of generation.
- Include **prefill workspace** in your peak estimate (it is transient, but can dominate memory).
- For hybrid models, sum all components (full-attention KV + linear/SSM cache + workspace).
- When using KV quantization, change `bytesPerElement` accordingly.
