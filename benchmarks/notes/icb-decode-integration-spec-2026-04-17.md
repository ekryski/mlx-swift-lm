# ICB Decode Integration — Design Spec — 2026-04-17

Spec for the next phase of the Metal Indirect Command Buffer prototype:
wiring ICB replay into the decode loop to produce a measurable tok/s
speedup. Self-contained so a fresh session can execute without
re-deriving decisions.

**Prior context:**
- Feasibility: [benchmarks/notes/cpu-encoding-optimization-strategies-2026-04-17.md](cpu-encoding-optimization-strategies-2026-04-17.md) (strategy)
- First measurement: [benchmarks/notes/icb-first-measurement-2026-04-17.md](icb-first-measurement-2026-04-17.md)
  — 1.27x on Gemma 4 E2B (100% capture), 1.55x on GPT-OSS 20B (85% capture)
- Baseline + projection: [benchmarks/notes/icb-tok-per-sec-projection-2026-04-17.md](icb-tok-per-sec-projection-2026-04-17.md)
  — Gemma 4 E2B baseline 89.8 tok/s / 143 ms TTFT; naive projection ~107 tok/s (+19%)
- Working branch: `ek/metal-icb-prototype` on all four repos (mlx / mlx-c / mlx-swift / mlx-swift-lm)

---

## Design decision 1: capture scope — per-layer

For each model, record **one ICB per distinct `layerType`** (full vs
sliding attention in GPT-OSS and Gemma, plus any future variants), then
replay that ICB once per matching layer per decode step.

### Why not the alternatives

| Scope | Why not |
|---|---|
| Per-segment (barrier block) | GPT-OSS is ~2 cmds/seg, Gemma is ~1.4 — chunks too small to amortize capture overhead |
| Whole-model forward | One replay per step per recording — no repetition leverage. Rebinding surface explodes (mutable slots at every layer position). Worse integration ergonomics. |
| Per attention / MLP block | Finer-grained than needed. Doubles the recordings without meaningful replay savings. |

### Why per-layer wins

1. **Repetition leverage.** One recording replayed ~15× per step × hundreds of steps. Whole-model gets you 1 replay per step per recording.
2. **`layerTypes` is already the natural split.** Both GPT-OSS (`sliding_attention` / `full_attention`) and Gemma 3/4 (`slidingWindowPattern`) cycle between two or three layer types. Two or three ICBs cover everything; each one is reused ~N/2 times per step.
3. **Small mutable surface.** Per-layer the only per-replay overrides are typically `(x, cache_read_ptr, cache_write_ptr)` and maybe a scalar offset. Whole-forward would need those at every layer position — dozens of slots vs ~4.
4. **Surgical integration.** Wraps the existing `for (i, layer) in layers.enumerated()` loop (see [GPTOSS.swift:383](/Libraries/MLXLLM/Models/GPTOSS.swift#L383), [Gemma3Text.swift:331](/Libraries/MLXLLM/Models/Gemma3Text.swift#L331)) with no cross-cutting refactor.
5. **Capture cost is negligible.** Feasibility data shows ~400 µs to record 1500 trivial commands. For 13-31 cmds per layer × 2 types, the one-time capture is sub-millisecond.

### Expected command counts per recording

| Model | Commands per layer | Distinct layer types |
|---|---|---|
| Gemma 4 E2B | ~13 | 2 (full + sliding) |
| GPT-OSS 20B | ~31 (captured only — MoE adds more via multi-stream) | 2 |

---

## Design decision 2: integration approach — `replay(overrides:)`

Add `IndirectCommandBuffer.replay(overrides: [Name: MLXArray])` rather
than doing a persistent-buffer model refactor.

### Why `replay(overrides:)`

1. **No model refactor.** Every mlx model in the library stays functional.
   Compatible with upstream mlx-swift's design.
2. **Metal-native.** `MTL::IndirectComputeCommand::setKernelBuffer(buffer, offset, slot)`
   is explicitly mutable; re-binding between `executeCommandsInBuffer`
   calls is the design pattern Apple intended.
3. **Reversible.** Additive API — can be ripped out with zero blast radius
   if results don't match projection.
4. **Fights the right battle.** mlx is a functional array API. A
   persistent-buffer refactor would fight that core design.
5. **Small rebinding cost.** 3–4 `setKernelBuffer` calls per replay × 30
   layer replays ≈ 120 Obj-C calls per decode step — microseconds vs. the
   milliseconds ICB saves on encoding.

### Why not persistent-buffer refactor

- Touches every primitive and every model in the library.
- Fights mlx's functional design (every op returns a fresh array).
- High blast radius — non-ICB users would pay the refactor cost too.
- Premature. If `replay(overrides:)` delivers the win, we're done. If
  its rebinding overhead eats too much of the win, THEN persistent
  buffers become a V2 optimization worth its cost.

---

## API sketch

### Swift (mlx-swift)

```swift
public final class IndirectCommandBuffer {
    // Existing static record with untagged bindings:
    public static func record(
        maxCommandsPerSegment: Int = 2048,
        bytesArenaCapacity: Int = 64 * 1024,
        stream: StreamOrDevice = .default,
        _ block: () throws -> Void
    ) rethrows -> IndirectCommandBuffer

    // NEW: named-binding record. During record, the closure tags specific
    // MLXArrays with names; the recorder remembers which (command_idx, slot)
    // each named MLXArray was bound to. At replay time, overrides map
    // name → new MLXArray and the recorder calls setKernelBuffer on the
    // corresponding command/slot.
    public static func recordWithBindings(
        names: [BindingName],
        maxCommandsPerSegment: Int = 2048,
        bytesArenaCapacity: Int = 64 * 1024,
        stream: StreamOrDevice = .default,
        _ block: (BindingTagger) throws -> Void
    ) rethrows -> IndirectCommandBuffer

    // Existing replay:
    public func replay(stream: StreamOrDevice = .default)

    // NEW: replay with named overrides. Overrides map name → MLXArray;
    // the recorder mutates the matching IndirectComputeCommand slots
    // before executing the ICB. Names not present in overrides keep
    // their recorded bindings.
    public func replay(
        overrides: [BindingName: MLXArray],
        stream: StreamOrDevice = .default
    )
}

public struct BindingName: Hashable, Sendable { ... }

public final class BindingTagger {
    // Caller uses this inside the record block to tell the recorder
    // "this MLXArray at this point in the execution is the `input` binding":
    public func tag(_ array: MLXArray, as name: BindingName)
}
```

### Usage — decoder loop integration

```swift
// Inside Gemma3Model.callAsFunction or equivalent:
var x = embedTokens(inputs)
var slidingICB: IndirectCommandBuffer? = nil
var fullICB: IndirectCommandBuffer? = nil

for (i, layer) in layers.enumerated() {
    let isFull = (layerTypes[i] == "full_attention")
    let icbSlot: IndirectCommandBuffer? = isFull ? fullICB : slidingICB
    let cacheI = cache[i]

    if let icb = icbSlot {
        // Fast path — replay with this step's bindings
        icb.replay(overrides: [
            .input: x,
            .cacheKeys: cacheI.keys,
            .cacheValues: cacheI.values,
            .output: /* fresh output buffer */
        ])
        // After replay, `x` needs to come from the output buffer; the
        // precise plumbing depends on how outputs flow. See integration
        // notes below.
    } else {
        // First encounter with this layer type — record
        let recorded = try IndirectCommandBuffer.recordWithBindings(
            names: [.input, .cacheKeys, .cacheValues, .output]
        ) { tagger in
            tagger.tag(x, as: .input)
            tagger.tag(cacheI.keys, as: .cacheKeys)
            tagger.tag(cacheI.values, as: .cacheValues)
            x = layer(x, mask: mask, cache: cacheI)
            tagger.tag(x, as: .output)
        }
        if isFull { fullICB = recorded } else { slidingICB = recorded }
    }
}
```

### Binding provenance (how the tagger works internally)

During recording, when `tagger.tag(array, as: name)` is called:
1. The recorder looks up `array.buffer().ptr()` in its recent-bindings
   history — scan backwards through commands to find the first command
   whose slot was bound with this buffer pointer.
2. If found, store `(name, command_idx, slot)` in a tag table.
3. If not found (caller tagged something that wasn't bound yet or was
   bound as output), defer: wait until the next `set_kernel_buffer` for
   this buffer and tag it then.

For output tagging (`tag(result, as: .output)` after the op ran), the
recorder needs to find commands whose `set_output_array` bound that
buffer. Since outputs are written, typically the last command's output
slot is the top-level output.

On replay with overrides, for each `(name → new_array)`:
1. Look up the tag table — find all `(command_idx, slot)` pairs for that name.
2. Before `executeCommandsInBuffer`, call
   `indirectCommandBuffer.indirectComputeCommandAtIndex(ci).setKernelBuffer(new_array.buffer, offset, slot)`.
3. Also update the recorder's `resource_set_` so `useResource` covers the
   new buffer on replay.

### C / C++ layer additions

Add to `mlx/c/metal.h`:
```c
// Named binding. `name` is an opaque 32-bit ID assigned by the recorder
// at tag time; the Swift layer maintains a name ↔ ID mapping.
int mlx_metal_icb_tag_binding(
    mlx_stream stream,
    uint32_t name_id,
    const mlx_array array);

int mlx_metal_icb_replay_with_overrides(
    mlx_stream stream,
    mlx_metal_icb_recorder rec,
    const uint32_t* override_name_ids,
    const mlx_array* override_arrays,
    size_t n_overrides);
```

Add to `CommandEncoder`:
```cpp
void tag_binding(uint32_t name_id, const array& a);
```

Add to `IndirectCommandRecorder`:
```cpp
// Called by CommandEncoder::tag_binding when recording.
// Looks up recent bindings of a.buffer().ptr() and stores (name, cmd_idx, slot).
void tag_binding(uint32_t name_id, const array& a);

// Replay with bindings patched pre-execute.
void replay_with_overrides(
    MTL::ComputeCommandEncoder* enc,
    const std::vector<std::pair<uint32_t, const array*>>& overrides) const;
```

---

## Work breakdown

| # | Task | Repo(s) | Est. |
|---|---|---|---|
| 1 | `IndirectCommandRecorder::tag_binding` + lookup table | mlx | 1 h |
| 2 | `IndirectCommandRecorder::replay_with_overrides` | mlx | 1 h |
| 3 | `CommandEncoder::tag_binding` pass-through | mlx | 15 m |
| 4 | C API: `mlx_metal_icb_tag_binding`, `_replay_with_overrides` | mlx-c | 45 m |
| 5 | Swift: `BindingTagger`, `BindingName`, `recordWithBindings`, `replay(overrides:)` | mlx-swift | 1.5 h |
| 6 | Decoder integration: Gemma 4 E2B per-layer ICB caching | mlx-swift-lm | 1.5 h |
| 7 | `--icb` flag on `simple` benchmark; measure tok/s + TTFT | mlx-swift-lm | 45 m |
| 8 | Numerical parity check: single-prompt live vs ICB-enabled output comparison | mlx-swift-lm | 45 m |
| **Total** | | | **~7 h** |

Parallelization: tasks 1-4 must be sequential. Task 5 can overlap with
task 6 prep (interface design). Tasks 7-8 sit on top of 6.

Validation gates:
- After task 4: C-level round trip (record + tag + replay with override
  on a synthetic 3-op graph, verify output matches non-replay eval).
- After task 6: one decoder step replay produces same logits as live.
- After task 7: tok/s measurement.
- After task 8: full-generation token identity with and without ICB.

Exit criterion: Gemma 4 E2B tok/s with `--icb` ≥ 100 tok/s (vs 89.8
baseline) and generated output is token-identical to baseline for a
deterministic prompt. If below 100 tok/s but numerically correct,
investigate rebinding overhead.

---

## Open design questions (resolve during task 1)

1. **Name IDs vs strings.** Simplest: caller-chosen `uint32_t` IDs. Swift
   side owns a registry to avoid collisions. Cleaner DX: `BindingName`
   is a `struct` wrapping a string, hashed to a uint32 at the C boundary.
   *Recommendation:* struct wrapping string for DX.

2. **Output binding semantics.** mlx's `set_output_array` calls
   `set_input_array` then `register_output_array`. Recording already
   tracks outputs via resource_set for useResource. The tag mechanism
   just needs to map `set_output_array(a, idx)` to `(cmd_idx, slot=idx)`
   the same as inputs.

3. **What happens if a tagged array gets re-bound multiple times?** E.g.,
   `x` is the input to layer 0, the output to layer 0's norm, the input
   to attention, ... In per-layer scope this won't happen (scope is small
   enough). Cross-check during implementation; if it does, store a list
   of `(cmd_idx, slot)` per name rather than one.

4. **Do we need `tag` to work for raw MTL::Buffer as well as MLXArray?**
   `set_buffer(buf, idx)` path is used by Fence and possibly custom
   kernels. Skip for V1; only mlx::array tagging.

5. **Stream handling.** Tagging happens on a specific stream's
   CommandEncoder. If a primitive schedules to a different stream,
   tagging misses it. This is the same multi-stream issue we saw with
   the 134-dispatch leak on GPT-OSS — tracked separately in the
   first-measurement note. For Gemma 4 E2B (100% single-stream), not a
   problem.

---

## Starting state (as of this spec)

- Branch: `ek/metal-icb-prototype` on all four repos, pushed to `ekryski`
- All existing tests pass (mlx C++ 264/265; the one failure is a
  pre-existing scheduler flake)
- ICB recorder, CommandEncoder integration, Swift API, and `--method icb`
  benchmark all shipped
- Diagnostic counters (dispatch calls routed, pre-pipeline skips,
  dispatches leaked during record) all wired up and printing to stderr

A fresh session can start at **task 1** without re-deriving any of the
above. The first compile-test-debug cycle is on
`/Users/eric/Development/personal/ai/mlx` building the mlx C++ recorder
extensions.
