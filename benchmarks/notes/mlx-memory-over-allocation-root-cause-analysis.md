# MLX Swift Memory Over-Allocation — Root Cause Analysis

**Date:** 2026-04-09
**Context:** Eric (ekryski) observed mlx-swift-lm retaining ~5.4 GB of buffer pool "cache" after a single prefill-4k + decode run, vs. mlx-lm Python retaining ~0.21 GB in the same workload. Both wrap the identical C++ MetalAllocator, so the discrepancy is not in allocator policy itself.

---

## 1. Root cause

**The Swift prefill chunk loop is missing an explicit `MLX.Memory.clearCache()` call between chunks.** Python's `mlx_lm/generate.py` calls `mx.clear_cache()` after every prefill chunk (and every 256 decode tokens). Swift's equivalent call sites have the *decode* clear already (Eric added it in `cb8e2af`), but **no clear between prefill chunks**.

Because the shared C++ `MetalAllocator` default `max_pool_size_` (the cache limit) is set to `block_limit_`, which is `min(1.5 * max_recommended_working_set_size, 0.95 * memsize)`—tens of GB on any modern Mac—the buffer pool **never** trims naturally during a single 4 k prefill. Every intermediate tensor (Q/K/V projections, SDPA workspace, MLP intermediates, logits slabs) produced by each prefill chunk gets `free()`'d into the pool via `recycle_to_cache` and sits there until either:

1. total memory pressure reaches `gc_limit_` (~0.95 × max_rec_size), or
2. the pool itself exceeds `max_pool_size_` (block_limit_), or
3. something explicitly calls `clear_cache()`.

For a 4 k prefill of a ~3 GB model, none of those fires—so every prefill chunk's garbage stacks up and Eric sees 5.4 GB of cache when only ~0.21 GB of genuinely reusable tensors exist.

### Evidence (file:line)

**Swift prefill loops — the bug sites (no `clearCache`):**

- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:22-37` — default `LLMModel.prepare`, used by most models:
  ```swift
  while y.tokens.size > prefillStepSize {
      let input = y[.newAxis, ..<prefillStepSize]
      _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
      eval(cache)
      y = y[prefillStepSize...]
  }                             // <-- no clearCache()
  ```
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:696-701` — same structure
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift:829-834` — same structure
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift:442-447` — same structure

**Shared C++ allocator defaults** (`/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp:46-54`):
```cpp
auto memsize = std::get<size_t>(info.at("memory_size"));
auto max_rec_size =
    std::get<size_t>(info.at("max_recommended_working_set_size"));
resource_limit_ = std::get<size_t>(info.at("resource_limit"));
block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);
gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);
max_pool_size_ = block_limit_;   // <-- cache limit == memory limit
```

**Cache reclaim only triggers under real pressure** (`allocator.cpp:127-175`):
```cpp
// Try the cache
MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
if (!buf) {
    size_t mem_required = get_active_memory() + get_cache_memory() + size;
    // Only reclaim if near gc_limit_ (~0.95 * max_rec_size)
    if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }
    ...
}
// ...
// Maintain the cache below the requested limit
if (get_cache_memory() > max_pool_size_) {   // almost never true with default
    num_resources_ -= buffer_cache_.release_cached_buffers(
        get_cache_memory() - max_pool_size_);
}
```

And `free()` always recycles into the pool:
```cpp
void MetalAllocator::free(Buffer buffer) {
  ...
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);   // <-- almost always this path
  } else { ... }
}
```

So on a fresh Mac with 36 GB unified memory, `max_pool_size_` is ~54 GB and the pool effectively never auto-shrinks during a single inference request.

---

## 2. Why Python doesn't have this

Python's `mlx_lm/generate.py` **explicitly** calls `mx.clear_cache()` at every prefill-chunk boundary and every 256 decode tokens. The underlying C++ allocator is identical, but mlx-lm Python compensates by proactively flushing the pool.

### Evidence (file:line)

`/Users/tom/dev/mlx-lm/mlx_lm/generate.py:429-450` — single-sequence prefill loop (`generate_step`):
```python
while total_prompt_tokens - prompt_processed_tokens > 1:
    remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
    n_to_process = min(prefill_step_size, remaining)
    _model_call(...)
    quantize_cache_fn(prompt_cache)
    mx.eval([c.state for c in prompt_cache])
    prompt_processed_tokens += n_to_process
    ...
    mx.clear_cache()                    # <-- missing in Swift
```

`/Users/tom/dev/mlx-lm/mlx_lm/generate.py:466-467` — decode loop:
```python
yield y.item(), logprobs
if n % 256 == 0:
    mx.clear_cache()                    # <-- Eric already added this in Swift
```

`/Users/tom/dev/mlx-lm/mlx_lm/generate.py:572-579` — speculative-decode prefill:
```python
def _prefill(model, cache, y):
    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=cache)
        quantize_cache_fn(cache)
        mx.eval([c.state for c in cache])
        y = y[prefill_step_size:]
        mx.clear_cache()                # <-- missing in Swift
    return y
```

Also `generate.py:1150-1162` (`BatchGenerator.prompt` loop — every prefill chunk and every finalize).

Git history: `d4701ba clear cache on prompt ingestion in server (#917)` (Feb 2026) and `179da77 Clear the cache during batch generation (#926)` put these in place upstream. So this is a known, intentional pattern in mlx-lm Python.

### Secondary confirmation

- Swift `MLX.Memory.clearCache()` literally just calls `mlx_clear_cache()` which is `mlx::core::clear_cache()` → `metal::allocator().clear_cache()`. Same entry point as Python. (`/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/MLX/Memory.swift:356-360`, `allocator.cpp:178-182`, `allocator.cpp:275-277`)
- No Swift-side buffer caching layer exists on top of the C++ allocator. `MLXArray` is a final class whose `deinit` calls `mlx_array_free` (`/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/MLX/MLXArray.swift:35-37`), which drops through to `MetalAllocator::free`. ARC is not the problem.
- The Python binding layer (`/Users/tom/dev/mlx/python/src/memory.cpp:10-124`) exposes the exact same six functions with the exact same semantics as the Swift `Memory` enum (`/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/MLX/Memory.swift:175-360`). There is no auto-shrink hook in either binding layer. Both rely on caller-driven `clear_cache()`.
- Python mlx-lm does **not** `set_cache_limit` anywhere in `generate.py`—it relies purely on explicit `clear_cache()`. So the fix for Swift is symmetric: match the call pattern, not fiddle with limits.

---

## 3. Eric's fork status (`ek/speed-improvements-2`, revision `d87ed12`)

The mlx-swift checkout in `.build/checkouts/mlx-swift` is pinned to `d87ed12b11be6f2a0fc0d48cc6c30a5332f08afb`, which is **one commit ahead of `origin/main`**:

- `d87ed12 perf: lower gather_qmm_rhs threshold from B>=16 to B>=4` — unrelated to memory

Nothing in `ek/speed-improvements-2` touches the allocator, buffer pool, or prefill memory management on the core `mlx-swift` side.

On the `mlx-swift-lm` side (Eric's wrapper, currently on `tom/turboquant-fixes`), the closest relevant work is:

- `cb8e2af periodic GPU cache clear, revert compiled sampler (deadlocks)` (Mar 24, 2026) — added `MLX.Memory.clearCache()` every 256 tokens in the decode loop of `Evaluate.swift:1408-1412`, with the comment "matching Python mlx-lm behavior to prevent memory fragmentation." **This is a partial fix; it addresses the decode side but not the prefill side.**
- `0055e08 feat: model-aware smart memory pinning with KV quantization support` — wired-memory budgeting; orthogonal to this bug.

No commit yet addresses prefill-chunk cache clearing.

Relevant upstream mlx-swift changes for context:
- `2755373 Fix wired mem race condition (#358)`
- `5bd59d0 Fix race condition in clearCache causing Metal crash (#331)` — NB: `Memory.clearCache()` now takes `evalLock`, which means it is safe to call from the prefill path as long as we're not holding the lock. The prefill loop is outside any eval-lock scope, so this is fine.
- No issues on `ml-explore/mlx-swift` related to buffer-pool over-retention (searched "memory", "leak", "cache buffer pool" — nothing hits). This bug is invisible to most users because clearCache masks it in any benchmark that triggers the 256-token decode path.

---

## 4. Proposed fix

### Option A — Minimal, mirrors Python exactly (recommended)

Add `MLX.Memory.clearCache()` at the end of each iteration of every prefill chunk loop in `mlx-swift-lm`. Four sites, four one-liners. Blast radius: only the prefill path; zero impact on decode; matches Python pattern one-to-one.

**Diff 1 — `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift`** (default `LLMModel.prepare` used by most models):
```diff
     public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
         -> PrepareResult
     {
         let prefillStepSize = windowSize ?? 512
         var y = input.text

         // Prepare the prompt in chunks if larger than the prefill size
         while y.tokens.size > prefillStepSize {
             let input = y[.newAxis, ..<prefillStepSize]
             _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
             eval(cache)
             y = y[prefillStepSize...]
+            // Match mlx-lm Python: clear GPU buffer pool between prefill chunks to
+            // prevent intermediate tensors from accumulating in the pool
+            // (mlx_lm/generate.py:450).
+            MLX.Memory.clearCache()
         }

         return .tokens(y)
     }
```

**Diff 2 — `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`** (around line 696):
```diff
         while y.tokens.size > prefillStepSize {
             let input = y[.newAxis, ..<prefillStepSize]
             _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
             eval(cache)
             y = y[prefillStepSize...]
+            MLX.Memory.clearCache()
         }
```

**Diff 3 — `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift`** (around line 829): identical one-line insertion after `y = y[prefillStepSize...]`.

**Diff 4 — `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift`** (around line 442): identical one-line insertion after `y = y[prefillStepSize...]`.

Verify the imports — `MLXLLM/LLMModel.swift` already `import MLX`, so `MLX.Memory.clearCache()` resolves. The three model files already import `MLX` at the top; confirm before PR.

**Note on the `eval(cache)` ordering**: Python evaluates the cache state (`mx.eval([c.state for c in cache])`) *before* `mx.clear_cache()`. The Swift code does `eval(cache)` already, which is the equivalent barrier — any buffers still alive through `cache.state` will survive the clear. This is what we want: KV cache survives; intermediates don't.

### Option B — Centralize in `Evaluate.swift` prepare path

Rather than touching every `prepare` implementation, gate on the return of `model.prepare` inside `TokenIterator.prepare`. But this misses the fact that the clear must happen *inside* the loop, not after, to be effective. Rejected.

### Option C — Flip allocator defaults (upstream mlx-swift change)

Make Swift default `cacheLimit` to something sane (e.g. a few hundred MB), or call `Memory.cacheLimit = 256 * 1024 * 1024` on library init. Risk: changes performance characteristics for users doing tight fixed-size loops where buffer reuse is cheap. Also blast radius is huge — every mlx-swift app would change behavior. **Rejected as the primary fix**, but worth considering as a belt-and-suspenders addition in `Libraries/MLXLMCommon/Evaluate.swift` for LLM inference specifically (since it's known intermediates-dominant).

### Option D — Fix upstream mlx-swift allocator C++

Change `max_pool_size_ = block_limit_` to something smaller, or add an auto-shrink heuristic on `free()`. This is the most invasive and would have to be re-upstreamed to ml-explore. **Do not pursue** — the allocator is shared with Python and Python users rely on the current behavior.

### Recommended plan
1. Ship **Option A** as a small PR against Eric's `ek/speed-improvements-2` (or `tom/turboquant-fixes`, depending on where Tom is iterating).
2. Optionally add a comment in `Evaluate.swift:1408-1412` noting the symmetric prefill clear location.
3. Consider filing an upstream issue on `ml-explore/mlx-swift` (and a PR on `mlx-swift-examples` / this repo since it owns the prefill loop) to get the same pattern into the canonical prepare path.

---

## 5. Validation plan

### Reproducing Eric's numbers

Eric's original test exercises a ~4 k prefill followed by some decode. The knob that matters is `prefillStepSize` (default 512), which forces at least 7 chunks for a 4 k prompt.

**Before-fix expected:**
- Pre-generation: ~2512 MB active, ~0 MB cache (similar to Eric's numbers depending on model)
- Post-generation: ~2528 MB active, ~5000+ MB cache, ~5700 MB peak
- After explicit `Memory.clearCache()`: ~2512 MB active

**After-fix expected:**
- Pre-generation: same
- Post-generation: ~2528 MB active, **~200-500 MB cache** (closer to the 210 MB Python sees, with some variance for the small number of live intermediates around the active decode step)
- Peak: ~3000-3500 MB instead of ~5700 MB
- `Memory.clearCache()` at the end should drop cache to near-zero (same as before).

### Concrete commands

1. Build Eric's benchmark with the fix:
   ```bash
   cd /Users/tom/dev/mlx-swift-lm
   swift build -c release
   ```

2. Run the existing benchmark harness with a 4 k prompt — use whatever the Eric test harness is (`benchmarks/` directory). If there's a standalone run command, something like:
   ```bash
   swift run -c release benchmark \
       --model mlx-community/Qwen2.5-3B-Instruct-4bit \
       --prompt-length 4096 \
       --max-tokens 512 \
       --memory-snapshot
   ```
   (Eric likely has a script; confirm path under `benchmarks/`.)

3. Instrument a tiny test harness that prints `Memory.snapshot()` at:
   - After model load
   - After first `eval(cache)` in the prefill loop
   - After the final prefill chunk
   - Before decode start
   - After 1 decode token
   - After 256 decode tokens (first clearCache fires)
   - After final decode
   - After an explicit `Memory.clearCache()` at the end

4. Compare cache delta:
   - **Pass:** after fix, `cacheMemory` stays below ~500 MB throughout the prefill phase
   - **Fail:** if `cacheMemory` grows above 1 GB during prefill, there is another retention site (possibly inside `eval(cache)` or the model forward pass)

### Diff against Python reference

Run the same workload under mlx-lm Python with `mx.get_cache_memory()` tapped at identical points, and expect the Swift numbers to be within ~50 MB of Python at each snapshot. Eric's original Python numbers (2.63→2.84→2.83→2.67 GB) are the gold standard.

### Regression check

Decode throughput should be essentially unchanged — we are only adding 7-ish extra `clearCache()` calls during prefill (one per chunk). Each call is a mutex lock + `BufferCache::clear()` which is microsecond-scale. Verify `tok/s` delta is within noise on a pure-decode benchmark (no prefill).

---

## 6. Open questions

1. **Are there other prefill loops I haven't found?** I only searched the top-level `Libraries` directory in `mlx-swift-lm`. VLM prefill paths (`MLXVLM/Models/*.swift`) do **not** have chunked-prefill loops today based on grep — they invoke the text model's `prepare`, which routes back to the same `LLMModel.prepare` default. So Option A's four edits should cover everything. But if a downstream consumer implements a custom `prepare`, they'll need the same pattern.

2. **Does `MLX.Memory.clearCache()` serialize with in-flight async eval?** The implementation wraps the call in `evalLock.withLock`. If `eval(cache)` submitted work that hasn't drained by the time we hit `clearCache`, the lock will serialize us behind it. This is correct (no crash) but could add prefill latency. Python does `mx.eval([c.state for c in cache])` immediately before `mx.clear_cache()` which acts as a barrier; Swift does `eval(cache)` likewise, so the behavior should match. Worth instrumenting prefill tok/s before and after to be sure.

3. **Why does `max_pool_size_` default to `block_limit_` in the first place?** The upstream rationale is "never evict buffers unless we're running out of memory." For small-model / fixed-shape workloads that is optimal. For variable-shape prefill chunks it is a pessimization. An upstream patch could condition on workload, but that's Apple's call to make.

4. **Is Eric's `ek/speed-improvements-2` ever going to land the prefill clear?** It's only one commit ahead of main today and that commit is orthogonal. Someone needs to cut the PR (either to `mlx-swift-lm` main or to Eric's branch).

5. **Is there an interaction with the new `WiredMemoryManager` ticket system?** The `0055e08` commit sizes wired-memory budgets assuming the pool doesn't balloon; if the pool grows 5 GB past expected during prefill, the wired budget is wrong by that much. Fixing the prefill retention also tightens the wired-memory math for free. Worth calling out in the PR description.

6. **LM Studio comparison.** The background implies LM Studio (Python-based) outperforming Swift-based MLX apps is at least partly explained by this bug. Could not verify independently — no access to LM Studio internals. If true, fixing this should close a meaningful chunk of the gap.

---

## Appendix — files touched / referenced

**Fix targets:**
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift` (line 32, insert after line 33)
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (line 700, insert after)
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift` (line 833, insert after)
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift` (line 446, insert after)

**Reference (Python golden pattern):**
- `/Users/tom/dev/mlx-lm/mlx_lm/generate.py:450, 467, 578, 1154, 1162, 1763`

**Shared C++ allocator (do not touch):**
- `/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp` (vendored; same logic as `/Users/tom/dev/mlx/mlx/backend/metal/allocator.cpp`)

**Swift binding (for context):**
- `/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/MLX/Memory.swift:356-360` — `clearCache()`
- `/Users/tom/dev/mlx-swift-lm/.build/checkouts/mlx-swift/Source/MLX/MLXArray.swift:35-37` — `deinit → mlx_array_free`

**Python binding (for context):**
- `/Users/tom/dev/mlx/python/src/memory.cpp:117-124` — `clear_cache` → `mx::clear_cache`

**Eric's prior art:**
- `mlx-swift-lm` commit `cb8e2af` — decode-side 256-token clearCache (the complementary half of this fix)
- `/Users/tom/dev/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:1408-1412` — where the decode clear lives today
