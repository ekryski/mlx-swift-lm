# Spec 003 — QKV projection fusion via `batchedQKVQuantizedGEMV`

**Status:** draft
**Target:** `Qwen35Attention`, `Qwen3NextAttention`, `Gemma4Attention`, and every dense attention block that calls `q_proj` / `k_proj` / `v_proj` as three separate Linears on the same (possibly pre-normed) input.
**Owner:** TBD
**Expected gain:** +3–5% decode tok/s. Smaller than the MLP fusion because attention is KV-read-bound at longer contexts, not matmul-dispatch-bound. Biggest win at short contexts where the attention scan is cheap.
**Depends on:** nothing. Orthogonal to specs 001/002 (MLP-side fusions). Stackable.

## Motivation

Every decoder attention call dispatches three quantized matmuls on the same hidden state:

```swift
// Qwen3.5 attention, today
let qProjOut = qProj(x)    // [..., 2 * nHeads * headDim]  (Qwen's q split)
let keys    = kProj(x)     // [..., kvHeads * headDim]
let values  = vProj(x)     // [..., kvHeads * headDim]
```

Three dispatches on the same input vector at decode. On a 32-layer model at ~50 µs/dispatch, two saved dispatches per layer is **3.2 ms/token**. Current decode on Qwen3.5-4B is ~12.5 ms/token post-001 → expected **+3–5%**.

`MLXFast.batchedQKVQuantizedGEMV` is the primitive we want. It exists in mlx-swift and takes a list of quantized weights of the same input dim (possibly different output dims) and issues a single kernel. We consume its output as a concatenated tensor and split by head counts.

## Non-goals

- Does not touch attention math itself (SDPA, RoPE, GQA routing) — only the projection-op fusion.
- Does not merge q/k/v *weights* on disk. The kernel accepts three separate quantized weight tensors as inputs. Weight layout is unchanged.
- Does not cover the output projection `o_proj` — that's a single Linear already.
- Not a prefill optimization. Like the other GEMV fusions, this path is decode-only.

## Design

### New helper: `fusedQKVProjection`

Add next to `AttentionUtils.swift` (or a new `FusedQKVProjection.swift`):

```swift
/// Apply `qProj`, `kProj`, `vProj` to the same input as a single
/// `batchedQKVQuantizedGEMV` kernel when guards are met. Returns three
/// arrays shaped as the individual projection outputs, so callers can keep
/// their existing reshape / normalize / RoPE path unchanged.
///
/// Guards (decode-only fast path):
/// - x.dim(-2) == 1  (GEMV, not GEMM)
/// - x.dtype in { bf16, fp16 }
/// - all three projections are quantized Linear with matching bits/groupSize
/// - no bias on any of the three (can extend later if needed)
public func fusedQKVProjection(
    _ x: MLXArray,
    qProj: Linear, kProj: Linear, vProj: Linear
) -> (q: MLXArray, k: MLXArray, v: MLXArray)
```

If guards fail, fall back to the three separate calls. The fallback is trivial, and callers are isolated from the decision.

### Attention block wiring

Replace:
```swift
let q = qProj(x)
let k = kProj(x)
let v = vProj(x)
```
with:
```swift
let (q, k, v) = fusedQKVProjection(x, qProj: qProj, kProj: kProj, vProj: vProj)
```

Three callers to update initially:
- `Qwen35Attention.callAsFunction` (also `Qwen3NextAttention`).
- `Gemma4Attention.callAsFunction` (note: Gemma 4 already uses `MLXFast.rmsNormQuantizedGEMV` via `fusedNormProj` — this spec composes with that by switching Gemma 4 to a combined `fusedNormQKV` that applies the shared norm *once* and runs the batched QKV. Design decision: either (a) keep `fusedNormProj` × 3 if `batchedQKVQuantizedGEMV` has no norm-integrated variant, or (b) add `fusedNormQKVQuantizedGEMV` upstream. Measure (a) first — the norm is already a single Metal dispatch, so saving two projection dispatches may be enough.)

Qwen3.5's `q_proj` is a double-wide projection (Qwen's interleaved Q + gate). The batched kernel handles different output dims per projection, so this is a non-issue — just pass `2 * nHeads * headDim` for the q-slot.

### Fallback & guard observability

Match spec 002's debug knob: `MLX_DEBUG_FUSED_QKV=1` prints the first rejection reason per call site. Catch silent fallbacks early — if every Gemma 4 call hits the fallback because of a dtype mismatch, we want to see that in bench logs, not discover it when decode numbers don't move.

## Acceptance criteria

1. All dense Qwen3.5 / Qwen3.6 / Gemma 4 models load and produce coherent summarization output at ctx=1024.
2. Decode tok/s on Qwen3.5-4B (M1 Max, 4bit, no-quant KV, ctx=1024): **≥ 83 tok/s** (baseline post-001 ≈ 80). Measure independently of spec 002.
3. Gemma 4 31B decode ≥ 15.0 tok/s at ctx=1024 (baseline post-001 ≈ 14.7).
4. Qwen3.6-27B decode ≥ 19.0 tok/s at ctx=1024 (baseline post-001 ≈ 18.1).
5. KLD vs bf16 baseline unchanged within measurement noise. (The batched kernel produces the same arithmetic result as three separate GEMVs when guards are met.)
6. Fallback path is exercised by at least one test (e.g. fp32 input) to confirm the non-fast path still works.

## Measurement plan

```bash
for m in qwen35-4b qwen35-9b qwen35-27b qwen36-27b gemma4-31b gemma4-e4b; do
    scripts/benchmark.sh --model $m --method summarization \
        --quant 4bit --kv none --context 1024,4096,8192
done
```

Report before/after deltas for ctx=1024 decode. Note anything outside acceptance ranges, especially if Gemma 4 regresses because the pre-norm broadcast changes (see design note above).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `batchedQKVQuantizedGEMV` is stricter about stride alignment than three separate GEMVs | Keep a fast-path guard and fall back. Log first fallback per call site. |
| Gemma 4 regresses because the existing `fusedNormProj` × 3 is actually faster than 1 norm + 1 batched QKV when norm is already fused | Measure Gemma 4 both ways. Gate behind per-model flag until we have data. |
| Mixed-bit quantized variants (e.g. Gemma 4 26B A4B mixed 4/8 per-layer) fail the "matching bits" guard | Guard falls back cleanly; this is expected and not a regression. Track how often it kicks in. |
| KLD drift from different reduction order inside the batched kernel | Gate on `--kld` equivalence test. If detected, keep a `MLX_FUSED_QKV=0` escape hatch. |

## Out of scope / follow-ups

- **`fusedNormQKVQuantizedGEMV`** (norm + batched QKV): would need a new kernel in mlx. File upstream if measurements show it matters.
- **`o_proj` and MLP `down_proj` fusions**: those are single matmuls; no obvious fusion partner unless we stack post-SDPA routing.
- **Non-quantized (bf16) path**: the primitive is quantized-only. bf16 models keep three separate `matmul` calls. Negligible decode relevance since prod weights are 4bit.

## Open questions

1. Does Qwen3Next's q-projection "2 × nHeads × headDim" layout work as a first-slot input to the batched kernel? Quick repro before committing to the helper signature.
2. Gemma 4 integration: one call-site takes a pre-normed input (`fusedNormProj` path), others don't. Do we want two helper shapes (`fusedQKV` and `fusedNormQKV`) or a single helper with an optional `preNorm` parameter?
