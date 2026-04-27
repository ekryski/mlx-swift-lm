# DeepSeek-V4 — Upstream Attribution

The Swift implementation of DeepSeek-V4 in this repository
(`DeepseekV4.swift`, `DeepseekV4Compressor.swift`,
`DeepseekV4Configuration.swift`, `DeepseekV4MathHelpers.swift`) is
adapted from the open-source port at:

    https://github.com/osaurus-ai/vmlx-swift-lm

published by the Osaurus AI team under the MIT license. The MIT
copyright notices on each Swift file remain on the originals; this
fork's modifications are layered on top under the same license.

## What was adapted

- The full `DeepseekV4Model` forward pass, including:
  - mHC (manifold hyper-connections) collapse / expand around each
    decoder block.
  - MLA attention with `head_dim=512`, `num_kv_heads=1`, partial RoPE
    on the last 64 dims, learned per-head attention sinks, inverse RoPE
    on the attention output, and the grouped low-rank `wo_a / wo_b`
    output projection.
  - Compressor + Indexer modules for the long-context pooled-attention
    path.
  - sqrtsoftplus MoE routing with hash routing for the first three
    layers via a `tid2eid` table, plus the shared expert.
  - DSV4 SwiGLU with `swiglu_limit=10` and YaRN RoPE on
    `compress_ratio>0` layers.
  - Per-layer `DeepseekV4Cache` composing `RotatingKVCache` with the
    Compressor's pooled buffer.

## What was changed

- File-level copyright headers updated to credit both the upstream
  Osaurus authors (MIT) and this fork's modifications.
- Comments referring to internal-to-osaurus research documents (e.g.
  `jang/research/...`, `EXHAUSTIVE-VARIABLES-GUIDE.md`,
  `JANGTQ-PROGRESS-LOG`, internal bug numbers) were rewritten to be
  self-contained.
- Dead helper `_osaurusSanitizeUnused` was renamed to
  `_bundleFormatSanitize` and documented as future-use rather than
  legacy.
- Factory registration added in `LLMModelFactory.swift` (single line).

## Phase 1 status

This port is **Phase 1**: the model loads from
`mlx-community/DeepSeek-V4-Flash-2bit-DQ` and produces coherent text
output on smoke tests, but the math is not yet bug-for-bug equivalent
to the Python reference. Known follow-ups (tracked for Phase 2):

1. Validate HC sinkhorn iteration count and fp32 boundaries against
   Python.
2. Apply the `swiglu_limit` clamp on the up-projection inside
   `SwitchGLU` (currently only the gate-side silu is clamped; the
   2-bit DQ codebook's natural boundedness regularises the
   up-projection well enough for smoke tests, but mismatches Python).
3. Verify Compressor + Indexer numerics for long contexts (`L >
   sliding_window=128`). Short prompts use the bypass fast-path which
   is exercised on every smoke test.
4. Check inverse partial RoPE sign convention against the upstream
   complex-conjugate path.
5. `attn_sink` dtype + casting boundary verification.
6. `q_norm` per-head fp32 rescale ordering vs the variance-only
   normalization in Python.
7. YaRN ramp / correction-range formula edge cases (low/high clamps).
8. Hash-routing weight normalization (currently uses uniform
   `1/topK`; Python does the same but worth a sanity check).
9. APE positional bias inside the Compressor — verify the broadcast
   axes match Python.
10. Overlap transform fill values for `compress_ratio=4` — Python
    uses `-inf` for the gate side (matched here) but we should
    confirm the kv side fill of `0.0` is correct, not `nan`.
11. Per-layer compress_ratio fallback table when `compress_ratios`
    is absent from `config.json` — currently uses the DSV4-Flash
    default; verify this matches official bundles.
12. Routed-expert weight stacking under `switch_mlp.*` — verify the
    `tq_packed` / `tq_norms` path against TurboQuant routed bundles.
13. MoE down-projection `bf16 → fp16` cast safety — confirmed for
    DSV3, needs spot-check for DSV4's wider hidden dim.

## License

Upstream MIT license (verbatim) preserved below.

```
MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

(Note: the LICENSE file shipped with osaurus-ai/vmlx-swift-lm carries
the standard mlx-explore MIT text — same copyright holder as the
upstream project both repos descend from.)
