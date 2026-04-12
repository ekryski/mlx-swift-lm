# Kernel Fusion Roadmap

**Date**: 2026-04-11
**Status**: All 14 custom kernels migrated to C framework dispatch. Zero JIT in hot paths.

## Current State: Ops Per Decode Token

| Model | Layers | Current Dispatches | Theoretical Min |
|-------|--------|:---:|:---:|
| Gemma4 E2B | 30 | ~420 | ~120 |
| GPT-OSS-20B | 24 | ~480 | ~200 |
| Qwen3.5-35B-A3B | 40 | ~1020 | ~800 |

## Already Fused (Completed)

| Fusion | Saves/layer | Kernel |
|--------|:---:|--------|
| RMSNorm + RoPE → 1 dispatch | 3 | `rms_norm_rope.metal` (C framework) |
| RMSNorm + Residual Add → 1 dispatch | 1 | `rms_norm_residual.metal` (C framework) |
| GEGLU (gelu×mul) → 1 dispatch | 2 | `compiledGeglu` (MLX graph JIT, fusable) |
| GatedDelta fused (norm+gate+beta+state) → 1 dispatch | 4-6 | `gated_delta_step_fused` (C framework) |
| TurboFlash (score+softmax+value) → 2 dispatches | entire attn | `turbo_flash_p1/p2` (C framework) |

## Top Fusion Opportunities (Priority Order)

### 1. Batched Q+K+V GEMV (HIGH — saves 60-90 dispatches)
Fuse 3 separate Q/K/V Linear projections into 1 batched dispatch.
- Gemma4: 2 saves × 30 layers = 60 dispatches
- GPT-OSS: 2 saves × 24 layers = 48 dispatches
- Requires `quantizedBatchedGEMV` Metal kernel
- Note: Gemma4 already has `compiledQKVProjection` that partially fuses via MLX graph

### 2. Full MLP Fusion (HIGH — saves 120-150 dispatches)
Fuse norm+gate+up+gelu×mul+down+norm+residual into 1 mega-kernel per layer.
- Current: 6 dispatches (preNorm, gate, up, gelu×mul, down, postNorm+residual)
- Target: 1 dispatch
- Gemma4: 5 saves × 30 = 150 dispatches
- Biggest single optimization remaining
- Challenge: Custom Metal kernel with multiple weight matrix reads

### 3. O Projection + Norm + Residual (MEDIUM — saves 30 dispatches)
Fuse output projection GEMV with post-attention rmsNormResidual.
- Extends existing `rms_norm_residual.metal` to absorb GEMV
- Or new `gemv_norm_residual.metal` kernel

### 4. MoE Router Pipeline (MEDIUM — saves 20-40 dispatches for MoE models)
Fuse router → TopK → softmax → expert index selection.
- Current: 4 dispatches per MoE layer
- Target: 1-2 dispatches

## Remaining JIT (Non-Hot Path)

| Kernel | File | Path | Action |
|--------|------|------|--------|
| FusedGateActivation | SwitchLayers.swift | Disabled by default | Keep as-is |
| Bitnet matmul | Bitnet.swift | Model-specific | Port to .metal when needed |
| Interpolation | InterpolationUtils.swift | Image preprocessing | Keep as-is |

## Framework Kernels Status

| Kernel | .metal | C++ Primitive | C Bridge | Swift Binding | Caller |
|--------|:---:|:---:|:---:|:---:|:---:|
| rms_norm_residual | Yes | Yes | Yes | Yes | **Active** |
| turbo_score | Yes | Yes | Yes | Yes | **Active** |
| turbo_fused_encode | Yes | Yes | Yes | Yes | **Active** |
| turbo_fused_encode_wht | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p1 | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p1_causal | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p1_nr0 | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p1_nr0_causal | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p2 | Yes | Yes | Yes | Yes | **Active** |
| turbo_flash_p2_fused_rot | Yes | Yes | Yes | Yes | **Active** |
| turbo_value | Yes | Yes | Yes | Yes | **Active** |
| gated_delta_step | Yes | Yes | Yes | Yes | **Active** |
| gated_delta_step_fused | Yes | Yes | Yes | Yes | **Active** |
| ssm_step | Yes | Yes | Yes | Yes | **Active** |
