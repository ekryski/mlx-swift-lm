#!/usr/bin/env python3
"""
Config-I Weight Compression for MLX

Converts HuggingFace models to MLX format with TurboQuant+ Config-I
mixed-precision quantization policy. Config-I protects attention and
boundary layers at higher precision while aggressively compressing
expert MLPs (MoE) or FFN layers (dense).

Usage:
  # MoE models (MiniMax, Qwen MoE, Mixtral, etc.)
  python3 convert.py --model MiniMaxAI/MiniMax-M2.7 --output ~/models/M2.7-ConfigI --type moe

  # Dense models (Qwen, Phi — NOT Llama, see below)
  python3 convert.py --model Qwen/Qwen2.5-72B --output ~/models/Qwen72B-ConfigI --type dense

  # Custom layer count and boundary size
  python3 convert.py --model MiniMaxAI/MiniMax-M2.7 --output ~/models/M2.7-ConfigI --type moe --layers 62 --boundary 2

  # Dry run (print policy without converting)
  python3 convert.py --model MiniMaxAI/MiniMax-M2.7 --type moe --dry-run

Supported model families:
  - MoE: MiniMax M2.7/M2.5, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Mixtral
  - Dense (Config I): Qwen (all sizes), Phi-3, Phi-4
  - Dense (Premium/Hybrid): Llama — Config I produces +17% PPL on Llama.
    Use --type llama for Llama-specific configs (Premium or Hybrid).

Config-I policy (MoE):
  - Expert MLP gate/up: 2-bit (bulk params, MoE-tolerant)
  - Expert MLP down: 3-bit (write-back sensitivity)
  - Attention Q/K/V/O: 4-bit uniform per layer
  - Boundary layers: 8-bit (first N + last N)
  - MoE router: f16 (never quantized)
  - Embeddings + lm_head: 8-bit

Config-I policy (Dense, Qwen/Phi):
  - Attention Q/K/V/O: TQ4_1S equivalent (4-bit, all same type per layer)
  - FFN gate/up: 4-bit
  - FFN down: native Q4_K equivalent (3-bit with higher group precision)
  - Boundary layers: 8-bit (first 2 + last 2)
  - Embeddings + lm_head: 8-bit

See: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md
"""

import argparse
import json
import sys
from mlx_lm.convert import convert


def make_moe_predicate(num_layers: int, boundary: int):
    """Config-I predicate for MoE models (MiniMax, Qwen MoE, Mixtral)."""
    def predicate(path: str, module) -> bool | dict:
        # Never quantize norms
        if "norm" in path or "layernorm" in path:
            return False

        # Embeddings and head: 8-bit
        if "embed_tokens" in path:
            return {"bits": 8, "group_size": 64}
        if "lm_head" in path:
            return {"bits": 8, "group_size": 64}

        # Router: never quantize (routing precision critical for MoE)
        # MiniMax: block_sparse_moe.gate, Qwen: mlp.gate, Mixtral: block_sparse_moe.gate
        gate_patterns = ["moe.gate", "mlp.gate"]
        if any(p in path for p in gate_patterns) and "gate_proj" not in path:
            return False

        # Parse layer index
        layer_idx = None
        parts = path.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None:
            return {"bits": 4, "group_size": 64}

        is_boundary = (layer_idx < boundary) or (layer_idx >= num_layers - boundary)

        if is_boundary:
            return {"bits": 8, "group_size": 64}

        # Attention: 4-bit uniform per layer
        if "self_attn" in path or "attention" in path:
            return {"bits": 4, "group_size": 64}

        # Expert MLPs: differentiate down (write-back) from gate/up
        if "down_proj" in path:
            return {"bits": 3, "group_size": 64}
        if "gate_proj" in path or "up_proj" in path:
            return {"bits": 2, "group_size": 64}

        # Fallback
        return {"bits": 4, "group_size": 64}

    return predicate


def make_dense_predicate(num_layers: int, boundary: int):
    """Config-I predicate for dense models (Qwen, Phi)."""
    def predicate(path: str, module) -> bool | dict:
        if "norm" in path or "layernorm" in path:
            return False

        if "embed_tokens" in path:
            return {"bits": 8, "group_size": 64}
        if "lm_head" in path:
            return {"bits": 8, "group_size": 64}

        layer_idx = None
        parts = path.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None:
            return {"bits": 4, "group_size": 64}

        is_boundary = (layer_idx < boundary) or (layer_idx >= num_layers - boundary)

        if is_boundary:
            return {"bits": 8, "group_size": 64}

        # Attention: 4-bit uniform per layer
        if "self_attn" in path or "attention" in path:
            return {"bits": 4, "group_size": 64}

        # FFN down: 3-bit (write-back protection)
        if "down_proj" in path:
            return {"bits": 3, "group_size": 64}

        # FFN gate/up: 4-bit
        if "gate_proj" in path or "up_proj" in path:
            return {"bits": 4, "group_size": 64}

        return {"bits": 4, "group_size": 64}

    return predicate


def make_llama_predicate(num_layers: int, boundary: int, profile: str = "premium"):
    """Config for Llama family (higher error amplification in FFN path).
    Premium: TQ4 attn + Q5K/Q6K FFN, boundary 4+4. +5.8% PPL, -29% size.
    Hybrid: TQ4 attn + Q4K FFN, boundary 2+2. +16% PPL, -42% size.
    """
    llama_boundary = 4 if profile == "premium" else boundary

    def predicate(path: str, module) -> bool | dict:
        if "norm" in path or "layernorm" in path:
            return False

        if "embed_tokens" in path:
            return {"bits": 8, "group_size": 64}
        if "lm_head" in path:
            return {"bits": 8, "group_size": 64}

        layer_idx = None
        parts = path.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None:
            return {"bits": 4, "group_size": 64}

        is_boundary = (layer_idx < llama_boundary) or (layer_idx >= num_layers - llama_boundary)

        if is_boundary:
            return {"bits": 8, "group_size": 64}

        if "self_attn" in path or "attention" in path:
            return {"bits": 4, "group_size": 64}

        if profile == "premium":
            # Q5_K equivalent for all FFN
            if "down_proj" in path or "gate_proj" in path or "up_proj" in path:
                return {"bits": 5, "group_size": 64}
        else:
            # Hybrid: Q4_K equivalent
            if "down_proj" in path or "gate_proj" in path or "up_proj" in path:
                return {"bits": 4, "group_size": 64}

        return {"bits": 4, "group_size": 64}

    return predicate


def main():
    parser = argparse.ArgumentParser(
        description="Config-I weight compression for MLX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniMax M2.7
  python3 convert.py --model MiniMaxAI/MiniMax-M2.7 --output ~/models/M2.7-ConfigI --type moe --layers 62

  # Qwen3.5-27B (dense)
  python3 convert.py --model Qwen/Qwen3.5-27B --output ~/models/Qwen27B-ConfigI --type dense --layers 64

  # Qwen3.5-35B MoE
  python3 convert.py --model Qwen/Qwen3.5-35B-A3B --output ~/models/Qwen35B-ConfigI --type moe --layers 40

  # Llama 70B (Premium config, boundary 4+4)
  python3 convert.py --model meta-llama/Llama-3.1-70B-Instruct --output ~/models/Llama70B-ConfigI --type llama --layers 80

  # Dry run
  python3 convert.py --model MiniMaxAI/MiniMax-M2.7 --type moe --layers 62 --dry-run

Supported families:
  moe     MiniMax, Qwen MoE, Mixtral (2-bit experts, 3-bit down, 4-bit attn)
  dense   Qwen, Phi (4-bit attn+gate/up, 3-bit down) — +1-4% PPL
  llama   Llama family (higher FFN sensitivity, uses Premium or Hybrid config)

Recommendations:
  - Source weights should be FP8, FP16, BF16, or Q8_0. Do NOT use Q4_K_M source
    (insufficient headroom, Config-I may increase size).
  - head_dim >= 128 required (head_dim=64 models like Qwen3-4B are unsupported).
  - Best results on Qwen and Phi families (+1-4% PPL). Llama has steeper quality
    tradeoff (+5-17% PPL depending on config).
  - MoE models benefit most: 98%+ of params are expert MLPs which tolerate
    aggressive 2-bit compression. Dense models compress 27-38%, MoE up to 62%.
  - Stacks with TurboQuant KV cache compression for additional memory savings.

Paper: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md
        """
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", default=None, help="Output directory (default: ./mlx_model)")
    parser.add_argument("--type", required=True, choices=["moe", "dense", "llama"],
                        help="Model type: moe (MiniMax, Qwen MoE), dense (Qwen, Phi), llama")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of transformer layers (auto-detected if not specified)")
    parser.add_argument("--boundary", type=int, default=2,
                        help="Boundary layer count (first N + last N protected at 8-bit, default: 2)")
    parser.add_argument("--llama-profile", choices=["premium", "hybrid"], default="premium",
                        help="Llama config profile (default: premium)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"],
                        help="Non-quantized parameter dtype (default: bfloat16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print policy summary without converting")
    parser.add_argument("--trust-remote-code", action="store_true", default=True,
                        help="Trust remote code when loading model (default: True)")

    args = parser.parse_args()

    output = args.output or "./mlx_model"
    num_layers = args.layers

    # Auto-detect layer count if not specified
    if num_layers is None:
        try:
            import json as _json
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(args.model, "config.json")
            with open(config_path) as f:
                config = _json.load(f)
            num_layers = config.get("num_hidden_layers", config.get("n_layer"))
            if num_layers:
                print(f"Auto-detected {num_layers} layers from config.json")
            else:
                print("ERROR: Could not auto-detect layer count. Use --layers N", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not auto-detect layers: {e}. Use --layers N", file=sys.stderr)
            sys.exit(1)

    # Build predicate
    if args.type == "moe":
        predicate = make_moe_predicate(num_layers, args.boundary)
        desc = f"MoE Config-I: 2-bit expert gate/up, 3-bit expert down, 4-bit attn, 8-bit boundary {args.boundary}+{args.boundary}"
    elif args.type == "dense":
        predicate = make_dense_predicate(num_layers, args.boundary)
        desc = f"Dense Config-I: 4-bit attn+gate/up, 3-bit down, 8-bit boundary {args.boundary}+{args.boundary}"
    else:
        predicate = make_llama_predicate(num_layers, args.boundary, args.llama_profile)
        bnd = 4 if args.llama_profile == "premium" else args.boundary
        ffn_bits = 5 if args.llama_profile == "premium" else 4
        desc = f"Llama {args.llama_profile}: 4-bit attn, {ffn_bits}-bit FFN, 8-bit boundary {bnd}+{bnd}"

    print(f"Model: {args.model}")
    print(f"Output: {output}")
    print(f"Layers: {num_layers}")
    print(f"Policy: {desc}")
    print(f"Router: f16 (never quantized)")
    print(f"Embeddings + lm_head: 8-bit")
    print()

    if args.dry_run:
        print("Dry run — no conversion performed.")
        return

    convert(
        hf_path=args.model,
        mlx_path=output,
        quantize=True,
        q_group_size=64,
        q_bits=4,
        dtype=args.dtype,
        quant_predicate=predicate,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"\nDone. Output at {output}")


if __name__ == "__main__":
    main()
