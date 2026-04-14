#!/usr/bin/env python3
"""Fix Gemma 4 MLX community quants that ship without chat_template.

MLX community quantizations of Gemma 4 (E2B, E4B, 26B-A4B, 31B) are missing
the chat_template field in tokenizer_config.json. Without it, the model echoes
input or produces garbage instead of following instructions.

Usage:
    python3 scripts/fix-gemma4-template.py ~/models/gemma-4-e2b-it-4bit
    python3 scripts/fix-gemma4-template.py ~/models/gemma-4-26b-a4b-4bit
"""
import json, sys, os

if len(sys.argv) < 2:
    print("Usage: python3 fix-gemma4-template.py <model-path>")
    sys.exit(1)

model_path = os.path.expanduser(sys.argv[1])
tc_path = os.path.join(model_path, "tokenizer_config.json")
jinja_path = os.path.join(model_path, "chat_template.jinja")

if not os.path.exists(tc_path):
    print(f"ERROR: {tc_path} not found")
    sys.exit(1)

tc = json.load(open(tc_path))

# Check if already has template
if "chat_template" in tc and len(tc["chat_template"]) > 100:
    print(f"OK: chat_template already present ({len(tc['chat_template'])} chars)")
    sys.exit(0)

# Download jinja from E2B repo (all Gemma 4 variants use the same template)
if not os.path.exists(jinja_path):
    print("Downloading chat_template.jinja from mlx-community/gemma-4-e2b-it-4bit...")
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("mlx-community/gemma-4-e2b-it-4bit", "chat_template.jinja")
    import shutil
    shutil.copy(src, jinja_path)
    print(f"  Saved: {jinja_path}")

template = open(jinja_path).read()
tc["chat_template"] = template
json.dump(tc, open(tc_path, "w"), indent=2)
print(f"FIXED: Injected chat_template ({len(template)} chars) into {tc_path}")
