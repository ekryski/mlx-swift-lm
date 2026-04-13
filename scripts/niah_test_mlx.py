#!/usr/bin/env python3
"""Needle-In-A-Haystack (NIAH) Benchmark for MLX models

Follows Kamradt (2023) methodology. Works with:
  - Direct Python mlx-lm inference (--mode direct)
  - MLXServer or any OpenAI-compatible server (--mode server)

Handles thinking models (strips <think> tags before scoring).
Uses 7-digit magic number needle for unambiguous extraction.

Usage:
  # Direct inference (no server needed)
  python3 scripts/niah_test_mlx.py --model ~/models/MiniMax-M2.7-ConfigI-MLX

  # Against running server
  python3 scripts/niah_test_mlx.py --mode server --port 8080

  # Custom contexts and depths
  python3 scripts/niah_test_mlx.py --model ~/models/Model --contexts 1024,4096,16384 --depths 10,50,90

  # Verbose (print full responses)
  python3 scripts/niah_test_mlx.py --model ~/models/Model -v
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

SEED = 42

# Diverse filler paragraphs (~50-60 tokens each) covering different topics.
FILLER_PARAGRAPHS = [
    "The observable universe spans roughly 93 billion light-years in diameter. "
    "Within this volume lie an estimated two trillion galaxies, each hosting "
    "hundreds of billions of stars. Our own Milky Way is a barred spiral galaxy "
    "approximately 100,000 light-years across.",

    "Roman engineers perfected the art of concrete construction over two thousand "
    "years ago. The Pantheon in Rome, completed around 125 AD, features an "
    "unreinforced concrete dome that remained the largest in the world for over "
    "a millennium. Roman concrete used volcanic ash as a key ingredient.",

    "Modern semiconductor fabrication is among the most complex manufacturing "
    "processes ever devised. A single microprocessor contains billions of "
    "transistors, each smaller than a virus, etched into silicon wafers using "
    "extreme ultraviolet lithography at a wavelength of 13.5 nanometers.",

    "The history of cartography reflects humanity's evolving understanding of "
    "the world. The oldest known maps were etched on clay tablets in Mesopotamia "
    "around 2300 BC depicting local land features. The Mercator projection "
    "introduced in 1569 preserved compass bearings but exaggerated polar regions.",

    "Fermentation is one of the oldest food preservation techniques known to "
    "humanity, predating written history by thousands of years. The process "
    "relies on microorganisms that convert sugars into alcohol, acids, or gases. "
    "Bread, beer, wine, yogurt, cheese, and miso all owe their existence to it.",

    "Human memory is not a single system but a collection of interrelated "
    "processes distributed across multiple brain regions. Short-term memory "
    "relies on the prefrontal cortex while long-term consolidation involves "
    "the hippocampus, a seahorse-shaped structure deep in the temporal lobe.",

    "Gothic cathedrals represent one of the most ambitious building programs in "
    "European history, spanning from the twelfth to the sixteenth century. The "
    "key innovation was the pointed arch which distributes weight more efficiently "
    "and allows for taller, thinner walls with enormous stained-glass windows.",

    "Pollination is essential for the reproduction of roughly 87 percent of "
    "flowering plant species and underpins an estimated 35 percent of global "
    "food production. A single honeybee colony can visit millions of flowers "
    "per day. Colony collapse disorder drew global attention in 2006.",

    "The Silk Road was not a single route but a network of interconnected trade "
    "paths linking East Asia to the Mediterranean. At its peak the network "
    "spanned over 6,400 kilometers from Chang'an in China to Constantinople, "
    "passing through deserts, mountain ranges, and steppes.",

    "The discovery of antibiotics transformed medicine in the twentieth century. "
    "Alexander Fleming noticed in 1928 that mold contamination killed bacteria "
    "in a petri dish, leading to the development of penicillin. Before antibiotics "
    "even minor infections could prove fatal.",

    "Plate tectonics explains the large-scale motion of Earth's lithosphere. "
    "The theory emerged in the 1960s unifying continental drift and seafloor "
    "spreading. Earthquakes and volcanic eruptions concentrate along plate "
    "boundaries where plates collide, separate, or slide past each other.",

    "The invention of the printing press by Johannes Gutenberg around 1440 "
    "revolutionized the spread of information. Before movable type, books were "
    "copied by hand at enormous cost. Within fifty years of Gutenberg's invention "
    "an estimated twenty million volumes had been printed across Europe.",
]

NEEDLE_KEY = "The special magic number is"
QUESTION = (
    "What is the special magic number mentioned in the above text? "
    "Reply with ONLY the number, nothing else."
)


def build_haystack(target_tokens, tokenizer=None):
    """Build a haystack of approximately target_tokens using diverse paragraphs."""
    paragraphs = []
    while True:
        for p in FILLER_PARAGRAPHS:
            paragraphs.append(p)
        text = "\n\n".join(paragraphs)
        if tokenizer:
            toks = len(tokenizer.encode(text))
        else:
            toks = len(text) // 4  # rough estimate
        if toks >= target_tokens:
            break
    return paragraphs


def strip_thinking(text):
    """Remove <think>...</think> blocks from response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def score_response(response, expected_value):
    """Extract first 7-digit number from response, exact match."""
    clean = strip_thinking(response)
    match = re.search(r"\b(\d{7})\b", clean)
    return match is not None and match.group(1) == expected_value


# ---------------------------------------------------------------------------
# Direct inference mode
# ---------------------------------------------------------------------------

def query_direct(model, tokenizer, user_content, max_tokens=500):
    """Generate via mlx-lm directly with chat template."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, sampler=make_sampler(temp=0)
    )
    return response


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------

def query_server(port, user_content, max_tokens=500):
    """Send chat completion request to server."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"].get("content", "")
        return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_niah(args):
    contexts = [int(c) for c in args.contexts.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    model = tokenizer = None
    if args.mode == "direct":
        from mlx_lm import load
        print(f"Loading {args.model}...")
        model, tokenizer = load(args.model)

    print(f"\n{'=' * 70}")
    print(f"NIAH: {'direct' if args.mode == 'direct' else f'server :{args.port}'}")
    if args.model:
        print(f"Model: {args.model}")
    print(f"Contexts: {contexts}")
    print(f"Depths: {depths}")
    print(f"{'=' * 70}\n")
    print(f"{'Tokens':>8} {'Depth':>6} {'Result':>6} {'Time':>7}  Expected -> Answer")
    print("-" * 70)

    results = []
    for ctx in contexts:
        random.seed(SEED + ctx)
        for depth in depths:
            needle_value = str(random.randint(1000000, 9999999))
            needle_sentence = f"{NEEDLE_KEY}: {needle_value}."

            paragraphs = build_haystack(ctx, tokenizer)
            insert_idx = max(0, int(len(paragraphs) * depth / 100) - 1)
            paragraphs.insert(insert_idx, needle_sentence)
            full_text = "\n\n".join(paragraphs)

            user_content = f"{full_text}\n\n{QUESTION}"

            if tokenizer:
                toks = len(tokenizer.encode(user_content))
            else:
                toks = ctx  # estimate

            try:
                t0 = time.time()
                if args.mode == "direct":
                    response = query_direct(model, tokenizer, user_content)
                else:
                    response = query_server(args.port, user_content)
                elapsed = time.time() - t0

                found = score_response(response, needle_value)
                status = "HIT" if found else "MISS"

                clean = strip_thinking(response)
                display = clean[:50].replace("\n", " ")
                print(f"{toks:>8} {depth:>5}% {status:>6} {elapsed:>6.1f}s  {needle_value} -> {display}")

                if args.verbose and not found:
                    print(f"         FULL: {clean[:200]}")

                results.append({"ctx": toks, "depth": depth, "found": found})
            except Exception as e:
                print(f"{toks:>8} {depth:>5}%  ERROR         {str(e)[:40]}")
                results.append({"ctx": toks, "depth": depth, "found": False})

    total = len(results)
    passed = sum(1 for r in results if r["found"])
    print(f"\nPassed: {passed}/{total} ({passed / total * 100:.0f}%)" if total else "")

    # Save results
    if args.output:
        output = {
            "model": args.model or f"server:{args.port}",
            "mode": args.mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "contexts": contexts,
            "depths": depths,
            "pass_rate": passed / total if total else 0,
            "results": results,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIAH benchmark for MLX models")
    parser.add_argument("--model", type=str, help="Model path (required for direct mode)")
    parser.add_argument("--mode", choices=["direct", "server"], default="direct")
    parser.add_argument("--port", type=int, default=8080, help="Server port (server mode)")
    parser.add_argument("--contexts", default="1024,2048,4096,8192",
                        help="Comma-separated context lengths")
    parser.add_argument("--depths", default="10,25,50,75,90",
                        help="Comma-separated depth percentages")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full responses on miss")
    args = parser.parse_args()

    if args.mode == "direct" and not args.model:
        parser.error("--model required for direct mode")

    run_niah(args)
