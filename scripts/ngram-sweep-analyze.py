#!/usr/bin/env python3
"""
Parse the [BENCH] log produced by `--method ngram-sweep` runs and reduce it
to a per-(category, cell) summary. The bench writes prefill/gen/TTFT to its
markdown file; this script extracts those plus computes speedup vs the
per-prompt baseline.

Usage:
  scripts/ngram-sweep-analyze.py LOGFILE [LOGFILE ...]

Outputs:
  - stdout: a markdown report with the best cell per category and the
    overall best cell across categories.
  - <logfile>.csv: per-row data (one row per benchmark cell + prompt).
"""

import re
import sys
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


# Each results block follows this shape, all from [BENCH] lines:
#   === RESULTS: <model> [<quant>] — ngram-sweep [<cat>/<name>] (baseline|n=N D=D H=H) [<kv>] ===
#   Method: ngram-sweep
#   Context: <ctx> tokens, Prompt Tokens: <P> (after template)
#   Prefill: <X> tok/s
#   Generation: <Y> tok/s (<N> tokens)
#   TTFT: <Z>ms
#   Total: <T>s
LABEL_RE = re.compile(
    r"=== RESULTS: (?P<model>[^[]+) \[(?P<quant>[^\]]+)\] — ngram-sweep "
    r"\[(?P<category>[^/]+)/(?P<name>[^\]]+)\] "
    r"(?P<config>baseline|n=\d+ D=\d+ H=\d+) "
    r"\[(?P<kv>[^\]]+)\] ===",
)
PREFILL_RE = re.compile(r"Prefill: ([\d.]+) tok/s")
GEN_RE = re.compile(r"Generation: ([\d.]+) tok/s")
TTFT_RE = re.compile(r"TTFT: ([\d.]+)ms")
TOTAL_RE = re.compile(r"Total: ([\d.]+)s")
PROMPT_TOKENS_RE = re.compile(r"Prompt Tokens: (\d+)")


def parse_log(path: Path):
    """Yield dicts, one per results block."""
    rows = []
    current = None
    with path.open() as f:
        for line in f:
            m = LABEL_RE.search(line)
            if m:
                if current is not None:
                    rows.append(current)
                current = {
                    "model": m["model"].strip(),
                    "quant": m["quant"],
                    "category": m["category"],
                    "name": m["name"],
                    "kv": m["kv"],
                }
                if m["config"] == "baseline":
                    current["n"], current["d"], current["h"] = 0, 0, 0
                else:
                    nm = re.match(r"n=(\d+) D=(\d+) H=(\d+)", m["config"])
                    current["n"] = int(nm[1])
                    current["d"] = int(nm[2])
                    current["h"] = int(nm[3])
                continue
            if current is None:
                continue
            for key, regex, cast in [
                ("prefill", PREFILL_RE, float),
                ("gen", GEN_RE, float),
                ("ttft_ms", TTFT_RE, float),
                ("total_s", TOTAL_RE, float),
                ("prompt_tokens", PROMPT_TOKENS_RE, int),
            ]:
                if key not in current:
                    mm = regex.search(line)
                    if mm:
                        current[key] = cast(mm[1])
            # Once all five fields are populated, lock in this row.
            if all(k in current for k in
                   ("prefill", "gen", "ttft_ms", "total_s", "prompt_tokens")):
                rows.append(current)
                current = None
    if current is not None:
        rows.append(current)
    return rows


def attach_speedup(rows):
    """For each non-baseline row, compute speedup vs the matching baseline."""
    baselines = {}
    for r in rows:
        if r["n"] == 0:
            baselines[(r["model"], r["category"], r["name"])] = r
    for r in rows:
        b = baselines.get((r["model"], r["category"], r["name"]))
        if b and "gen" in b and "gen" in r:
            r["speedup"] = r["gen"] / b["gen"]
            r["ttft_ratio"] = r.get("ttft_ms", 0) / b.get("ttft_ms", 1) if b.get("ttft_ms") else None
        else:
            r["speedup"] = None
            r["ttft_ratio"] = None
    return rows


def write_csv(rows, path):
    cols = ["model", "quant", "kv", "category", "name", "n", "d", "h",
            "prompt_tokens", "prefill", "gen", "ttft_ms", "total_s",
            "speedup", "ttft_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def report(rows):
    """Print a markdown report."""
    cells = [r for r in rows if r["n"] > 0 and r.get("speedup") is not None]
    if not cells:
        print("No sweep cells parsed.")
        return

    by_category = defaultdict(list)
    for r in cells:
        by_category[r["category"]].append(r)

    by_cell = defaultdict(list)
    for r in cells:
        by_cell[(r["n"], r["d"], r["h"])].append(r)

    print("# N-gram speculative decoding sweep — Phase B results\n")

    # Per-category summary.
    print("## Per-category best cells (median speedup across that category's prompts)\n")
    print("| Category | Best cell (n, D, H) | median speedup | mean gen tok/s | accept count |")
    print("|---|---|---:|---:|---:|")
    for category, crows in sorted(by_category.items()):
        cell_med = defaultdict(list)
        for r in crows:
            cell_med[(r["n"], r["d"], r["h"])].append(r["speedup"])
        if not cell_med:
            continue
        best_cell = max(cell_med, key=lambda c: median(cell_med[c]))
        med_speedup = median(cell_med[best_cell])
        sub = [r for r in crows if (r["n"], r["d"], r["h"]) == best_cell]
        mean_gen = mean(r["gen"] for r in sub)
        print(f"| {category} | n={best_cell[0]} D={best_cell[1]} H={best_cell[2]} "
              f"| {med_speedup:.3f}× | {mean_gen:.1f} | {len(sub)} |")
    print()

    # Overall — geometric-mean style: median speedup ACROSS ALL prompts per cell.
    print("## Overall best cells (median speedup across all prompts)\n")
    print("| Rank | Cell (n, D, H) | median speedup | mean speedup | min speedup | wins/total |")
    print("|---:|---|---:|---:|---:|---:|")
    cell_scores = []
    for cell, crows in by_cell.items():
        speedups = [r["speedup"] for r in crows]
        wins = sum(1 for s in speedups if s > 1.0)
        cell_scores.append({
            "cell": cell,
            "median": median(speedups),
            "mean": mean(speedups),
            "min": min(speedups),
            "max": max(speedups),
            "wins": wins,
            "total": len(speedups),
        })
    cell_scores.sort(key=lambda x: x["median"], reverse=True)
    for i, s in enumerate(cell_scores[:10], 1):
        c = s["cell"]
        print(f"| {i} | n={c[0]} D={c[1]} H={c[2]} | {s['median']:.3f}× | "
              f"{s['mean']:.3f}× | {s['min']:.3f}× | {s['wins']}/{s['total']} |")
    print()

    # Worst cells (so we can document what NOT to use).
    print("## Worst cells — these LOSE on average\n")
    print("| Cell (n, D, H) | median speedup | min speedup | wins/total |")
    print("|---|---:|---:|---:|")
    for s in cell_scores[-5:]:
        c = s["cell"]
        print(f"| n={c[0]} D={c[1]} H={c[2]} | {s['median']:.3f}× | "
              f"{s['min']:.3f}× | {s['wins']}/{s['total']} |")
    print()

    # Sanity: how many cells actually beat baseline?
    winning_cells = [s for s in cell_scores if s["median"] > 1.0]
    print(f"**{len(winning_cells)} of {len(cell_scores)} cells beat baseline at "
          f"median**. Best median speedup: **{cell_scores[0]['median']:.3f}×**, "
          f"worst median: **{cell_scores[-1]['median']:.3f}×**.\n")

    # Per-category × cell heatmap.
    print("## Per-category × cell speedup heatmap (median)\n")
    cells_sorted = sorted(by_cell.keys())
    cats_sorted = sorted(by_category.keys())
    print("| Category \\\\ Cell | " + " | ".join(
        f"n={c[0]}D{c[1]}H{c[2]}" for c in cells_sorted) + " |")
    print("|---|" + ":---:|" * len(cells_sorted))
    for cat in cats_sorted:
        row = [f"**{cat}**"]
        for cell in cells_sorted:
            xs = [r["speedup"] for r in by_category[cat]
                  if (r["n"], r["d"], r["h"]) == cell]
            row.append(f"{median(xs):.2f}×" if xs else "—")
        print("| " + " | ".join(row) + " |")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    all_rows = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        rows = parse_log(path)
        rows = attach_speedup(rows)
        write_csv(rows, path.with_suffix(".csv"))
        print(f"# Parsed {len(rows)} rows from {path} → {path.with_suffix('.csv')}\n",
              file=sys.stderr)
        all_rows.extend(rows)
    report(all_rows)


if __name__ == "__main__":
    main()
