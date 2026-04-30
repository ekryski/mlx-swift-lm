#!/usr/bin/env python3
"""
spec-decode-compare.py — A vs B compare of two `spec-decode-sweep.sh` runs.

Reads two log files in the parser-friendly format produced by
`scripts/spec-decode-sweep.sh`, computes per-cell medians + speedup-ratios,
and prints a side-by-side table.

USAGE:
    spec-decode-compare.py BASELINE.log TREATMENT.log [--csv]

The two logs must use the same (model, prompt, cell-set) — only the
trial-by-trial gen tok/s numbers should differ. Speedup ratios are
computed cell-by-cell as TREATMENT_median / BASELINE_median.

EXAMPLE:
    # Generate baseline + treatment logs
    scripts/spec-decode-sweep.sh \\
        --model gemma4-26b-a4b --quant 4bit \\
        --prompts Tests/Benchmarks/Resources/ngram-sweep-prompts/recipe-bulk \\
        --cells "TI@0:0:0:,NGgreedy@0:3:0:" \\
        --trials 5 --output /tmp/baseline.log

    scripts/spec-decode-sweep.sh \\
        --model gemma4-26b-a4b --quant 4bit \\
        --prompts Tests/Benchmarks/Resources/ngram-sweep-prompts/recipe-bulk \\
        --cells "TI@0:0:0:,NGtreatment@0:3:0:MY_KNOB=1" \\
        --trials 5 --output /tmp/treatment.log

    # Compare
    scripts/spec-decode-compare.py /tmp/baseline.log /tmp/treatment.log

EXIT CODE: 0 always (the script doesn't make pass/fail decisions; that's
the human reviewer's job).
"""

import argparse
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(r"MODEL: (\S+)\s+PROMPT: (.+)")
# Cell line accepts both the new `extra=...` field (from spec-decode-sweep.sh)
# and the older `leviathan=...` field (from /tmp/ ad-hoc sweeps before
# spec-decode-sweep.sh was committed). Either is matched as group(4).
CELL_RE = re.compile(r"--- Cell (\S+): ngram=(\d+) temp=(\S+) (?:extra|leviathan)=(.+) ---")
TRIAL_RE = re.compile(r"\s+trial \S+: gen=(\S+) tok/s accept=(\S+)")


def parse_log(path: Path):
    """Return cells: dict[(model, prompt_label, cell_label) -> (median, accept)]."""
    cells = {}
    section = None
    cell = None
    trials = []

    def commit():
        nonlocal cell, trials
        if cell is not None and section is not None:
            gens = [g for g, _ in trials]
            accepts = [a for _, a in trials if a != "—"]
            med = sorted(gens)[len(gens) // 2] if gens else None
            cells[(section[0], section[1], cell)] = (med, accepts[0] if accepts else "—")
        cell = None
        trials = []

    for line in path.read_text().splitlines():
        m = SECTION_RE.search(line)
        if m:
            commit()
            section = (m.group(1), m.group(2))
            continue
        m = CELL_RE.match(line)
        if m:
            commit()
            cell = m.group(1)
            continue
        m = TRIAL_RE.match(line)
        if m:
            try:
                trials.append((float(m.group(1)), m.group(2)))
            except ValueError:
                pass  # skip malformed trial lines
    commit()
    return cells


def fmt_speedup(ratio: float) -> str:
    """Speedup string with directional emoji."""
    if ratio is None:
        return "—"
    if ratio >= 1.05:
        return f"{ratio:.2f}× ✅"
    if ratio <= 0.95:
        return f"{ratio:.2f}× ⚠️"
    return f"{ratio:.2f}× ≈"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("baseline", type=Path, help="path to baseline log")
    ap.add_argument("treatment", type=Path, help="path to treatment log")
    ap.add_argument("--csv", action="store_true", help="output CSV instead of table")
    args = ap.parse_args()

    base = parse_log(args.baseline)
    treat = parse_log(args.treatment)

    keys = sorted(set(base) | set(treat))
    if args.csv:
        print("model,prompt,cell,baseline_tok_s,treatment_tok_s,speedup,baseline_accept,treatment_accept")
    else:
        print(f"{'model':<16}{'prompt':<30}{'cell':<14}{'baseline':<12}{'treatment':<12}{'speedup':<14}{'accept (b/t)':<20}")
        print("-" * 120)

    for key in keys:
        model, prompt, cell = key
        b_med, b_acc = base.get(key, (None, "—"))
        t_med, t_acc = treat.get(key, (None, "—"))
        speedup = (t_med / b_med) if (b_med and t_med and b_med > 0) else None

        if args.csv:
            print(f"{model},{prompt},{cell},"
                  f"{'' if b_med is None else f'{b_med:.2f}'},"
                  f"{'' if t_med is None else f'{t_med:.2f}'},"
                  f"{'' if speedup is None else f'{speedup:.4f}'},"
                  f"{b_acc},{t_acc}")
        else:
            b_str = f"{b_med:.1f}" if b_med is not None else "—"
            t_str = f"{t_med:.1f}" if t_med is not None else "—"
            sp_str = fmt_speedup(speedup) if speedup is not None else "—"
            acc_str = f"{b_acc} / {t_acc}"
            # Truncate long prompt names for terminal display.
            prompt_disp = prompt if len(prompt) <= 28 else prompt[:25] + "..."
            print(f"{model:<16}{prompt_disp:<30}{cell:<14}{b_str:<12}{t_str:<12}{sp_str:<14}{acc_str:<20}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
