"""Visualize OpsBench results: comparison tables across baselines/algorithms.

Reads one or more eval_report.json files and prints a side-by-side table.
Optionally renders bar charts with matplotlib (if installed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def print_table(reports: list[tuple[str, dict]]) -> None:
    cols = ["n_scenarios", "success_rate", "diagnostic_accuracy", "mean_steps_to_resolution", "command_efficiency", "pass@1"]
    header = f"{'model':28s} " + " ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, rep in reports:
        s = rep.get("summary", rep)
        row = f"{name:28s} "
        for c in cols:
            v = s.get(c, "")
            row += f"{v:>14}" if isinstance(v, str) else f"{v:>14.4f}" if isinstance(v, float) else f"{v:>14}"
        print(row)
    # per-category success rate
    print("\nper-category success rate:")
    cats = sorted({c for _, rep in reports for c in rep.get("summary", rep).get("by_category", {})})
    print(f"  {'model':28s} " + " ".join(f"{c[:12]:>12s}" for c in cats))
    for name, rep in reports:
        bc = rep.get("summary", rep).get("by_category", {})
        print(f"  {name:28s} " + " ".join(f"{bc.get(c,{}).get('success_rate',0):>12.2%}" for c in cats))


def plot(reports: list[tuple[str, dict]], out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping chart.")
        return
    names = [n for n, _ in reports]
    srs = [rep.get("summary", rep).get("success_rate", 0) for _, rep in reports]
    das = [rep.get("summary", rep).get("diagnostic_accuracy", 0) for _, rep in reports]
    x = range(len(names))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], srs, width=0.4, label="Success Rate")
    plt.bar([i + 0.2 for i in x], das, width=0.4, label="Diagnostic Accuracy")
    plt.xticks(list(x), names, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("score")
    plt.title("OpsBench: Success Rate & Diagnostic Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"[viz] chart saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True, help="eval_report.json files")
    ap.add_argument("--names", nargs="*", default=None, help="display names (default: filenames)")
    ap.add_argument("--chart", default=None, help="output png (optional)")
    args = ap.parse_args()
    names = args.names or [Path(p).stem for p in args.reports]
    reports = [(n, _load(p)) for n, p in zip(names, args.reports)]
    print_table(reports)
    if args.chart:
        plot(reports, args.chart)


if __name__ == "__main__":
    main()
