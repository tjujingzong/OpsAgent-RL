"""Explore the generated dataset: per-split/category/difficulty stats and samples."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from data.dataset import load_jsonl


def describe(path: str) -> None:
    recs = load_jsonl(path)
    if not recs:
        print(f"{path}: (empty or missing)")
        return
    by_cat = Counter(r["category"] for r in recs)
    by_diff = Counter(r["difficulty"] for r in recs)
    by_cat_diff = defaultdict(int)
    for r in recs:
        by_cat_diff[(r["category"], r["difficulty"])] += 1
    print(f"\n== {path} ==")
    print(f"  total: {len(recs)}")
    print(f"  by category: {dict(by_cat)}")
    print(f"  by difficulty: {dict(by_diff)}")
    print("  category x difficulty:")
    cats = sorted(by_cat)
    diffs = sorted(by_diff)
    print("    " + " ".join(f"{d:>8}" for d in diffs))
    for c in cats:
        print(f"    {c:18s}" + " ".join(f"{by_cat_diff[(c,d)]:>8}" for d in diffs))
    s = recs[0]
    print(f"  sample task_id: {s['task_id']}")
    print(f"  sample desc: {s['description'][:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default=None, help="train|val|test (default: all)")
    args = ap.parse_args()
    splits = [args.split] if args.split else ["train", "val", "test"]
    for s in splits:
        describe(str(Path(args.data_dir) / f"{s}.jsonl"))


if __name__ == "__main__":
    main()
