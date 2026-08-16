"""Dataset utilities: load JSONL task/prompt files and stratified splitting."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_split(
    records: list[dict[str, Any]],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 42,
    key: str = "category",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test preserving the per-category proportion.

    Falls back to a plain random split if `key` is absent.
    """
    rng = random.Random(seed)
    total_target = train_n + val_n + test_n
    n = len(records)
    if n == 0:
        return [], [], []

    by_group: dict[str, list[dict]] = defaultdict(list)
    has_key = bool(records and key in records[0])
    for r in records:
        by_group[str(r.get(key, "all")) if has_key else "all"].append(r)

    for g in by_group:
        rng.shuffle(by_group[g])

    train, val, test = [], [], []
    # allocate proportionally per group
    for g, items in by_group.items():
        g_frac = len(items) / n
        g_test = max(1, round(total_target * g_frac * (test_n / total_target)))
        g_val = max(1, round(total_target * g_frac * (val_n / total_target)))
        g_test = min(g_test, len(items))
        g_val = min(g_val, len(items) - g_test)
        g_train = len(items) - g_val - g_test
        test.extend(items[:g_test])
        val.extend(items[g_test : g_test + g_val])
        train.extend(items[g_test + g_val : g_test + g_val + g_train])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # Trim / top up to exact targets (best-effort).
    def _trim(lst, k):
        return lst[:k] if len(lst) >= k else lst

    train = _trim(train, train_n)
    val = _trim(val, val_n)
    test = _trim(test, test_n)

    # top-up: if any split is short, borrow from the largest other split
    def _topup(target_lst, target_n, source_lst):
        need = target_n - len(target_lst)
        if need > 0 and source_lst:
            target_lst.extend(source_lst[:need])
            del source_lst[:need]
        return target_lst

    if len(test) < test_n:
        test = _topup(test, test_n, train if len(train) > len(val) else val)
    if len(val) < val_n:
        val = _topup(val, val_n, train)
    if len(train) < train_n:
        # can't create more than available; keep what we have
        pass
    return train, val, test
