"""Convert OpsAgent-RL jsonl splits into verl RLHFDataset parquet format.

verl expects columns (see verl/utils/dataset/rl_dataset.py:386):
  - prompt        : list[dict] chat messages (system + user scenario). No tokenization.
  - data_source   : str (selects reward; we compute reward in-loop, so just a label)
  - reward_model  : dict with "ground_truth"
  - extra_info    : dict; our OpsAgentLoop reads extra_info["task"] to rebuild the Task

Our data/*.jsonl records already carry prompt + the full task spec
(setup_commands, inject_fault, verification_criteria, root_cause_keywords,
reward_spec, ...). We pass the whole record through extra_info["task"].

Usage:
    python -m verl_integration.convert_data          # writes data/{train,val}.parquet
    python -m verl_integration.convert_data --in data/train.jsonl --out data/train.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _records(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _to_verl_row(rec: dict) -> dict:
    return {
        "prompt": rec["prompt"],
        "data_source": "opsagent",
        "reward_model": {"ground_truth": rec.get("task_id", "")},
        "extra_info": {"task": rec},
    }


def convert(in_path: Path, out_path: Path) -> int:
    recs = _records(in_path)
    rows = [_to_verl_row(r) for r in recs]
    # Store nested dicts as JSON strings so parquet/arrow can hold arbitrary depth
    # robustly; RLHFDataset reads via pyarrow and json.loads them back if needed.
    table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    print(f"[convert] {in_path} -> {out_path}  ({len(rows)} rows)")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Convert OpsAgent-RL jsonl -> verl parquet.")
    ap.add_argument("--in", dest="inp", default=None, help="input jsonl (default: auto both splits)")
    ap.add_argument("--out", dest="out", default=None, help="output parquet")
    args = ap.parse_args()

    if args.inp:
        convert(Path(args.inp), Path(args.out or args.inp.replace(".jsonl", ".parquet")))
        return

    base = Path(__file__).resolve().parents[2] / "data"
    for split in ("train", "val"):
        inp = base / f"{split}.jsonl"
        outp = base / f"{split}.parquet"
        if inp.exists():
            convert(inp, outp)


if __name__ == "__main__":
    main()
