"""Parameterized dataset generator for OpsAgent-RL.

Expands the 48 hand-crafted scenario templates into concrete tasks, augments the
port dimension to reach the planned ~300+ variants, stratified-splits into
train/val/test, and writes verl-compatible JSONL prompt records.

Usage:
    python3 -m src.data.generator --templates src/env/scenarios \
        --out data --train 200 --val 30 --test 100 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from env.task_loader import TaskLoader, ScenarioTemplate
from agent.prompts import SYSTEM_PROMPT, build_user_message
from data.dataset import write_jsonl, stratified_split

# Extended port dimension used to reach the planned ~300+ variants.
DEFAULT_PORTS = [8080, 9090, 3000, 8888, 5000, 8000, 7000]


def _augment_port_dimension(templates: list[ScenarioTemplate], ports: list[int]) -> None:
    """For templates that already parameterize on `port`, extend the value list."""
    for tpl in templates:
        if "port" in tpl.params:
            tpl.params["port"] = list(ports)


def task_to_record(task) -> dict:
    """Serialize a Task to a verl-compatible training prompt record."""
    pd = task.to_prompt_dict()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(task)},
        ],
        "task_id": task.task_id,
        "category": task.category,
        "difficulty": task.difficulty,
        "description": task.description,
        "expected_verification": pd["expected_verification"],
        "root_cause_keywords": pd["root_cause_keywords"],
        "max_steps": task.max_steps,
        "setup_commands": task.setup_commands,
        "inject_fault": task.inject_fault,
        "verification_criteria": [
            {"command": c.command, "expected": c.expected, "contains": c.contains, "exit_zero": c.exit_zero}
            for c in task.verification.criteria
        ],
        "reward_spec": {
            "success_reward": task.reward_spec.success_reward,
            "partial_rewards": [
                {"condition": p.condition, "reward": p.reward, "check": p.check, "contains": p.contains, "expected": p.expected}
                for p in task.reward_spec.partial_rewards
            ],
            "step_penalty": task.reward_spec.step_penalty,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Generate the OpsAgent-RL dataset.")
    ap.add_argument("--templates", default="src/env/scenarios")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ports", nargs="*", type=int, default=DEFAULT_PORTS)
    args = ap.parse_args()

    loader = TaskLoader(args.templates)
    templates = loader.load_templates()
    print(f"[generator] loaded {len(templates)} templates")

    _augment_port_dimension(templates, args.ports)

    # Expand each (possibly augmented) template.
    tasks = []
    for tpl in templates:
        tasks.extend(tpl.expand())
    print(f"[generator] expanded to {len(tasks)} concrete tasks")
    from collections import Counter
    print(f"[generator] by category: {dict(Counter(t.category for t in tasks))}")

    records = [task_to_record(t) for t in tasks]

    train, val, test = stratified_split(records, args.train, args.val, args.test, seed=args.seed)
    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test, out / "test.jsonl")
    print(f"[generator] wrote train={len(train)} val={len(val)} test={len(test)} to {out}/")
    print(f"[generator]   {out/'train.jsonl'}")
    print(f"[generator]   {out/'val.jsonl'}")
    print(f"[generator]   {out/'test.jsonl'}")


if __name__ == "__main__":
    main()
