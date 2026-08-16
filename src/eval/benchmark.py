"""OpsBench evaluation runner (planning doc Section 7).

Loads the test split, runs the agent against the Docker sandbox for each
scenario, scores with the multi-level reward engine, and aggregates metrics.

Usage:
    # rule-based sanity check (no model needed, exercises the full harness):
    PYTHONPATH=src python3 -m eval.benchmark --config configs/eval/opsbench.yaml \
        --test-file data/test.jsonl --rule-based --limit 5

    # against a vLLM server:
    python3 -m eval.benchmark --test-file data/test.jsonl \
        --server-url http://localhost:8000/v1 --model-name Qwen3.5-9B

    # against a local HF checkpoint:
    python3 -m eval.benchmark --test-file data/test.jsonl --model-path checkpoints/grpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.dataset import load_jsonl
from data.sft_generator import task_from_record
from agent.policy import AgentPolicy
from agent.prompts import SYSTEM_PROMPT
from env.docker_env import DockerShellEnv
from reward.reward_model import RewardEngine
import eval.metrics as metrics
from model_backend import build_backend


def main():
    ap = argparse.ArgumentParser(description="Run OpsBench evaluation.")
    ap.add_argument("--config", default="configs/eval/opsbench.yaml")
    ap.add_argument("--test-file", default=None, help="override test file from config")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--server-url", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rule-based", action="store_true", help="no-model harness sanity check")
    ap.add_argument("--limit", type=int, default=0, help="limit #scenarios (0=all)")
    ap.add_argument("--pass-k", type=int, nargs="*", default=[1, 2, 5])
    ap.add_argument("--out", default="data/eval_report.json")
    ap.add_argument("--image", default="opsagent-sandbox:latest")
    args = ap.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
    test_file = args.test_file or cfg.get("eval", {}).get("test_file", "data/test.jsonl")

    records = load_jsonl(test_file)
    if args.limit:
        records = records[: args.limit]
    print(f"[eval] loaded {len(records)} test scenarios from {test_file}")

    env = DockerShellEnv(
        image=args.image,
        max_steps=cfg.get("env", {}).get("max_steps_per_episode", 20),
        command_timeout=cfg.get("env", {}).get("command_timeout", 15),
        pool_size=cfg.get("env", {}).get("parallel_envs", 4),
    )
    env._init_pool()
    engine = RewardEngine()

    results: list[dict] = []
    try:
        for i, rec in enumerate(records):
            task = task_from_record(rec)
            try:
                # rule-based backend is stateful per-task; rebuild each time
                backend = build_backend(args, task) if args.rule_based else (
                    build_backend(args, task) if i == 0 else backend  # reuse HTTP/HF backend
                )
                policy = AgentPolicy(backend.generate, max_turns=task.max_steps)
                summary = policy.run_episode(env, task, SYSTEM_PROMPT)
                breakdown = engine.compute(env, task, summary["trajectory"], steps=summary["steps"])
                results.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "difficulty": task.difficulty,
                        "steps": summary["steps"],
                        "truncated": summary.get("truncated", False),
                        "trajectory": summary["trajectory"],
                        "reward": breakdown.as_dict(),
                    }
                )
            except Exception as e:  # pragma: no cover
                results.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "difficulty": task.difficulty,
                        "steps": 0,
                        "truncated": False,
                        "trajectory": [],
                        "reward": {"total": 0.0, "success": False, "error": str(e)},
                    }
                )
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                sr = metrics.success_rate(results)
                print(f"[eval] {i+1}/{len(records)} done | running SR={sr:.2%}")
    finally:
        env.shutdown()

    report = metrics.aggregate(results, pass_k=args.pass_k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "summary": report, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[eval] report written to {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
