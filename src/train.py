"""Training entrypoint for OpsAgent-RL.

Real RL training is delegated to verl (GRPO/DAPO/PPO). This module provides:
  * `build_reward_fn()`  - a verl-compatible reward function wired to the
    Docker sandbox + multi-level reward engine.
  * `--smoke-test` mode  - runs a few end-to-end episodes through the env,
    agent policy and reward engine WITHOUT a model (rule-based backend), to
    validate the whole harness before spending GPU time.

Launch real training (after `pip install -e .[train]`) with:
    bash scripts/train_grpo.sh
    bash scripts/train_dapo.sh
    bash scripts/train_ppo.sh
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
from env.mock_env import MockShellEnv
from reward.reward_model import RewardEngine, group_relative_advantage
from model_backend import RuleBasedBackend


def build_reward_fn(image: str = "opsagent-sandbox:latest"):
    """Return a verl-compatible reward callable.

    Signature: reward_fn(prompt, response_or_trajectory, task_meta) -> float
    The verl agentic worker passes the final multi-turn trajectory; we look up
    the task by task_id, run verification in a fresh sandbox, and score it.
    """
    engine = RewardEngine()
    # A short-lived env per call keeps reward computation isolated. verl workers
    # should pool these; here we create one per call for simplicity.
    def reward_fn(task_meta: dict, trajectory: list[dict], steps: int) -> float:
        task = task_from_record(task_meta) if isinstance(task_meta, dict) else task_meta
        env = DockerShellEnv(image=image)
        env.reset(task)
        try:
            # Re-apply the agent's fix by replaying its commands, then verify.
            for step in trajectory:
                cmd = step.get("action")
                if cmd:
                    env.step(cmd)
            breakdown = engine.compute(env, task, trajectory, steps=steps)
            return breakdown.total
        finally:
            env.close()

    return reward_fn


def smoke_test(test_file: str, limit: int = 5, image: str = "opsagent-sandbox:latest",
               use_mock: bool = False) -> None:
    """Run a few episodes with the rule-based backend to validate the harness."""
    records = load_jsonl(test_file)[:limit]
    if not records:
        print(f"[smoke] no records in {test_file}; run the generator first.")
        return
    env = MockShellEnv() if use_mock else DockerShellEnv(image=image)
    if not use_mock:
        env._init_pool()
    engine = RewardEngine()
    try:
        for rec in records:
            task = task_from_record(rec)
            backend = RuleBasedBackend(task)
            policy = AgentPolicy(backend.generate, max_turns=task.max_steps)
            summary = policy.run_episode(env, task, SYSTEM_PROMPT)
            bd = engine.compute(env, task, summary["trajectory"], steps=summary["steps"])
            print(
                f"[smoke] {task.task_id:40s} cat={task.category:20s} "
                f"steps={summary['steps']:2d} reward={bd.total:7.2f} success={bd.success}"
            )
    finally:
        if use_mock:
            env.close()
        else:
            env.shutdown()


def main():
    ap = argparse.ArgumentParser(description="OpsAgent-RL training launcher.")
    ap.add_argument("--config", default="configs/train/grpo.yaml")
    ap.add_argument("--smoke-test", action="store_true", help="validate env+policy+reward without a model")
    ap.add_argument("--mock", action="store_true", help="use MockShellEnv (no Docker) for smoke test")
    ap.add_argument("--smoke-file", default="data/test.jsonl")
    ap.add_argument("--smoke-limit", type=int, default=5)
    ap.add_argument("--image", default="opsagent-sandbox:latest")
    args = ap.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
    algo = cfg.get("algorithm", {}).get("type", "grpo")
    print(f"[train] algorithm={algo} config={args.config}")

    if args.smoke_test:
        smoke_test(args.smoke_file, args.smoke_limit, image=args.image, use_mock=args.mock)
        return

    # Real training: delegate to verl if installed.
    try:
        import verl  # noqa: F401
        from verl.trainer.main_ppo import main as verl_main  # type: ignore
    except ImportError:
        print(
            "[train] verl not installed. To run real RL training:\n"
            "  1. pip install -e .[train]   (torch / vllm / verl / ray)\n"
            "  2. ensure data/train.jsonl & data/val.jsonl exist (bash scripts/generate_data.sh)\n"
            "  3. bash scripts/train_grpo.sh   (or train_dapo.sh / train_ppo.sh)\n"
            "The reward function for verl is available via opsagent.train.build_reward_fn().\n"
            "Run `python3 -m train --smoke-test` to validate the harness without a model."
        )
        return

    # If verl is present, hand off. (verl's own hydra entry reads the config.)
    print("[train] delegating to verl ...")
    verl_main()


if __name__ == "__main__":
    main()
