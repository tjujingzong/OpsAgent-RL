"""Analyze the reward distribution by running the rule-based expert over tasks.

Useful for sanity-checking that scenarios are solvable and rewards are sensible
before spending GPU on real rollouts.
"""
from __future__ import annotations

import argparse
import statistics

from data.dataset import load_jsonl
from data.sft_generator import task_from_record
from agent.policy import AgentPolicy
from agent.prompts import SYSTEM_PROMPT
from env.docker_env import DockerShellEnv
from env.mock_env import MockShellEnv
from reward.reward_model import RewardEngine
from model_backend import RuleBasedBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--image", default="opsagent-sandbox:latest")
    ap.add_argument("--mock", action="store_true", help="use MockShellEnv (no Docker)")
    args = ap.parse_args()

    recs = load_jsonl(args.file)[: args.limit]
    env = MockShellEnv() if args.mock else DockerShellEnv(image=args.image)
    if not args.mock:
        env._init_pool()
    engine = RewardEngine()
    rewards, successes = [], []
    try:
        for rec in recs:
            task = task_from_record(rec)
            backend = RuleBasedBackend(task)
            policy = AgentPolicy(backend.generate, max_turns=task.max_steps)
            summary = policy.run_episode(env, task, SYSTEM_PROMPT)
            bd = engine.compute(env, task, summary["trajectory"], steps=summary["steps"])
            rewards.append(bd.total)
            successes.append(bd.success)
            tag = "OK " if bd.success else "FAIL"
            print(f"[{tag}] {task.task_id:40s} r={bd.total:7.2f} l1={bd.l1_task:5.1f} "
                  f"part={bd.partial:4.1f} l2={bd.l2_diagnostic:4.1f} l3={bd.l3_efficiency:5.2f}")
    finally:
        if args.mock:
            env.close()
        else:
            env.shutdown()

    print("\n== reward summary ==")
    print(f"  n={len(rewards)}  success_rate={sum(successes)/max(1,len(successes)):.2%}")
    if rewards:
        print(f"  mean={statistics.mean(rewards):.2f}  stdev={statistics.stdev(rewards) if len(rewards)>1 else 0:.2f}")
        print(f"  min={min(rewards):.2f}  max={max(rewards):.2f}")


if __name__ == "__main__":
    main()
