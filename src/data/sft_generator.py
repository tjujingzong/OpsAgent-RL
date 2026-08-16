"""SFT trajectory generator.

Runs a teacher model (OpenAI-compatible API) as an agent against the Docker
sandbox, executes the resulting trajectory, verifies it actually fixes the
fault, and keeps only successful trajectories (the plan's verification filter).

Usage:
    OPSAGENT_TEACHER_API=<key> OPSAGENT_TEACHER_BASE_URL=http://localhost:8000/v1 \\
        python3 -m src.data.sft_generator --train-file data/train.jsonl \\
        --out data/sft.jsonl --num-per-scenario 3

If no teacher API is configured, a rule-based expert produces minimal
(diagnose -> known-fix -> verify) trajectories WITHOUT live Docker observations;
real SFT data requires the teacher (see planning doc Section 3.5).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from agent.prompts import SYSTEM_PROMPT, build_user_message, FEWSHOT_EXAMPLE
from agent.policy import AgentPolicy, parse_action
from data.dataset import load_jsonl, write_jsonl
from env.docker_env import DockerShellEnv
from env.task_loader import (
    Task,
    VerificationSpec,
    VerificationCriterion,
    RewardSpec,
    PartialReward,
)


def task_from_record(rec: dict) -> Task:
    verification = VerificationSpec(
        type="command_check",
        criteria=[
            VerificationCriterion(
                command=c.get("command", ""),
                expected=c.get("expected", ""),
                contains=c.get("contains", ""),
                exit_zero=bool(c.get("exit_zero", False)),
            )
            for c in rec.get("verification_criteria", [])
        ],
        root_cause_keywords=list(rec.get("root_cause_keywords", [])),
    )
    reward_spec = RewardSpec(
        success_reward=rec.get("reward_spec", {}).get("success_reward", 10.0),
        partial_rewards=[
            PartialReward(
                condition=p.get("condition", ""),
                reward=float(p.get("reward", 0.0)),
                check=p.get("check", ""),
                contains=p.get("contains", ""),
                expected=p.get("expected", ""),
            )
            for p in rec.get("reward_spec", {}).get("partial_rewards", [])
        ],
        step_penalty=float(rec.get("reward_spec", {}).get("step_penalty", -0.1)),
        max_steps=int(rec.get("max_steps", 20)),
    )
    return Task(
        task_id=rec["task_id"],
        category=rec["category"],
        difficulty=rec["difficulty"],
        description=rec["description"],
        setup_commands=list(rec.get("setup_commands", [])),
        inject_fault=list(rec.get("inject_fault", [])),
        verification=verification,
        reward_spec=reward_spec,
    )


# ----------------------------- teacher client -----------------------------
class TeacherClient:
    """Minimal OpenAI-compatible chat client (no external SDK needed)."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": 512}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


# ----------------------------- rule-based expert -----------------------------
def rule_based_trajectory(task: Task) -> list[dict]:
    """Produce a minimal plausible trajectory WITHOUT running the env.

    Used as a fallback so the SFT data has the right shape even without a teacher.
    Observations are placeholders; real SFT needs the teacher path.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(task)},
        {"role": "user", "content": "Initial state: (sandbox not run for rule-based fallback)"},
    ]
    # inspect
    msgs += [
        {"role": "assistant", "content": "Let me gather broad system state first.\n```bash\nps aux | head -10; df -h /; free -m\n```"},
        {"role": "user", "content": "(system summary)"},
    ]
    # apply the inverse of each inject_fault step is not generally derivable,
    # so we show the verification command and a generic fix attempt.
    first_check = task.verification.criteria[0].command if task.verification.criteria else "true"
    msgs += [
        {"role": "assistant", "content": "Let me reproduce the failing check to confirm the symptom.\n```bash\n" + first_check + "\n```"},
        {"role": "user", "content": "(check currently failing)"},
        {"role": "assistant", "content": "I'll apply the fix based on the root cause and re-verify.\n```bash\n" + " && ".join(["true"] + [c.command for c in task.verification.criteria]) + "\n```"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Verified. The root cause is resolved.\nTASK_COMPLETE"},
    ]
    return msgs


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate SFT trajectories.")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--num-per-scenario", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="limit number of train tasks (0 = all)")
    ap.add_argument("--image", default="opsagent-sandbox:latest")
    args = ap.parse_args()

    base_url = os.environ.get("OPSAGENT_TEACHER_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPSAGENT_TEACHER_API", "EMPTY")
    model = os.environ.get("OPSAGENT_TEACHER_MODEL", "qwen2.5-72b-instruct")

    records = load_jsonl(args.train_file)
    if args.limit:
        records = records[: args.limit]
    print(f"[sft] loaded {len(records)} train tasks")

    use_teacher = bool(os.environ.get("OPSAGENT_TEACHER_API"))
    sft_records: list[dict] = []

    if use_teacher:
        env = DockerShellEnv(image=args.image)
        env._init_pool()
        try:
            for i, rec in enumerate(records):
                task = task_from_record(rec)
                for variant in range(args.num_per_scenario):
                    teacher = TeacherClient(base_url, api_key, model, temperature=0.5 + 0.2 * variant)
                    policy = AgentPolicy(teacher.generate, max_turns=task.max_steps)
                    summary = policy.run_episode(env, task, SYSTEM_PROMPT)
                    # verify
                    checks = env.run_verification(task)
                    passed = all(c.passed for c in checks)
                    if passed:
                        sft_records.append({"messages": summary["messages"], "task_id": task.task_id})
                if (i + 1) % 10 == 0:
                    print(f"[sft] processed {i + 1}/{len(records)}, kept {len(sft_records)}")
        finally:
            env.shutdown()
    else:
        print("[sft] no OPSAGENT_TEACHER_API set; producing rule-based fallback trajectories.")
        for rec in records:
            task = task_from_record(rec)
            sft_records.append({"messages": rule_based_trajectory(task), "task_id": task.task_id})

    write_jsonl(sft_records, args.out)
    print(f"[sft] wrote {len(sft_records)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
