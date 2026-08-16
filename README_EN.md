# OpsAgent-RL

> Train an LLM agent to **think like an SRE** — diagnose, fix, and verify system failures via reinforcement learning.

## What is this?

OpsAgent-RL trains a language model agent (Qwen3.5-9B) to autonomously troubleshoot Linux system failures inside Docker sandboxes. The agent issues shell commands, observes outputs, forms hypotheses, narrows down the root cause, applies a fix, and verifies — just like a human SRE engineer.

We compare three RL algorithms (**GRPO / DAPO / PPO**) on this task, built on top of [verl](https://github.com/volcengine/verl).

## Key Features

- **48 hand-crafted fault scenarios** across 5 categories: service failures, misconfigurations, resource exhaustion, network issues, security incidents
- **Parameterized expansion** to ~340 task variants, stratified into **200 train / 30 val / 100 test**
- **Docker sandbox**: pre-warmed container pool, `init=tini` for zombie reaping, `NET_ADMIN` for network scenarios, runtime network disabled for safety
- **Multi-level reward**: task completion (L1) + partials + diagnostic quality (L2) + efficiency (L3) + methodology (L4)
- **OpsBench**: a 100-scenario benchmark with SR / DA / steps / pass@k metrics
- **GRPO group-relative advantage** naturally fits "multiple valid fix paths for one fault"

## Tech Stack

`Qwen3.5-9B` · `verl (GRPO/DAPO/PPO)` · `vLLM` · `Docker` · `Ray` · `RCAEval` (seed data)

## Quick Start

```bash
# 1. install (core deps; train deps optional)
pip install -e ".[dev]"            # add [train] for torch/vllm/verl

# 2. build the sandbox image (network needed at build time; container is offline at runtime)
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .

# 3. generate the dataset (48 templates -> 342 variants -> 200/30/100)
bash scripts/generate_data.sh

# 4. validate the full harness end-to-end (no model, no Docker)
PYTHONPATH=src python3 -m train --smoke-test --mock --smoke-limit 10

# 5. train (GPU + verl required)
bash scripts/train_grpo.sh        # or train_dapo.sh / train_ppo.sh

# 6. evaluate
bash scripts/evaluate.sh checkpoints/grpo
```

## Sandbox

- Base image: `python:3.11-slim` + nginx / apache2 / redis / mariadb / sqlite / cron / sshd / iptables
- No systemd: scenarios start daemons via `svc <name>` (image helper)
- Runtime network disabled; all hosts are `127.0.0.1`
- Limits: 0.5 CPU, 256MB RAM; pool of 4 pre-warmed containers
- `NET_ADMIN` cap for iptables/route scenarios; `init=tini` reaps zombies

## Reward (4 levels)

| Level | What | Score | How |
|-------|------|-------|-----|
| L1 completion | fault fixed | 0 / +10 | run `verification.criteria`; all pass => success_reward |
| partials | intermediate state | +stack | each passing `partial_rewards` check |
| L2 diagnostic | root cause found | +0..+5 | root_cause_keywords coverage in agent cmds/outputs |
| L3 efficiency | #commands | -0.1*steps | cap -2.0 |
| L4 methodology | reasoning | +0..+2 | LLM-as-judge (optional, off by default) |

GRPO samples N=8 rollouts per prompt and computes group-relative advantage.

## OpsBench Metrics

Success Rate · Diagnostic Accuracy · Mean Steps to Resolution · Command Efficiency · pass@k, plus per-category breakdown.

## Hardware

Designed for **2× NVIDIA A30 (24 GB)** (4× also fine, more parallelism). Full pipeline (SFT + 3 RL algos + eval) takes ~4–6 days.

## Status

The data pipeline (templates / generator / split), environment, reward engine, evaluation, training entrypoint and tests are **implemented and verified** (8 unit/integration tests PASS, 342 tasks generated and split 200/30/100, end-to-end mock smoke run passes). Real RL training requires a **network-enabled GPU host** to build the sandbox image and install `.[train]` deps; SFT trajectory generation needs a teacher model via `OPSAGENT_TEACHER_API`.
