# OpsAgent-RL

> Train an LLM agent to **think like an SRE** — diagnose, fix, and verify system failures via reinforcement learning.

## What is this?

OpsAgent-RL trains a language model agent (Qwen3.5-9B) to autonomously troubleshoot Linux system failures inside Docker sandboxes. The agent issues shell commands, observes outputs, forms hypotheses, and iterates — just like a human SRE engineer.

We compare three RL algorithms (**GRPO / DAPO / PPO**) on this task, built on top of [verl](https://github.com/volcengine/verl).

## Key Features

- **48 hand-crafted fault scenarios** across 5 categories: service failures, misconfigurations, resource exhaustion, network issues, security incidents
- **Docker sandbox** with container pooling for safe, parallel, reproducible training
- **Multi-level reward**: task completion + diagnostic quality + efficiency + methodology
- **OpsBench**: a 100-scenario evaluation benchmark for system troubleshooting agents

## Tech Stack

`Qwen3.5-9B` · `verl (GRPO/DAPO/PPO)` · `vLLM` · `Docker` · `Ray` · `RCAEval` (seed data)

## Quick Start

```bash
pip install -e ".[dev]"
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .
bash scripts/generate_data.sh
bash scripts/train_grpo.sh
```

## Hardware

Tested on **2× NVIDIA A30 (24 GB)**. Full training pipeline (SFT + 3 RL algorithms + eval) takes ~4–6 days.
