# OpsAgent-RL

> 用强化学习训练 LLM Agent，使其像 SRE 工程师一样**诊断、修复并验证**系统故障。

## 项目简介

OpsAgent-RL 通过强化学习训练语言模型 Agent（Qwen3.5-9B），使其能在 Docker 沙盒中自主排查 Linux 系统故障。Agent 通过执行 shell 命令、观察输出、形成假设、逐步迭代来完成诊断修复——如同真实 SRE 工程师的工作方式。

我们在该任务上对比了三种 RL 算法（**GRPO / DAPO / PPO**），基于 [verl](https://github.com/volcengine/verl) 框架构建。

## 核心特性

- **48 个手工故障场景**，覆盖 5 大类：服务故障、配置错误、资源耗尽、网络故障、安全事件
- **Docker 沙盒环境**，容器池预热，支持安全、并行、可复现的训练
- **多层奖励设计**：任务完成 + 诊断质量 + 效率 + 方法论
- **OpsBench**：100 个场景的系统运维 Agent 评估基准

## 技术栈

`Qwen3.5-9B` · `verl (GRPO/DAPO/PPO)` · `vLLM` · `Docker` · `Ray` · `RCAEval`（种子数据）

## 快速开始

```bash
pip install -e ".[dev]"
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .
bash scripts/generate_data.sh
bash scripts/train_grpo.sh
```

## 硬件要求

在 **2× NVIDIA A30 (24 GB)** 上测试。完整训练流程（SFT + 3 种 RL 算法 + 评估）约需 4–6 天。
