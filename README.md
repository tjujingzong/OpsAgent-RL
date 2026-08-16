# OpsAgent-RL

> 用强化学习训练 LLM Agent，使其像 SRE 工程师一样**诊断、修复并验证** Linux 系统故障。

## 项目简介

OpsAgent-RL 在 Docker 沙盒中训练一个语言模型 Agent（Qwen3.5-9B），使其能自主排查系统故障：执行 shell 命令、观察输出、形成假设、逐步缩小根因、实施修复并验证——如同真实 SRE 的工作方式。我们在该任务上对比三种 RL 算法（**GRPO / DAPO / PPO**），基于 [verl](https://github.com/volcengine/verl) 框架构建。

核心价值：Agent 需要像 SRE 一样思考——形成假设、收集证据、缩小根因、修复并验证，而非简单命令执行。

## 核心特性

- **48 个手工故障场景模板**，覆盖 5 大类：服务故障、配置错误、资源耗尽、网络故障、安全事件
- **参数化扩展**为 ~340 个任务变体，分层抽样拆分为 **200 train / 30 val / 100 test**
- **Docker 沙盒环境**：容器池预热、`init=tini` 回收僵尸、`NET_ADMIN` 支持网络场景、运行时禁网保安全
- **多层奖励设计**：任务完成(L1) + 部分奖励 + 诊断质量(L2) + 效率(L3) + 方法论(L4)
- **OpsBench**：100 场景的系统运维 Agent 评估基准，含 SR / DA / 步数 / pass@k 等指标
- **GRPO 群组相对优势**天然适配"同一故障多种修复路径"

## 技术栈

`Qwen3.5-9B` · `verl (GRPO/DAPO/PPO)` · `vLLM` · `Docker` · `Ray` · `RCAEval`（种子数据）

## 目录结构

```
opsagent-rl/
├── configs/                    # 模型 / 训练(GRPO/DAPO/PPO) / 评估配置
├── docker/Dockerfile.base      # 沙盒基础镜像 (python:3.11-slim + nginx/redis/mysql/apache/...)
├── src/
│   ├── agent/                  # 系统提示词、动作解析、会话与 episode 运行器
│   ├── env/                    # Docker 沙盒、任务加载器、48 个 YAML 场景、MockShellEnv
│   ├── data/                   # 数据集 IO、参数化生成器、SFT 轨迹生成器
│   ├── reward/                # 多层奖励引擎 + 3 个类别验证器
│   ├── eval/                   # OpsBench 基准 + 指标
│   ├── model_backend.py        # vLLM-HTTP / 本地 HF / 规则式后端
│   └── train.py                # 训练入口 + reward 函数工厂 + --smoke-test
├── scripts/                    # 环境搭建 / 下载数据 / 生成数据 / 训练 / 评估
├── tools/                      # 数据探索 / 奖励分析 / 结果可视化
├── tests/                      # 单元测试 + 端到端 harness 测试
└── data/                       # 生成的 train/val/test + SFT + 评估报告
```

## 快速开始

```bash
# 1. 安装（核心依赖；训练依赖可选）
pip install -e ".[dev]"            # 可加 [train] 拉取 torch/vllm/verl

# 2. 构建沙盒镜像（需联网主机；运行时容器禁网）
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .

# 3. 生成数据集（48 模板 → 342 变体 → 200/30/100）
bash scripts/generate_data.sh

# 4. 端到端验证 harness（无需模型 / 无需 Docker）
PYTHONPATH=src python3 -m train --smoke-test --mock --smoke-limit 10

# 5. 训练（需 GPU + verl）
bash scripts/train_grpo.sh        # 或 train_dapo.sh / train_ppo.sh

# 6. 评估
bash scripts/evaluate.sh checkpoints/grpo
```

## 沙盒环境

- 基础镜像：`python:3.11-slim` + nginx / apache2 / redis / mariadb / sqlite / cron / sshd / iptables
- 无 systemd：场景用 `svc <name>` 启动守护进程（镜像内置启动助手）
- 运行时禁用外网（`network_disabled`），所有主机为 `127.0.0.1`
- 资源限制：CPU 0.5 core、内存 256MB；容器池预热 4 个，避免冷启动
- `NET_ADMIN` capability 支持 iptables / ip route 类网络场景；`init=tini` 回收僵尸进程

## 奖励设计（4 层）

| 层级 | 内容 | 分值 | 计算方式 |
|------|------|------|---------|
| L1 任务完成 | 故障是否修复 | 0 / +10 | 执行 `verification.criteria`，全部通过得 success_reward |
| 部分奖励 | 中间状态达标 | +叠加 | 通过 `partial_rewards` 中各 check 得对应分 |
| L2 诊断质量 | 是否定位根因 | +0~+5 | root_cause_keywords 在 agent 命令/输出中的覆盖率 |
| L3 效率 | 命令数量 | -0.1×步 | 上限 -2.0 |
| L4 方法论 | 诊断思路 | +0~+2 | LLM-as-judge（可选，默认关闭） |

GRPO 对同一 prompt 采样 N=8 条轨迹，用 `group_relative_advantage` 计算群组内优势。

## 评估指标（OpsBench）

Success Rate · Diagnostic Accuracy · Mean Steps to Resolution · Command Efficiency · pass@k，并提供按类别的细分。

## 硬件要求

在 **2× NVIDIA A30 (24 GB)** 上设计（4× 亦可，可扩大并行）。完整流程（SFT + 3 种 RL + 评估）约需 4–6 天。

## 状态说明

- 数据管线（模板 / 生成 / 拆分）、环境、奖励、评估、训练入口与测试均已**实现并通过验证**（8 个单元/集成测试 PASS，342 任务生成 200/30/100 拆分，mock 模式端到端跑通）。
- 真实 RL 训练需在**联网 GPU 主机**上构建沙盒镜像并安装 `.[train]` 依赖后执行；SFT 轨迹需配置 `OPSAGENT_TEACHER_API` 调用教师模型。
