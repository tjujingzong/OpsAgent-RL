# OpsAgent-RL: RL-Trained System Troubleshooting Agent

## 项目定位

训练一个 LLM Agent，使其能在 Docker 沙盒中通过 shell 命令**主动调查、诊断并修复**系统级故障。区别于现有项目：

| 现有项目 | 侧重 | 我们的差异 |
|---------|------|----------|
| Endless Terminals | 通用终端任务的程序化生成 | 聚焦**诊断推理**，非简单命令执行 |
| SWE-bench | 软件工程 bug 修复 | 面向**基础设施/系统运维**场景 |
| Open-AgentRL | 通用 RL 训练框架 | **垂直领域应用** + 自建评估基准 |

核心价值：**Agent 需要像 SRE 工程师一样思考** -- 形成假设、收集证据、逐步缩小根因范围、实施修复并验证。

---

## 计划修订说明

> 本文档在初版规划基础上进行了以下关键修订：

| 修订项 | 原计划 | 修订后 |
|--------|--------|--------|
| 数据集 | 5 个开源数据集（LogHub 77GB 等） | 1 个种子数据集 RCAEval RE1（390MB） |
| 场景模板 | ~60 个（模糊） | 48 个（精确到每个类别） |
| 训练数据 | 500+（模糊） | 200 train + 30 val + 100 test（精确拆分） |
| SFT 轨迹 | ~1000 条（来源不明） | 500 条（API 生成 + verification 过滤） |
| 训练步数/episode | 30 步 | 20 步（加速 33%） |
| 命令超时 | 30s | 15s（加速 50%） |
| 训练时间 | 未估算 | 94-135 小时（4-6 天，精确估算） |
| 容器资源 | 512MB | 256MB（支持更多并行） |
| Docker 基础镜像 | ubuntu:22.04 (~800MB) | python:3.11-slim (~200MB) |
| 数据总量 | 未估算 | ~465MB（不含模型） |

---

## 技术架构

```
                    +-------------------+
                    | Qwen3.5-9B Agent  |
                    |  (策略模型/Actor)  |
                    +---------+---------+
                              |
                         shell_command
                              |
                    +---------v---------+
                    |  Docker Sandbox   |
                    |  (隔离环境/工具)   |
                    +---------v---------+
                              |
                       observation
                              |
                    +---------v---------+
                    |   Reward Engine   |
                    |  (奖励计算引擎)    |
                    +---------v---------+
                              |
                    +---------v---------+
                    |  verl RL Trainer  |
                    | GRPO/DAPO/PPO    |
                    +-------------------+
```

### 核心技术栈

| 组件 | 选型 | 理由 |
|-----|------|------|
| 基座模型 | Qwen3.5-9B | 2026年3月发布，最新一代，推理能力强，9B 参数 bf16 ~18GB，2×A30 24GB 可训 |
| RL 框架 | verl (vLLM + Ray) | 原生支持 GRPO/DAPO/PPO，agentic RL 支持好，server-based async rollout |
| 推理引擎 | vLLM | 高效 KV cache、TP 支持、高吞吐 |
| 沙盒环境 | Docker | 隔离安全、可复现、易于并行 |
| 评估框架 | 自建 OpsBench | 针对性评估 + RCAEval 基线参考 |
| 种子数据集 | RCAEval RE1 | 390MB，375 个故障案例，5 种故障类型，结构化 |

---

## 数据集选型与规模预估

### 种子数据集：RCAEval RE1

#### 候选对比与选型理由

| 数据集 | 下载大小 | 故障案例数 | 故障类型 | 数据格式 | 适配难度 | 是否选用 |
|--------|---------|-----------|---------|---------|---------|---------|
| **RCAEval RE1** | **390MB** | **375** | **5种**(CPU/MEM/DISK/DELAY/LOSS) | 结构化CSV+JSON | 低 | **是（种子源）** |
| RCAEval RE2 | 4.21GB | 270 | 6种(+SOCKET) | 多源(指标+日志+trace) | 中 | 否（体积过大） |
| RCAEval RE3 | 534MB | 90 | 5种代码级(F1-F5) | 多源+堆栈跟踪 | 中 | 测试集参考 |
| OpenStack Failure | 216MB | ~100 | 子系统级 | 原始日志 | 高 | 否 |
| LogHub-2.0 | 77GB+ | 无标签 | 19+系统 | 原始日志 | 极高 | 否（不适合） |
| AIOpsLab | 框架级 | ~60 problems | K8s层 | 需K8s集群 | 极高 | 架构参考 |

**选择 RCAEval RE1 的理由：**
1. **规模适中**：390MB 下载，375 个案例，半小时内完成下载和解压
2. **结构化好**：每个案例包含 `metrics.json` + `inject_time.txt`，故障类型和根因已标注
3. **故障类型映射直接**：5 种故障类型（CPU/MEM/DISK/DELAY/LOSS）可直接映射到我们的 5 大场景类别
4. **无需额外基础设施**：纯数据集，不需要 K8s 集群
5. **有评估框架**：RCAEval 自带 15 个 RCA 基线方法，可参考其评估设计

**AIOpsLab 作为架构参考**（不直接使用其数据）：
- 问题结构：Detection → Localization → Analysis → Mitigation（对应我们的诊断→修复流程）
- Evaluator 设计：定量指标 + 定性指标（LLM-as-judge）
- Agent 接口：`get_action(state) -> action` 的简单交互协议

#### RCAEval RE1 种子提取

从 RCAEval RE1 的 375 个案例中提取故障模式：

| RCAEval 故障类型 | 提取数量 | 映射到我们的类别 |
|-----------------|---------|-----------------|
| CPU 故障 | 10 | 资源耗竭 |
| MEM 故障 | 10 | 资源耗竭 |
| DISK 故障 | 10 | 资源耗竭 |
| DELAY 故障 | 10 | 网络故障 |
| LOSS 故障 | 10 | 网络故障 |
| **小计** | **50** | |

提取内容：故障注入方式、受影响指标、根因标注 → 转化为 Docker 场景的 `inject_fault` 脚本和 `verification` 规则。

### 数据集总体规模

```
RCAEval RE1 (种子, 390MB)
    ↓ 提取 ~50 个故障模式
    ↓
48 个手工场景模板 (5大类)
    ↓ 参数化扩展 (端口×服务名×难度)
    ↓
~300 个训练场景变体
    ↓ 拆分
    ├── 200 个 RL 训练 prompt
    ├── 30 个验证场景
    └── 100 个测试场景 (OpsBench)
    ↓
+ 500 条 SFT 轨迹 (API 生成 + verification 过滤)
```

### 详细规模分解

#### 场景模板（48 个手工设计）

| 类别 | 模板数 | 具体场景举例 |
|------|--------|-------------|
| 服务故障 | 12 | Nginx 502/503/404、MySQL 启动失败、Redis 连接拒绝、Apache 配置语法错误、SSH 服务停止、Cron 服务异常 |
| 配置错误 | 12 | 错误 DNS 配置、错误 /etc/hosts、错误环境变量、Nginx upstream 端口错误、MySQL 权限配置错误、fstab 挂载错误 |
| 资源耗尽 | 8 | 磁盘满(大文件占满)、内存泄漏(模拟)、僵尸进程堆积、文件描述符耗尽、inode 耗尽、CPU 被占满 |
| 网络故障 | 8 | iptables 规则阻断、错误路由表、DNS 解析失败、端口冲突、broken symlink 导致服务异常、firewall 配置错误 |
| 安全事件 | 8 | 异常 crontab 条目、可疑后台进程、错误文件权限(777)、后门脚本、SSH authorized_keys 异常、SUID 权限异常 |
| **合计** | **48** | |

#### 参数化扩展规则

| 变化维度 | 取值数 | 示例 |
|---------|--------|------|
| 端口号变化 | 3 | 8080 / 9090 / 3000 |
| 服务名变化 | 2 | nginx / apache2 |
| 难度级别 | 3 | easy(直接线索) / medium(需2步排查) / hard(需3步+误导项) |
| 平均扩展倍数 | ~6x | 48 × 6 ≈ 288 变体 |

#### 最终数据集

| 数据集 | 数量 | 格式 | 预估大小 | 用途 |
|--------|------|------|---------|------|
| RCAEval RE1 种子 | 375 cases | CSV+JSON | 390MB | 提取故障模式 |
| 场景模板 YAML | 48 个 | YAML | ~2MB | 参数化生成基础 |
| RL 训练集 | 200 scenarios | JSONL | ~15MB | GRPO/DAPO/PPO 训练 |
| 验证集 | 30 scenarios | JSONL | ~2MB | 训练中评估 |
| OpsBench 测试集 | 100 scenarios | JSONL | ~8MB | 最终评估 |
| SFT 轨迹 | 500 条 | JSONL | ~50MB | SFT 预训练 |
| **总数据量** | | | **~465MB** | 不含模型权重 |

### 训练计算量预估 (2×A30 24GB)

#### 单 episode 耗时分析

| 环节 | 耗时 | 说明 |
|------|------|------|
| vLLM 生成(每步) | ~0.1s | TP=2, ~1000 tok/s, 平均 100 token/命令 |
| Docker 执行(每步) | ~1.5s | 含 exec_run + 命令执行 + 结果返回 |
| 平均步数/episode | 15 步 | 非所有 episode 都跑满 20 步 |
| **单 episode 总耗时** | **~24s** | 15 × (0.1 + 1.5) = 24s |

#### GRPO 训练耗时

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练 prompts | 200 个 | |
| N (rollouts/prompt) | 8 | GRPO 采样数 |
| micro_batch_size | 1 | |
| grad_accumulation | 8 | effective batch = 8 prompts |
| 并行 Docker 环境 | 4 个 | |
| 每步 rollout | 8×8=64 episodes | 64/4 = 16 批顺序执行 |
| 每步 rollout 耗时 | 16 × 24s = 384s ≈ 6.4min | |
| 每步 backward 耗时 | ~1.5min | FSDP backward + optimizer step |
| **每步总耗时** | **~8min** | |
| 训练步数 | 200-300 步 | |
| **GRPO 总训练时间** | **~27-40 小时** | 200×8min ~ 300×8min |

#### 全部算法训练时间

| 阶段 | 耗时 | 说明 |
|------|------|------|
| SFT (1-2 epochs) | ~2 小时 | 500 条轨迹, Qwen3.5-9B |
| GRPO (200-300 steps) | ~27-40 小时 | 核心实验 |
| DAPO (200-300 steps) | ~27-40 小时 | 动态采样，耗时类似 |
| PPO (200-300 steps) | ~35-50 小时 | 额外 critic 模型开销 |
| 评估 | ~3 小时 | OpsBench 100 场景 |
| **总计** | **~94-135 小时 ≈ 4-6 天** | 可在一周内完成 |

---

## Task 1: 项目骨架搭建 (Week 1)

### 1.1 目录结构

```
agentic-rl/
├── pyproject.toml                  # 项目依赖管理
├── configs/
│   ├── model/
│   │   └── qwen35_9b.yaml         # 模型配置
│   ├── train/
│   │   ├── grpo.yaml              # GRPO 训练配置
│   │   ├── dapo.yaml              # DAPO 训练配置
│   │   └── ppo.yaml               # PPO 训练配置
│   └── eval/
│       └── opsbench.yaml          # 评估配置
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── prompts.py             # Agent 系统提示词
│   │   └── policy.py              # 策略封装 (动作解析、历史管理)
│   ├── env/
│   │   ├── __init__.py
│   │   ├── docker_env.py          # Docker 沙盒环境
│   │   ├── task_loader.py         # 任务加载器
│   │   └── scenarios/             # 预定义故障场景 (48 个 YAML 模板)
│   │       ├── service_failure/
│   │       ├── misconfiguration/
│   │       ├── resource_exhaustion/
│   │       ├── network_issues/
│   │       └── security_incidents/
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── reward_model.py        # 奖励计算主逻辑
│   │   └── verifiers/             # 各场景验证器
│   │       ├── service_verifier.py
│   │       ├── config_verifier.py
│   │       └── state_verifier.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # 数据集定义与拆分
│   │   ├── generator.py           # 参数化任务生成
│   │   ├── sft_generator.py       # SFT 轨迹生成 (API + 过滤)
│   │   └── seed/                  # RCAEval 种子数据
│   └── eval/
│       ├── __init__.py
│       ├── benchmark.py           # OpsBench 评估主逻辑
│       └── metrics.py             # 评估指标计算
├── docker/
│   └── Dockerfile.base            # 沙盒基础镜像
├── scripts/
│   ├── setup_env.sh               # 环境搭建脚本
│   ├── download_datasets.sh       # 下载 RCAEval RE1
│   ├── generate_data.sh           # 数据生成脚本
│   ├── train_grpo.sh              # GRPO 训练启动
│   ├── train_dapo.sh              # DAPO 训练启动
│   ├── train_ppo.sh               # PPO 训练启动
│   └── evaluate.sh                # 评估脚本
├── tools/
│   ├── explore_data.py             # 数据探索脚本
│   ├── reward_analysis.py          # 奖励分布分析
│   └── eval_visualization.py       # 评估结果可视化
├── data/                           # 生成的训练/验证/测试数据
└── assets/
    └── figures/                    # 项目图表
```

### 1.2 依赖管理 (`pyproject.toml`)

```toml
[project]
name = "opsagent-rl"
version = "0.1.0"
description = "RL-trained System Troubleshooting Agent"
requires-python = ">=3.10"
dependencies = [
    "verl>=0.3.0",
    "vllm>=0.6.0",
    "torch>=2.3.0",
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "docker>=7.0.0",
    "ray>=2.35.0",
    "datasets>=3.0.0",
    "wandb>=0.18.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "pandas>=2.0",           # 处理 RCAEval CSV 数据
    "scikit-learn>=1.3.0",   # 数据拆分
    "pyyaml>=6.0",            # 场景模板解析
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ipython>=8.0.0", "asciinema>=2.0.0"]
```

### 1.3 环境搭建脚本 (`scripts/setup_env.sh`)

```bash
#!/bin/bash
set -e

echo "=== OpsAgent-RL Environment Setup ==="

# 1. Python 虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 2. 安装项目依赖
pip install -e ".[dev]"

# 3. 构建 Docker 沙盒镜像
echo "Building Docker sandbox image..."
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .

# 4. 下载 RCAEval RE1 种子数据
echo "Downloading RCAEval RE1 dataset (390MB)..."
bash scripts/download_datasets.sh

echo "=== Setup complete! ==="
```

### 1.4 数据下载脚本 (`scripts/download_datasets.sh`)

```bash
#!/bin/bash
set -e

mkdir -p src/data/seed
cd src/data/seed

python3 -c "
from RCAEval.utility import download_re1_dataset
print('Downloading RCAEval RE1 (390MB, ~1 min)...')
download_re1_dataset()
print('Done! Data saved to src/data/seed/')
"

cd -
```

---

## Task 2: Docker 沙盒环境 (Week 1-2)

### 2.1 基础镜像 (`docker/Dockerfile.base`)

**关键改进**：基于 `python:3.11-slim` 而非 `ubuntu:22.04`，镜像从 ~800MB 降到 ~200MB。

```dockerfile
FROM python:3.11-slim

# 安装常用系统工具（精简版）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget net-tools iputils-ping dnsutils procps \
    nginx redis-server sqlite3 jq \
    && rm -rf /var/lib/apt/lists/*

# 预装故障排查常用 Python 工具
RUN pip3 install --no-cache-dir psutil pyyaml

# 设置工作目录
WORKDIR /workspace
```

### 2.2 沙盒管理 (`src/env/docker_env.py`)

**关键改进（相比初版计划）：**

1. **容器池预热**：启动时创建 4 个容器保持在 pool 中，避免每次 episode 冷启动
2. **故障注入分离**：`setup_commands`（环境初始化）和 `inject_fault`（注入故障）分离执行
3. **轻量化资源限制**：CPU 0.5 core, Memory 256MB（比原计划 512MB 更省，可跑更多并行）

核心设计：
- 每个 task 从容器池取一个容器，执行 setup + inject_fault
- Agent 通过 `exec_run()` 执行 shell 命令
- 每次返回 stdout + stderr + exit_code 作为 observation
- 超时保护：单命令 15s（加速 50%），整个 episode 5min
- 资源限制：CPU 0.5 core, Memory 256MB, 无网络（安全）
- 支持并行运行 4 个容器（batch rollout 时使用）

```python
class DockerShellEnv:
    """Docker-based shell environment for agent interaction."""

    def __init__(self, image: str, pool_size: int = 4, command_timeout: int = 15):
        self.client = docker.from_env()
        self.image = image
        self.pool_size = pool_size
        self.command_timeout = command_timeout
        self.container_pool = []   # 预热容器池
        self.container = None
        self.current_task = None

    def _init_pool(self):
        """启动时预热 pool_size 个容器"""

    def reset(self, task: Task) -> str:
        """从容器池取容器 → 执行 setup_commands → 执行 inject_fault → 返回初始 observation"""

    def step(self, command: str) -> Tuple[str, bool, float, dict]:
        """执行命令，返回 (observation, done, reward, info)
        - 截断输出：保留前 200 + 后 300 字符，中间用 ...[truncated]... 替代
        """

    def close(self):
        """停止并移除容器（或重置后放回池中）"""
```

### 2.3 性能优化参数

| 参数 | 原计划 | 修订后 | 理由 |
|------|--------|--------|------|
| 基础镜像 | ubuntu:22.04 (~800MB) | python:3.11-slim (~200MB) | 减少 75% 镜像体积 |
| 容器内存 | 512MB | 256MB | 支持更多并行容器 |
| 命令超时 | 30s | 15s | 加速 50% |
| episode 超时 | 10min | 5min | 加速训练 |
| max_steps | 30 | 20 | 加速 33% |
| 容器池 | 无 | 4 个预热 | 避免冷启动 |

---

## Task 3: 故障场景与合成数据构建 (Week 2-3) — 核心

### 3.1 五大故障类别（48 个模板）

| 类别 | 模板数 | 示例 | 验证方式 |
|------|--------|------|---------|
| 服务故障 | 12 | Nginx 502/503/404、MySQL 启动失败、Redis 连接拒绝、Apache 配置语法错误 | 检查服务状态/端口 |
| 配置错误 | 12 | 错误 DNS 配置、错误 /etc/hosts、错误环境变量、Nginx upstream 端口错误 | diff 修复前后配置 |
| 资源耗尽 | 8 | 磁盘满(大文件占满)、内存泄漏(模拟)、僵尸进程堆积、文件描述符耗尽 | 检查资源使用率 |
| 网络故障 | 8 | iptables 规则阻断、错误路由表、DNS 解析失败、端口冲突 | ping/curl 目标 |
| 安全事件 | 8 | 异常 crontab 条目、可疑后台进程、错误文件权限(777)、后门脚本 | 检查进程/文件状态 |

### 3.2 场景模板设计

每个场景模板 YAML 包含以下字段（以 nginx_502 为例）：

```yaml
# scenarios/service_failure/nginx_502.yaml
scenario:
  id: "service_nginx_502"
  category: "service_failure"
  difficulty: "medium"

  description: "Web server returning 502 errors. Users report the site is down."

  setup_commands:
    - "apt-get install -y nginx"
    - "echo 'server { listen 80; location / { proxy_pass http://127.0.0.1:{port}; } }' > /etc/nginx/conf.d/default.conf"
    - "nginx"

  inject_fault:
    - "pkill -f 'python.*http_server' || true"  # 杀掉后端服务

  verification:
    type: "service_check"
    criteria:
      - command: "curl -s -o /dev/null -w '%{http_code}' http://localhost"
        expected: "200"
    root_cause_keywords: ["backend", "upstream", "port", "service down"]

  reward_spec:
    success_reward: 10.0
    partial_rewards:
      - condition: "nginx config syntax valid"
        reward: 2.0
        check: "nginx -t"
      - condition: "backend service running"
        reward: 5.0
        check: "pgrep -f http_server"
    step_penalty: -0.1
    max_steps: 20  # 从30降到20，加速训练

  # 参数化变量
  params:
    port: [8080, 9090, 3000]
    difficulty: [easy, medium, hard]
```

### 3.3 种子数据复用：RCAEval RE1

不再使用多个开源数据集，仅复用 RCAEval RE1（390MB，375 个案例）作为种子源：

**数据适配流程：**
```
RCAEval RE1 (375 cases) → 提取 50 个故障模式 → 转化为 Docker 场景模板 → 参数化扩展 → 训练数据
```

具体做法：
1. 从 RCAEval RE1 提取 CPU/MEM/DISK/DELAY/LOSS 五种故障的注入方式和根因标签
2. 将微服务级故障简化为单机 Docker 场景
3. 结合 SRE 领域知识设计 48 个手工场景模板（含 RCAEval 提取的 50 个故障模式的映射）
4. 用 Qwen3.5-27B API 基于真实故障模式生成更多变体

### 3.4 参数化生成 Pipeline (`src/data/generator.py`)

```
generator.py 核心逻辑:

1. 加载 48 个 YAML 模板
2. 对每个模板:
   a. 读取 params 中的变化维度
   b. 笛卡尔积生成所有变体
   c. 对每个变体:
      - 替换模板中的 {port} 等占位符
      - 根据 difficulty 调整 inject_fault 复杂度:
        easy:   单一故障，症状明显
        medium: 故障 + 1 个误导项
        hard:   故障 + 2 个误导项 + 更隐蔽的注入方式
      - 生成唯一 task_id (如 service_nginx_502_port8080_medium)
3. 总计生成 ~300 个场景变体
4. 按类别分层抽样拆分: train(200) / val(30) / test(100)
5. 输出为 JSONL 格式
```

### 3.5 SFT 轨迹生成 (`src/data/sft_generator.py`)

使用 Qwen3.5-27B API 为 200 个训练场景生成专家诊断轨迹：

**生成策略：**
- 每个训练场景生成 2-3 条轨迹（不同修复路径）→ 共 ~500 条
- Prompt 设计：给出场景描述 + 正确根因 + 要求生成 step-by-step 诊断修复过程
- 质量控制：生成后用 verification 命令在 Docker 中实际执行验证，只保留成功修复的轨迹
- 预计成功率 ~70%，需生成 ~720 条 → 过滤后得 ~500 条

**轨迹格式（verl 兼容）：**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SRE agent..."},
    {"role": "user", "content": "Scenario: Web server returning 502..."},
    {"role": "assistant", "content": "Let me check the nginx status first.\n```bash\nsystemctl status nginx\n```"},
    {"role": "user", "content": "nginx is active (running)..."},
    {"role": "assistant", "content": "The service is running. Let me check the upstream..."}
  ]
}
```

### 3.6 训练数据格式（verl 兼容）

```json
{
  "prompt": [
    {"role": "system", "content": "You are a system troubleshooting agent..."},
    {"role": "user", "content": "Scenario: Web server returning 502 errors..."}
  ],
  "task_id": "service_nginx_502_port8080_medium",
  "category": "service_failure",
  "difficulty": "medium",
  "expected_verification": {
    "commands": ["curl -s -o /dev/null -w '%{http_code}' http://localhost"],
    "expected_outputs": ["200"]
  }
}
```

---

## Task 4: Agent 设计与系统提示词 (Week 2)

### 4.1 Agent 交互协议（细化版）

```
System: You are an expert SRE (Site Reliability Engineer) agent with access to a Linux shell.

Your goal: Diagnose and fix the system issue described below.

Rules:
1. Issue ONE shell command per turn, wrapped in ```bash``` tags
2. Follow systematic diagnosis: observe → hypothesize → test → fix → verify
3. Start with broad checks (systemctl, ps, df, netstat) before diving deep
4. After applying a fix, VERIFY it works before declaring completion
5. Say "TASK_COMPLETE" only after verification passes
6. Maximum 20 commands per task

Available: Any valid bash command in the container.
```

### 4.2 Agent 工具集

Agent 直接使用 shell 命令作为 action，无需额外工具封装。verl 的 agentic RL 模式下，模型生成 shell 命令文本，环境执行并返回结果。

verl agentic RL 支持特性：
- Server-based 异步 rollout：避免 GPU 等待工具调用时空闲
- Multi-turn conversations and tool calls
- LangGraph-based Agent 框架支持

### 4.3 对话历史管理（优化版）

多轮交互需要管理上下文窗口：
- 保留最近 **16 轮**对话（比原计划 20 轮更紧凑，节省 token）
- 超长输出截断：保留前 **200** + 后 **300** 字符，中间用 `...[truncated]...` 替代
- 关键诊断信息摘要：每 5 步自动注入一次系统状态摘要（`ps aux | head -5; df -h; free -m`）

---

## Task 5: 奖励模型设计 (Week 3)

### 5.1 分层奖励结构 (`src/reward/reward_model.py`)

```python
class TroubleshootingReward:
    """Multi-level reward for system troubleshooting tasks."""

    def compute(self, trajectory, task) -> float:
        # Level 1: 任务完成奖励 (0 或 +10)
        task_reward = self.verify_task_completion(trajectory, task)

        # Level 2: 诊断质量奖励 (+0~+5)
        diagnostic_reward = self.evaluate_diagnostic_quality(trajectory)

        # Level 3: 效率奖励/惩罚
        efficiency_reward = self.evaluate_efficiency(trajectory)

        # Level 4: 方法论奖励 (+0~+2)
        methodology_reward = self.evaluate_methodology(trajectory)

        return task_reward + diagnostic_reward + efficiency_reward + methodology_reward
```

### 5.2 各层奖励细则（实现细节）

| 层级 | 内容 | 分值 | 计算方式 | 实现细节 |
|------|------|------|---------|---------|
| L1-任务完成 | 故障是否修复 | 0 / +10 | 执行 verification.criteria 中的所有命令 | 逐条执行 check 命令，比较 expected 输出 |
| L2-诊断质量 | 是否正确定位根因 | +0~+5 | 检查轨迹中是否查看了关键日志/配置 | 关键词匹配：root_cause_keywords 出现在 agent 执行的命令或输出中 |
| L3-效率 | 命令数量和步骤 | 动态 | -0.1 × steps，最多扣 -2.0 | step_penalty × min(steps, 20) |
| L4-方法论 | 诊断思路是否系统 | +0~+2 | LLM-as-judge 评估诊断逻辑 | 用 Qwen3.5-9B 本身做 judge（无需额外 API） |

### 5.3 GRPO 的特殊处理

GRPO 不需要单独的 reward model，而是对同一 prompt 的多个 rollout 计算 group-level 优势：
- 同一故障场景采样 N=8 条轨迹
- 计算 group 内奖励的均值和方差
- 优势 = (单条奖励 - group均值) / (group标准差 + 1e-8)

这天然适合我们的场景：同一故障有多种修复路径，GRPO 自动学习哪些路径更优。

---

## Task 6: RL 训练配置 (Week 4-6)

### 6.1 硬件分配 (2 x A30 24GB = 48GB)

```
GPU 0 (A30 24GB)              GPU 1 (A30 24GB)
┌───────────────────┐        ┌───────────────────┐
│ Actor (FSDP分片)   │        │ Reference Model    │
│ Qwen3.5-9B bf16   │        │ Qwen3.5-9B bf16   │
│ ~9GB (FSDP半量)    │        │ ~9GB (FSDP半量)    │
│                   │        │                   │
│ vLLM KV Cache     │        │ Critic (PPO only) │
│ ~8GB              │        │ ~9GB              │
│                   │        │                   │
│ Docker envs (CPU) │        │ Free ~6GB         │
│ ~2GB (host RAM)   │        │                   │
└───────────────────┘        └───────────────────┘
TP=2 for vLLM rollout | FSDP for training
```

### 6.2 GRPO 训练配置 (`configs/train/grpo.yaml`)

```yaml
model:
  name: "Qwen/Qwen3.5-9B"
  use_flash_attention: true

algorithm:
  type: "grpo"
  num_generation_per_prompt: 8
  kl_coef: 0.001        # 低 KL 约束，允许较大策略更新
  clip_range: 0.2

rollout:
  engine: "vllm"
  tensor_parallel_size: 2
  max_new_tokens: 4096
  temperature: 1.0      # 高温度探索不同修复路径
  top_p: 0.95
  gpu_memory_utilization: 0.85

train:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-6  # 保守学习率
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  max_epochs: 3
  bf16: true

data:
  train_file: "data/train.jsonl"
  val_file: "data/val.jsonl"
  max_prompt_length: 2048
  num_workers: 4

env:
  max_steps_per_episode: 20   # 从30降到20
  command_timeout: 15          # 从30降到15
  parallel_envs: 4

logging:
  project: "opsagent-rl"
  run_name: "grpo_qwen35_9b"
  log_interval: 10
  eval_interval: 50
```

### 6.3 DAPO 训练配置 (`configs/train/dapo.yaml`)

DAPO 与 GRPO 的核心区别：

```yaml
algorithm:
  type: "dapo"
  # 动态采样：reward 方差大时增加 N，方差小时减少
  dynamic_sampling: true
  min_n: 4
  max_n: 16
  # 过长惩罚
  length_penalty: -0.05      # 每超出 1000 token 扣分
  max_length: 4096
  # Clip-higher：放宽正优势上界，鼓励探索
  clip_range_high: 0.28      # 比标准 0.2 更宽
  clip_range_low: 0.2
  # Token-level loss：使用 token 级别的策略梯度而非 sequence 级别
  token_level_loss: true
  kl_coef: 0.001
```

### 6.4 PPO 训练配置 (`configs/train/ppo.yaml`)

PPO 额外需要 Critic 模型。48GB 放不下完整独立 critic，使用 **共享 backbone + 独立 value head** 的方式：

```yaml
algorithm:
  type: "ppo"
  critic_shared_backbone: true   # 共享 backbone + 独立 value head
  critic_lr: 1.0e-6
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  value_loss_coef: 0.5
  kl_coef: 0.001
```

备选方案：使用 Qwen3.5-4B 作为独立 critic 模型以节省显存。

### 6.5 训练流程

```
Phase 1: SFT (2小时)
   └── 500 条轨迹, 2 epochs, lr=2e-5
   └── 输出: sft_checkpoint/

Phase 2: GRPO (27-40小时)
   └── 从 sft_checkpoint 初始化
   └── 200-300 steps, 每50步评估一次
   └── 输出: grpo_checkpoint/

Phase 3: DAPO (27-40小时)
   └── 从 sft_checkpoint 初始化（非从GRPO继续）
   └── 200-300 steps
   └── 输出: dapo_checkpoint/

Phase 4: PPO (35-50小时)
   └── 从 sft_checkpoint 初始化
   └── 200-300 steps
   └── 输出: ppo_checkpoint/

Phase 5: 评估与对比
   └── 在 OpsBench 测试集 (100 场景) 上评估所有模型
```

---

## Task 7: OpsBench 评估基准 (Week 7)

### 7.1 基准设计

**OpsBench 构成（100 场景，独立于训练集）：**

| 来源 | 数量 | 说明 |
|------|------|------|
| 手工设计（独立于训练集） | 50 | 使用不同的端口/服务名/故障模式变体 |
| RCAEval RE3 适配 | 30 | 代码级故障，从 RCAEval RE3 90 个案例中选取并适配到 Docker |
| 程序化生成（独立种子） | 20 | 使用不同随机种子生成，确保与训练集无重叠 |
| **合计** | **100** | |

| 维度 | 说明 |
|------|------|
| 分类 | 5 大类别，3 个难度级别 |
| 隔离 | 每个场景独立 Docker 容器 |
| 可复现 | Dockerfile + 种子固定 |

### 7.2 评估指标

| 指标 | 说明 | 实现方式 | 权重 |
|------|------|---------|------|
| Success Rate (SR) | 故障是否被成功修复 | L1 奖励 > 0 的比例 | 核心 |
| Diagnostic Accuracy (DA) | 是否正确识别根因 | L2 奖励 > 3 的比例 | 核心 |
| Mean Steps to Resolution | 平均修复步骤数 | 成功 episode 的平均步数 | 效率 |
| Command Efficiency | 有效命令占比 | 有效命令数 / 总命令数 | 效率 |
| Pass@k | k 次尝试中至少成功一次 | k 次采样中至少 1 次成功 | 鲁棒性 |

### 7.3 对比基线

| 基线 | 说明 |
|------|------|
| Qwen3.5-9B (base) | 未经 RL 训练的原始模型 |
| Qwen3.5-9B + SFT | 仅做 SFT 不做 RL |
| Qwen3.5-9B + GRPO | 我们的方法 |
| Qwen3.5-9B + DAPO | 我们的方法 |
| Qwen3.5-9B + PPO | 我们的方法 |
| GPT-4o / Claude (API) | 闭源模型对比（可选） |

---

## Task 8: 实验与消融 (Week 8)

### 8.1 主实验

三种 RL 算法的对比，记录训练曲线（reward、success rate、KL divergence）。

### 8.2 消融实验

| 实验 | 变量 | 预计时间 |
|------|------|---------|
| 主实验 | GRPO vs DAPO vs PPO vs Base vs SFT | ~3小时(评估) |
| 消融1: Reward 设计 | L1-only vs L1+L2 vs Full | ~1.5小时 |
| 消融2: 采样数 N | N=4 vs N=8 vs N=16 | ~2小时 |
| 消融3: SFT 影响 | with SFT vs without SFT | ~1小时 |
| 消融4: 训练数据量 | 100 vs 200 vs 300 场景 | ~2小时 |
| 消融5: 模型大小(如条件允许) | 4B vs 9B | ~2小时 |

### 8.3 定性分析

- 选取典型案例展示 Agent 的诊断推理过程
- 分析 RL 训练前后 Agent 行为的差异
- 展示 Agent 学到的诊断策略

---

## Task 9: 项目展示与文档 (Week 9-10)

### 9.1 README 内容

- 项目动机与核心思路
- 快速开始（5 分钟跑通一个 demo）
- 完整的训练/评估/推理流程
- 实验结果与消融分析
- 硬件要求与优化建议

### 9.2 Demo / 可视化

- 录制 Agent 排查故障的终端交互过程（asciinema），3 个典型案例（成功/失败/边缘）
- 训练曲线对比图（WandB 导出）
- 成功案例 vs 失败案例的可视化对比
- 消融实验对比表格

### 9.3 项目亮点（面试 talking points）

1. **完整后训练 pipeline**：从数据构建 -> SFT -> RL(GRPO/DAPO/PPO) -> 评估
2. **自建环境**：Docker 沙盒 + 多进程并行 + 容器池预热，可扩展到其他场景
3. **自建 Benchmark**：OpsBench，填补系统运维领域 Agent 评估空白
4. **三种 RL 算法对比**：展示对不同算法的深入理解
5. **Reward Engineering**：多层奖励设计，展示 RL 调参经验
6. **实际价值**：SRE/DevOps 是真实高价值场景

---

## Task 10: 修订时间线规划

| 阶段 | 时间 | 产出 | 预估工时 |
|------|------|------|---------|
| **Phase 1: 基础设施** | Week 1 | 项目骨架 + Docker 环境 + RCAEval 下载 | 15h |
| **Phase 2: 数据构建** | Week 2-3 | 48 模板 + 300 变体 + 500 SFT 轨迹 + 验证器 | 25h |
| **Phase 3: SFT 预训练** | Week 3 | SFT 模型 | 4h (2h训练+2h调试) |
| **Phase 4: GRPO 训练** | Week 4-5 | GRPO 模型 + 初步结果 | 45h (含调试) |
| **Phase 5: DAPO+PPO** | Week 6-7 | 三种算法对比 | 80h |
| **Phase 6: 评估分析** | Week 8 | OpsBench 评估 + 消融实验 | 15h |
| **Phase 7: 文档展示** | Week 9-10 | README + Demo + 报告 | 15h |
| **总计** | **10 周** | | **~200h** |

---

## 风险评估与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| 2×A30 显存不足 | 中 | 无法训练 9B | 降级到 Qwen3.5-4B (bf16 ~8GB)，或使用 LoRA + GRPO |
| Docker 环境开销过大 | 高 | 训练慢 | 容器池预热 + 轻量镜像 + 减少步数到 20 + 并行 4 环境 |
| SFT 轨迹质量差 | 中 | RL 起点低 | 用 verification 命令过滤失败轨迹 + 人工抽检 50 条 |
| GRPO 训练不稳定 | 中 | reward 不收敛 | 先用 L1-only reward 跑 50 步验证 pipeline，再加 L2-L4 |
| 3 种算法时间不够 | 中 | 无法完成全部 | 优先 GRPO，DAPO 其次，PPO 最后（可降为 100 步） |
| 场景泛化性差 | 低 | 过拟合训练场景 | 测试集使用独立种子和 RCAEval RE3 数据 |
