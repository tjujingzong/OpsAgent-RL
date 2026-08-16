# OpsAgent-RL 进度与上手速查

> 最后更新: 2026-08-16。本文档是"当前真实状态"的单一事实源,README 偏介绍,这里偏实操。
> 任何环境/配置/进度变化后请同步更新本文件。

## 1. 一句话状态

代码 / 数据 / Docker 沙盒 / 评估 harness 全部就绪并通过 mock 端到端验证;**真正的 RL 训练尚未启动**。当前正在 opsagent conda env 内安装训练依赖(torch / vllm / verl / ray),并用本机已起的 **Qwen3-8B**(vllm:8000)替代原计划的 Qwen3.5-9B。

## 2. 已完成 ✅

| 模块 | 状态 | 位置 / 证据 |
|------|------|------|
| 项目骨架与全部源码 | ✅ | `src/{agent,env,data,reward,eval,model_backend.py,train.py}` |
| Docker 沙盒镜像 | ✅ 已构建 | `opsagent-sandbox:latest` (788MB);3 个预热容器 `sleep infinity` 在跑 |
| 数据集生成 (48 模板 → 342 变体 → 200/30/100) | ✅ | `data/{train,val,test}.jsonl` |
| 多层奖励引擎 (L1~L4) + 3 类验证器 | ✅ | `src/reward/` |
| OpsBench 评估 + 指标 | ✅ | `src/eval/` |
| 单元/集成测试 (8 个) | ✅ PASS | `tests/{test_core,test_harness}.py` |
| Mock 端到端 harness 验证 | ✅ | `python -m train --smoke-test --mock` |
| 包安装(核心依赖) | ✅ | `opsagent-rl 0.1.0` 已装于 opsagent env |

## 3. 进行中 / 待办 ⏳

- [⏳] opsagent env 安装 `[train]` extra(torch 2.11+cu130 / vllm 0.23.0 / verl 0.9.0 / ray)
- [ ] 用真实 vllm rollout 跑一次非 mock 的 smoke(连 Docker 沙盒)
- [ ] 启动 GRPO 真实训练(`bash scripts/train_grpo.sh`)
- [ ] DAPO / PPO 训练
- [ ] OpsBench 评估 + 出报告

## 4. 硬件与环境

- GPU: **4× NVIDIA A30 (24GB)**;当前 vllm:8000 占用 2 张(TP=2),训练计划用 2 张(可扩到 4)。
- 系统 CUDA: 13.0,驱动 580.159.03。
- 训练 conda env: `opsagent`(Python 3.11.15,`/root/miniconda3/envs/opsagent`)。
- 模型 rollout env: `pd_vllm`(Python 3.12,运行 vllm 0.23.0 serve,不要在里面训练)。

## 5. 依赖版本对齐(关键!)

CUDA 13 很新,版本必须严格对齐,否则 torch/vllm 装出来跑不了。已知能跑的栈(来自 `pd_vllm` env):

| 包 | 版本 | 来源 |
|----|------|------|
| torch | **2.11.0+cu130** | `--index-url https://download.pytorch.org/whl/cu130` |
| torchvision | 0.26.0 | 同上 |
| torchaudio | 2.11.0 | 同上 |
| vllm | **0.23.0** | PyPI(钉死 torch==2.11.0) |
| transformers | **5.10.x**(`>=5.5.3,<5.11`) | verl 0.9.0 与 vllm 0.23.0 的交集 |
| verl | **0.9.0** | PyPI |
| ray[default] | >=2.41.0 | PyPI |
| numpy | 2.x | 随 torch |

> ⚠️ transformers 必须落在 `[5.5.3, 5.11)`。verl 0.9.0 拒绝 >=5.11;vllm 0.23.0 拒绝 5.0~5.5.0。所以装 **5.10.x**。

## 6. 模型与 rollout 接入(改用 8B)

原计划 Qwen3.5-9B,因本机现成可用的是 Qwen3-8B,改用之(见 `configs/model/qwen3_8b.yaml`)。

- 权重路径: `/path/to/Qwen3-8B`
- vllm serve(已在 `pd_vllm` env 跑着,别重启):
  ```
  vllm serve /path/to/Qwen3-8B --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 2 --dtype bfloat16 \
      --max-model-len 8192 --gpu-memory-utilization 0.85 --trust-remote-code
  ```
- 健康检查: `curl -s http://127.0.0.1:8000/v1/models`
- model id(填到 config 的 `server.model_name`): `/path/to/Qwen3-8B`
- eval/benchmark 走 `model_backend.HTTPBackend` 打这个 endpoint;verl rollout worker 也指过来。

## 7. 关键路径速查

| 操作 | 命令 |
|------|------|
| 进训练 env | `source activate opsagent` 或直接用 `/root/miniconda3/envs/opsagent/bin/python` |
| 装/补依赖 | `pip install -e ".[train]"`(env 已对齐版本后) |
| 重建沙盒镜像 | `docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .` |
| 重新生成数据 | `bash scripts/generate_data.sh` |
| Mock smoke(无模型/Docker) | `PYTHONPATH=src python3 -m train --smoke-test --mock --smoke-limit 10` |
| 真实 smoke(连 Docker) | `PYTHONPATH=src python3 -m train --smoke-test --smoke-limit 5` |
| 启动训练 | `bash scripts/train_grpo.sh`(或 `train_dapo.sh` / `train_ppo.sh`) |
| 评估 | `bash scripts/evaluate.sh checkpoints/grpo` |
| 跑测试 | `PYTHONPATH=src pytest tests/` |

## 8. 代码结构要点(便于继续开发)

- `src/train.py`:训练入口。`--smoke-test` 用 `RuleBasedBackend`(回放专家脚本轨迹,无需模型)验证 harness;真实训练 `import verl` 后交给 `verl.trainer.main_ppo.main`。**当前 train.py 还没把 server_url/model_path 暴露成 CLI 参数**,verl 路径靠 hydra 读 config。
- `src/model_backend.py`:`HTTPBackend`(vllm/TGI)/`HFBackend`(本地 transformers)/`RuleBasedBackend`(脚本回放)。eval 用 HTTPBackend。
- `src/agent/policy.py`:框架无关的 episode runner,`generate_fn` 注入即可;`parse_action` 从 ```` ```bash ``` ```` 围栏取命令。
- `src/reward/reward_model.py`:`RewardEngine.compute(env, task, trajectory, steps)`;`group_relative_advantage` 给 GRPO 用。
- `src/env/docker_env.py`:`DockerShellEnv` + 容器池(`pool_size`)。

## 9. 待解决的工程问题(继续任务时会碰到)

1. **verl agentic rollout 适配**:verl 原生支持多轮 agent rollout,但需要把 `AgentPolicy.run_episode` 包成 verl 的 `Worker.rollout` 接口,并把 Docker 容器池挂到 ray worker 上(避免每步起容器)。
2. **reward_fn 注入**:`train.py:build_reward_fn` 已给出 verl 兼容签名,需在 verl 的 hydra config 里挂上 `opsagent.train.build_reward_fn`。
3. **GPU 规划**:vllm:8000 已占 2 张;训练 actor 还要显存。2×A30 24G 同时跑 8B 的 vllm rollout + actor 训练可能 OOM,需 `gpu_memory_utilization` 调低或用 4 卡(2 给 rollout / 2 给 actor)。
4. **SFT 轨迹**:`data/sft_generator.rule_based_trajectory` 已能产出脚本轨迹;若要更强的 SFT 起点,需配 `OPSAGENT_TEACHER_API` 调教师模型。

## 10. 决策记录

- 2026-08-16:模型从 Qwen3.5-9B 切到 **Qwen3-8B**(本机现成可用,权重在 `/path/to/Qwen3-8B`)。所有 train/eval 配置 `defaults: model` 已改为 `qwen3_8b`,`run_name` 同步改名。
- 2026-08-16:训练依赖装在 `opsagent` conda env(Python 3.11),而非 pd_vllm(后者只跑 vllm serve)。
