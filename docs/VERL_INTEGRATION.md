# verl GRPO 集成笔记与踩坑手册

> 本文档记录把 OpsAgent-RL 任务接入 verl 0.9.0 GRPO 的架构决策、踩过的坑与解法、
> 以及最终的显存结论。是 `docs/PROGRESS.md` §11 的完整展开版。最后更新: 2026-08-16。

## 0. 一图流(数据怎么走)

```
data/train.parquet (prompt, data_source, extra_info.task)
   │  (verl RLHFDataset → DataLoader)
   ▼
AgentLoopManagerTQ.generate_sequences(batch)
   │  按 rollout.agent.num_workers 并行
   ▼
OpsAgentLoop.run(sampling_params, raw_prompt, extra_info)   ← 我们的 AgentLoopBase
   │  1. task_from_record(extra_info["task"]) → Task
   │  2. DockerShellEnv(image=opsagent-sandbox) 一轨迹一容器: reset→setup+inject_fault
   │  3. 循环: server_manager.generate(prompt_ids) → 解析 ```bash``` 命令 → env.step →
   │     把 observation 渲染成 user 消息拼回 token 流 (mask=0)
   │  4. RewardEngine.compute(env, task, trajectory) → reward (in-loop, SWE-agent 模式)
   │  5. env.close()
   ▼
AgentLoopOutput(prompt_ids, response_ids, response_mask, reward_score, ...)
   │  as_dict(): reward_score → rm_scores (reward 放在最后一个有效 token)
   ▼
verl trainer (PPOTrainerSync / colocate): reward 已有 → NaiveRewardManager 跳过 compute_score
   │  GRPO 群组相对优势 → actor.backward (FSDP2) → optim.step
   ▼
checkpoint_engine.update_weights → 同步到 vLLM 引擎 → 下一轮 rollout
```

## 1. 为什么不用 verl 内置 ToolAgentLoop,而写自定义 OpsAgentLoop

verl 0.9.0 自带多轮 agentic 框架 `verl/experimental/agent_loop/`,内置 `ToolAgentLoop`
(ReAct 状态机,PENDING→GENERATING→PROCESSING_TOOLS→TERMINATED)。但对我们的任务它**不适配**:

- `ToolAgentLoop._call_tool`(`tool_agent_loop.py:519-530`)对 `BaseTool` 是**每次 tool call 都
  create→execute→release**:`instance_id` 每次 `uuid4` 新生成 → 容器**不跨命令持久**。我们的 SRE
  任务要在**同一个故障容器**里跑 10-20 条命令(diagnose→fix→verify),这直接破坏。
- `calc_reward` 在整个 `tool_agent_loop.py` 里**从未被调用**。终端奖励只能走 reward manager 的
  `compute_score`,而我们的奖励需要在**同一容器**里跑 `verification.criteria` 验证 —— compute_score
  在 reward worker 进程里、拿不到容器。

所以采用 **SWE-agent 模式**:自定义 `OpsAgentLoop(AgentLoopBase)`,`run()` 里管一整个轨迹的容器
生命周期 + token mask + 奖励。这是 verl 官方支持的模式(`agent_loop.py:745` 注释明说
"Some AgentLoop may have already computed the reward score, e.g SRE-agent")。

关键契约(读 verl 源码确认):
- `_compute_score`(`agent_loop.py:966`):**仅当 `reward_score is None 且 async reward 开启时**才算
  奖励。我们在 loop 内设了 `reward_score` → `as_dict()` 生成 `rm_scores` → `NaiveRewardManager`
  在 `abstract.py:64` 检测到 `rm_scores` 已存在 → **跳过 `compute_score`**。所以奖励权威来自 loop。
- `AgentLoopOutput.as_dict()`(`:143-147`):`reward_score` → `rm_scores`,`reward 放在最后一个有效 token`。
- token mask:`1`=LLM 生成 token,`0`=观察/tool 模板 token(不训练)。observation 用
  `apply_chat_template([user_msg], remove_system_prompt=True)` 渲染(含 generation prompt)+ `turn_separator` 还原。
- `rollout.agent.agent_loop_config_path` 指向 yaml → `AgentLoopWorker.__init__`(`:548-553`)在
  **每个 ray worker 进程**里加载并注册自定义 loop(见 §2.4)。

## 2. 踩过的坑(9 个,按出现顺序)

### 2.1 `ModuleNotFoundError: No module named 'transfer_queue'`
- **现象**:verl V1 trainer 在 `main_ppo.py:138` 无条件 `import transfer_queue as tq`,启动即崩。
- **根因**:`transfer_queue` 是独立开源包 **TransferQueue**(`github.com/Ascend/TransferQueue`,
  verl 只 import 不声明依赖),PyPI verl wheel 不带它,verl 源码树里也没有(只在
  `docs/data/transfer_queue.md` 提到)。
- **解**:`pip install TransferQueue`(纯 python,装出 0.1.9,`tq.init/async_kv_put` 都有)。

### 2.2 `FlashAttention2 ... package doesn't seem to be installed`
- **现象**:actor 加载 Qwen3-8B 时报 flash_attention_2 不可用。
- **根因**:`model.py:185` `attn_implementation = override_config.get("attn_implementation", "flash_attention_2")`
  默认 flash_attention_2,而 flash-attn 包没装(编译难、跟不上 torch 2.11/CUDA 13)。
- **解**:`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`(torch 自带 SDPA)。
  注意键是 `override_config.attn_implementation`,不是 `model.attn_implementation`(后者 `+` 加了也
  不被 `HFModelConfig.__init__` 接收,会 `unexpected keyword argument`)。

### 2.3 wandb `No API key configured`
- **现象**:训练即将开始第一步时崩 `wandb.errors.errors.UsageError`。
- **根因**:verl 默认开 wandb(`logging.wandb_enabled`),本机没配 key。
- **解**:`export WANDB_DISABLED=true`(再加 `WANDB_MODE=disabled` 双保险)。

### 2.4 `Agent loop ops_agent not registered`(只在 ray worker 里)
- **现象**:driver 里 `@register("ops_agent")` 注册了,但 `AgentLoopWorkerTQ`(独立 ray 进程)
  报 `registered agent loops: dict_keys(['single_turn_agent', 'tool_agent'])` —— 没有 `ops_agent`。
- **根因**:`@register` 只在 import 它的进程注册;driver import 了我们的模块,但 ray worker 没 import →
  registry 空。
- **解**:建 `configs/agent/ops_agent.yaml`(yaml **list**,每项 `name` + `_target_` FQN),设
  `actor_rollout_ref.rollout.agent.agent_loop_config_path` 指过去。verl 在 `AgentLoopWorker.__init__`
  里 `OmegaConf.load` + 注册;运行时 `hydra.utils.instantiate(_target_=...)` import 我们的模块 →
  `@register` 触发。前提:我们的包 `pip install -e .` 过(env 里 importable,ray worker 继承)。

### 2.5 vLLM `max_model_len (40960) ... KV cache ... larger than available`
- **现象**:vLLM 引擎起不来 `ValueError: ... estimated maximum model length is 14560`。
- **根因**:Qwen3-8B config 默认 `max_seq_len=40960`;colocate 下 vLLM 只拿到少量 KV(<2GB) →
  装不下一个 40960 的请求。
- **解**:`actor_rollout_ref.rollout.max_model_len=8192`(任务实际 prompt 2048+response 4096=6144,够)。

### 2.6 FSDP1 加载期整模 OOM → 切 FSDP2
- **现象**:`FlatParamHandle._get_shard`/`chunk.clone()` 时 OOM,GPU0 已占 23GB。
- **根因**:FSDP1 把 8B 整模先加载到单卡(16GB)+ vLLM 残留 → flat_param clone 再加 → >24GB。
- **解**:`actor_rollout_ref.actor.strategy=fsdp2`。FSDP2 per-param 分片,不做整模 flat clone,
  加载阶段内存友好。verl README 也推荐 FSDP2(更好内存/吞吐)。

### 2.7 3 卡 `TP=3` → 32 heads 不整除
- **现象**:`Total number of attention heads (32) must be divisible by tensor parallel size (3)`。
- **根因**:Qwen3-8B 32 个注意力头,TP 必须整除 32 → 只能 1/2/4/8/16/32,3 不行。
- **结论**:用 3 卡时 vLLM TP 不能取 3。要么 2 卡 TP=2,要么 4 卡 TP=2(rollout)+2(actor)。
  最终走 separate 放置(§4)。

### 2.8 全参 8B actor backward OOM(~1.5GB)→ 改 LoRA
- **现象**:FSDP2 all_gather(全参 16GB)+ 全参梯度(8GB)+ vLLM 残留 1.55GB → 反向 OOM(差 14-96MB)。
  降 `response_length`、`param_offload/optimizer_offload` 都救不了(瓶颈是全参梯度,不是激活)。
- **解**:`actor_rollout_ref.model.lora_rank=32`(+`lora_alpha=64`)。LoRA 冻结 8B base(无全参梯度,
  只有适配器梯度)→ backward 跑通。代价:训的是适配器,非全参(与项目原意不符,但 2 卡上的现实选择)。
- **注意**:LoRA 仍需 `param_offload=true`(否则 actor 整模占 GPU → vLLM 起不来,见 §2.6 同因)。

### 2.9 colocate weight-sync 重叠 OOM(20MB,未解)→ separate 放置
- **现象**:LoRA backward 跑通了,但 `checkpoint_manager.update_weights`(把 actor 权重同步回 vLLM)
  阶段 OOM:`actor 13.1GB + vLLM 10.46GB = 23.5GB > 24GB`,差 20MB。
- **根因**:colocate 下 vLLM 与 actor 同卡,weight-sync 需要**两者同时驻留**(vLLM 要 awake 接权重,
  actor 还在 GPU 上),2×24GB 放不下这个重叠。降 `gpu_memory_utilization` 无用(vLLM 常驻 10.46GB 不变)。
- **结论**:colocate-2卡到此为止。正解是 **separate 放置**(§4)。

## 3. 最终能跑到哪 + 显存账

实测配置(见 `scripts/verl_run_grpo.sh`)能跑到 **actor backward 完成、卡在 weight-sync**:

| 阶段 | 状态 | 备注 |
|------|------|------|
| 配置校验 / Ray init / 数据加载 | ✅ | train 200/val 30 parquet |
| actor+ref 引擎初始化(FSDP2,offload) | ✅ | params on CPU |
| vLLM 引擎启动(TP=2, max_model_len=8192) | ✅ | |
| rollout(真实 Qwen3-8B 跑 Docker 沙盒故障诊断) | ✅ | `finished:N failure:0` |
| 多层奖励计算(in-loop RewardEngine) | ✅ | |
| val 指标 | ✅ | `step:0 val-core/opsagent/reward/mean@1≈9.4, num_turns mean≈11` |
| actor backward(LoRA r=32) | ✅ | 全参会 OOM,LoRA 通过 |
| weight-sync(权重复制回 vLLM) | ❌ OOM 20MB | colocate 同卡重叠放不下 |

显存账(colocate,单卡 24GB):
- vLLM 常驻 ~10.46GB(权重 8B/2 TP=4GB + KV/CUDA graph)
- actor backward:全参 16GB(all-gather)+ 全参梯度 8GB = 24GB(已爆)→ LoRA:16GB(冻结 base,无梯度)+ 适配器梯度(小)≈13GB
- 重叠 10.46+13.1=23.5GB → 差 0.5GB,实测差 20MB(碎片)。

## 4. 下一步:separate 放置(解开 weight-sync OOM)

把 vLLM(rollout)和 actor 放到**不同 GPU**,消除 colocate 同卡重叠:
- vLLM rollout:GPU0+GPU1(TP=2,32 heads ✓),满血 24GB×2
- actor:GPU3 独占(满血 24GB,16GB all-gather + 梯度/激活,LoRA 或全参都装得下)

verl 配置方向(`_generated_ppo_trainer.yaml`):
- `trainer.v1.trainer_mode=separate_async`(用 `PPOTrainerSeparateAsync`)
- `enable_resource_pool` + 给 `actor_rollout_ref.actor` 和 `actor_rollout_ref.rollout` 各分一个
  resource pool(`n_gpus_per_node` 分池:rollout pool=2 GPU、actor pool=1 GPU)
- `CUDA_VISIBLE_DEVICES=0,1,3` 让 verl 只见这 3 张物理卡
- 保持 LoRA r=32(或试着回全参,因为 actor 独占 GPU 后全参也能装)

## 5. 速查

- 单测(不烧 GPU,验证 OpsAgentLoop 接线):`PYTHONPATH=src pytest tests/`(10/10 pass)
- mock smoke(验证 harness):`PYTHONPATH=src python3 -m train --smoke-test --mock --smoke-limit 10`
- 真实 GRPO 启动(当前 LoRA+2卡colocate,会跑到 weight-sync 才 OOM):
  `export MODEL_PATH=/path/to/Qwen3-8B && bash scripts/verl_run_grpo.sh`
- 关键日志:`/tmp/opencode/verl_grpo*.log`;ray dashboard:`http://127.0.0.1:8265`
