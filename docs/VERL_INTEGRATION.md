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

## 2. 踩过的坑(11 个,按出现顺序)

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

### 2.9 colocate weight-sync 重叠 OOM(20MB,8B 未解)→ 换 4B
- **现象**:LoRA backward 跑通了,但 `checkpoint_manager.update_weights`(把 actor 权重同步回 vLLM)
  阶段 OOM:`actor 13.1GB + vLLM 10.46GB = 23.5GB > 24GB`,差 20MB。
- **根因**:colocate 下 vLLM 与 actor 同卡,weight-sync 需要**两者同时驻留**(vLLM 要 awake 接权重,
  actor 还在 GPU 上),2×24GB 放不下这个重叠。降 `gpu_memory_utilization` 无用(vLLM 常驻 10.46GB 不变)。
- **解**:8B 在 2×24GB colocate 走不通。换成 **Qwen3.5-4B**(见 2.10/2.11)后峰值降到 ~10-15GB,宽裕。

### 2.10 4B 全参:`optimizer_offload/param_offload/offload_policy` 对 FSDP2 + qwen3_5 不生效
- **现象**:4B 全参 actor backward 仍 OOM,actor 占 22.7GB(fp32 AdamW 状态堆在 GPU 上)。
- **根因**:FSDP2 + qwen3_5 这个组合下,FSDP1 的 `optimizer_offload/param_offload` 键和 FSDP2 的
  `offload_policy=True` 都**没把 optimizer 真正卸到 CPU**(改前改后都是 22.70GB)。
- **解**:用 **LoRA**(r=32)。冻结 4B base(无全参梯度、无大 optimizer 状态),只剩适配器梯度 →
  backward 峰值 ~10GB。这绕开了 offload 失效问题。

### 2.11 `_compute_metrics` 崩 `min_global_steps` None(TypeError)
- **现象**:4B+LoRA 的 actor update + weight-sync 都跑通了,却在训练步**完成后**的
  `_compute_metrics` 崩:`TypeError: int() argument ... not 'NoneType'`。
- **根因**:`agent_loop_tq.py:216-218` 把 `batch.tags["min_global_steps"]` /
  `["max_global_steps"]` 取自 `output.extra_fields.get("min/max_global_steps")`。我们的 OpsAgentLoop
  没设这两个 key → None → `trainer_base.py:1742` 的 `np.array(..., dtype=int)` 崩。
- **解**:在 `OpsAgentLoop` 的 `AgentLoopOutput.extra_fields` 里补 `min_global_steps=0`、
  `max_global_steps=0`(on-policy sync 训练下都是当前步)。补完即出 step:1。

## 3. 最终能跑到哪 + 显存账

> **更新 2026-08-17**:换 Qwen3.5-4B + LoRA(r=32)后,**step:1 完整跑通**(rollout→reward→
> actor update→weight-sync→metrics),GRPO 正式训练中。8B 的 colocate weight-sync OOM 是
> 模型太大;4B + LoRA 在 2×A30 上峰值仅 ~10GB,宽裕。

实测能跑到 step:1(且持续)的配置(`scripts/verl_run_grpo.sh`):

| 项 | 值 |
|----|----|
| 模型 | Qwen3.5-4B(`/root/ljz/Qwen3.5-4B`,ModelScope 下,`$MODEL`) |
| 算法 | GRPO(n=4, group-relative,no critic) |
| actor | FSDP2 + **LoRA r=32/α=64** + `param_offload=true`(base 卸 CPU)+ sdpa attn |
| rollout | vLLM TP=2 + `enforce_eager=true` + `max_model_len=8192` + `gpu_memory_utilization=0.45` |
| batch | train_batch_size=8 × n=4 = 32 响应/步,ppo_mini_batch_size=8 |
| 卡 | GPU0/1 colocate(`CUDA_VISIBLE_DEVICES=0,1`) |

step:1 实测指标:
- `actor/perf/max_memory_allocated_gb: 9.997`(峰值 10GB,24GB 卡余量充足)
- `timing_s/gen: 341`(rollout:32 条轨迹 × ~33 轮命令)
- `timing_s/update_actor: 159`、`timing_s/update_weights: 4.1`、`timing_s/step: 541`(约 9 分钟/步)
- `critic/rewards/mean: 11.08`、`actor/pg_loss: 0.052`、`actor/grad_norm: 0.175`、`actor/entropy: 0.621`
- `training/off_policy/trajectory_staleness: 0`(on-policy,正确)
- 750 步 ETA ≈ 113h(慢主要在多轮 rollout)

显存账(colocate,单卡 24GB):
- vLLM 常驻 ~872MB(4B/2 TP 权重 ~2GB,enforce_eager 无 graph)
- actor backward(LoRA):冻结 4B base all-gather ~8GB + 适配器梯度(小)≈ 10GB → 远低于 24GB
- weight-sync:vLLM ~5GB + actor ~10GB ≈ 15GB < 24GB(8B 时是 23.5GB 爆掉)

### 为什么 8B 不行、4B 行
8B 全参:actor backward 全参梯度 ~8GB → 16+8+1.55(vllm)≈ 23.5GB,差 20MB;weight-sync
vLLM(8.7GB)+actor(13-14.8GB)= 23.5GB 也差 20MB。2×24GB colocate 放不下 8B。
4B + LoRA:无全参梯度(LoRA),base 仅 8GB all-gather,vLLM 仅 ~5GB → 各阶段峰值 ~10-15GB,宽裕。

## 4. 下一步 / 调优方向

- **速度**:当前 ~541s/step(rollout 32 条 × 33 轮占 341s)。可调:`max_assistant_turns` 降到 12、
  `n=4`→`n=2`(GRPO group 变小)、rollout `agent.num_workers` 调高(更多并行容器)。
- **奖励信号偏弱**:`rewards/mean≈11`、`advantages/mean≈0`(组内奖励方差小)→ GRPO 学习信号弱。
  需调 `RewardEngine`:收紧 partial_rewards(别在没修复时也给),拉开 success(10)与失败的差距。
- **回全参**:若腾出 4 卡(GPU0/1 vLLM + GPU2/3 actor separate),4B 全参应能装(无 LoRA,offload
  也不再必需)。verl V1 的 `ActorRolloutRefWorker` 是 colocate 设计,separate 放置需自定义 worker。
- **DAPO/PPO**:复用同一 OpsAgentLoop,换 `algorithm.adv_estimator`(dapo/rloo)+ `policy_loss.loss_mode`。

## 5. 速查

- 单测(不烧 GPU,验证 OpsAgentLoop 接线):`PYTHONPATH=src pytest tests/`(10/10 pass)
- mock smoke(验证 harness):`PYTHONPATH=src python3 -m train --smoke-test --mock --smoke-limit 10`
- 真实 GRPO 启动(4B+LoRA,2 卡 colocate,已跑通 step:1):
  `export MODEL=/path/to/Qwen3.5-4B && bash scripts/verl_run_grpo.sh`
- 关键日志:`/tmp/opencode/verl_grpo*.log`;ray dashboard:`http://127.0.0.1:8265`
