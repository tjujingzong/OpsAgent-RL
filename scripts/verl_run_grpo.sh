#!/bin/bash
# Launch OpsAgent-RL GRPO training via verl on 2xA30 (colocated actor + vLLM rollout).
# Prereqs (done): opsagent conda env has torch/vllm/verl/ray; data/{train,val}.parquet exist.
# Usage: bash scripts/verl_run_grpo.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/opsagent/bin/python
export PYTHONPATH="src:${PYTHONPATH:-}"
export OPSAGENT_IMAGE="${OPSAGENT_IMAGE:-opsagent-sandbox:latest}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1
export ray_temp_dir="${ray_temp_dir:-/tmp/opencode/ray}"

MODEL=/path/to/Qwen3-8B
TRAIN_PQ=data/train.parquet
VAL_PQ=data/val.parquet

# Ensure parquet dataset exists; (re)generate if missing.
if [ ! -f "$TRAIN_PQ" ] || [ ! -f "$VAL_PQ" ]; then
  echo "[run] generating verl parquet from jsonl..."
  "$PY" -m verl_integration.convert_data
fi

# --- Hydra overrides on verl's packaged ppo_trainer config ---
OVERRIDES=(
  trainer.nnodes=1
  trainer.n_gpus_per_node=2
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync

  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false

  actor_rollout_ref.model.path=$MODEL
  actor_rollout_ref.model.tokenizer_path=$MODEL
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=64

  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.optim.lr=1.0e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=30
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.fsdp_config.param_offload=true

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  actor_rollout_ref.rollout.max_model_len=8192
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.data_parallel_size=1
  actor_rollout_ref.rollout.load_format=hf
  actor_rollout_ref.rollout.enable_prefix_caching=true
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.agent.num_workers=4
  actor_rollout_ref.rollout.agent.default_agent_loop=ops_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent/ops_agent.yaml
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20
  actor_rollout_ref.rollout.multi_turn.max_user_turns=20
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle

  reward.reward_model.enable=false
  reward.reward_manager.source=register
  reward.reward_manager.name=naive
  reward.num_workers=4

  data.train_files=[$TRAIN_PQ]
  data.val_files=[$VAL_PQ]
  data.train_batch_size=8
  data.gen_batch_size=8
  data.max_prompt_length=2048
  data.max_response_length=4096
  data.reward_fn_key=data_source
)

if [ "${1:-}" = "--dry-run" ]; then
  echo "[dry-run] would run: $PY -m verl_train --config-name ppo_trainer ${OVERRIDES[*]}"
  exit 0
fi

echo "=== OpsAgent-RL GRPO (verl) ==="
exec "$PY" -m verl_train --config-name ppo_trainer "${OVERRIDES[@]}"
