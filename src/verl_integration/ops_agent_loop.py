"""Custom verl AgentLoop for OpsAgent-RL: multi-turn shell agent in a Docker sandbox.

Why a custom AgentLoop instead of verl's built-in ToolAgentLoop + BaseTool?
verl's ToolAgentLoop._call_tool creates/releases a BaseTool instance PER tool call
(verl/.../tool_agent_loop.py:519-530), so a stateful sandbox cannot persist across
the 10-20 commands of a single troubleshooting trajectory. It also never calls
BaseTool.calc_reward. Our task needs (a) one persistent fault-injected container
per trajectory and (b) terminal reward via RewardEngine verification in that same
container. So we implement AgentLoopBase.run directly (the SWE-agent pattern,
explicitly supported by verl — see agent_loop.py:745, _compute_score at :966, and
NaiveRewardManager._extract_reward_from_rm_scores which skips compute_score when
the loop already set reward_score -> rm_scores).

Contract (from verl 0.9.0):
  run(sampling_params, **kwargs) -> AgentLoopOutput
    kwargs carries RLHFDataset fields: raw_prompt (list[dict]), extra_info (dict),
    data_source (str), reward_model (dict), ...
  self.server_manager.generate(request_id=, prompt_ids=, sampling_params=,
    image_data=, video_data=, audio_data=, mm_processor_kwargs=) -> TokenOutput
    TokenOutput has .token_ids, .log_probs, .num_preempted, .extra_fields
  self.apply_chat_template(messages, ..., remove_system_prompt=False) -> list[int]
    (hardcodes add_generation_prompt=True)
  self.turn_separator : list[int]  (restored at turn boundaries, see base :265)
  self.rollout_config.prompt_length / response_length
  AgentLoopOutput(prompt_ids, response_ids, response_mask, response_logprobs,
    reward_score, num_turns, metrics=AgentLoopMetrics(), extra_fields=...)
    as_dict() turns reward_score into rm_scores with the reward at the last token.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from agent.policy import parse_action
from agent.prompts import SYSTEM_PROMPT
from data.sft_generator import task_from_record
from env.docker_env import DockerShellEnv
from reward.reward_model import RewardEngine

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))


@register("ops_agent")
class OpsAgentLoop(AgentLoopBase):
    """Multi-turn SRE agent loop over a per-trajectory Docker sandbox.

    Per trajectory: spin up one container, inject the fault, let the policy
    issue bash commands (parsed from fenced ```bash blocks or TASK_COMPLETE),
    feed observations back, and on completion run the multi-level reward engine
    in the same container to score the trajectory.
    """

    def _sandbox_image(self) -> str:
        # Prefer a verl config key (set via `+opsenv.image=...` override),
        # fall back to env var, then the default image.
        try:
            node = self.config.opsenv
            if getattr(node, "image", None):
                return node.image
        except Exception:
            pass
        return os.getenv("OPSAGENT_IMAGE", "opsagent-sandbox:latest")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt: list[dict] = list(kwargs["raw_prompt"])
        extra_info: dict = kwargs.get("extra_info") or {}
        # The dataset converter stashes the full task record under extra_info["task"];
        # fall back to the record itself for robustness.
        task_rec = extra_info.get("task") or extra_info
        task = task_from_record(task_rec)

        image = self._sandbox_image()
        env = DockerShellEnv(image=image, max_steps=task.max_steps)
        engine = RewardEngine()

        metrics = AgentLoopMetrics()
        request_id = uuid4().hex

        # --- Build the initial prompt tokens: system + scenario + initial state ---
        messages: list[dict] = list(raw_prompt)
        # Ensure a system prompt is present (our dataset already includes one).
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        prompt_ids: list[int] = []
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = []
        trajectory: list[dict] = []
        steps = 0
        reward = 0.0
        breakdown: dict[str, Any] = {}

        try:
            init_obs = env.reset(task)
            messages.append({"role": "user", "content": f"Initial state:\n{init_obs}"})
            prompt_ids = await self.apply_chat_template(messages)

            done = False
            while not done and steps < task.max_steps:
                # Stop if the running sequence already exceeds the response budget.
                if len(response_mask) >= self.rollout_config.response_length:
                    break

                # --- Generate one assistant turn ---
                full_ids = prompt_ids + response_ids
                gen = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs=None,
                )
                asst_ids = list(gen.token_ids or [])
                if not asst_ids:
                    break
                asst_lp = list(gen.log_probs or []) if gen.log_probs else []
                # capacity guard before committing
                room = self.rollout_config.response_length - len(response_mask)
                if len(asst_ids) > room:
                    asst_ids = asst_ids[:room]
                    asst_lp = asst_lp[:room]
                response_ids += asst_ids
                response_mask += [1] * len(asst_ids)
                response_logprobs += asst_lp

                asst_text = self.tokenizer.decode(asst_ids, skip_special_tokens=True)
                action = parse_action(asst_text)
                trajectory.append(
                    {"response": asst_text, "action": action.command, "complete": action.is_complete}
                )
                steps += 1

                if action.is_complete and not action.command:
                    done = True
                    break

                # --- Run the command (or nudge) and render the observation ---
                if action.command is None:
                    obs_text = "[no command parsed; issue ONE bash command in a ```bash block.]"
                else:
                    result = env.step(action.command)
                    obs_text = result.observation
                    if result.done:
                        done = True

                obs_msg = {"role": "user", "content": obs_text}
                messages.append({"role": "assistant", "content": asst_text})
                messages.append(obs_msg)

                obs_ids = await self.apply_chat_template([obs_msg], remove_system_prompt=True)
                obs_ids = list(self.turn_separator) + list(obs_ids)
                room = self.rollout_config.response_length - len(response_mask)
                if len(obs_ids) > room:
                    obs_ids = obs_ids[:room]
                response_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                response_logprobs += [0.0] * len(obs_ids)

            # --- Terminal reward: run the multi-level reward engine in-sandbox ---
            bd = engine.compute(env, task, trajectory, steps=steps)
            reward = float(bd.total)
            breakdown = bd.as_dict()
        except Exception as e:  # pragma: no cover - surfaced in metrics for debugging
            logger.warning("OpsAgentLoop trajectory failed: %s", e)
            breakdown = {"success": False, "total": 0.0, "error": str(e)}
            reward = 0.0
        finally:
            try:
                env.close()
            except Exception:
                pass

        response_length = self.rollout_config.response_length
        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:response_length],
            response_mask=response_mask[:response_length],
            response_logprobs=(response_logprobs[:response_length] if response_logprobs else None),
            reward_score=reward,
            num_turns=steps * 2 + 1,
            metrics=metrics,
            extra_fields={
                "ops_task_id": task.task_id,
                "ops_category": task.category,
                "ops_success": breakdown.get("success", False),
                "ops_reward": reward,
                "ops_breakdown": breakdown,
                # verl's AgentLoopWorkerTQ builds batch.tags from these two fields
                # (agent_loop_tq.py:216-218): tag["min/max_global_steps"] = extra_fields.get(...).
                # If absent they're None and _compute_metrics np.array(...,dtype=int) crashes
                # (trainer_base.py:1742-1743). For on-policy sync training both = current step.
                "min_global_steps": 0,
                "max_global_steps": 0,
            },
        )
        return out
