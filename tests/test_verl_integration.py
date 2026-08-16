"""Unit test for OpsAgentLoop: validates token/mask/reward wiring without GPU or Docker.

Uses the REAL Qwen3-8B tokenizer (so apply_chat_template + turn_separator are
real) but mocks the LLM server, the Docker sandbox and the reward engine.
Exercises one bash-command turn followed by a TASK_COMPLETE turn.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, _agent_loop_registry

# importing the module registers "ops_agent"
import verl_integration.ops_agent_loop as ops_mod
from verl_integration.ops_agent_loop import OpsAgentLoop

# Real Qwen3 tokenizer is used to validate token/mask wiring. Set
# OPSAGENT_MODEL_PATH to your local Qwen3-8B dir to exercise this test;
# skipped otherwise (no personal paths hardcoded in the repo).
_MODEL_PATH = os.environ.get("OPSAGENT_MODEL_PATH", "")


def _build_loop():
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 4096,
                    "multi_turn": {
                        "max_assistant_turns": 20,
                        "max_user_turns": 20,
                        "max_tool_response_length": 1024,
                        "tool_response_truncate_side": "middle",
                    },
                    "agent": {"num_workers": 4, "default_agent_loop": "ops_agent"},
                },
                "model": {"path": _MODEL_PATH},
            },
            "data": {
                "apply_chat_template_kwargs": {},
                "mm_processor_kwargs": {},
                "continuous_token": {"enable": False},
            },
            "opsenv": {"image": "opsagent-sandbox:latest"},
        }
    )
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        _MODEL_PATH, trust_remote_code=True
    )

    class MockServer:
        def __init__(self):
            self._script = ["```bash\nnginx -t\n```", "TASK_COMPLETE"]
            self._i = 0

        async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
            text = self._script[self._i]
            self._i += 1
            ids = tokenizer.encode(text, add_special_tokens=False)
            return SimpleNamespace(token_ids=ids, log_probs=None, num_preempted=0, extra_fields={})

    return cfg, tokenizer, MockServer()


def test_ops_agent_loop_registered():
    assert "ops_agent" in _agent_loop_registry
    assert _agent_loop_registry["ops_agent"]["_target_"].endswith("OpsAgentLoop")


def test_run_trajectory_masks_and_reward():
    if not _MODEL_PATH:
        pytest.skip("set OPSAGENT_MODEL_PATH to a Qwen3-8B dir to run this test")
    cfg, tokenizer, server = _build_loop()
    # A minimal task record (matches data/*.jsonl shape).
    task_rec = {
        "task_id": "test_dummy",
        "category": "service_failure",
        "difficulty": "easy",
        "description": "dummy",
        "max_steps": 20,
        "setup_commands": [],
        "inject_fault": [],
        "verification_criteria": [],
        "root_cause_keywords": [],
        "reward_spec": {"success_reward": 10.0, "partial_rewards": [], "step_penalty": -0.1},
        "prompt": [{"role": "system", "content": "You are an SRE."}, {"role": "user", "content": "Fix it."}],
    }

    class FakeResult:
        def __init__(self, obs):
            self.observation = obs
            self.done = False

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def reset(self, task):
            return "INITIAL_STATE"

        def step(self, cmd):
            return FakeResult(f"OUTPUT[{cmd}]")

        def close(self):
            self.closed = True

    class FakeBreakdown:
        total = 9.5

        def as_dict(self):
            return {"total": 9.5, "success": True}

    class FakeEngine:
        def compute(self, env, task, trajectory, steps=None):
            assert trajectory, "trajectory must not be empty"
            assert env.__class__ is FakeEnv
            return FakeBreakdown()

    import asyncio

    async def _go():
        # Construct the loop INSIDE the running event loop so self.loop (captured
        # in AgentLoopBase.__init__ via get_event_loop()) matches the loop we run on.
        with patch.object(ops_mod, "DockerShellEnv", lambda **k: FakeEnv()), patch.object(
            ops_mod, "RewardEngine", FakeEngine
        ):
            loop = OpsAgentLoop(
                trainer_config=DictConfigWrap(cfg),
                server_manager=server,
                tokenizer=tokenizer,
                processor=None,
                dataset_cls=type("DS", (), {}),
                data_config=DictConfigWrap(cfg.data),
            )
            return await loop.run(
                {"temperature": 0.0}, raw_prompt=task_rec["prompt"], extra_info={"task": task_rec}
            )

    out = asyncio.run(_go())

    # --- assertions on the AgentLoopOutput ---
    assert out.reward_score == 9.5
    assert out.metrics is not None
    # response_mask: 1s for assistant tokens, 0s for observation tokens; both present
    assert 1 in out.response_mask and 0 in out.response_mask
    assert sum(out.response_mask) > 0  # at least some assistant tokens
    # prompt_ids = initial prompt; response_ids = everything after
    assert len(out.prompt_ids) > 0
    assert len(out.response_ids) == len(out.response_mask)
    # reward stashed in extra_fields for the fallback reward fn
    assert out.extra_fields["ops_reward"] == 9.5
    assert out.extra_fields["ops_success"] is True
