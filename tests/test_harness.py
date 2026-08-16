"""End-to-end harness test using MockShellEnv (no Docker / GPU required).

Validates the full pipeline: task -> agent policy (scripted backend) ->
MockShellEnv -> RewardEngine -> metrics, for both a fixing and a no-op agent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.mock_env import MockShellEnv
from env.task_loader import load_single_template
from agent.policy import AgentPolicy
from agent.prompts import SYSTEM_PROMPT
from reward.reward_model import RewardEngine
import eval.metrics as metrics

SCEN = os.path.join(os.path.dirname(__file__), "..", "src", "env", "scenarios")


class _ScriptedBackend:
    """Issues a fixed list of commands, then declares TASK_COMPLETE."""

    def __init__(self, commands: list[str]):
        self._cmds = list(commands)

    def generate(self, messages):
        if self._cmds:
            cmd = self._cmds.pop(0)
            return f"Running: {cmd}\n```bash\n{cmd}\n```"
        return "Done.\nTASK_COMPLETE"


def _run(task, backend):
    env = MockShellEnv(max_steps=task.max_steps)
    policy = AgentPolicy(backend.generate, max_turns=task.max_steps)
    summary = policy.run_episode(env, task, SYSTEM_PROMPT)
    engine = RewardEngine()
    # use the docker env's recorded issued commands via the same env instance
    bd = engine.compute(env, task, summary["trajectory"], steps=summary["steps"])
    return summary, bd


def test_fixing_agent_succeeds_noxop_agent_fails():
    task = load_single_template(os.path.join(SCEN, "service_failure", "nginx_502.yaml")).expand()[0]
    # fix: restart the backend http server on the task's port
    port = task.params.get("port", 8080)
    fix_cmds = [
        f"python3 -m http.server {port}",
        "curl -s -o /dev/null -w '%{http_code}' http://localhost",
    ]
    summary_fix, bd_fix = _run(task, _ScriptedBackend(fix_cmds))
    summary_noop, bd_noop = _run(task, _ScriptedBackend([]))

    assert bd_fix.success, f"fixing agent should succeed, reward={bd_fix.as_dict()}"
    assert not bd_noop.success, "no-op agent should fail"
    assert bd_fix.total > bd_noop.total
    # L3 efficiency: fix agent used steps; penalty applied (non-positive L3)
    assert bd_fix.l3_efficiency <= 0
    # metrics aggregate runs without error and reports SR
    results = [
        {"task_id": task.task_id, "category": task.category, "difficulty": task.difficulty,
         "steps": summary_fix["steps"], "trajectory": summary_fix["trajectory"], "reward": bd_fix.as_dict()},
        {"task_id": task.task_id, "category": task.category, "difficulty": task.difficulty,
         "steps": summary_noop["steps"], "trajectory": summary_noop["trajectory"], "reward": bd_noop.as_dict()},
    ]
    rep = metrics.aggregate(results, pass_k=[1])
    assert rep["n_scenarios"] == 2
    assert 0.0 <= rep["success_rate"] <= 1.0
