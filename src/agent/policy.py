"""Agent policy: action parsing, conversation history management, and a
stateful wrapper that drives a single episode against the Docker environment.

This module is deliberately framework-agnostic so it can be used:
  * directly in eval/benchmark.py (greedy or sampled rollout against vLLM),
  * adapted into a verl agentic rollout worker for RL training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from env.docker_env import DockerShellEnv, TASK_COMPLETE_TOKEN
from env.task_loader import Task

# Regex to extract the bash command from a model response fenced in ```bash ... ```
_BASH_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_NOFENCE_COMMAND = re.compile(r"^(?:sudo\s+|)\S.*$", re.MULTILINE)

# Keep the last N turns of conversation to control context length.
DEFAULT_HISTORY_TURNS = 16


@dataclass
class AgentAction:
    command: str | None
    is_complete: bool
    raw: str


def parse_action(model_output: str) -> AgentAction:
    """Extract the shell command (if any) and whether the agent declared done."""
    text = model_output or ""
    is_complete = TASK_COMPLETE_TOKEN in text

    m = _BASH_BLOCK.search(text)
    if m:
        command = m.group(1).strip()
    elif is_complete:
        # The model declared completion without issuing a new command.
        command = None
    else:
        # No fenced block: take the first non-empty line that looks like a command.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        command = lines[0] if lines else ""
    # strip a trailing TASK_COMPLETE if it leaked into the command
    command = (command or "").replace(TASK_COMPLETE_TOKEN, "").strip()
    if not command and not is_complete:
        command = None
    return AgentAction(command=command or None, is_complete=is_complete, raw=text)


@dataclass
class ConversationState:
    """A bounded multi-turn conversation history."""

    messages: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = DEFAULT_HISTORY_TURNS

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        # Always keep the system + first user (scenario) messages.
        if len(self.messages) <= 2 + 2 * self.max_turns:
            return
        head = self.messages[:2]
        tail = self.messages[-2 * self.max_turns :]
        self.messages = head + tail

    def to_messages(self) -> list[dict[str, str]]:
        return list(self.messages)


class AgentPolicy:
    """Stateful agent that drives one episode against a Docker environment.

    The model interaction is abstracted through `generate(messages) -> str` so
    this class works with any backend (vLLM, HuggingFace, or an API client).
    """

    def __init__(self, generate_fn, max_turns: int = DEFAULT_HISTORY_TURNS):
        self.generate = generate_fn
        self.max_turns = max_turns

    def run_episode(self, env: DockerShellEnv, task: Task, system_prompt: str) -> dict[str, Any]:
        """Run a full episode. Returns a trajectory summary."""
        from agent.prompts import build_user_message

        state = ConversationState(max_turns=self.max_turns)
        state.append("system", system_prompt)
        state.append("user", build_user_message(task))

        observation = env.reset(task)
        state.append("user", f"Initial state:\n{observation}")

        trajectory: list[dict[str, Any]] = []
        done = False
        steps = 0
        while not done and steps < task.max_steps:
            response = self.generate(state.to_messages())
            action = parse_action(response)
            trajectory.append({"response": response, "action": action.command, "complete": action.is_complete})

            if action.is_complete and not action.command:
                done = True
                break
            if action.command is None:
                state.append("assistant", response)
                state.append("user", "[no command parsed; issue ONE bash command in a ```bash block.]")
                steps += 1
                continue

            result = env.step(action.command)
            state.append("assistant", response)
            state.append("user", result.observation)
            steps += 1
            if result.done:
                done = True

        return {
            "task_id": task.task_id,
            "steps": steps,
            "trajectory": trajectory,
            "messages": state.to_messages(),
            "truncated": steps >= task.max_steps and not done,
        }
