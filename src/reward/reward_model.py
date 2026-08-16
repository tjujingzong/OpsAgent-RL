"""Multi-level reward engine for system troubleshooting tasks.

Implements the 4-level reward from the planning doc (Section 5.1):

  L1 - task completion   : success_reward (10.0) if ALL verification criteria pass
                           + sum of passing partial_rewards (partial credit).
  L2 - diagnostic quality: 0..+5 proportional to root-cause keyword coverage.
  L3 - efficiency         : step_penalty * steps, capped at -2.0.
  L4 - methodology        : LLM-as-judge bonus (0..+2); off by default (set a
                           judge callback via `methodology_judge`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from env.task_loader import Task
from reward.verifiers import get_verifier, VerificationOutcome

EFFICIENCY_CAP = -2.0
DIAGNOSTIC_MAX = 5.0
METHODOLOGY_MAX = 2.0


@dataclass
class RewardBreakdown:
    l1_task: float
    partial: float
    l2_diagnostic: float
    l3_efficiency: float
    l4_methodology: float
    total: float
    success: bool
    keywords_hit: list[str] = field(default_factory=list)
    outcome: Any = None  # VerificationOutcome

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "l1_task": self.l1_task,
            "partial": self.partial,
            "l2_diagnostic": self.l2_diagnostic,
            "l3_efficiency": self.l3_efficiency,
            "l4_methodology": self.l4_methodology,
            "success": self.success,
            "keywords_hit": self.keywords_hit,
        }


class RewardEngine:
    """Stateless reward engine with access to a Docker sandbox for verification."""

    def __init__(self, methodology_judge: Callable[[list, Task], float] | None = None):
        self.methodology_judge = methodology_judge

    def compute(self, env, task: Task, trajectory: list[dict], steps: int | None = None) -> RewardBreakdown:
        verifier = get_verifier(task.category)
        outcome: VerificationOutcome = verifier.verify_completion(env, task)

        # L1 + partials
        l1 = task.reward_spec.success_reward if outcome.success else 0.0
        partial = outcome.partial_reward

        # L2 diagnostic quality
        l2, hit = verifier.score_diagnostic_quality(trajectory, task, max_score=DIAGNOSTIC_MAX)

        # L3 efficiency
        n = steps if steps is not None else len(trajectory)
        l3 = max(task.reward_spec.step_penalty * n, EFFICIENCY_CAP)

        # L4 methodology (optional LLM judge)
        l4 = 0.0
        if self.methodology_judge is not None:
            try:
                l4 = max(0.0, min(METHODOLOGY_MAX, float(self.methodology_judge(trajectory, task))))
            except Exception:
                l4 = 0.0

        total = l1 + partial + l2 + l3 + l4
        return RewardBreakdown(
            l1_task=l1,
            partial=partial,
            l2_diagnostic=l2,
            l3_efficiency=l3,
            l4_methodology=l4,
            total=total,
            success=outcome.success,
            keywords_hit=hit,
            outcome=outcome,
        )


# GRPO helper: group-relative advantage within a prompt's N rollouts.
def group_relative_advantage(rewards: list[float]) -> list[float]:
    """GRPO advantage: (r - mean) / (std + 1e-8)."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    return [(r - mean) / (std + 1e-8) for r in rewards]
