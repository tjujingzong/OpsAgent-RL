"""Fallback reward function for verl (belt-and-suspenders).

OpsAgentLoop sets `output.reward_score` directly (SWE-agent pattern); verl's
NaiveRewardManager then sees existing `rm_scores` and SKIPS calling compute_score
(verl/.../reward_manager/abstract.py:64-72). So this function is normally never
invoked. It exists only in case a reward path that ignores precomputed rm_scores
is triggered: it returns the reward stashed by the loop in extra_fields, so the
value stays consistent.

Signature verl expects (verl/.../reward_manager/naive.py:131-135):
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
"""
from __future__ import annotations

from typing import Any


def _stash(extra_info: dict[str, Any] | None) -> dict[str, Any]:
    if not extra_info:
        return {}
    # verl may pass the loop's extra_fields under "tool_extra_fields".
    for key in ("tool_extra_fields", "ops_breakdown", "extra_fields"):
        v = extra_info.get(key)
        if isinstance(v, dict):
            return v
    return extra_info


def compute_score(
    data_source: str,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
):
    stash = _stash(extra_info)
    # Prefer the scalar the loop stashed; else read breakdown.total; else 0.0.
    if isinstance(stash.get("ops_reward"), (int, float)):
        return float(stash["ops_reward"])
    bd = stash.get("ops_breakdown")
    if isinstance(bd, dict) and isinstance(bd.get("total"), (int, float)):
        return float(bd["total"])
    return 0.0
