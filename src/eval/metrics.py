"""OpsBench evaluation metrics (planning doc Section 7.2)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def success_rate(results: list[dict]) -> float:
    """Fraction of episodes whose L1 task-completion reward is positive."""
    if not results:
        return 0.0
    n = sum(1 for r in results if r.get("reward", {}).get("success"))
    return n / len(results)


def diagnostic_accuracy(results: list[dict], threshold: float = 3.0) -> float:
    """Fraction of episodes whose L2 diagnostic score >= threshold."""
    if not results:
        return 0.0
    n = sum(1 for r in results if r.get("reward", {}).get("l2_diagnostic", 0) >= threshold)
    return n / len(results)


def mean_steps_to_resolution(results: list[dict]) -> float:
    """Average number of steps among successful episodes."""
    succ = [r for r in results if r.get("reward", {}).get("success")]
    if not succ:
        return 0.0
    return sum(r.get("steps", 0) for r in succ) / len(succ)


def command_efficiency(results: list[dict]) -> float:
    """Effective-command ratio: valid commands / total commands issued."""
    total = 0
    valid = 0
    for r in results:
        for step in r.get("trajectory", []):
            total += 1
            if step.get("action"):
                valid += 1
    return (valid / total) if total else 0.0


def pass_at_k(results_by_scenario: dict[str, list[dict]], k: int) -> float:
    """pass@k: fraction of scenarios with at least one success in k samples."""
    if not results_by_scenario:
        return 0.0
    passed = 0
    for sid, runs in results_by_scenario.items():
        sample = runs[:k]
        if any(r.get("reward", {}).get("success") for r in sample):
            passed += 1
    return passed / len(results_by_scenario)


def aggregate(results: list[dict], pass_k: list[int] | None = None) -> dict[str, Any]:
    """Compute the full OpsBench metric suite."""
    out: dict[str, Any] = {
        "n_scenarios": len(results),
        "success_rate": round(success_rate(results), 4),
        "diagnostic_accuracy": round(diagnostic_accuracy(results), 4),
        "mean_steps_to_resolution": round(mean_steps_to_resolution(results), 2),
        "command_efficiency": round(command_efficiency(results), 4),
        "mean_reward": round(sum(r.get("reward", {}).get("total", 0) for r in results) / max(1, len(results)), 4),
    }
    # pass@k: group by task_id (each scenario sampled k times)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r.get("task_id", "?")].append(r)
    for k in pass_k or [1]:
        out[f"pass@{k}"] = round(pass_at_k(by_task, k), 4)
    # per-category breakdown
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r.get("category", "?")].append(r)
    out["by_category"] = {
        cat: {
            "n": len(rs),
            "success_rate": round(success_rate(rs), 4),
            "diagnostic_accuracy": round(diagnostic_accuracy(rs), 4),
            "mean_reward": round(sum(r.get("reward", {}).get("total", 0) for r in rs) / max(1, len(rs)), 4),
        }
        for cat, rs in by_cat.items()
    }
    return out
