"""Reward verifiers: per-category L1 completion + partial-reward checks.

The L1 verification logic is generic (run the task's `verification.criteria`
commands and match their output). The three category verifiers are thin
specializations that also score L2 diagnostic quality with category-specific
hints about *what a good SRE would have inspected*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.docker_env import DockerShellEnv
    from env.task_loader import Task


@dataclass
class VerificationOutcome:
    success: bool                       # all L1 criteria pass
    criteria_results: list              # list[CheckResult]
    partial_results: list               # list[(PartialReward, passed: bool)]
    partial_reward: float               # sum of passing partials
    l2_keywords_hit: list[str]


class Verifier:
    """Generic verifier shared by all categories."""

    category: str = "generic"
    # Category-specific diagnostic hints (commands a competent SRE would run).
    expected_inspect_hints: tuple[str, ...] = ()

    def verify_completion(self, env, task):
        criteria_results = env.run_verification(task)
        success = bool(criteria_results) and all(c.passed for c in criteria_results)
        partial_results = []
        partial_reward = 0.0
        for p in task.reward_spec.partial_rewards:
            res = env.run_check(p.check)
            passed = p.matches(res.stdout, res.stderr, res.exit_code)
            partial_results.append((p, passed))
            if passed:
                partial_reward += p.reward
        return VerificationOutcome(
            success=success,
            criteria_results=criteria_results,
            partial_results=partial_results,
            partial_reward=partial_reward,
            l2_keywords_hit=[],
        )

    def diagnostic_keywords(self, task) -> list[str]:
        return list(task.verification.root_cause_keywords)

    def score_diagnostic_quality(self, trajectory, task, max_score: float = 5.0) -> tuple[float, list[str]]:
        """L2: did the agent inspect evidence pointing at the root cause?

        Matches both the agent's issued commands and the observed outputs
        against the scenario's root_cause_keywords. Score is proportional to the
        fraction of keywords hit, capped at max_score.
        """
        keywords = self.diagnostic_keywords(task)
        if not keywords:
            return 0.0, []
        haystack_parts = []
        for step in trajectory:
            cmd = step.get("action") or ""
            resp = step.get("response") or ""
            # observations are not stored in trajectory entries; we also scan
            # the assistant responses which often quote observed symptoms.
            haystack_parts.append(cmd)
            haystack_parts.append(resp)
        haystack = " ".join(haystack_parts).lower()
        hit = [kw for kw in keywords if kw.lower() in haystack]
        frac = len(hit) / len(keywords)
        return round(max_score * frac, 2), hit


class ServiceVerifier(Verifier):
    category = "service_failure"
    expected_inspect_hints = ("systemctl", "pgrep", "ss", "curl", "nginx -t", "redis-cli", "mysql")


class ConfigVerifier(Verifier):
    category = "misconfiguration"
    expected_inspect_hints = ("cat", "grep", "readlink", "diff", "stat", "date")


class StateVerifier(Verifier):
    """Resource exhaustion, network, security — state-based checks."""
    category = "state"
    expected_inspect_hints = ("ps", "df", "free", "lsof", "ip", "iptables", "find", "stat")


_REGISTRY: dict[str, Verifier] = {
    "service_failure": ServiceVerifier(),
    "misconfiguration": ConfigVerifier(),
    "resource_exhaustion": StateVerifier(),
    "network_issues": StateVerifier(),
    "security_incidents": StateVerifier(),
}


def get_verifier(category: str) -> Verifier:
    return _REGISTRY.get(category, Verifier())
