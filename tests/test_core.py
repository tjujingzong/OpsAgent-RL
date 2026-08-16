"""Unit tests for the task loader, generator, reward math, and action parser.

These run without Docker / GPUs: `PYTHONPATH=src python3 -m pytest -q`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.task_loader import TaskLoader, load_single_template
from agent.policy import parse_action
from reward.reward_model import group_relative_advantage
from data.dataset import stratified_split
from agent.prompts import build_user_message


HERE = os.path.dirname(__file__)
SCEN = os.path.join(HERE, "..", "src", "env", "scenarios")


def test_all_templates_parse_and_expand():
    tasks = TaskLoader(SCEN).load_tasks()
    assert len(tasks) >= 200, f"expected >=200 expanded tasks, got {len(tasks)}"
    cats = {t.category for t in tasks}
    assert cats == {"service_failure", "misconfiguration", "resource_exhaustion",
                    "network_issues", "security_incidents"}
    # every task has at least one verification criterion + a unique id
    ids = set()
    for t in tasks:
        assert t.verification.criteria, f"{t.task_id} has no verification criteria"
        assert t.task_id not in ids
        ids.add(t.task_id)
        assert t.max_steps == 20


def test_placeholder_substitution():
    tpl = load_single_template(os.path.join(SCEN, "service_failure", "nginx_502.yaml"))
    tasks = tpl.expand()
    ports = {t.params.get("port") for t in tasks}
    assert 8080 in ports and 9090 in ports and 3000 in ports
    # {port} must be substituted in setup commands
    assert all("{port}" not in c for t in tasks for c in t.setup_commands)
    # and in verification commands
    assert all("{port}" not in c.command for t in tasks for c in t.verification.criteria)


def test_difficulty_misdirection_tiers():
    tpl = load_single_template(os.path.join(SCEN, "resource_exhaustion", "mem_hog.yaml"))
    tasks = tpl.expand()
    by_diff = {t.difficulty: t for t in tasks}
    assert len(by_diff["easy"].misdirection) == 0
    assert len(by_diff["medium"].misdirection) == 1
    assert len(by_diff["hard"].misdirection) == 2


def test_parse_action_fenced_and_complete():
    out = "Let me check.\n```bash\nnginx -t\n```\n"
    a = parse_action(out)
    assert a.command == "nginx -t"
    assert not a.is_complete
    b = parse_action("Fixed and verified.\nTASK_COMPLETE")
    assert b.is_complete and b.command is None
    c = parse_action("```bash\nsystemctl status nginx\nTASK_COMPLETE\n```")
    assert c.command == "systemctl status nginx"
    assert c.is_complete


def test_group_relative_advantage_basic():
    adv = group_relative_advantage([1.0, 3.0, 5.0])
    # mean=3, the middle element -> advantage 0
    assert abs(adv[1]) < 1e-6
    assert adv[0] < 0 and adv[2] > 0
    assert abs(sum(adv)) < 1e-6  # zero mean


def test_stratified_split_counts_and_disjoint():
    cats = ["service_failure"] * 20 + ["misconfiguration"] * 20 + ["network_issues"] * 20
    recs = [{"task_id": i, "category": c} for i, c in enumerate(cats)]
    tr, va, te = stratified_split(recs, 30, 10, 10, seed=0)
    assert len(tr) + len(va) + len(te) <= len(recs)
    assert len(te) <= 10 and len(va) <= 10
    ids_tr = {r["task_id"] for r in tr}
    ids_va = {r["task_id"] for r in va}
    ids_te = {r["task_id"] for r in te}
    assert not (ids_tr & ids_te)
    assert not (ids_tr & ids_va)
    assert not (ids_va & ids_te)


def test_build_user_message_contains_scenario_and_checks():
    tpl = load_single_template(os.path.join(SCEN, "network_issues", "iptables_block.yaml"))
    task = tpl.expand(max_variants=1)[0]
    msg = build_user_message(task)
    assert task.description in msg
    assert task.verification.criteria[0].command in msg
