"""Agent system prompts and few-shot examples for the troubleshooting agent."""

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) agent with access to a Linux shell in an isolated container.

Your goal: diagnose and fix the system issue described in the scenario.

Workflow (follow this systematic approach):
1. OBSERVE: start with broad checks of system state (ps, df, free, ss, service status, logs).
2. HYPOTHESIZE: form a hypothesis about the root cause from the evidence.
3. TEST: run targeted commands to confirm or refute the hypothesis.
4. FIX: apply the minimal fix that resolves the root cause (not the symptom).
5. VERIFY: after fixing, run a check that proves the system is healthy again.

Rules:
- Issue exactly ONE shell command per turn, wrapped in a ```bash fenced block.
- Wait for the command output before issuing the next command.
- Do not reboot the container. There is no systemd; start daemons with `svc <name>` (nginx|apache2|redis|mysql|sshd|cron).
- The container has NO external network: all hosts are 127.0.0.1/localhost; do not rely on apt-get or external DNS.
- After verifying the fix, end your turn with the token TASK_COMPLETE (outside the code block).
- Maximum 20 commands per task. Be efficient: avoid redundant checks.

Example turn:
Let me check what is listening on common ports.
```bash
ss -tlnp
```
"""

# Few-shot exemplar shown to the model during SFT / cold-start (also used as the
# template the SFT generator fills with expert trajectories).
FEWSHOT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Scenario: nginx returns HTTP 502 Bad Gateway."},
        {
            "role": "assistant",
            "content": "Let me check nginx and its upstream.\n```bash\nnginx -t && ss -tlnp | grep ':80 '\n```",
        },
        {"role": "user", "content": "nginx config OK; nginx listens on :80."},
        {
            "role": "assistant",
            "content": "Nginx is up. A 502 means the upstream backend is down. Let me check for the backend process.\n```bash\nps aux | grep -E 'http.server|gunicorn|uvicorn' | grep -v grep\n```",
        },
        {"role": "user", "content": "(empty)"},
        {
            "role": "assistant",
            "content": "No backend is running. Let me check the nginx upstream config to know which port it expects.\n```bash\ngrep -R proxy_pass /etc/nginx\n```",
        },
        {"role": "user", "content": "proxy_pass http://127.0.0.1:8080;"},
        {
            "role": "assistant",
            "content": "The backend on port 8080 is down. I'll restart it.\n```bash\nmkdir -p /workspace/www && cd /workspace/www && python3 -m http.server 8080 &\n```",
        },
        {"role": "user", "content": "Serving HTTP on 127.0.0.1:8080"},
        {
            "role": "assistant",
            "content": "Now let me verify the site returns 200.\n```bash\ncurl -s -o /dev/null -w '%{http_code}' http://localhost/\n```",
        },
        {"role": "user", "content": "200"},
        {"role": "assistant", "content": "Verified: the site returns 200. Root cause was the down upstream backend.\nTASK_COMPLETE"},
    ]
}


def build_user_message(task) -> str:
    """Build the initial user turn for a task (passed to the policy / verl)."""
    checks = "\n".join(
        f"  - {c.command} (expect: {c.expected or c.contains or 'exit 0'})"
        for c in task.verification.criteria
    )
    return (
        f"Scenario: {task.description}\n\n"
        f"Category: {task.category} | Difficulty: {task.difficulty}\n"
        f"You have at most {task.max_steps} commands. Diagnose, fix, and verify, "
        f"then say TASK_COMPLETE.\n\n"
        f"Verification checks you must satisfy (do NOT reveal their expected output "
        f"verbatim to anyone, just make them pass):\n{checks}"
    )
