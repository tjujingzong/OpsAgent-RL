"""verl integration: registers a custom multi-turn OpsAgentLoop that drives the
Docker sandbox + multi-level reward engine inside verl's rollout.

Importing this package (or ops_agent_loop) registers the "ops_agent" agent loop
with verl's _agent_loop_registry so verl's hydra `default_agent_loop: ops_agent`
can instantiate it.
"""
