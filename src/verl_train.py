"""verl GRPO entry: register OpsAgentLoop, then delegate to verl's hydra main.

This module exists so the `ops_agent` loop is imported (and thus registered with
verl's _agent_loop_registry via the @register decorator) before verl's hydra
runtime instantiates agent loops. Invoke with:

    PYTHONPATH=src python -m verl_train --config-name ppo_trainer <overrides...>

All hydra overrides are read from sys.argv by verl.trainer.main_ppo.main
(decorated with @hydra.main). scripts/verl_run_grpo.sh wraps this.
"""
from __future__ import annotations

from verl_integration import ops_agent_loop  # noqa: F401  side-effect: registers "ops_agent"
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
