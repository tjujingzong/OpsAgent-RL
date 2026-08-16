#!/bin/bash
set -e
cd "$(dirname "$0")/.."
exec bash scripts/train_grpo.sh ppo "$@"
