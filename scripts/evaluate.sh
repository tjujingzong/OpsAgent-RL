#!/bin/bash
set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${1:-checkpoints/grpo}"
EXTRA="${@:2}"

echo "=== OpsAgent-RL evaluation on OpsBench ==="

export PYTHONPATH="src:${PYTHONPATH:-}"
python3 -m eval.benchmark --config configs/eval/opsbench.yaml \
    --model-path "${MODEL_PATH}" ${EXTRA}
