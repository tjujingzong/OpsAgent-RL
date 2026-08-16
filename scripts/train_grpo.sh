#!/bin/bash
set -e
cd "$(dirname "$0")/.."

ALGO="${1:-grpo}"
CONFIG="configs/train/${ALGO}.yaml"
EXTRA="${@:2}"

echo "=== OpsAgent-RL training: ${ALGO} ==="

export PYTHONPATH="src:${PYTHONPATH:-}"
python3 -m train --config-name "${ALGO}" ${EXTRA}
