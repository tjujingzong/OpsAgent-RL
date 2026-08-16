#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "=== Generating OpsAgent-RL dataset ==="

python3 -m data.generator --templates src/env/scenarios \
    --out data \
    --train 200 --val 30 --test 100 \
    --seed 42

echo "Generating SFT trajectories (requires API or teacher model)..."
python3 -m src.data.sft_generator --train-file data/train.jsonl \
    --out data/sft.jsonl --num-per-scenario 3 || \
    echo "[warn] SFT generation skipped (set OPSAGENT_TEACHER_API to enable)"

echo "=== Dataset generation complete ==="
