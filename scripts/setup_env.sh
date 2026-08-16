#!/bin/bash
set -e

echo "=== OpsAgent-RL Environment Setup ==="

# 1. Python virtual environment (skip if already inside one)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# 2. Install project (core deps first, train deps optional for GPU nodes)
pip install -e ".[dev]"
if [ "${WITH_TRAIN_DEPS:-0}" = "1" ]; then
    echo "Installing training dependencies (torch/vllm/verl)..."
    pip install -e ".[train]"
fi

# 3. Build Docker sandbox image
echo "Building Docker sandbox image..."
docker build -f docker/Dockerfile.base -t opsagent-sandbox:latest .

# 4. Download RCAEval RE1 seed data
echo "Downloading RCAEval RE1 dataset (390MB)..."
bash scripts/download_datasets.sh

echo "=== Setup complete! ==="
