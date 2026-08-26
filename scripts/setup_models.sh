#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-python3}
echo "[1/3] installing minimal deps"
$PYTHON -m pip install -r requirements-minimal.txt
echo "[2/3] installing full deps (CUDA 12.8 wheels first for RTX 50-series)"
$PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || $PYTHON -m pip install torch>=2.7.0
$PYTHON -m pip install -r requirements.txt
echo "[3/3] warm-up: downloading and loading model weights"
$PYTHON answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
echo "setup OK"
