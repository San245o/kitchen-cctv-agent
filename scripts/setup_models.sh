#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-python3}
echo "[1/2] installing deps"
$PYTHON -m pip install -r requirements.txt
echo "[2/2] warm-up: downloading and loading model weights (YOLO11n + Qwen3-VL-2B)"
$PYTHON answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
echo "setup OK"
