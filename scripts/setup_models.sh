#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:-python3}
echo "[1/2] installing deps"
if command -v nvidia-smi >/dev/null 2>&1; then
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    || "$PYTHON" -m pip install "torch>=2.7.0"
fi
$PYTHON -m pip install -r requirements.txt
echo "[2/2] warm-up: downloading and loading model weights (YOLO26s + SigLIP2 + Qwen3-VL-2B)"
$PYTHON answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
echo "setup OK"
