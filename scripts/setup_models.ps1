$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host "[1/3] installing minimal deps"
& $py -m pip install -r requirements-minimal.txt
Write-Host "[2/3] installing full deps (CUDA 12.8 wheels first for RTX 50-series)"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { & $py -m pip install "torch>=2.7.0" }
& $py -m pip install -r requirements.txt
Write-Host "[3/3] warm-up: downloading and loading model weights"
& $py answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
if ($LASTEXITCODE -eq 0) { Write-Host "setup OK" }
