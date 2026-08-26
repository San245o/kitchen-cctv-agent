$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host "[1/2] installing deps (CUDA 12.8 wheels for RTX 50-series Blackwell)"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { & $py -m pip install "torch>=2.7.0" }
& $py -m pip install -r requirements.txt
Write-Host "[2/2] warm-up: downloading and loading model weights"
& $py answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
if ($LASTEXITCODE -eq 0) { Write-Host "setup OK" }
