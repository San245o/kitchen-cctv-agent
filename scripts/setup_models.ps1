$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host "[1/2] installing deps (CUDA 12.8 wheels for RTX 50-series Blackwell)"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA wheel install failed; trying the default package index"
    & $py -m pip install "torch>=2.7.0"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
}
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
Write-Host "[2/2] warm-up: downloading and loading model weights"
& $py answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
if ($LASTEXITCODE -ne 0) { throw "Model warm-up failed" }
Write-Host "setup OK"
