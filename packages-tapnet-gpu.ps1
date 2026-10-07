# Lean Windows setup for tapnetCreate with CUDA torch.
#
#   powershell -ExecutionPolicy Bypass -File packages-tapnet-gpu.ps1
#   .\.venv\Scripts\Activate.ps1
#   python src/tapnetCreate.py --data-dir data --calib calib/wrist_cam.json --out tasks/pick_place_task.npz

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> packages-tapnet-gpu.ps1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Test-Path .venv)) {
    uv venv --python 3.12
}
.\.venv\Scripts\Activate.ps1

Write-Host "==> ensuring tapnet git submodule"
git submodule update --init --recursive tapnet
if (-not (Test-Path "tapnet\tapnet")) {
    throw "tapnet submodule missing. Run: git submodule update --init --recursive tapnet"
}

$ckptDir = "tapnet\checkpoints"
$ckpt = Join-Path $ckptDir "causal_bootstapir_checkpoint.pt"
New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
$needCkpt = -not (Test-Path $ckpt) -or ((Get-Item $ckpt).Length -lt 1000000)
if ($needCkpt) {
    Write-Host "==> downloading causal_bootstapir_checkpoint.pt"
    $urls = @(
        "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt",
        "https://huggingface.co/google/tapnet/resolve/main/causal_bootstapir_checkpoint.pt"
    )
    $ok = $false
    foreach ($url in $urls) {
        try {
            curl.exe -L --fail -o $ckpt $url
            if ((Test-Path $ckpt) -and ((Get-Item $ckpt).Length -gt 1000000)) {
                $ok = $true
                break
            }
        } catch {
            Write-Host "download failed from $url"
        }
    }
    if (-not $ok) { throw "Failed to download BootsTAPIR checkpoint" }
}

# Install tapnet WITHOUT pulling a CPU torch from PyPI.
Write-Host "==> installing local tapnet (editable, --no-deps)"
uv pip install -e .\tapnet --no-deps

Write-Host "==> installing requirements-tapnet-gpu.txt"
uv pip install -r requirements-tapnet-gpu.txt

# CUDA torch LAST so nothing overwrites it with +cpu.
Write-Host "==> installing PyTorch CUDA 12.4 (force reinstall)"
uv pip uninstall torch torchvision torchaudio 2>$null
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "==> GPU sanity check"
python -c @"
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        'ERROR: still on CPU torch. Check nvidia-smi and that the wheel is +cu124, not +cpu.'
    )
print('GPU', torch.cuda.get_device_name(0))
from tapnet.torch import tapir_model
print('tapnet.torch OK')
"@

Write-Host ""
Write-Host "Setup complete."
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python src/tapnetCreate.py --data-dir data --calib calib/wrist_cam.json --out tasks/pick_place_task.npz"
