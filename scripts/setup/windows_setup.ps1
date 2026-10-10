# Windows environment setup for TAPIR with CUDA PyTorch.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup/windows_setup.ps1
#   .\.venv\Scripts\Activate.ps1
#   python -m robot101.legacy.tapnet.tapnetCreate --data-dir data --calib calib/wrist_cam.json --out tasks/pick_place_task.npz

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "../.."))

Write-Host "==> scripts/setup/windows_setup.ps1"

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

# Install tapnet WITHOUT pulling a CPU torch from PyPI.
Write-Host "==> installing local tapnet (editable, --no-deps)"
uv pip install -e .\tapnet --no-deps

Write-Host "==> installing TAPIR requirements"
python scripts/install_requirements.py --profile tapnet --installer uv

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
import einshape
import einops
print('TAPIR dependencies OK')
"@

Write-Host "Package setup complete. Run commands: docs/legacy/TAPNET_COMMANDS.md"
