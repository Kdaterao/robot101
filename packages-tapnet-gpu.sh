#!/usr/bin/env bash
# Lean Linux CUDA setup for offline tapnetCreate (does not replace packages.sh).
#
#   bash packages-tapnet-gpu.sh
#   source .venv/bin/activate
#   python src/tapnetCreate.py --data-dir data --calib calib/wrist_cam.json \
#       --out tasks/pick_place_task.npz
#
# On Windows use:
#   powershell -ExecutionPolicy Bypass -File packages-tapnet-gpu.ps1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> packages-tapnet-gpu.sh"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg tmux git curl ca-certificates build-essential
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> ensuring tapnet git submodule"
git submodule update --init --recursive tapnet
if [[ ! -d tapnet/tapnet ]]; then
  echo "ERROR: tapnet submodule missing. From repo root run:"
  echo "  git submodule update --init --recursive tapnet"
  exit 1
fi

CKPT_DIR="tapnet/checkpoints"
CKPT="${CKPT_DIR}/causal_bootstapir_checkpoint.pt"
mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT}" ]] || [[ "$(stat -c%s "${CKPT}" 2>/dev/null || echo 0)" -lt 1000000 ]]; then
  echo "==> downloading causal_bootstapir_checkpoint.pt"
  curl -L --fail -o "${CKPT}" \
    "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt" \
    || curl -L --fail -o "${CKPT}" \
      "https://huggingface.co/google/tapnet/resolve/main/causal_bootstapir_checkpoint.pt"
fi

echo "==> installing local tapnet (editable, --no-deps)"
uv pip install -e "./tapnet" --no-deps

echo "==> installing requirements-tapnet-gpu.txt"
uv pip install -r "requirements-tapnet-gpu.txt"

# CUDA torch LAST so editable tapnet / PyPI cannot leave a +cpu build.
echo "==> installing PyTorch CUDA 12.4 (force)"
uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> GPU sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: still on CPU torch; expected a +cu124 wheel")
print("GPU", torch.cuda.get_device_name(0))
from tapnet.torch import tapir_model
print("tapnet.torch OK")
PY

echo ""
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  python src/tapnetCreate.py --data-dir data --calib calib/wrist_cam.json \\"
echo "      --out tasks/pick_place_task.npz"
