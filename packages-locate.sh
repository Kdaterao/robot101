#!/usr/bin/env bash
# Linux CUDA setup for SO-101 LocateAnything / Florence-2 / locate_gripper / HF preprocess.
# Does not replace packages.sh or packages-tapnet-gpu.sh.
#
#   bash packages-locate.sh
#   source .venv/bin/activate
#   hf auth login
#   python src/locate_collect_label.py batch
#   python src/testing2.py --locator florence2 --prompt "SO-101 gripper"
#   python src/hf_preprocess_smolvla.py --episodes 0-2 --dry-run
#
# Docs: src/so101Locate.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> packages-locate.sh (LocateAnything + Florence-2 + locate_gripper)"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    ffmpeg tmux git curl ca-certificates build-essential \
    libgl1 libglib2.0-0
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "==> installing Hugging Face CLI"
  curl -LsSf https://hf.co/cli/install.sh | bash || true
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> ensuring git submodules (tapnet, lerobot if present)"
git submodule update --init --recursive || true

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

echo "==> installing PyTorch CUDA 12.4"
uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> installing local tapnet + lerobot (editable, --no-deps)"
uv pip install -e "./tapnet" --no-deps
if [[ -f lerobot/pyproject.toml ]] || [[ -f lerobot/setup.py ]]; then
  uv pip install -e "./lerobot" --no-deps
fi

echo "==> installing requirements-locate.txt"
uv pip install -r "requirements-locate.txt"

mkdir -p data/locate_gripper/images work_dirs

echo "==> sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: still on CPU torch; expected a +cu124 wheel")
print("GPU", torch.cuda.get_device_name(0))

import cv2  # noqa: F401
import transformers
from PIL import Image  # noqa: F401
print("transformers", transformers.__version__)
from tapnet.torch import tapir_model  # noqa: F401
print("tapnet.torch OK")

# Workers import without loading weights.
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from locateanything_worker import LocateAnythingWorker  # noqa: F401
from florence2_worker import Florence2Worker, DEFAULT_FLORENCE_MODEL  # noqa: F401
print("locate workers OK; Florence default:", DEFAULT_FLORENCE_MODEL)
PY

echo ""
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  hf auth login"
echo ""
echo "Label + push locate_gripper:"
echo "  python src/locate_collect_label.py batch"
echo "  python src/locate_push_hf.py"
echo "  # or: python src/locate_finetune.py export --push-to-hub"
echo ""
echo "Live grounding A/B:"
echo "  python src/testing2.py --prompt \"SO-101 gripper\""
echo "  python src/testing2.py --locator florence2 --prompt \"SO-101 gripper\""
echo ""
echo "HF SmolVLA preprocess:"
echo "  python src/hf_preprocess_smolvla.py --episodes 0-2 --dry-run"
echo "  python src/hf_preprocess_smolvla.py --episodes 0-99 --locator florence2 --resume"
echo ""
echo "See src/so101Locate.md"
