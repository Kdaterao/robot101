#!/usr/bin/env bash
set -euo pipefail

# Fresh Ubuntu setup for Molmo2-4B SO-101 training/preprocessing on one RTX A6000.
# Run from the robot101 repository root:
#   bash packages-molmo2-so101.sh
# Then:
#   source .venv-molmo2/bin/activate
#   hf auth login
#   bash scripts/prepare_molmo2_so101_data.sh
# Run scripts/train_molmo2_so101.sh only when you intend to start a new fine-tune.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script is for the Ubuntu Linux VM." >&2
  exit 2
fi

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential ca-certificates curl ffmpeg git python3.12 \
    python3.12-dev python3.12-venv
fi

if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "${ROOT}/.venv-molmo2/bin/python" ]]; then
  uv venv --seed --python 3.12 "${ROOT}/.venv-molmo2"
fi

VENV_PYTHON="${ROOT}/.venv-molmo2/bin/python"
uv pip install --python "${VENV_PYTHON}" pip
MOLMO2_WORK="${SO101_MOLMO2_WORKDIR:-${ROOT}/molmo2_so101_run}"
MOLMO2_REPO="${MOLMO2_REPO_ROOT:-${MOLMO2_WORK}/molmo2}"
MOLMO2_REF="${SO101_MOLMO2_REF:-f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a}"

if [[ ! -f "${MOLMO2_REPO}/launch_scripts/sft.py" ]]; then
  mkdir -p "$(dirname "${MOLMO2_REPO}")"
  git clone --depth 1 https://github.com/allenai/molmo2.git "${MOLMO2_REPO}"
  git -C "${MOLMO2_REPO}" fetch --depth 1 origin "${MOLMO2_REF}"
  git -C "${MOLMO2_REPO}" checkout --detach FETCH_HEAD
fi

echo "Installing CUDA 13.0 PyTorch for the RTX A6000..."
uv pip install --python "${VENV_PYTHON}" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130

# Molmo2 SO-101 training uses images only. CPU TorchCodec avoids CUDA NPP
# runtime-library requirements while retaining FFmpeg-backed CPU decoding.
uv pip install --python "${VENV_PYTHON}" --reinstall --no-deps \
  torchcodec==0.12.0 --index-url https://download.pytorch.org/whl/cpu

echo "Installing the pinned Molmo2 training stack (includes datasets and HF Hub)..."
uv pip install --python "${VENV_PYTHON}" -e "${MOLMO2_REPO}[train]"

echo "Checking CUDA and required imports..."
"${VENV_PYTHON}" - <<'PY'
import torch
import datasets
import huggingface_hub
import torchcodec

print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check that the Thunder Compute VM has its NVIDIA driver installed.")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {name}; VRAM: {memory:.1f} GiB")
if "A6000" not in name:
    print("Warning: this VM is not reporting an RTX A6000.")
print(f"datasets: {datasets.__version__}; huggingface_hub: {huggingface_hub.__version__}")
print(f"torchcodec: {torchcodec.__version__}; import OK")
PY

echo
echo "Package setup complete. Next run:"
echo "  source .venv-molmo2/bin/activate"
echo "  hf auth login"
echo "  bash scripts/prepare_molmo2_so101_data.sh"
echo "To start a new fine-tune, run: bash scripts/train_molmo2_so101.sh"
