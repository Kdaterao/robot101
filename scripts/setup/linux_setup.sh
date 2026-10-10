#!/usr/bin/env bash
# Ubuntu environment setup. Processing/training commands live in docs/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PROFILE=grounded
usage() {
  cat <<'HELP'
Usage: bash scripts/setup/linux_setup.sh [--profile grounded|molmo2|robot|all]

  grounded  Molmo2 inference + TAPIR + video preprocessing (default; .venv-grounded)
  molmo2    Molmo2 connector training dependencies (.venv-molmo2)
  robot     Robot control and SmolVLA dependencies (.venv)
  all       Prepare all three isolated environments

Set SKIP_APT=1 to skip Ubuntu system packages when already installed.
Model checkpoint downloads and job execution are documented in docs/.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  grounded|molmo2|robot|all) ;;
  *) echo "Unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Use scripts/setup/windows_setup.ps1 on Windows.' >&2
  exit 2
fi
if [[ "${SKIP_APT:-0}" != 1 ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo 'This installer targets Ubuntu. Install system dependencies and set SKIP_APT=1.' >&2
    exit 2
  fi
  sudo apt-get update -y
  sudo apt-get install -y git curl ca-certificates ffmpeg tmux python3.12 \
    python3.12-venv python3.12-dev build-essential
fi
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

setup_grounded() {
git submodule update --init tapnet lerobot
ENV_DIR="${ROOT}/.venv-grounded"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --seed --python 3.12 "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
"$ENV_DIR/bin/python" scripts/install_requirements.py --profile grounded \
  --installer uv --pin torch==2.10.0 --pin torchvision==0.25.0
# Match Torch 2.10; CPU decoding avoids CUDA NPP library requirements.
uv pip install --python "$ENV_DIR/bin/python" --no-deps torchcodec==0.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
# Only the PyTorch TAPIR path is used; do not install its JAX training stack.
uv pip install --python "$ENV_DIR/bin/python" --no-deps -e ./tapnet
"$ENV_DIR/bin/python" -m spacy download en_core_web_sm
export PYTHONPATH="$ROOT/src:$ROOT/lerobot/src:${PYTHONPATH:-}"
"$ENV_DIR/bin/python" - <<'PY'
import torch
import spacy
import einshape
import einops
from robot101.data.utilities.episode_helpers import _load_lerobot
from robot101.perception.molmo2 import Molmo2Worker
from robot101.perception.tracking import BootsTAPIR
_load_lerobot()
spacy.load('en_core_web_sm')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check the VM NVIDIA driver.')
print('Ready for preprocessing on', torch.cuda.get_device_name(0))
PY
echo 'Next: source .venv-grounded/bin/activate && hf auth login'
echo 'Run commands: docs/preprocessing.md'
}

setup_molmo2() {
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
uv pip install --python "${VENV_PYTHON}" --no-deps -e .

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

echo "Package setup complete. Run commands: docs/molmo2_training.md"
}

setup_robot() {
git submodule update --init lerobot
if [[ ! -x .venv/bin/python ]]; then
  uv venv --seed --python 3.12 .venv
fi
.venv/bin/python scripts/install_requirements.py --profile robot --installer uv
echo 'Package setup complete. Run commands: README.md'
}

case "$PROFILE" in
  grounded) setup_grounded ;;
  molmo2) setup_molmo2 ;;
  robot) setup_robot ;;
  all)
    setup_grounded
    setup_molmo2
    setup_robot
    ;;
esac
