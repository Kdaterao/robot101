#!/usr/bin/env bash
# Ubuntu/A6000 package setup for episode clustering and Molmo2 inference.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this package installer on the Ubuntu GPU VM.' >&2
  exit 2
fi
sudo apt-get update -y
sudo apt-get install -y git curl ffmpeg python3.12 python3.12-venv python3.12-dev build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
git submodule update --init tapnet lerobot
ENV_DIR="${ROOT}/.venv-grounded"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --seed --python 3.12 "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$ENV_DIR/bin/python" 'torch==2.10.0' 'torchvision==0.25.0' \
  -r requirements-grounded-preprocess.txt
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
from hf_preprocess_smolvla import _load_lerobot
from molmo2_worker import Molmo2Worker
from tapnet_utils import BootsTAPIR
_load_lerobot()
spacy.load('en_core_web_sm')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check the VM NVIDIA driver.')
print('Ready for preprocessing on', torch.cuda.get_device_name(0))
PY
echo 'Next: source .venv-grounded/bin/activate && hf auth login'
echo 'Then: bash scripts/preprocess_so101_grounded.sh'
