#!/usr/bin/env bash
# Linux CUDA setup for Florence-2 SO-101 gripper fine-tune / inference.
# Leaner than packages-locate.sh (no Eagle / tapnet / deepspeed).
#
#   bash packages-florence.sh
#   source .venv/bin/activate
#   source .venv/florence.env
#   hf auth login
#   python src/florence2_finetune.py train --dataset-repo kdaterao/so101_locate_gripper
#
# Optional:
#   TORCH_CUDA_INDEX=cu124 bash packages-florence.sh
#   SKIP_APT=1 bash packages-florence.sh
#
# Docs: src/so101Locate.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu124}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}}"

echo "==> packages-florence.sh (Florence-2 fine-tune / inference)"

if [[ "${SKIP_APT:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    git curl ca-certificates build-essential \
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

echo "==> installing PyTorch CUDA (${TORCH_CUDA_INDEX})"
uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
uv pip install --reinstall torch torchvision --index-url "${TORCH_INDEX_URL}"

echo "==> installing Florence requirements"
uv pip install -r requirements-florence.txt

# Re-pin CUDA torch in case a transitive dep pulled a CPU wheel.
uv pip install --reinstall torch torchvision --index-url "${TORCH_INDEX_URL}"

ENV_FILE=".venv/florence.env"
cat > "${ENV_FILE}" <<EOF
# Sourced by packages-florence.sh — use: source .venv/florence.env
export PYTHONPATH="${ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Single-GPU Florence train; override if multi-GPU:
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
# Prefer copy over symlink on some networked filesystems:
export HF_HUB_DISABLE_SYMLINKS="\${HF_HUB_DISABLE_SYMLINKS:-0}"
EOF
echo "Wrote ${ENV_FILE}"

# shellcheck disable=SC1091
source "${ENV_FILE}"

echo "==> smoke: torch + Florence import"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CPU torch; expected a CUDA wheel from packages-florence.sh")
print("GPU", torch.cuda.get_device_name(0))
print("VRAM_GiB", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from florence2_finetune import DEFAULT_DATASET_REPO, DEFAULT_TRAIN_MODEL
from florence2_worker import TASK_OPEN_VOCAB, resolve_florence_model
print("dataset_repo", DEFAULT_DATASET_REPO)
print("train_model", DEFAULT_TRAIN_MODEL)
print("task", TASK_OPEN_VOCAB)
print("resolve_florence_model", resolve_florence_model())
print("florence env OK")
PY

cat <<EOF

Done. On each new shell:

  source .venv/bin/activate
  source .venv/florence.env
  hf auth login   # once

Train from Hub dataset:

  python src/florence2_finetune.py train \\
    --dataset-repo kdaterao/so101_locate_gripper \\
    --model-id microsoft/Florence-2-base-ft \\
    --output-dir work_dirs/florence2_so101_gripper \\
    --batch-size 2 --grad-accum 4 --epochs 5 --push-to-hub

Or local labels:

  python src/florence2_finetune.py train --data-root data/locate_gripper

Infer:

  python src/testing2.py --locator florence2 --prompt "SO-101 gripper"

EOF
