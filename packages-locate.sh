#!/usr/bin/env bash
# Linux CUDA setup for SO-101 LocateAnything / Florence-2 / locate_gripper /
# HF preprocess / locate_finetune (Eagle Embodied).
# Does not replace packages.sh or packages-tapnet-gpu.sh.
#
#   bash packages-locate.sh
#   source .venv/bin/activate
#   hf auth login
#   python src/locate_finetune.py export
#   python src/locate_finetune.py train --eagle-root third_party/Eagle/Embodied --push-model-to-hub
#
# Override Eagle clone location:
#   EAGLE_ROOT=/path/to/Eagle bash packages-locate.sh
#
# Docs: src/so101Locate.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EAGLE_ROOT="${EAGLE_ROOT:-$ROOT/third_party/Eagle}"
EAGLE_EMBODIED="${EAGLE_ROOT}/Embodied"

echo "==> packages-locate.sh (LocateAnything + Florence-2 + locate_finetune/Eagle)"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    ffmpeg tmux git curl ca-certificates build-essential \
    libgl1 libglib2.0-0 ninja-build
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

echo "==> cloning NVlabs/Eagle (LocateAnything trainer) -> ${EAGLE_ROOT}"
if [[ ! -d "${EAGLE_EMBODIED}" ]]; then
  mkdir -p "$(dirname "${EAGLE_ROOT}")"
  git clone --depth 1 https://github.com/NVlabs/Eagle.git "${EAGLE_ROOT}"
else
  echo "    already present: ${EAGLE_EMBODIED}"
fi
if [[ ! -f "${EAGLE_EMBODIED}/eaglevl/train/locany_finetune_magi_stream.py" ]]; then
  echo "ERROR: Eagle train script missing under ${EAGLE_EMBODIED}"
  exit 1
fi

echo "==> installing PyTorch CUDA 12.4"
uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> installing local tapnet + lerobot (editable, --no-deps)"
uv pip install -e "./tapnet" --no-deps
if [[ -f lerobot/pyproject.toml ]] || [[ -f lerobot/setup.py ]]; then
  uv pip install -e "./lerobot" --no-deps
fi

echo "==> installing Eagle Embodied (editable, --no-deps) + finetune requirements"
uv pip install -e "${EAGLE_EMBODIED}" --no-deps
# deepspeed / liger / triton can fail on some GPUs; keep going for inference-only use.
set +e
uv pip install -r "requirements-locate-finetune.txt"
FINETUNE_RC=$?
set -e
if [[ "${FINETUNE_RC}" -ne 0 ]]; then
  echo "WARNING: requirements-locate-finetune.txt had install errors."
  echo "  Retry without optional kernels:"
  echo "  uv pip install transformers==4.57.1 tokenizers==0.22.0 sentencepiece==0.2.0 \\"
  echo "    accelerate==1.5.2 peft==0.12.0 deepspeed==0.15.4 bitsandbytes decord wandb tensorboard"
fi

echo "==> installing requirements-locate.txt (collect / Florence / preprocess)"
uv pip install -r "requirements-locate.txt"

# Ensure CUDA torch was not replaced by a CPU wheel from transitive deps.
echo "==> re-asserting PyTorch CUDA 12.4"
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

mkdir -p data/locate_gripper/images work_dirs

echo "==> sanity check"
python - <<PY
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

try:
    import peft  # noqa: F401
    print("peft", peft.__version__)
except Exception as e:
    print("peft MISSING:", e)

try:
    import deepspeed  # noqa: F401
    print("deepspeed OK")
except Exception as e:
    print("deepspeed MISSING (train may need --deepspeed none):", e)

import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from locateanything_worker import LocateAnythingWorker  # noqa: F401
from florence2_worker import Florence2Worker, DEFAULT_FLORENCE_MODEL  # noqa: F401
print("locate workers OK; Florence default:", DEFAULT_FLORENCE_MODEL)

eagle = Path(r"${EAGLE_EMBODIED}")
train_py = eagle / "eaglevl" / "train" / "locany_finetune_magi_stream.py"
print("Eagle Embodied:", eagle, "train_script:", train_py.is_file())
PY

echo ""
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  hf auth login"
echo ""
echo "Eagle root for locate_finetune:"
echo "  ${EAGLE_EMBODIED}"
echo ""
echo "Fine-tune LocateAnything on locate_gripper:"
echo "  # pull labels if needed:"
echo "  hf download kdaterao/so101_locate_gripper --repo-type dataset --local-dir data/locate_gripper"
echo "  python src/locate_finetune.py export"
echo "  python src/locate_finetune.py train --eagle-root ${EAGLE_EMBODIED} --push-model-to-hub"
echo "  # if DeepSpeed fails on this GPU:"
echo "  python src/locate_finetune.py train --eagle-root ${EAGLE_EMBODIED} --deepspeed none --push-model-to-hub"
echo ""
echo "Label / live / preprocess:"
echo "  python src/locate_collect_label.py batch"
echo "  python src/testing2.py --prompt \"SO-101 gripper\""
echo "  python src/hf_preprocess_smolvla.py --episodes 0-2 --dry-run"
echo ""
echo "See src/so101Locate.md"
