#!/usr/bin/env bash
set -euo pipefail

# Single-GPU connector fine-tune of Molmo2-4B on SO-101 third-person point labels.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${SO101_MOLMO2_WORKDIR:-${REPO_ROOT}/molmo2_so101_run}"
MOLMO2_REPO_ROOT="${MOLMO2_REPO_ROOT:-${WORK_ROOT}/molmo2}"
MOLMO2_REF="${SO101_MOLMO2_REF:-f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a}"
DATA_CACHE="${SO101_DATA_CACHE:-${WORK_ROOT}/hub_dataset}"
PREPARED_DATA="${SO101_PREPARED_DATA:-${WORK_ROOT}/prepared_dataset}"
MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-${WORK_ROOT}/molmo_data}"
CHECKPOINT_ARCHIVE="${SO101_MOLMO2_CHECKPOINT_ARCHIVE:-${WORK_ROOT}/Molmo2-4B-SFT.tar}"
CHECKPOINT_ROOT="${SO101_MOLMO2_CHECKPOINT_DIR:-${WORK_ROOT}/Molmo2-4B-SFT}"
SAVE_FOLDER="${SO101_SAVE_FOLDER:-${WORK_ROOT}/checkpoints/so101_molmo2_4b}"
DATASET_REPO="${SO101_DATASET_REPO:-kdaterao/so101_locate_gripper}"
MAX_DURATION="${SO101_MAX_DURATION:-500}"
CHECKPOINT_URL="https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B-SFT.tar"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on your Linux NVIDIA compute machine." >&2
  exit 2
fi
if [[ -f "${SAVE_FOLDER}/config.yaml" && ! -e "${SAVE_FOLDER}/latest" && ! -L "${SAVE_FOLDER}/latest" ]]; then
  SAVE_FOLDER="${SAVE_FOLDER}_retry_$(date +%Y%m%d_%H%M%S)"
  echo "Existing run has no resumable checkpoint; using fresh output directory: ${SAVE_FOLDER}"
fi

if [[ ! -f "${MOLMO2_REPO_ROOT}/launch_scripts/sft.py" ]]; then
  mkdir -p "$(dirname "${MOLMO2_REPO_ROOT}")"
  echo "Cloning the official Molmo2 training repository..."
  git clone --depth 1 https://github.com/allenai/molmo2.git "${MOLMO2_REPO_ROOT}"
  git -C "${MOLMO2_REPO_ROOT}" fetch --depth 1 origin "${MOLMO2_REF}"
  git -C "${MOLMO2_REPO_ROOT}" checkout --detach FETCH_HEAD
fi
if [[ ! -f "${MOLMO2_REPO_ROOT}/launch_scripts/sft.py" || ! -f "${MOLMO2_REPO_ROOT}/olmo/train/run_trainer.py" ]]; then
  echo "Molmo2 training repository not found or incomplete at ${MOLMO2_REPO_ROOT}." >&2
  echo "Set MOLMO2_REPO_ROOT to an allenai/molmo2 checkout." >&2
  exit 2
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Molmo2 recommends Python 3.11 or newer")
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-enabled PyTorch build before running this script") from exc
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate this VM's NVIDIA PyTorch environment")
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA devices are visible")
props = torch.cuda.get_device_properties(0)
print(f"Using one GPU: {props.name}, {props.total_memory / 1024**3:.1f} GiB VRAM")
if props.total_memory < 40 * 1024**3:
    print("WARNING: Molmo2-4B SFT may need reduced image crops or a smaller batch on this GPU.")
PY

mkdir -p "${WORK_ROOT}" "${MOLMO_DATA_DIR}"
python3 -m pip install --upgrade pip
python3 -m pip install -e "${MOLMO2_REPO_ROOT}[train]"

python3 "${REPO_ROOT}/scripts/install_molmo2_so101_adapter.py" --repo "${MOLMO2_REPO_ROOT}"
python3 "${REPO_ROOT}/scripts/prepare_molmo_so101.py" \
  --repo-id "${DATASET_REPO}" \
  --cache-dir "${DATA_CACHE}" \
  --output-dir "${PREPARED_DATA}"

if [[ ! -f "${CHECKPOINT_ARCHIVE}" ]]; then
  echo "Downloading Molmo2-4B SFT checkpoint..."
  curl --fail --location --retry 5 --retry-delay 2 \
    "${CHECKPOINT_URL}" \
    --output "${CHECKPOINT_ARCHIVE}"
fi
CONFIG_FILE="$(find "${CHECKPOINT_ROOT}" -name config.yaml -type f -print -quit 2>/dev/null || true)"
if [[ -z "${CONFIG_FILE}" ]]; then
  mkdir -p "${CHECKPOINT_ROOT}"
  tar -xf "${CHECKPOINT_ARCHIVE}" -C "${CHECKPOINT_ROOT}"
  CONFIG_FILE="$(find "${CHECKPOINT_ROOT}" -name config.yaml -type f -print -quit)"
fi
if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Could not find config.yaml after extracting ${CHECKPOINT_ARCHIVE}" >&2
  exit 2
fi
CHECKPOINT_DIR="$(dirname "${CONFIG_FILE}")"

mkdir -p "${SAVE_FOLDER}"
export MOLMO_DATA_DIR
export SO101_POINT_DATA_ROOT="${PREPARED_DATA}"
export SO101_SAVE_TRAINABLE_ONLY=1
export WANDB_MODE="disabled"
export PYTHONPATH="${MOLMO2_REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ROWS="$(wc -l < "${PREPARED_DATA}/train.jsonl" | tr -d ' ')"
if [[ "${TRAIN_ROWS}" -lt 1 ]]; then
  echo "No training points were prepared; check ${DATASET_REPO} and point_labels.jsonl." >&2
  exit 2
fi

echo "Starting single-GPU Molmo2-4B SO-101 point fine-tuning"
echo "  checkpoint: ${CHECKPOINT_DIR}"
echo "  dataset:    ${DATASET_REPO} (${TRAIN_ROWS} train rows)"
echo "  trainable:  vision-language connector only"
echo "  steps:      ${MAX_DURATION}"
echo "  output:     ${SAVE_FOLDER}/so101_connector.pt"

cd "${MOLMO2_REPO_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  launch_scripts/sft.py "${CHECKPOINT_DIR}" so101_point \
  --save_folder="${SAVE_FOLDER}" \
  --max_duration="${MAX_DURATION}" \
  --global_train_batch_size=1 \
  --device_batch_size=1 \
  --seq_len=1024 \
  --num_workers=2 \
  --prefetch_factor=2 \
  --model.mm_preprocessor.video=null \
  --model.mm_preprocessor.image.max_images=null \
  --model.mm_preprocessor.image.max_crops=2 \
  --model.mm_preprocessor.image.high_res_max_crops=4 \
  --model.mm_preprocessor.image.p_high_res=0 \
  --ft_llm=false \
  --ft_vit=false \
  --ft_connector=true \
  --save_num_checkpoints_to_keep=0 \
  --save_final_unsharded_checkpoint=false \
  --save_final_optim=false \
  --eval_interval=-1 \
  --inf_eval_interval=-1 \
  --wandb=null \
  --compile=null \
  --compile_loss=false \
  --save_overwrite=true
