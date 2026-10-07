#!/usr/bin/env bash
set -euo pipefail

# Single-GPU, connector-only Molmo fine-tune on the labeler's default Hub dataset.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOLMO_REPO_ROOT="${MOLMO_REPO_ROOT:-${REPO_ROOT}/molmo}"
WORK_ROOT="${SO101_MOLMO_WORKDIR:-${REPO_ROOT}/molmo_so101_run}"
DATA_CACHE="${SO101_DATA_CACHE:-${WORK_ROOT}/hub_dataset}"
PREPARED_DATA="${SO101_PREPARED_DATA:-${WORK_ROOT}/prepared_dataset}"
MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-${WORK_ROOT}/molmo_data}"
CHECKPOINT_CACHE="${MOLMO_CHECKPOINT_ARCHIVE:-${WORK_ROOT}/Molmo-7B-D-0924.tar}"
CHECKPOINT_DIR="${MOLMO_CHECKPOINT_DIR:-${WORK_ROOT}/Molmo-7B-D-0924}"
SAVE_FOLDER="${SO101_SAVE_FOLDER:-${WORK_ROOT}/checkpoints/so101_molmo}"
DATASET_REPO="${SO101_DATASET_REPO:-kdaterao/so101_locate_gripper}"
MAX_DURATION="${SO101_MAX_DURATION:-500}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on your Linux NVIDIA compute machine." >&2
  exit 2
fi
if [[ ! -f "${MOLMO_REPO_ROOT}/launch_scripts/train_multitask_model.py" ]]; then
  echo "Molmo training repository not found at ${MOLMO_REPO_ROOT}." >&2
  echo "Set MOLMO_REPO_ROOT to the workspace's allenai/molmo checkout." >&2
  exit 2
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Molmo requires Python 3.10 or newer")
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-enabled PyTorch build before running this script") from exc
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate this CM's NVIDIA PyTorch environment")
props = torch.cuda.get_device_properties(0)
print(f"Using one GPU: {props.name}, {props.total_memory / 1024**3:.1f} GiB VRAM")
if props.total_memory < 32 * 1024**3:
    print("WARNING: Molmo-7B connector fine-tuning may exceed this GPU's memory.")
PY

mkdir -p "${WORK_ROOT}" "${MOLMO_DATA_DIR}"
python3 -m pip install --upgrade pip
python3 -m pip install -e "${MOLMO_REPO_ROOT}[train]"

export MOLMO_DATA_DIR
export SO101_POINT_DATA_ROOT="${PREPARED_DATA}"
python3 "${REPO_ROOT}/scripts/prepare_molmo_so101.py" \
  --repo-id "${DATASET_REPO}" \
  --cache-dir "${DATA_CACHE}" \
  --output-dir "${PREPARED_DATA}"

echo "Preparing general point-grounding examples for replay..."
python3 - <<'PY'
from olmo.data.pixmo_datasets import PixMoPoints
PixMoPoints.download(n_procs=4)
PY

CONFIG_FILE="$(find "${CHECKPOINT_DIR}" -name config.yaml -type f -print -quit 2>/dev/null || true)"
if [[ -z "${CONFIG_FILE}" ]]; then
  if [[ ! -f "${CHECKPOINT_CACHE}" ]]; then
    echo "Downloading Molmo-7B-D-0924 training checkpoint..."
    curl --fail --location --retry 5 --retry-delay 2 \
      "https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-D-0924.tar" \
      --output "${CHECKPOINT_CACHE}"
  fi
  mkdir -p "${CHECKPOINT_DIR}"
  tar -xf "${CHECKPOINT_CACHE}" -C "${CHECKPOINT_DIR}"
  CONFIG_FILE="$(find "${CHECKPOINT_DIR}" -name config.yaml -type f -print -quit)"
  if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Could not find config.yaml after extracting ${CHECKPOINT_CACHE}" >&2
    exit 2
  fi
  CHECKPOINT_DIR="$(dirname "${CONFIG_FILE}")"
fi

mkdir -p "${SAVE_FOLDER}"
export PYTHONPATH="${MOLMO_REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="disabled"

echo "Starting single-GPU SO-101 Molmo fine-tuning"
echo "  checkpoint: ${CHECKPOINT_DIR}"
echo "  dataset:    ${DATASET_REPO}"
echo "  train rows: $(wc -l < "${PREPARED_DATA}/train.jsonl" | tr -d ' ')"
echo "  val rows:   $(wc -l < "${PREPARED_DATA}/validation.jsonl" | tr -d ' ')"
echo "  steps:      ${MAX_DURATION}"
echo "  output:     ${SAVE_FOLDER}"

cd "${MOLMO_REPO_ROOT}"
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_multitask_model.py \
  so101-point "${CHECKPOINT_DIR}" \
  --save_folder="${SAVE_FOLDER}" \
  --global_batch_size=1 \
  --device_train_batch_size=1 \
  --device_eval_batch_size=1 \
  --device_inf_batch_size=1 \
  --seq_len=1024 \
  --max_duration="${MAX_DURATION}" \
  --save_interval=100 \
  --save_interval_unsharded="${MAX_DURATION}" \
  --eval_interval=-1 \
  --inf_eval_interval=-1 \
  --model.max_crops=1 \
  --ft_llm=false \
  --ft_vit=false \
  --ft_connector=true \
  --fsdp.precision=pure \
  --optimizer.connector_learning_rate=5e-5 \
  --wandb=null

echo "Training finished. Checkpoints: ${SAVE_FOLDER}"
