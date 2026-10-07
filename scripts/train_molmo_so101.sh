#!/usr/bin/env bash
set -euo pipefail

# Single-GPU, connector-only Molmo fine-tune on the labeler's default Hub dataset.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOLMO_REPO_ROOT="${MOLMO_REPO_ROOT:-${REPO_ROOT}/molmo}"
WORK_ROOT="${SO101_MOLMO_WORKDIR:-${REPO_ROOT}/molmo_so101_run}"
DATA_CACHE="${SO101_DATA_CACHE:-${WORK_ROOT}/hub_dataset}"
PREPARED_DATA="${SO101_PREPARED_DATA:-${WORK_ROOT}/prepared_dataset}"
MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-${WORK_ROOT}/molmo_data}"
MOLMO_MODEL="${SO101_MOLMO_MODEL:-MolmoE-1B-0924}"
CHECKPOINT_CACHE="${MOLMO_CHECKPOINT_ARCHIVE:-${WORK_ROOT}/${MOLMO_MODEL}.tar}"
CHECKPOINT_DIR="${MOLMO_CHECKPOINT_DIR:-${WORK_ROOT}/${MOLMO_MODEL}}"
SAVE_FOLDER="${SO101_SAVE_FOLDER:-${WORK_ROOT}/checkpoints/so101_molmo}"
DATASET_REPO="${SO101_DATASET_REPO:-kdaterao/so101_locate_gripper}"
MAX_DURATION="${SO101_MAX_DURATION:-500}"

case "${MOLMO_MODEL}" in
  MolmoE-1B-0924)
    CHECKPOINT_URL="https://storage.googleapis.com/oe-training-public/Molmo-0924/MolmoE-1B-0924.tar"
    NEEDS_MEGABLOCKS=1
    ;;
  Molmo-7B-D-0924)
    CHECKPOINT_URL="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-D-0924.tar"
    NEEDS_MEGABLOCKS=0
    ;;
  *)
    echo "Unsupported SO101_MOLMO_MODEL: ${MOLMO_MODEL}" >&2
    echo "Supported values: MolmoE-1B-0924, Molmo-7B-D-0924" >&2
    exit 2
    ;;
esac

# A prior startup can leave config.yaml behind before step 1 is checkpointed.
# Preserve that directory and start into a fresh one instead of overwriting it.
if [[ -f "${SAVE_FOLDER}/config.yaml" && ! -e "${SAVE_FOLDER}/latest" && ! -L "${SAVE_FOLDER}/latest" ]]; then
  SAVE_FOLDER="${SAVE_FOLDER}_retry_$(date +%Y%m%d_%H%M%S)"
  echo "Existing run has no latest checkpoint; using fresh output directory: ${SAVE_FOLDER}"
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on your Linux NVIDIA compute machine." >&2
  exit 2
fi
if [[ ! -f "${MOLMO_REPO_ROOT}/launch_scripts/train_multitask_model.py" ]]; then
  if [[ "${MOLMO_REPO_ROOT}" == "${REPO_ROOT}/molmo" && -f "${REPO_ROOT}/.gitmodules" ]]; then
    echo "Initializing the Molmo submodule..."
    git -C "${REPO_ROOT}" submodule update --init molmo
  fi
fi
if [[ ! -f "${MOLMO_REPO_ROOT}/launch_scripts/train_multitask_model.py" ]]; then
  echo "Molmo training repository not found at ${MOLMO_REPO_ROOT}." >&2
  echo "Initialize the molmo submodule with: git submodule update --init molmo" >&2
  echo "Or set MOLMO_REPO_ROOT to an AllenAI Molmo checkout." >&2
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
    print("WARNING: Molmo connector fine-tuning may exceed this GPU's memory.")
PY

mkdir -p "${WORK_ROOT}" "${MOLMO_DATA_DIR}"
python3 - "${MOLMO_REPO_ROOT}" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
data_init = repo / "olmo/data/__init__.py"
launcher = repo / "launch_scripts/train_multitask_model.py"
trainer = repo / "scripts/train.py"
adapter = repo / "olmo/data/so101_point_dataset.py"
if not data_init.is_file() or not launcher.is_file() or not trainer.is_file():
    raise SystemExit(f"Not an AllenAI Molmo training checkout: {repo}")

adapter.write_text('''"""SO-101 point annotations in the Molmo training example format."""

import json
import os
from pathlib import Path

import numpy as np

from olmo.data.dataset import Dataset


class So101GripperPoints(Dataset):
    def __init__(self, split):
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported SO-101 split: {split}")
        root = os.environ.get("SO101_POINT_DATA_ROOT")
        if not root:
            raise RuntimeError("Set SO101_POINT_DATA_ROOT to prepared SO-101 point data")
        path = Path(root) / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if split == "train" and not self.rows:
            raise ValueError(f"No SO-101 training annotations found in {path}")

    def __len__(self):
        return len(self.rows)

    def get(self, item, rng):
        row = self.rows[item]
        return {
            "image": row["image"],
            "message_list": [{
                "label": row["label"],
                "points": np.asarray(row["point_xy_100"], dtype=np.float32).reshape(1, 2),
                "point_scale": 100,
                "style": "pointing",
            }],
            "metadata": {
                "episode": row["episode"],
                "camera": row["camera"],
                "source_image": row["source_image"],
            },
        }
''', encoding="utf-8")

def insert_once(path, marker, addition):
    text = path.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if marker not in text:
        raise SystemExit(f"Could not find expected Molmo integration point in {path}")
    path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")

insert_once(
    data_init,
    "from olmo.torch_util import get_global_rank, get_world_size" + chr(10),
    "from olmo.data.so101_point_dataset import So101GripperPoints" + chr(10),
)
insert_once(
    data_init,
    '    elif dataset_name in ["point_count", "pixmo_points_counting"]:' + chr(10),
    '    elif dataset_name == "so101_gripper_points":' + chr(10)
    + '        return So101GripperPoints(split)' + chr(10),
)
launcher_text = launcher.read_text(encoding="utf-8")
branch_start = '    elif args.mixture == "so101-point":' + chr(10)
branch_end = '    elif args.mixture in ["small1", "debug"]:' + chr(10)
branch = (
    branch_start
    + '        eval_tasks = []' + chr(10)
    + '        tasks = [["so101_pointing", ["so101_gripper_points"], 1.0]]' + chr(10)
)
if branch_start in launcher_text:
    start = launcher_text.index(branch_start)
    end = launcher_text.index(branch_end, start)
    launcher_text = launcher_text[:start] + branch + launcher_text[end:]
else:
    if branch_end not in launcher_text:
        raise SystemExit(f"Could not find expected Molmo mixture point in {launcher}")
    launcher_text = launcher_text.replace(branch_end, branch + branch_end, 1)
launcher.write_text(launcher_text, encoding="utf-8")

trainer_text = trainer.read_text(encoding="utf-8")
if "import os" not in trainer_text:
    trainer_text = trainer_text.replace("import logging" + chr(10), "import logging" + chr(10) + "import os" + chr(10), 1)
old_checkpoint_load = (
    '                state_dict = torch.load(join(cfg.initial_model_checkpoint, "model.pt"), map_location="cpu")' + chr(10)
    + '                olmo_model.load_state_dict(state_dict)' + chr(10)
    + '                del state_dict'
)
memory_efficient_load = (
    '                checkpoint_file = join(cfg.initial_model_checkpoint, "model.pt")' + chr(10)
    + '                state_dict = torch.load(checkpoint_file, map_location="cpu", mmap=True, weights_only=True)' + chr(10)
    + '                olmo_model.load_state_dict(state_dict, assign=True)' + chr(10)
    + '                del state_dict'
)
if old_checkpoint_load in trainer_text:
    trainer_text = trainer_text.replace(old_checkpoint_load, memory_efficient_load, 1)
elif memory_efficient_load not in trainer_text:
    raise SystemExit(f"Could not find the expected Molmo checkpoint loader in {trainer}")

old_fit_end = (
    '            trainer.fit()' + chr(10)
    + '            log.info("Training complete")'
)
connector_fit_end = (
    '            trainer.fit()' + chr(10)
    + '            if os.environ.get("SO101_SAVE_CONNECTOR_ONLY") == "1":' + chr(10)
    + '                model = trainer.fsdp_model.module' + chr(10)
    + '                connector_state = {' + chr(10)
    + '                    name: param.detach().to(device="cpu", copy=True)' + chr(10)
    + '                    for name, param in model.named_parameters() if param.requires_grad' + chr(10)
    + '                }' + chr(10)
    + '                if not connector_state:' + chr(10)
    + '                    raise RuntimeError("No trainable connector parameters to save")' + chr(10)
    + '                if get_global_rank() == 0:' + chr(10)
    + '                    connector_path = Path(cfg.save_folder) / "so101_connector.pt"' + chr(10)
    + '                    torch.save(connector_state, connector_path)' + chr(10)
    + '                    log.info(f"Saved {len(connector_state)} trainable tensors to {connector_path}")' + chr(10)
    + '                barrier()' + chr(10)
    + '            log.info("Training complete")'
)
if old_fit_end in trainer_text:
    trainer_text = trainer_text.replace(old_fit_end, connector_fit_end, 1)
elif connector_fit_end not in trainer_text:
    raise SystemExit(f"Could not find the expected Molmo training-completion point in {trainer}")
trainer.write_text(trainer_text, encoding="utf-8")
print(f"Installed SO-101 point adapter in {repo}")
PY
python3 -m pip install --upgrade pip
python3 -m pip install 'transformers==4.57.1' 'huggingface_hub<1.0'
python3 -m pip install -e "${MOLMO_REPO_ROOT}[train]"
if [[ "${NEEDS_MEGABLOCKS}" == "1" ]]; then
  CUDA_BUILD_HOME="${SO101_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
  if [[ ! -x "${CUDA_BUILD_HOME}/bin/nvcc" ]]; then
    echo "CUDA compiler not found at ${CUDA_BUILD_HOME}/bin/nvcc." >&2
    echo "MolmoE needs Megablocks compiled with nvcc. Install the CUDA toolkit matching PyTorch, then set SO101_CUDA_HOME to its root." >&2
    echo "Current PyTorch CUDA build: $(python3 -c 'import torch; print(torch.version.cuda)')" >&2
    echo "For example, if nvcc is at /usr/local/cuda-13.0/bin/nvcc, run:" >&2
    echo "  export SO101_CUDA_HOME=/usr/local/cuda-13.0" >&2
    exit 2
  fi
  CUDA_VERSION="$("${CUDA_BUILD_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
  TORCH_CUDA_VERSION="$(python3 -c 'import torch; print(torch.version.cuda or "")')"
  if [[ -z "${CUDA_VERSION}" || "${CUDA_VERSION%%.*}" != "${TORCH_CUDA_VERSION%%.*}" ]]; then
    echo "CUDA compiler ${CUDA_VERSION:-unknown} does not match PyTorch CUDA ${TORCH_CUDA_VERSION:-unknown}." >&2
    echo "Install a matching CUDA toolkit or use a PyTorch build for the installed toolkit." >&2
    exit 2
  fi
  export CUDA_HOME="${CUDA_BUILD_HOME}"
  export PATH="${CUDA_BUILD_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_BUILD_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "Building Megablocks with PyTorch CUDA ${TORCH_CUDA_VERSION} and nvcc ${CUDA_VERSION}"
  # Avoid pip's isolated build environment selecting a different CUDA PyTorch
  # wheel than the one used for this training run.
  python3 -m pip install --no-build-isolation 'git+https://github.com/Muennighoff/megablocks.git@olmoe'
fi

export MOLMO_DATA_DIR
export SO101_POINT_DATA_ROOT="${PREPARED_DATA}"
export SO101_SAVE_CONNECTOR_ONLY=1
python3 "${REPO_ROOT}/scripts/prepare_molmo_so101.py" \
  --repo-id "${DATASET_REPO}" \
  --cache-dir "${DATA_CACHE}" \
  --output-dir "${PREPARED_DATA}"

CONFIG_FILE="$(find "${CHECKPOINT_DIR}" -name config.yaml -type f -print -quit 2>/dev/null || true)"
if [[ -n "${CONFIG_FILE}" ]]; then
  CHECKPOINT_DIR="$(dirname "${CONFIG_FILE}")"
else
  if [[ ! -f "${CHECKPOINT_CACHE}" ]]; then
    echo "Downloading ${MOLMO_MODEL} training checkpoint..."
    curl --fail --location --retry 5 --retry-delay 2 \
      "${CHECKPOINT_URL}" \
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
echo "  model:      ${MOLMO_MODEL}"
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
  --save_interval_unsharded=null \
  --save_num_checkpoints_to_keep=0 \
  --save_num_unsharded_checkpoints_to_keep=0 \
  --eval_interval=-1 \
  --inf_eval_interval=-1 \
  --data.num_workers=0 \
  --model.max_crops=1 \
  --ft_llm=false \
  --ft_vit=false \
  --ft_connector=true \
  --fsdp.precision=pure \
  --optimizer.connector_learning_rate=5e-5 \
  --wandb=null

echo "Training finished. Connector weights: ${SAVE_FOLDER}/so101_connector.pt"
