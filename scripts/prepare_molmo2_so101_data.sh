#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${SO101_MOLMO2_WORKDIR:-${REPO_ROOT}/molmo2_so101_run}"
DATA_CACHE="${SO101_DATA_CACHE:-${WORK_ROOT}/hub_dataset}"
PREPARED_DATA="${SO101_PREPARED_DATA:-${WORK_ROOT}/prepared_dataset}"
SOURCE_DATASET="${SO101_DATASET_REPO:-kdaterao/so101_locate_gripper}"
OUTPUT_DATASET="${SO101_PREPROCESSED_DATASET_REPO:-kdaterao/so101_molmo2_gripper_preprocessed}"

args=(
  --repo-id "${SOURCE_DATASET}"
  --cache-dir "${DATA_CACHE}"
  --output-dir "${PREPARED_DATA}"
  --push-to-hub
  --hub-repo-id "${OUTPUT_DATASET}"
)
if [[ "${SO101_PREPROCESSED_DATASET_PUBLIC:-0}" == "1" ]]; then
  args+=(--public)
fi

python3 "${REPO_ROOT}/scripts/prepare_molmo_so101.py" "${args[@]}"
