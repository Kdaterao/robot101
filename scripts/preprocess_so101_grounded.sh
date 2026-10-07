#!/usr/bin/env bash
# Generate clustered episode videos + Molmo2 third-person heatmaps, then upload.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${SO101_PYTHON:-$ROOT/.venv-grounded/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Install preprocessing packages first: bash packages-grounded-preprocess.sh' >&2
  exit 2
fi
CHECKPOINT="$ROOT/tapnet/checkpoints/causal_bootstapir_checkpoint.pt"
if [[ ! -s "$CHECKPOINT" ]]; then
  mkdir -p "$(dirname "$CHECKPOINT")"
  curl --fail --location --retry 3 \
    https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt \
    --output "$CHECKPOINT.part"
  mv "$CHECKPOINT.part" "$CHECKPOINT"
fi
echo 'Starting episode preprocessing: wrist clusters + Molmo2 connector inference.'
echo 'Default: episodes 0-2; resulting LeRobot dataset will upload to Hugging Face.'
exec "$PYTHON" src/hf_preprocess_smolvla_grounded.py \
  --episodes "${SO101_EPISODES:-0-2}" \
  --dst-repo-id "${SO101_DST_REPO:-kdaterao/community_v3_ee_smolvla_molmo_grounded}" \
  --molmo-model allenai/Molmo2-4B --molmo-backend molmo2 \
  --molmo-connector-repo kdaterao/so101-molmo2-4b-gripper \
  --molmo-dtype bf16 --device cuda --video-backend pyav \
  --cluster-tail-frames 30 --tapnet-checkpoint "$CHECKPOINT" \
  --push-to-hub "$@"
