# Grounded episode preprocessing

## Install packages once on the Ubuntu GPU VM

```bash
cd ~/robot101
bash scripts/setup/linux_setup.sh
source .venv-grounded/bin/activate
hf auth login
```

This installs inference, video, spaCy and TAPIR dependencies. The existing Molmo2
connector is applied during inference. New weights are created using the
separate [training guide](molmo2_training.md).

## Download the TAPIR checkpoint once

```bash
mkdir -p tapnet/checkpoints
curl --fail --location --retry 3 \
  https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt \
  --output tapnet/checkpoints/causal_bootstapir_checkpoint.pt.part
mv tapnet/checkpoints/causal_bootstapir_checkpoint.pt.part \
  tapnet/checkpoints/causal_bootstapir_checkpoint.pt
```

Skip this step if the checkpoint already exists. The Molmo2 base weights and
connector download automatically into the Hugging Face cache on first use.

## Process one episode and upload it, with telemetry

```bash
export SO101_HF_DOWNLOAD_WORKERS=8
python scripts/profile_preprocessing.py -- \
  python -m robot101.data.preprocess \
    --src-repo-id felsager/community_dataset_v3_ee_smolVLA \
    --episodes 0 --dst-repo-id kdaterao/so101_grounded_test \
    --molmo-model allenai/Molmo2-4B --molmo-backend molmo2 \
    --molmo-connector-repo kdaterao/so101-molmo2-4b-gripper \
    --molmo-batch-size 4 --molmo-dtype bf16 \
    --device cuda --video-backend pyav \
    --tapir-frame-batch-size 256 --third-person-tracking-fps 1 \
    --cluster-tail-frames 30 \
    --tapnet-checkpoint tapnet/checkpoints/causal_bootstapir_checkpoint.pt \
    --push-to-hub
```

The example uses the batch size from recent A6000 runs. The Python default is
16; reduce the argument if GPU memory runs out. Download workers control file
fetching independently of TAPIR frame batching and Molmo prompt batching.

- Change `--episodes` to `0-9` or `0,3,8`; the Python default is one episode, `0`.
- Episode shards are fetched on demand. If metadata is cached but a selected episode's data shard is missing, preprocessing syncs that episode's data and camera files from the Hub and retries.
- Points from the previous subtask are TAPIR-tracked into the next subtask for 1.5 seconds by default, for both wrist and third-person views. Set `--transition-persist-seconds` to adjust the overlap or `0` to disable it.
- Change `--dst-repo-id` to your output dataset. Choose a fresh destination for a new experiment; use `--resume` to continue a compatible existing destination.
- Remove `--push-to-hub` to keep the dataset local. Upload happens only when episodes are written.
- Replace `--molmo-connector-repo` with `--molmo-connector /absolute/path/so101_connector.pt` for local weights.
- Add `--viz-dir outputs/grounded_viz` for visualizations.
- To omit telemetry, run the inner `python -m robot101.data.preprocess ...` command directly.
- To select a profile directory, put `--out-dir outputs/profiles/my_run` before the profiler's `--` separator.

Profiles appear in `outputs/profiles/<timestamp>/`. Incompatible episodes are
logged in the destination's `point_tracks/skipped_episodes.jsonl`.

## What it produces

Gripper motion divides episodes into subtasks. Wrist candidates are clustered
in each stage's final window using the shared query bank and cross-demo
selection. Selected points track backward through their stages, with previous
stage points carried forward for continuity. Molmo2 grounds task entities and
the gripper in third-person images; third-person tracking uses the configured
FPS. Dataset videos keep the original clean camera frames, while tracked
coordinates and visibility are saved in point reports. `--viz-dir` writes
separate overlay images for review. Endpoint gripper recovery uses the exact
stage-end image. Unresolved goals are recorded in stage reports.

The LeRobot dataset contains clean camera videos and state/action data, with
point tracks in `point_tracks/`.
See [TAPIR commands](legacy/TAPNET_COMMANDS.md) for algorithm details.

## Inspect stages before processing

```bash
python -m robot101.data.preprocess --episodes 0-2 \
  --dst-repo-id kdaterao/grounded_local --dry-run
python scripts/inspect_gripper_stages.py --episode 0
```

Use `python -m robot101.data.preprocess --help` for all options.
