# Build the grounded robot training dataset

This pipeline converts SO-101 episodes into a LeRobot dataset with gripper-based
subtasks, wrist-camera point clusters, and third-person point heatmaps grounded
by Molmo2 and tracked by TAPIR.

On an Ubuntu GPU VM, install the preprocessing environment, authenticate to
Hugging Face, and download the TAPIR checkpoint once:

```bash
bash scripts/setup/linux_setup.sh
source .venv-grounded/bin/activate
hf auth login
mkdir -p tapnet/checkpoints
curl -L https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt \
  -o tapnet/checkpoints/causal_bootstapir_checkpoint.pt
```

Process one episode and upload the result:

```bash
python -m robot101.data.preprocess \
  --src-repo-id felsager/community_dataset_v3_ee_smolVLA \
  --episodes 0 \
  --dst-repo-id YOUR_USER/grounded-dataset \
  --push-to-hub
```

The default selection is episode `0`. To process every fifth episode from 0
through 25, set `--episodes 0-25:5` (start-stop:step). Remove `--push-to-hub` to
write locally. Use `--resume` to continue a compatible local output dataset.
Run `python -m robot101.data.preprocess --help` for tracking FPS, batch sizes,
and gripper stage options.

## Batching and download workers

These settings control different parts of the run:

- `--molmo-batch-size 4` batches image-and-prompt requests for Molmo on the GPU.
- `--tapir-frame-batch-size 256` batches ordered frames per TAPIR call. The
  default is 16; raise it while GPU memory allows.
- `--decode-batch-size 64` controls video frames decoded together and uses CPU
  memory.
- `SO101_HF_DOWNLOAD_WORKERS=64` controls parallel Hugging Face file downloads;
  it does not batch GPU inference. Start lower if downloads hit rate limits.

For example, set download concurrency and run the spaced episode sample like this:

```bash
SO101_HF_DOWNLOAD_WORKERS=64 python -m robot101.data.preprocess \
  --episodes 0-25:5 --dst-repo-id YOUR_USER/grounded-dataset \
  --molmo-batch-size 4 --tapir-frame-batch-size 256 --decode-batch-size 64
```

Each episode is split at gripper events. Wrist candidates are clustered from
stage-tail frames and tracked through their stages. Molmo2 grounds task objects
and the gripper in available third-person views; TAPIR tracks those points and
the pipeline saves clean camera videos, point reports, and robot state/action
data. Point coordinates and visibility are in `point_tracks/`; use `--viz-dir`
to save separate overlay images for review. Invalid episode shapes are recorded
as skipped episodes.
