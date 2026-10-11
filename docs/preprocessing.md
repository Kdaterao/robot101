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
    --cluster-tail-frames 30 --pov-next-stage-frames 10 \
    --tapnet-checkpoint tapnet/checkpoints/causal_bootstapir_checkpoint.pt \
    --push-to-hub
```

The example uses the batch size from recent A6000 runs. The Python default is
256; reduce the argument if GPU memory runs out. Download workers control file
fetching independently of TAPIR frame batching and Molmo prompt batching.

POV clustering uses each subtask's final 30 wrist frames plus the first 10 frames
of the following subtask by default. The extra opening frames help retain points
through gripper-triggered boundaries; selected tracks are still written over
their original subtask. Adjust `--pov-next-stage-frames` or set it to `0` to
disable this overlap. The final subtask has no following window.

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
stage points carried forward for continuity. SAM 2 excludes POV gripper pixels and weights nearby candidates. Molmo2
grounds third-person object and gripper candidates at each subtask start,
tracks them forward at the configured FPS, and selects objects by proximity
to the gripper at the endpoint. Dataset videos keep the original clean camera frames, while tracked
coordinates and visibility are saved in point reports. `--viz-dir` writes
separate overlay images for review. Valid object goals take priority. If unresolved, a gripper detected on that
exact endpoint may supply an explicitly marked fallback. Missing masks and
unresolved third-person stages are flagged for manual correction.

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

## SAM 2 POV gripper filtering

The preprocessor now enables SAM 2 gripper exclusion by default, using
`facebook/sam2.1-hiera-tiny`. Rerun Linux package setup to install the official
SAM 2 video predictor. Linux setup disables its optional CUDA cleanup extension,
so no nvcc installation is required for that extension. The first run downloads its checkpoint from the Hub.
BootsTAPIR remains the point tracker. Target-object segmentation is not used.

Useful arguments:

- `--sam2-model facebook/sam2.1-hiera-tiny`: SAM video model.
- `--gripper-proximity-weight 2`: maximum clustering weight near the gripper.
- `--gripper-proximity-radius 0.05`: radius as a fraction of the image diagonal.
- `--no-pov-gripper-filter`: explicitly run the old unfiltered POV behavior.

Masks cover each clustering tail and its next-subtask extension. Missing or
invalid masks produce no valid POV points for that stage and create review
items, rather than treating missing masks as background. Inspect initial runs:
a geometrically valid mask can still segment the wrong object.

Masks are compressed sidecars under `point_tracks/gripper_masks`, with
seed/end mask-preview JPEGs for visual inspection. Review images
and requests are under `point_tracks/mask_review`. These are uploaded with the
point sidecars when `--push-to-hub` is enabled. Training videos stay clean.

On a computer with a display and Tkinter, collect mask prompt corrections:

```bash
python scripts/fix_gripper_masks.py \
  --review-dir /path/to/output/point_tracks/mask_review
```

Left-click visible gripper body/fingers; right-click regions to exclude. Enter
saves, U undoes, N skips, and Q quits. Add `--overwrite` to revisit corrections. Click coordinates are mapped back to the
source image. Copy the resulting `mask_corrections.json` to the processing VM
and rerun the same episode selection with:

```bash
python -m robot101.data.preprocess \
  --episodes 0 --dst-repo-id YOUR_ACCOUNT/corrected_grounding \
  --sam2-corrections /path/to/mask_corrections.json
```

Use a fresh destination to regenerate episodes already written: `--resume`
skips completed episodes, including ones containing unresolved stages.
Corrections use `episode:subtask` keys and exact source `frame_index` values;
keep the source dataset, stage settings, and clustering windows unchanged.

Third-person object and gripper candidates are grounded at each subtask start
and tracked forward at the configured sparse FPS. Object goals are selected
using their proximity to the gripper at the exact endpoint, with the configured
proximity threshold and ambiguity margin. If the gripper track is lost, Molmo
may re-detect it only at that exact endpoint for association. When selection
fails, a gripper detected on that exact endpoint may supply `gripper_fallback`,
which is tracked backward through the subtask.
No nearby-frame substitution or POV fallback is allowed. Unresolved
third-person results retain `manual_review_required` for the existing point
correction viewer.
