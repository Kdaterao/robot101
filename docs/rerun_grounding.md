# Rerun grounding and replace it in the combined dataset

Keep the existing annotation captions, selected subtask boundaries and Claude
results. Rerun **only grounding** using gripper opening/closing subdivisions,
then assemble a new unified dataset containing the old text and new point tracks.
Videos remain full episodes. Grounding subdivisions do not become caption subtasks.

## 1. Set up each fresh Ubuntu GPU VM

Commit and push the latest development changes from your laptop first. These
instructions assume four fresh Ubuntu VMs with NVIDIA GPUs and working drivers.
Each VM runs one worker. It needs room for the Python environment, model weights,
source video cache and generated dataset; those caches are initially empty.

Run on **each new VM**:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
cd ~
git clone https://github.com/Kdaterao/robot101.git
cd ~/robot101
nvidia-smi
bash scripts/setup/linux_setup.sh --profile grounded
source .venv-grounded/bin/activate
hf auth login
hf auth whoami
mkdir -p tapnet/checkpoints
curl --fail --location --retry 5 \
  https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt \
  -o tapnet/checkpoints/causal_bootstapir_checkpoint.pt
```

The grounded installer initializes the `tapnet` and `lerobot` submodules and
installs CUDA PyTorch, Molmo2 inference, SAM 2, TAPIR, spaCy and video dependencies.
Use an HF token with read access to existing worker datasets/model weights and
write access to your new grounding repos. Each fresh VM must authenticate itself;
your laptop's saved login is not copied to it.

Only the **grounded** profile is needed. Molmo base weights, your existing trained
connector and SAM weights download on the first actual GPU run; source metadata
and videos download as needed. The dry run does not preload inference weights.
Do not install the annotation or Molmo training profile, or rerun captioning or Claude.

For an existing checkout instead of a fresh VM, replace the clone commands with
`cd ~/robot101` and `git pull`. Keep all four workers on the same code revision:

```bash
git rev-parse HEAD
```

### Keep the run alive after disconnecting

After setup, start a persistent terminal **before** setting worker variables:

```bash
tmux new -s grounding
cd ~/robot101
source .venv-grounded/bin/activate
```

Then continue with the worker block below and run inside tmux. Detach with
**Ctrl+B, then D**. Reconnect with `tmux attach -t grounding`. This preserves the
process while SSH disconnects; it does not survive deleting/recreating the VM.
After a reboot/recreated shell, activate the environment and repeat the worker
assignment/variables before running commands. `--resume` relies on compatible
local output, so keep the VM disk until its upload is complete. A newly recreated
VM with an empty disk does not automatically resume processing from Hub output.

## 2. Assign one worker to each VM

Use **one block per VM**. These annotation SHAs are from the existing unified
dataset's `manifest.json`; pinning them keeps the cuts/text used by existing Claude
outputs. The raw source dataset SHA is loaded automatically from those annotations.

### VM 1

```bash
export WORKER=1
export ANNOTATION_REVISION=fe68b919703eb867fc70b27cdc73114b75d89ecc
export TEST_EPISODE=10
```

### VM 2

```bash
export WORKER=2
export ANNOTATION_REVISION=06e5fcfd1fcefca1a464b7cdb37b79a7a7a76cdf
export TEST_EPISODE=896
```

### VM 3

```bash
export WORKER=3
export ANNOTATION_REVISION=ba86982b7bc65ddd1d26ca00097a244bad0451d5
export TEST_EPISODE=3406
```

### VM 4

```bash
export WORKER=4
export ANNOTATION_REVISION=03ab39b836c14ffdf463a4c95107a801ed096325
export TEST_EPISODE=7700
```

On each VM after its assignment:

```bash
export ANNOTATION_REPO="kdaterao/so101_spread73_v1_annotations_vm${WORKER}"
export GROUNDING_REPO="kdaterao/so101_spread73_v2_grounded_vm${WORKER}"
```

## 3. Inspect the new point intervals first

This checks source metadata and prints grounding intervals without loading models
or writing output videos:

```bash
python -m robot101.data.preprocess \
  --annotation-repo-ids "$ANNOTATION_REPO" \
  --annotation-revision "$ANNOTATION_REVISION" \
  --annotation-stage captioning --annotation-text caption \
  --episodes "$TEST_EPISODE" \
  --grounding-gripper-subdivision \
  --dst-repo-id "${GROUNDING_REPO}_smoke" \
  --dry-run
```

For an actual one-episode GPU smoke run, repeat it with
`--tapir-frame-batch-size 256 --resume --push-to-hub` instead of `--dry-run`.
Keep that `_smoke` destination separate from the full worker destination.
Inspect point quality before committing the worker batch; a metadata dry run does
not validate tracking or SAM masks. The new GPU path still needs this inspection.

## 4. Rerun all available episodes on each VM

```bash
python -m robot101.data.preprocess \
  --annotation-repo-ids "$ANNOTATION_REPO" \
  --annotation-revision "$ANNOTATION_REVISION" \
  --annotation-stage captioning --annotation-text caption \
  --grounding-gripper-subdivision \
  --tapir-frame-batch-size 256 \
  --dst-repo-id "$GROUNDING_REPO" \
  --resume --push-to-hub
```

`--episodes` is intentionally omitted: each VM processes all available annotation
episodes in its pinned worker repo. This avoids the earlier errors caused by
requesting episodes that never produced annotation artifacts. This does not
recover missing annotation episodes or guarantee all 73 planned episodes exist.

The new grounding flag is on by default and is explicit above for clarity.
It reuses the existing movement-based open/close detector and settings, including
`--gripper-min-change-frac 0.25` and `--min-stage-frames 1`. Detected events split
point computation inside each parent annotation interval; no interval crosses a
parent boundary. Tail-window clustering, SAM masks, endpoint association,
backtracking and overlap run on these point intervals. When no event is detected,
the parent retains one grounding interval. Detection is not proof of a successful grasp.

New `grounding_segment_id` values identify point intervals. Their nested
`annotation.node_id/start_frame/end_frame` preserve the original caption node.
Frame task text and original caption boundaries stay unchanged.

Rerun the identical command after interruptions. Do not resume a `v1` grounding
destination or change its episode selection/annotation revision mid-run. If the
point-interval count exceeds `--max-stages`, inspect it and raise that limit in a
fresh destination. Never silently merge caption boundaries to satisfy the limit.

## 5. Combine on the laptop after the four new uploads finish

The fresh GPU VMs do not need the old unified dataset or its images. Keep that
existing dataset and manifest on the laptop; combining happens there after the
new worker point tracks have been uploaded. Update the laptop checkout to the
same code revision as the VMs before combining.

Keep the original local manifest:

```text
outputs/so101_spread73_v1_unified/manifest.json
```

The combiner's new `--base-manifest` / `--replace-repo` arguments preserve the
exact pinned annotation/Claude inputs and substitute the four grounding repos.
Existing cached source files are reused; the new output directory requires disk
space for another assembled bundle. Do not include both old and new grounding for
the same episode, and do not pass the old unified repo as an ordinary worker repo.

```bash
source .venv-viewer/bin/activate
python -m robot101.data.combine_datasets \
  --base-manifest outputs/so101_spread73_v1_unified/manifest.json \
  --replace-repo kdaterao/so101_spread73_v1_grounded_vm1=kdaterao/so101_spread73_v2_grounded_vm1 \
  --replace-repo kdaterao/so101_spread73_v1_grounded_vm2=kdaterao/so101_spread73_v2_grounded_vm2 \
  --replace-repo kdaterao/so101_spread73_v1_grounded_vm3=kdaterao/so101_spread73_v2_grounded_vm3 \
  --replace-repo kdaterao/so101_spread73_v1_grounded_vm4=kdaterao/so101_spread73_v2_grounded_vm4 \
  --output-dir outputs/so101_spread73_v2_unified \
  --download-workers 1 --rate-limit-wait-minutes 5 \
  --resume
```

This writes locally first. Original inputs retain their existing SHAs; replacements
resolve to their current SHAs once and are pinned in the new manifest. On resume,
later uploads to replacement repos are not automatically picked up. Wait for their
completion before combining, or use another fresh output directory to refresh.

The table still has one row per original caption subtask. Multiple point intervals
are saved under `grounding_json.segments`; original per-interval tracks and masks
remain in the copied sidecars. Missing/failed grounding stays flagged in
`review.json`. Existing unreviewed Claude labels do not become accepted during joining.

For a quick index-only test, add `--lightweight` and use a separate output folder,
for example `outputs/so101_spread73_v2_test`. The viewer fetches media/masks from
the pinned source repos on demand. A lightweight bundle is not a complete media archive.

## 6. View, then optionally upload

```bash
.venv-viewer/bin/python -m streamlit run scripts/view_combined_dataset.py -- \
  --root outputs/so101_spread73_v2_unified
```

The viewer shows full-episode playback. Point/mask intervals can change more often
than the revised caption, which still switches at the original annotation bounds.

After inspection, upload separately without rerunning GPU processing or combining:

```bash
hf upload-large-folder kdaterao/so101_spread73_v2_unified \
  outputs/so101_spread73_v2_unified --repo-type dataset
```

Alternatively add `--push-to-hub --dst-repo-id kdaterao/so101_spread73_v2_unified`
to the combine command. The combiner now uses the large-folder uploader, which
splits uploads into multiple commits instead of exceeding the single-commit limit.

Keep `v1` until the new point tracks and uploads are inspected. No old dataset
is overwritten by this workflow. Existing Claude text is reused, not regenerated
or rewritten into the robot videos' stored frame tasks.
