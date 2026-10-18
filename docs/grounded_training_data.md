# Build the grounded robot training dataset

This pipeline converts SO-101 episodes into a LeRobot dataset with wrist-camera
point clusters and third-person grounding from Molmo2/TAPIR. Normal output videos
are clean; points and masks are sidecars. Subtasks come from gripper events for
raw inputs, or the existing selected caption/aggregation cut for annotation inputs.

## Ground an annotated dataset

Annotation Hub datasets contain JSON/text artifacts and references to source
LeRobot videos, rather than rewritten source videos. Use the annotation adapter
instead of passing an annotation repository as `--src-repo-id`:

```bash
python -m robot101.data.preprocess \
  --annotation-repo-ids kdaterao/so101_annotations_plm3b_verified_spread5 \
  --episodes 0 \
  --dst-repo-id kdaterao/so101_annotated_grounded_test \
  --push-to-hub
```

To ground your **local Claude output**, including its rewritten actions:

```bash
python -m robot101.data.preprocess \
  --annotation-input-dirs outputs/aggregated_plm3b_spread5_stillness_001 \
  --episodes 0 \
  --dst-repo-id kdaterao/so101_aggregated_grounded_test \
  --push-to-hub
```

Run grounding in the GPU preprocessing environment described below; the viewer
environment used for Claude may not contain Molmo2/SAM/TAPIR dependencies.

The adapter:

- Loads the exact source dataset **revision and episode IDs** referenced by the
  annotations, in a revision-isolated cache. Later source releases may trim or
  reindex episodes, so switching to the current source revision is not allowed.
- Uses `selected_node_ids` as unchanged caption boundaries. By default, detected
  gripper opening/closing events further subdivide each annotation interval for
  **grounding points only**. The detector uses the existing movement thresholds,
  smoothing and dwell settings. Each grounding segment stays inside its parent
  caption, and inherits its text; no caption nodes or video clips are created.
  Event frames belong to the preceding grounding segment and the next starts
  one frame later. No events means one grounding interval for that caption.
  `--no-grounding-gripper-subdivision` restores one tracking interval per caption.
  Half-open annotation bounds `[start,end)` map to inclusive grounding intervals.
  Gaps, overlaps, incomplete coverage, source FPS/frame-count/timestamp mismatch,
  or too many stages are errors; boundaries are never silently merged or replaced.
- Uses each subtask's `aggregation.brief_action` when aggregation is available,
  otherwise its completed original caption. This text supplies the destination
  frame `task` and the noun extraction for that subtask. Structured manipulated
  object/destination fields are also prioritized as third-person candidates.
- Preserves the POV tail window, next-stage extension, SAM exclusion/weighting,
  backward TAPIR tracking, transition overlap and exact-endpoint stationary
  third-person fallback behavior. Annotated stages use local point selection:
  **cross-demo funneling by numeric stage index is disabled**, because different
  episodes need not contain matching actions at the same stage number. The shared
  appearance-query bank remains; begin validation with one or a few episodes.
  If no usable POV queries exist, annotated runs continue third-person grounding
  and record the missing POV points explicitly.
- Saves source node IDs, original annotation bounds/text provenance and review
  status in `point_tracks/epNNNNNN.json`, copies the annotation episode to
  `point_tracks/annotations/epNNNNNN.json`, and records input fingerprints in
  `meta/grounding_annotations.json`. These sidecars are uploaded with the output.
  `annotation_review_required` stays true for unreviewed/truncated annotations;
  successful tracking does not validate the original caption.

Each point-track record has a `grounding_segment_id`, `interval_role=grounding_only`,
its own tracking bounds, and an `annotation` link retaining the parent's original
node ID/bounds. Tail clustering, SAM windows, endpoint selection and transition
overlap operate on these grounding intervals. Annotation frame tasks and Claude
captions retain their original boundaries. The combiner groups grounding intervals
under their parent caption; the viewer switches captions at annotation boundaries.
Use a **new destination repo** when enabling this on older grounding runs: resume
rejects the changed point-subdivision policy. Gripper thresholds still need tuning
for the data; extra subdivisions do not certify grasp/release success.

Defaults: `--annotation-stage auto` prefers `aggregation/` when it exists on a
Hub repo, otherwise `captioning/`. For local inputs, point to the run directory
containing `episodes/`. `--annotation-text auto` prefers aggregated brief actions
and explicitly records any fallback to an original caption. Choose `caption`,
`brief_action`, `detailed_action`, `brief_caption` or `detailed_caption` to require
a specific text field. An explicit unavailable field is an error. Source task
text is not used as a hidden replacement for a missing annotation.

Multiple disjoint VM outputs are supported:

```bash
python -m robot101.data.preprocess \
  --annotation-repo-ids YOUR_USER/annotations_vm1 YOUR_USER/annotations_vm2 \
  --dst-repo-id YOUR_USER/combined_grounded \
  --push-to-hub
```

The same works with multiple paths after `--annotation-input-dirs`. All inputs
must refer to one raw source dataset/revision; duplicate episode IDs are rejected.
Omit `--episodes` to process every available annotation episode, or supply the
same inclusive comma-separated/stepped syntax as annotation generation. Raw mode
still defaults to episode 0. For large runs, process manageable episode groups;
this adapter does not benchmark or remove the shared query bank's memory cost.

Use `--dry-run` to validate source metadata and print the annotation stages
without loading models or writing a dataset (source metadata/state parquet is
downloaded as needed, but videos are not). Use a fresh destination for annotation-based output. `--resume`
rejects changing annotation sources, revisions, text mode, selection or boundaries,
and rejects mixing raw gripper-stage output with annotated output.

## Environment and raw-input workflow

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
  default is 256; lower it if GPU memory runs out.
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
stage-tail frames plus the first 10 frames of the following stage by default,
then tracked backward and saved over the original stage. Set
`--pov-next-stage-frames 0` to disable that overlap. Molmo2 grounds task objects
and uses the gripper to associate them in available third-person views. The
gripper is not used as a fallback goal label: unresolved stages are marked for
human correction. TAPIR tracks selected points and the pipeline saves clean
camera videos, point reports, and robot state/action data. Point coordinates
and visibility are in `point_tracks/`; use `--viz-dir` to save separate overlay
images for review. Invalid episode shapes are recorded as skipped episodes.

## Manually correct third-person points

Open unresolved stages, click the goal object in the top or side camera, and
let TAPIR track the clicks through that subtask:

```bash
python scripts/fix_grounded_points.py \
  --repo-id YOUR_USER/grounded-dataset \
  --episodes 105 --push-to-hub
```

The script opens one source episode by default. Use `--episodes 105-120:5` to
select a stepped range, `--max-episodes 3` to open up to three episodes when no
range is given, or `--review-all` to replace existing model tracks too. Left
click adds a point, right click or Backspace removes the last click, `a`/`d`
steps through valid frames, Enter tracks and saves, `n` skips, and `q` quits.
Corrections are written to the episode sidecar; `--push-to-hub` uploads them.
