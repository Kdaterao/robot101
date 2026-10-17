# Episode segmentation and annotation

This is an **Action100M adaptation**, not an exact reproduction. It preserves
temporal Ward hierarchies and the Tree-of-Captions: midpoint image captions on
retained leaves, video captions on retained internal nodes, and text aggregation
using descendant and root context. It substitutes V-JEPA **2.1 ViT-L**, Qwen3-VL
**8B**, and Claude **Haiku** for the paper's encoder/image/aggregation models.
Qwen2.5-VL-7B-Instruct replaces PerceptionLM for internal-node video captions.
Qwen3-VL-8B-Instruct remains the leaf-image captioner. Both use SDPA. This further
model substitution needs quality and runtime evaluation; a speedup is not assumed.

These commands annotate existing LeRobot episodes. They do **not** run Molmo,
TAPIR, or SAM 2, modify the grounding preprocessor, or train models. Original
videos remain clean. JEPA runs only in offline data creation. Outputs stay local
unless `--push-to-hub --dst-repo-id OWNER/DATASET` is supplied.

## Ubuntu / RTX A6000 setup

```bash
git submodule update --init lerobot
bash scripts/setup/linux_setup.sh --profile annotation
source .venv-annotation/bin/activate
hf auth login
```

The default video model is [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
PerceptionLM approval is unnecessary for the default configuration. Model downloads
occur on the first GPU run; package setup does not download weights or start jobs.
The isolated environment uses Transformers 5.5.4; existing grounding pins remain
unchanged. The first run resolves dataset/model references and the official JEPA
repository ref to immutable commits in `run.json`. Resume uses those same commits.
The encoder checkpoint is downloaded explicitly from Meta's published
`https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` URL;
the adapter does not use upstream testing URLs such as `localhost:8300`.

## 1. GPU segmentation and captioning

Start with 10 episodes:

```bash
python -m robot101.data.annotate \
  --episodes 0-9 \
  --output-dir outputs/annotations_test \
  --device cuda
```

To choose every fifth episode from 0 through 25:

```bash
python -m robot101.data.annotate \
  --episodes 0-25:5 \
  --output-dir outputs/annotations_spaced
```

Resume an interrupted run with the same configuration:

```bash
python -m robot101.data.annotate \
  --episodes 0-9 --output-dir outputs/annotations_test --resume
```

Inspect `reports/ep000000.html` in a browser. It shows hierarchy intervals,
selected subtask spans, captions, and sampled images. All bounds are **[start,
end)** source-frame indices; the final included frame is `end - 1`.

You can run segmentation first with `--stage segment`, then captions using
`--stage caption --resume`. The same command configuration is required. Completed
feature, segmentation and per-node caption artifacts are reused; failed node
captions are retried. Incompatible settings require another output directory.

### Useful arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `--src-repo-id` | `felsager/community_dataset_v3_ee_smolVLA` | Source LeRobot dataset |
| `--src-revision` | Compatible LeRobot revision | Explicit source version |
| `--camera` | `top` | Segmentation and captioning camera |
| `--sample-step` | `4` | JEPA source-frame sampling interval |
| `--window-frames`, `--window-stride` | `64`, `8` | Sampled frames per window and stride; even multiples, stride ≤ window |
| `--jepa-batch-size` | `1` | Concurrent encoder windows |
| `--video-batch-size`, `--image-batch-size` | `1`, `1` | Caption generation requests per GPU batch |
| `--download-workers` | `1` | Hub download workers; higher values can hit rate limits |
| `--decode-batch-size` | `64` | CPU video decode batch |
| `--video-frames` | `32` | Uniform sparse frames per internal-node caption |
| `--max-caption-tokens` | `1024` | Generation cap per caption |
| `--helper-view` / `--no-helper-view` | enabled | Supply synchronized helper-camera frames during captioning |
| `--helper-camera` | `wrist` | Helper camera; primary view and JEPA still use `--camera` |
| `--min-duration` | `0.5` | Keep hierarchy nodes strictly longer than this many seconds |
| `--max-segments` | `8` | Largest candidate count for the exported subtask cut |
| `--silhouette-threshold` | `0.15` | Minimum score for a multi-subtask cut |
| `--jepa-ref` | `main`, resolved once | Official JEPA repository ref |
| `--video-model`, `--image-model` | Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct | Caption model IDs |
| `--video-revision`, `--image-revision` | `main`, resolved once | Model revisions |
| `--dtype` | `bf16` | Inference dtype |

Example increased GPU batching and download concurrency:

```bash
python -m robot101.data.annotate \
  --episodes 0-9 --output-dir outputs/annotations_batched \
  --jepa-batch-size 2 --video-batch-size 2 --image-batch-size 4 \
  --download-workers 4
```

Benchmark before increasing these further. An out-of-memory error is recorded;
the script does not silently change batch size. Use another output directory
when changing settings. `--video-frames 16` and smaller token caps are explicit
efficiency deviations from the paper's captioning recipe.

Caption batches print their starting size, succeeded/failed counts and elapsed
seconds. CUDA OOM and other caption errors print immediately to stderr with
episode, caption kind and affected node IDs; progress counts mean processed
nodes, not guaranteed successes. The final summary prints caption counts, writes
`caption_failures.json` and includes caption failures in `failures.json`.
`summary.json` records `caption_counts`. Failed captions remain review items;
uploads can contain partial results and print an explicit warning when they do.
Fixed batching does not automatically shrink or retry an OOM batch.

Each annotation invocation samples hardware usage approximately once per second.
At completion or interruption it prints peak process-tree CPU usage, host CPU
usage, process-tree RAM, and per-GPU utilization/VRAM, and writes
`hardware_usage.json`. Successful runs also include these measurements in
`summary.json` and upload the report with annotation artifacts.
Process CPU can exceed 100%: 200% means approximately two logical cores. NVIDIA
measurements are whole-device usage, including other jobs, rather than this
process alone. Missing `nvidia-smi` measurements remain unavailable. Sampled peaks
may miss brief spikes; summed process RSS can double-count shared memory.
Reports describe the current invocation, including cache reuse, and are replaced
on resume. The existing PyTorch peak-allocation metric remains separate from
device-wide VRAM.

At the end of processing, the script prints each episode's elapsed time with
segmentation, video-caption and image-caption breakdowns, plus total processing
wall time before upload. Episode time is the sum of its processing passes, not
the wait between passes. It includes data/model loading and inspection-image
writing; shared caption model loading is charged to the first episode using it.
Timings describe the current invocation, so resumed cached work takes only its
cache-read time. Partial caption passes retain elapsed timing on interruption.
Per-episode timings are saved in episode sidecars and `summary.json` under
`episode_processing_times`. Hardware/report cleanup is included in total wall
time; Hub upload is excluded from all processing times.

Ward clustering links only neighboring spans. Spatial pooling and overlap
averaging do not add feature normalization. Tubelet features are assigned to
their real sampled frames, ignoring padded positions; padding still provides
model context. The full hierarchy is preserved. A separate adaptive cut chooses
among feasible counts 2–8 using silhouette scores and duration constraints;
constant/weak evidence produces one root span. That cut is a robotics adaptation,
not a method claimed by Action100M. No gripper-event segmentation fallback occurs.

Internal-node video frames are resized to 320×320 and supplied directly without
temporal resampling. Qwen2.5's processor aligns these to 308×308 for its 28-pixel
patch/merge grid, an explicit spatial deviation from the paper. Source FPS,
node duration and sampled indices determine the effective sparse-video FPS;
Qwen2.5 represents timing using uniform spacing rather than per-frame timestamps.
Very short clips may repeat frame indices to satisfy the
requested count; the actual indices and timestamps are saved. Episode task text
is not used to imply observed actions or success. A synchronized wrist/POV view
is supplied as visual helper context by default, an explicit robotics adaptation.

### Synchronized POV helper view

Captioning uses the primary camera (`--camera top` by default) together with a
wrist/POV helper (`--helper-camera wrist`). Each video caption receives two
labeled clips with the exact same source frame IDs, 32 sparse frames per view
by default; image captions receive both camera images at the same midpoint.
Models receive these as separate visual inputs, not a stitched image. JEPA still
segments only the primary camera and its boundaries are unchanged.

Prompts ask for observed actions, object movement and spatial relationships,
without guessing intent or adding grasp-success/failure verdicts. The wrist view
helps clarify contact and occlusion; it depicts the same scene, not additional
objects or actions. Gripper sensor input has been removed from Qwen and Haiku
prompts. Raw gripper metadata may remain in episode sidecars for diagnostics.
The old `--gripper-hints` and `--gripper-open-direction` options are removed.

The POV helper is restricted to local interaction cues (visible contact,
object/finger relationships or separation). Its camera moves with the robot;
an object moving upward in that image does not establish an object lift. World
motion, lifting and translation must be supported by the primary third-person
view. Prompts request detailed, concrete observed actions while excluding generic
scene filler and success/failure audits. A token cap is a maximum, not a target;
untruncated captions can finish well below it. This prompt change requires new
caption and aggregation output directories.

```bash
python -m robot101.data.annotate --episodes 0 \
  --output-dir outputs/annotations_pov_helper \
  --camera top --helper-camera wrist \
  --max-caption-tokens 512 --video-batch-size 4 --image-batch-size 4 \
  --push-to-hub --dst-repo-id kdaterao/so101_action_annotations_pov_helper
```

Two views increase visual tokens and VRAM per node. Start with smaller batches
than the old single-view batch of 36 and measure usage before increasing them.
Use `--no-helper-view` for a primary-only comparison. If both camera options name
the same camera, it is sent only once. If any requested helper frame for a node
is missing, or helper decoding fails, only its primary view is used; the result's
`helper_status` explains why. No black placeholder or nearby-frame substitution
is sent as a helper observation.

Each caption records its actual camera views, sample IDs, prompt version and
helper availability. Primary inspection images retain the existing paths; helper
images are saved under `images/epXXXXXX/wrist/`. The browser viewer and HTML
reports display both views with camera labels. Original videos remain clean.

Use new caption and aggregation output directories for this prompt/input change;
resuming old sensor-biased captions will not replace them. Compatible JEPA feature
and segment caches can be reused with the procedure below. For Haiku, use the
new annotation repository with `--no-review` if only the initial annotation is
wanted; no images or gripper measurements are sent to that text-only API.

### More detailed descriptions

Caption prompts request thorough descriptions of the visible action phases,
temporal sequence, contact, object movement, appearance and spatial relationships.
Haiku keeps `brief_action`/`brief_caption` short and preserves supported detail
in `detailed_action`/`detailed_caption`. Prompts ask for useful specificity without
inventing events or padding the result with repeated sentences.

For low-motion intervals, prompts request comparison across video frames before
using action verbs. Mostly unchanged scenes should be described as mostly still
or with only the observed small adjustments, rather than inferred reaching or
grasping from posture. Clearly visible finger/contact/object changes are still
reported even if arm movement is small. This is prompt guidance, not a numeric
motion threshold or a change to segmentation.

Token defaults are unchanged. If a previous command used a 512-token cap, replace
it with `--max-caption-tokens 2048` for longer Qwen descriptions. For aggregation,
`--max-tokens 4096` leaves room for detailed fields and diagnostics. These are
ceilings, not minimum lengths, and longer generations may use more time and VRAM.
Start with a small two-view batch when increasing the cap.

```bash
python -m robot101.data.annotate --episodes 0 \
  --output-dir outputs/annotations_detailed --max-caption-tokens 2048 \
  --video-batch-size 4 --image-batch-size 4

python -m robot101.data.aggregate_annotations \
  --input-dir outputs/annotations_detailed \
  --output-dir outputs/haiku_detailed --no-review --max-tokens 4096
```

The detailed-description prompts are versioned. Use new caption and aggregation
output directories so the earlier short captions are not reused from cache.

### Reusing JEPA caches

Changing models requires a new output directory. To reuse completed segmentation
from the previous PerceptionLM run, copy only its feature and segment caches.
Keep source, JEPA and segmentation settings identical; incompatible caches are
rejected. Existing PerceptionLM captions are not copied.

```bash
mkdir -p outputs/annotations_qwen25
cp -r outputs/annotations/features outputs/annotations/segments outputs/annotations_qwen25/
python -m robot101.data.annotate --episodes 0-2 --resume \
  --output-dir outputs/annotations_qwen25 \
  --push-to-hub --dst-repo-id kdaterao/so101_action_annotations
```

For aggregation, pass `--input-dir outputs/annotations_qwen25`.

## 2. Text-only annotation aggregation

### Interactive segmentation and caption viewer

```bash
python scripts/view_episode_annotations.py \
  --repo-id kdaterao/so101_action_annotations --max-episodes 3 --open-browser
```

The viewer shows the selected-subtask timeline, the complete retained hierarchy,
node bounds, caption status/text, and saved sampled frames. Select a timeline
segment or hierarchy node to inspect it. It loads at most three episodes by
default; use `--episodes 0,2` for specific IDs. Hub mode reads the uploaded
`captioning/episodes/` artifacts and downloads inspection images only when
displayed. It pins the dataset revision at startup and does not download feature
caches or raw videos. Restart to see a newer Hub upload. Hub mode requires
`huggingface_hub` and uses `HF_TOKEN` or your saved login for private datasets.
The viewer loads no models and does not play videos.
For local results instead, pass `--root outputs/annotations_batched`; local mode
uses only Python's standard library and refreshes partial captions every ten seconds.

To inspect local Haiku outputs, including rewritten action labels/captions:

```bash
python scripts/view_episode_annotations.py --root outputs/haiku_test --open-browser
```

The Text selector defaults to the rewritten annotation when present; switch to
the original caption to compare. Review diagnostics and critique are collapsed
separately. Original sampled frames remain associated with the node. For Hub-fed
aggregation runs, `caption_source.json` supplies the pinned source repository for
inspection images, downloaded only when displayed. Captions remain viewable if
source images are unavailable. Context-only nodes show original captions.

After uploading aggregation results, view them directly from the Hub with:

```bash
python scripts/view_episode_annotations.py \
  --repo-id kdaterao/so101_action_annotations11 --stage aggregation --open-browser
```

This reads `aggregation/episodes/` rather than `captioning/episodes/`. It uses
inspection images from the same repository's `captioning/` folder or the original
source specified in the uploaded `caption_source.json`.

On a VM, omit `--open-browser` and forward its viewer port from your laptop:

```bash
ssh -L 8767:127.0.0.1:8767 ubuntu@YOUR_VM_HOST
```

Open `http://127.0.0.1:8767` on your laptop. The server binds to localhost by
default. Keep the annotation process and viewer running in separate terminals.

### Run aggregation

Run commands from the repository root. This is a **text-only API step**: it does
not load local vision models or require a GPU. By default it standardizes only
the selected controller subtasks, preserves IDs/boundaries, and flags uncertain evidence for review.
Descendant and root captions remain input evidence, without separate API calls
for those context nodes. The root is processed only if it is itself selected.
This selected-only aggregation is an efficiency adaptation of the full-tree method.
It does **not** merge subtasks, verify video pixels, or run grounding.

#### 1. Activate the environment and authenticate

If you already installed the annotation environment:

```bash
source .venv-annotation/bin/activate
```

For first-time Linux environment setup, use the annotation installer in the setup
section above. Two separate credentials are used:

- `HF_TOKEN` or `hf auth login`: download private source artifacts and upload results.
- `ANTHROPIC_API_KEY`: call Claude. A Hugging Face token cannot replace this key.

For an actual API run, set the key in the current terminal:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
```

Replace the placeholder with your API key; do not put it in a tracked file. A dry
run does not need this key. It still needs network access for Hub input and Hub
credentials when that dataset is private.

If the API key is not scoped to a single workspace, also set:

```bash
export ANTHROPIC_WORKSPACE_ID="YOUR_WORKSPACE_ID"
```

Find the ID in the Claude Console under **Settings → Workspaces**, in the ID
column. The script sends it as the `anthropic-workspace-id` header on every
request. It is optional for workspace-scoped keys. See
[Anthropic authentication](https://platform.claude.com/docs/en/manage-claude/authentication#select-a-workspace).
The script reads these environment variables; it does not automatically load a
`.env` file. After correcting credentials/workspace settings, use `--resume` to
retry failed nodes without changing model/round/token settings.

#### 2. Validate one episode without calling Claude

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id kdaterao/so101_action_annotations11 \
  --episodes 0 \
  --output-dir outputs/haiku_test \
  --dry-run
```

This automatically fetches only the selected `captioning/episodes/ep*.json`
files, validates the trees, and writes text prompts under
`outputs/haiku_test/request_previews/`. It does not download images, videos or
feature caches, call Claude, or upload results. Inspect the previews to confirm
that captions, source bounds, descendant evidence and root context are present.

#### 3. Run the real three-round aggregation

With the API key set, remove `--dry-run`:

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id kdaterao/so101_action_annotations11 \
  --episodes 0 \
  --output-dir outputs/haiku_test \
  --model claude-haiku-5-5 \
  --rounds 3 --concurrency 2 --max-tokens 2048
```

This makes paid API requests and saves results locally. The same output directory
can be used immediately after its dry run. Each selected subtask receives an initial
annotation and two critique/revision rounds. For 3–4 selected subtasks this is
9–12 requests, plus any repairs/retries, rather than 207 requests for 69 nodes.
The script prints the requested, cached and pending node counts before API work.
Missing or truncated captions remain review items;
task text is intended context, not proof of success. Desired completion conditions
are not observed success labels.

To skip the two critique/revision passes, add `--no-review`:

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id kdaterao/so101_action_annotations11 --episodes 0 \
  --output-dir outputs/haiku_single_pass --no-review
```

`--no-review` overrides `--rounds` and makes the effective round count one. Five
selected subtasks therefore need five initial requests, before any repairs or
transient retries. JSON validation, immutable-bound checks, the one malformed
output repair attempt, uncertainty flags and `review.json` remain active. The
flag skips model critique passes; it does not declare annotations human-verified.
Use a new output directory when changing effective rounds. The updated
multiview/temporal evidence prompt also requires a new aggregation output
directory when upgrading older runs.

#### 4. Inspect results, resume, and optionally upload

```bash
cat outputs/haiku_test/summary.json
cat outputs/haiku_test/failures.json
```

Inspect `annotations_subtasks.jsonl` for selected-subtask annotations and
`review.json` for uncertainty/conflicts. `annotations.jsonl` contains the full
hierarchy, with unrequested context nodes marked `aggregation.status=not_requested`;
`nodes/ep000000/` contains requested nodes' requests, refinement rounds,
final annotation and token usage. Check labels against the original footage;
zero schema failures does not establish annotation accuracy.

To continue an interrupted run or upload completed results:

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id kdaterao/so101_action_annotations11 \
  --episodes 0 \
  --output-dir outputs/haiku_test \
  --resume --push-to-hub \
  --dst-repo-id kdaterao/so101_action_annotations11
```

The model/round/token defaults match the explicit command above. Keep these
settings unchanged on resume; completed nodes and checkpointed successful
responses are reused. Failed nodes retry. Upload writes under `aggregation/`,
allowing the same repository to retain `captioning/`. `--push-to-hub` requires a
`--dst-repo-id` and cannot be combined with `--dry-run`.

You can resume an earlier full-tree run in the same directory using the new
selected-only default: compatible selected-node results are reused, unselected
nodes receive no new requests, and they are excluded from failure/review counts
and flat aggregation rows. Old per-node cache files are retained. The complete
hierarchy and original captions remain in episode JSON sidecars.

API requests use [schema-constrained JSON output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
followed by local field and immutable-boundary validation. New suggested changes
use `null` or an object containing a `note`; existing cached suggestion objects
remain compatible. Complete Markdown JSON fences from older responses are
accepted, but malformed JSON is not silently repaired into invented annotations.
One model repair request is still allowed after validation failure. Refusals,
empty responses, unexpected stop reasons and token-limit truncation remain
explicit failures. A valid JSON response does not establish visual accuracy.
After updating this script, rerun the same command with `--resume` to retry
previous parse failures while retaining completed nodes and valid rounds. If the
SDK rejects `output_config`, update it in the active environment:

```bash
python -m pip install --upgrade "anthropic>=0.125,<1"
```

#### Useful arguments and source behavior

| Argument | Default | Purpose |
| --- | --- | --- |
| `--src-repo-id` | unset | Read caption-tree artifacts from a Hub dataset |
| `--input-dir` | `outputs/annotations` when no Hub source is given | Read local `episodes/ep*.json`; mutually exclusive with Hub input |
| `--src-revision` | `main` | Source dataset revision, resolved to a pinned commit |
| `--episodes` | all available | Select episodes, e.g. `0`, `0-2`, or `0-25:5` |
| `--model` | `claude-haiku-5-5` | Claude API model ID |
| `--rounds` | `3` | Initial annotation plus critique/revision rounds |
| `--no-review` | off | Override rounds to one; skip critique/revision passes |
| `--node-scope` | `selected` | Only selected subtasks; `all` restores aggregation of every retained node |
| `--concurrency` | `2` | Concurrent node jobs; rounds within each node stay sequential |
| `--max-tokens` | `2048` | API output cap per request; independent of GPU caption token limits |
| `--resume` | off | Reuse compatible local checkpoints |

Use an explicit episode selection for initial tests. `--episodes 0-25:5` selects
0, 5, 10, 15, 20, 25; all must exist in the annotation repository. Concurrency
and node scope can change on resume; changing model, rounds or token cap requires a new output
directory. Fewer rounds are a departure from the three-round recipe.

Hub input uses the saved Hugging Face login or `HF_TOKEN`. The source commit is
recorded in `caption_source.json` and stays pinned for that output directory,
including after a dry run. Choose a new output directory to read a newer source
snapshot or different repository. Local input instead uses:

```bash
python -m robot101.data.aggregate_annotations \
  --input-dir outputs/annotations_qwen25 \
  --episodes 0 --output-dir outputs/haiku_local --dry-run
```

The core annotation fields are `brief_action`, `detailed_action`, `actor`,
`brief_caption`, and `detailed_caption`. The `robotics` object adds manipulated
object, destination, completion condition, evidence, conflicts, review status
and suggested changes. Suggestions do not rewrite IDs or bounds. Optional
`--grounding-diagnostics diagnostics.jsonl` adds text diagnostics keyed by source
`episode_index`; it does not invoke grounding.

| Problem | Action |
| --- | --- |
| Missing caption artifacts | Use an annotation repo containing `captioning/episodes/ep*.json` and episode IDs it actually contains. A raw LeRobot video dataset is not this input format. |
| Hub 401/403 | Run `hf auth login` with an account/token that can read the source and write the destination. |
| API authentication error | Set a valid `ANTHROPIC_API_KEY` in the same terminal. |
| API 400 asking for `anthropic-workspace-id` | Set `ANTHROPIC_WORKSPACE_ID` to the ID from Claude Console Settings → Workspaces, or use a workspace-scoped key; rerun with `--resume`. |
| Rate limiting | Lower `--concurrency`; transient errors are retried with backoff. |
| Response reached `max_tokens` | Raise `--max-tokens` in a new output directory; truncated responses are not accepted. |
| Incompatible cache | Keep processing arguments/source snapshot unchanged or choose another output directory. |

## Artifacts and validation

### Publish to Hugging Face

Both commands support `--push-to-hub`, using `HF_TOKEN` or your saved `hf auth
login`. They create the destination dataset if needed and upload at the end:

```bash
python -m robot101.data.annotate \
  --episodes 0-9 --output-dir outputs/annotations_test \
  --push-to-hub --dst-repo-id kdaterao/so101_action_annotations

python -m robot101.data.aggregate_annotations \
  --input-dir outputs/annotations_test --output-dir outputs/standardized_test \
  --push-to-hub --dst-repo-id kdaterao/so101_action_annotations
```

The same repository can contain both stages: `captioning/` and `aggregation/`.
Each has a flat `dataset_rows.jsonl` for the Hub viewer, plus full sidecars and
provenance. Choose the `captioning` or `aggregation` dataset configuration to
load rows. Use `selected_subtask` to filter the exported controller cut.
Videos and robot actions are referenced by source dataset/revision and episode,
not copied; this is an annotation dataset, not a grounded LeRobot dataset.

Add `--hub-private` to create a new private repository; existing visibility is
unchanged. API dry runs do not upload and reject `--push-to-hub`. Upload failures
leave local results intact. Rerun with the same processing arguments plus
`--resume --push-to-hub --dst-repo-id ...` to retry upload without reprocessing
completed nodes. Hub options do not invalidate inference caches.

GPU output:

- `features/`: pooled embeddings, sample IDs, overlap counts and metadata.
- `segments/`: complete hierarchy, retained nodes and selected cut.
- `node_captions/`: atomic independent caption results and failures.
- `episodes/`, `captions.jsonl`, `captions_subtasks.jsonl`: caption-tree exports.
- `images/`, `reports/`: sampled frames and HTML inspection.
- `run.json`, `failures.json`, `summary.json`: provenance and timings.

API output: per-node requests/rounds under `nodes/`, complete episode records,
`annotations.jsonl`, `annotations_subtasks.jsonl`, `review.json`, `failures.json`,
and token usage in `summary.json`. Usage includes refinement and repair attempts;
no hardcoded dollar price is reported.

Local tests use synthetic features and mock encoders, captioners, and API clients:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_annotation_pipeline.py
```

Real model compatibility and quality still require the GPU test. Review 10
episodes first, then 100, including quick actions, occlusion, missing frames and
failed demonstrations. Inspect boundaries, caption disagreements, review rate,
CPU memory, VRAM and stage timings. A full caption tree can contain substantially
more nodes than the selected subtasks; earlier per-subtask runtime estimates do
not establish this pipeline's throughput.

Reference: [Action100M Section 3](https://arxiv.org/html/2601.10592v1#S3).
