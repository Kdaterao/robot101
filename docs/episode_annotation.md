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
requested count; the actual indices and timestamps are saved. Prompts describe
images/videos neutrally; intended tasks and gripper values remain separate context.

### Reusing JEPA results after changing video models

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

On a VM, omit `--open-browser` and forward its viewer port from your laptop:

```bash
ssh -L 8767:127.0.0.1:8767 ubuntu@YOUR_VM_HOST
```

Open `http://127.0.0.1:8767` on your laptop. The server binds to localhost by
default. Keep the annotation process and viewer running in separate terminals.

### Run aggregation

First inspect requests without credentials, GPU inference, or API spending:

```bash
python -m robot101.data.aggregate_annotations \
  --input-dir outputs/annotations_test \
  --output-dir outputs/standardized_test --dry-run
```

Set `ANTHROPIC_API_KEY` in your environment, then run:

```bash
python -m robot101.data.aggregate_annotations \
  --input-dir outputs/annotations_test \
  --output-dir outputs/standardized_test
```

Add `--resume` to continue. Defaults: `--model claude-haiku-5-5`, `--rounds 3`,
`--concurrency 2`, `--max-tokens 2048`. Each retained node receives an initial
annotation followed by two critique/revision rounds. Fewer rounds are an explicit
adaptation. Every successful API response is checkpointed before validation;
resume can reuse it after an interruption before round completion.

The API receives captions, source bounds, descendants in depth-first order, root
context, and task metadata. No images are sent. It cannot verify pixel grounding
or certify success. Desired completion conditions are not success labels.
Missing/truncated captions and suggested changes require review. Unlike the
paper's final four-second cutoff, short robot subtasks remain eligible.

The core fields are `brief_action`, `detailed_action`, `actor`, `brief_caption`,
and `detailed_caption`. A `robotics` object adds manipulated object, destination,
completion condition, evidence, conflicts, review status, and suggested changes.
IDs and bounds are immutable; suggestions never rewrite them. Optional
`--grounding-diagnostics diagnostics.jsonl` adds text diagnostics keyed by source
`episode_index` for later integration. It does not invoke grounding.

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
