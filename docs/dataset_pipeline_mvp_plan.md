# Two-day dataset pipeline MVP plan

## Goal and architecture

Build and validate an offline pipeline that annotates existing SO-101 episodes,
adds spatial grounding, and exports training data for a **VLM planner/validator
and a smaller VLA controller**. The production target is 1,000–1,500 episodes.
These are existing demonstrations being annotated, not newly recorded episodes.

**Implementation status:** the segmentation/caption-tree and text aggregation
commands are now documented in [episode annotation](episode_annotation.md).
That guide records the implementation's Action100M hierarchy, leaf midpoint
captions, 32-frame internal-node video captions, and three aggregation rounds.
The earlier lightweight captioning proposals below are historical alternatives.
Connecting these sidecars to the grounding preprocessor remains a separate change.

```text
Source robot episodes
  → V-JEPA 2.1 + temporal clustering
  → PerceptionLM video captions + Qwen keyframe captions
  → Molmo + BootsTAPIR + SAM 2 grounding
  → Claude Haiku annotation standardization and consistency checks
  → human review + clean videos, robot data, and annotation sidecars
```

JEPA is used **only for offline dataset segmentation**. The deployed robot uses
VLM + VLA; no JEPA planner or world-model controller is included.

## Schedule

| Task | Working budget | Deliverable |
| --- | --- | --- |
| Correct gripper point labels and rerun existing Molmo fine-tune | 1–3 hours | Updated connector and held-out localization checks |
| Implement segmentation and annotation via Codex | 1 hour target | Runnable GPU pass with cached outputs |
| Validate/debug segmentation and captions on roughly 10 episodes | 1–2 hours | Inspected boundaries, captions, and sampling |
| Validate/debug the full pipeline on 100 episodes | 5 hours | Reviewed outputs, throughput measurements, and resume checks |
| Generate 1,000–1,500 annotated and grounded episodes | 24–48 hours target | Exported dataset and review queue |

The one-hour implementation slot is optimistic: dependency compatibility,
feature-to-frame alignment, and integration fixes may extend it. The production
runtime is a target, not a benchmark. Launch Molmo training while implementing
the annotation pass. The two-day goal is a validated MVP; a production run may
finish afterward. Training the final planner/VLA and robot evaluation are separate.

## Implementation: separate, resumable passes

### 1. GPU segmentation and captioning script

Use one script with independent cached stages, starting with the existing
LeRobot/Hugging Face source loading and episode selection conventions. Default
source: `felsager/community_dataset_v3_ee_smolVLA`. Preserve source episode IDs,
frame indices, FPS, camera names, and task text throughout processing.

**Segmentation:**

- Initial encoder: official **V-JEPA 2.1 ViT-L at 384 resolution** (~300M
  parameters). Extract dense temporal features without a planning predictor.
- Adapt Action100M's method: spatially pool features, average overlapping
  temporal observations, and apply hierarchical agglomerative clustering with
  Ward linkage and adjacency constraints. Only contiguous spans may merge.
- Use Action100M's sampling recipe as an initial experiment: one in four source
  frames, 64 sampled frames per window, stride eight. Expose these settings and
  window batch size; verify their compatibility with the actual 2.1 encoder.
- Map temporal tokens back to source frame intervals explicitly. Padding must
  not generate observations or segments beyond the episode. Handle short clips
  and final windows without losing endpoint coverage.
- Choose a usable segmentation level rather than exporting every hierarchy
  node as a controller subtask. Expose granularity and duration settings and
  tune on the inspected batch. Do not copy Action100M's four-second annotation
  cutoff: quick robot grasp/release phases must remain representable.
- Gripper events remain useful context and a comparison baseline. The new path
  derives boundaries from video features; it must not silently fall back to
  gripper splitting when JEPA processing fails.
- Start with the top camera, recording camera provenance. Missing input must
  produce an explicit skipped/review record rather than fabricated observations.

**Complementary captions:**

- Planned video captioner: **PerceptionLM-3B** for action sequence, interaction,
  and state changes. Default to **16 sampled frames per subtask**; expose a
  configurable cap, including 32 for longer/ambiguous spans.
- Sample ordered frames evenly, include endpoints, and prioritize frames around
  important gripper events within the cap. Pass sampled timing information when
  supported; do not mistake sparse frames for a contiguous full-FPS clip.
- Planned image captioner: **Qwen3-VL-8B-Instruct** for objects, spatial relations,
  and fine details. Initially caption two keyframes per subtask: start and end,
  with timestamps and their roles recorded. Expose keyframe selection and model
  choice. Qwen3-VL-4B in both captioning roles is a lower-compute comparison.
- Supply the original task as context, while instructing models to describe
  visible evidence, distinguish uncertainty, and avoid assuming task success.
- Keep descriptions concise and cap generated tokens. Save video and image
  captions separately so disagreements remain inspectable.
- Run JEPA, PerceptionLM, and Qwen sequentially in model passes on the A6000,
  releasing each before loading the next. Batch compatible requests within each
  pass; avoid swapping models for every subtask.

**Output:** feature caches, segmentation records, and per-episode caption JSONL
with subtask IDs, source frame bounds, sample frame IDs/timestamps, task context,
captions, provisional targets, uncertainty, and model/config provenance.
Provisional action/object/destination fields feed grounding; final labels come later.

### 2. Separate GPU grounding pass

Modify the current preprocessing entrypoint to accept the segmentation/caption
sidecars. Use the supplied boundaries and per-subtask targets instead of inferring
all targets from episode-level task nouns. Preserve states, actions, timestamps,
and source-to-destination episode mappings.

- **POV:** retain tail-window candidate clustering, the default 10-frame extension
  into the next subtask, selected-point backward BootsTAPIR tracking, and transition
  overlap. SAM 2 stays POV-only, excluding gripper-mask endpoints/observations and
  applying configurable proximity weighting (default maximum 2×, taper to 1× over
  5% of image diagonal). Missing/invalid masks require review.
- **Across demonstrations:** do not align new segments solely by ordinal stage
  index. Group compatible provisional task/action/object/destination labels;
  uncertain or unmatched segments use local selection rather than forced alignment.
- **Third person:** discover object candidates at the exact subtask start, track
  forward, and select by endpoint proximity to the gripper using the existing
  tracking settings (default 1 FPS). Use the provisional target fields to guide
  candidate queries. No SAM 2 segmentation on these cameras in this milestone.
- **Fallback:** only a valid detection on the exact subtask-end frame may supply
  the third-person gripper fallback. Freeze that point's pixel coordinates through
  the subtask and transition overlap; never follow the moving gripper or substitute
  a nearby frame. Record `gripper_fallback`; unresolved cases go to human review.
- Store points, visibility, mask sidecars, failures, and provenance. Keep normal
  videos clean; use `--debug-overlays` in a separate test dataset for masks and
  dots without heatmaps. Preserve manual correction tools.

The existing tracker is **BootsTAPIR**, not TAPNext++. Tracker migration is outside
this implementation. Molmo grounding is learned inference; deterministic selection
rules do not guarantee that the proposed points are correct.

### 3. Separate cloud API aggregation script

Use **Claude Haiku** with text-only inputs: episode task, temporal boundaries,
video/image captions, provisional targets, and summarized grounding diagnostics.
Caption generation and grounding run on the cloud GPU, not through Haiku.

- Standardize action, manipulated object, destination, goal/completion condition,
  supporting evidence, conflicts, and review status into structured JSON.
- Start with one pass. Retry/refine uncertain annotations rather than automatically
  running three full rounds for every subtask. Make refinement count configurable.
- Haiku checks textual consistency; it does **not** ground pixel points or verify
  masks visually. Self-refinement does not establish independent factual correctness.
- Preserve original boundaries, captions, and targets. If aggregation proposes a
  changed target or boundary, mark the grounding as requiring review/reprocessing;
  do not attach old tracks to a new interpretation silently.
- Require schema validation, per-request checkpoints, bounded concurrency, retry
  backoff, and input/output usage logging. Resume must skip completed requests.
- Take credentials from the environment. Record provider/model and prompt version.
  No API calls are required to build or locally test the script.

## Molmo label correction

The gripper training workflow already exists. The task is to correct examples
that need two points but were labeled with only one, then rerun preparation and
fine-tuning using the existing multiple-point format.

Keep the point meaning consistent between labels and inference prompts. The
existing collection guide describes jaw-center points per visible gripper; that
is distinct from finger/body points used to prompt a SAM 2 mask. Do not assume
that every image needs exactly two points or that a jaw-center detector establishes
reliable body localization. Label visible targets only, and check open jaws,
occlusion, POV views, and third-person views on held-out images.

## Caching, interfaces, and export

- Use a stable episode/subtask identity and one explicit source-frame boundary
  convention across all passes. Validate ranges and source dataset revision.
- Cache embeddings, captions, grounding, and aggregation separately. Include
  model revision, sampling parameters, and prompt/config fingerprints so changing
  a stage cannot silently reuse incompatible results downstream.
- Write records atomically and keep completion/failure manifests. A failed episode
  must not erase completed work; distinguish retryable processing errors from
  valid but unresolved annotations needing a person.
- Retain both raw evidence and standardized annotations alongside clean videos
  and robot state/action data. Support local output and explicit Hugging Face upload.
- Record per-stage wall time, throughput, CPU RAM, peak VRAM, API usage, and human
  correction rate. Time downloading/decoding/encoding separately from inference.

## Validation and production gate

**First roughly 10 episodes:** inspect feature/frame alignment, short/final clips,
subtask granularity, sparse frame coverage, and video/image caption consistency.
Check that captions distinguish observed actions from intended task outcomes.

**Then 100 full episodes:** use reproducible random selection and add coverage
of quick gripper reversals, occlusion, small objects, intermediate release phases,
missing camera frames, and unsuccessful demonstrations. Inspect every subtask's
boundaries, labels, point alignment, and mask/fallback behavior using the viewer.

Test interruption/resume, malformed captions/API output, partial episode failures,
cache invalidation, unresolved targets, and API-induced target changes. Confirm
that exported videos remain clean and sidecars refer to the correct frames.

Before production, summarize boundary errors, unsupported labels, incorrect goals,
review/correction counts, and measured processing time. Fix systematic errors and
inspect a fresh sample after tuning; do not use the tuning batch as the final
quality claim. Launch the large run only after these checks pass.

The earlier ~90-second grounding baseline implies approximately **25 hours for
1,000 episodes or 37.5 hours for 1,500** on one worker, before segmentation and
captioning. New segmentation/SAM settings may change this. Prior captioning-time
tables were scenarios, not A6000 measurements. Use the 100-episode benchmark to
estimate production time; separate VMs can reduce wall time with disjoint shards.

Goal/completion labels describe desired outcomes. Training a reliable VLM validator
also requires incomplete/failed/successful examples and separate evaluation; goal
points alone are not success supervision.

## Open-source packaging after validation

Expose annotation, grounding, and aggregation as separate resumable commands.
Provide one documented setup entrypoint with appropriate environment profiles,
model/config options, episode ranges, batching, worker settings, API usage logging,
and a small reproducible example. Reuse the review viewers and correction tools.
This milestone does not require a new general-purpose UI or broad dataset support.

## References

- [Action100M methodology](https://arxiv.org/html/2601.10592v1#S3): segmentation,
  complementary image/video captions, and annotation aggregation. Adapt the method;
  do not claim that this pipeline reproduces its model choices or results.
- [Official V-JEPA 2/2.1 repository](https://github.com/facebookresearch/vjepa2).
- [PerceptionLM-3B](https://huggingface.co/facebook/Perception-LM-3B): gated model
  access and noncommercial research license; check output-use terms when packaging.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- Existing guides: [Molmo labels](molmo_gripper_point_data.md),
  [Molmo training](molmo2_training.md), [grounding](preprocessing.md),
  and [SAM 2 filtering](sam2_contact_point_filtering.md).
