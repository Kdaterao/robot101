# Finish the episode batch on four VMs

Run annotation, grounding, then Claude aggregation for each VM's assigned episodes.
Ranges have **inclusive endpoints**. This batch contains **73 episodes**;
episodes **3–5 are already computed** and are not rerun here.

Use one GPU job at a time on each VM. Annotation and grounding use different
Python environments. Claude aggregation uses the annotation environment and an API key.
Each worker publishes to separate repositories; these commands do not merge the
four grounded LeRobot datasets into one destination.

## 1. Set up each Ubuntu GPU VM

First commit and push the current code from your development machine. Then run
on **each VM**:

```bash
git clone https://github.com/Kdaterao/robot101.git
cd robot101
bash scripts/setup/linux_setup.sh --profile annotation
bash scripts/setup/linux_setup.sh --profile grounded
source .venv-annotation/bin/activate
hf auth login
mkdir -p tapnet/checkpoints
curl --fail --location \
  https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt \
  -o tapnet/checkpoints/causal_bootstapir_checkpoint.pt
```

For an existing checkout, use `cd ~/robot101` and `git pull` instead of cloning.
The Hugging Face account needs approved access to Perception-LM-3B and permission
to create datasets under `kdaterao`.

## 2. Choose the assignment for this VM

Run **only one** of these blocks per VM. Keep the same shell for the later commands.
If reconnecting, rerun its assignment block and the shared variables block.

### VM 1 — 19 episodes

```bash
export WORKER=1
export EPISODES='10-20:2,490-500:2,688-694:2,890-894:2'
```

### VM 2 — 18 episodes

```bash
export WORKER=2
export EPISODES='896-900:2,1700-1710:2,2000-2010:2,3400-3404:2'
```

### VM 3 — 18 episodes

```bash
export WORKER=3
export EPISODES='3406-3410:2,3690-3700:2,4700-4710:2,5300-5304:2'
```

### VM 4 — 18 episodes

```bash
export WORKER=4
export EPISODES='6840-6850:2,7700-7710:2,10590-10600:2'
```

### Shared variables — run on every VM after its assignment

```bash
export SOURCE_REVISION=e0fda14c2d82b30aa4cbd32b5f972c32af10fd2d
export RUN_TAG=spread73_v1
export ANNOTATION_DIR="outputs/${RUN_TAG}_annotations_vm${WORKER}"
export AGGREGATION_DIR="outputs/${RUN_TAG}_aggregated_vm${WORKER}"
export ANNOTATION_REPO="kdaterao/so101_${RUN_TAG}_annotations_vm${WORKER}"
export AGGREGATION_REPO="kdaterao/so101_${RUN_TAG}_aggregated_vm${WORKER}"
export GROUNDING_REPO="kdaterao/so101_${RUN_TAG}_grounded_vm${WORKER}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

This source revision matches the earlier local PLM3B annotation artifacts. Keep
it identical across workers. Before combining already-computed episodes 3–5,
confirm their `source_revision` matches too; episode IDs can change across source
releases. Use a different `RUN_TAG` if intentionally changing processing settings.

## 3. Annotate — run on all four VMs

```bash
source .venv-annotation/bin/activate
python -m robot101.data.annotate \
  --src-repo-id felsager/community_dataset_v3_ee_smolVLA \
  --src-revision "$SOURCE_REVISION" \
  --episodes "$EPISODES" \
  --output-dir "$ANNOTATION_DIR" \
  --device cuda \
  --video-model facebook/Perception-LM-3B \
  --image-model Qwen/Qwen3-VL-8B-Instruct \
  --caption-quantization none \
  --max-caption-tokens 512 \
  --video-batch-size 24 --image-batch-size 24 \
  --resume --push-to-hub --dst-repo-id "$ANNOTATION_REPO"
```

These are explicit VM assignments; do not add sharding flags. Before continuing,
inspect the printed caption results, local `failures.json`, and HTML reports.
An upload can contain failed captions. Rerun the same command to retry failures
and reuse compatible completed work. If batch 24 runs out of VRAM, lower the
batch sizes and use a fresh run tag/output configuration as required by caching.

## 4. Ground the annotated subtasks

Run on each GPU VM after its annotation output is ready. Claude is not required:

```bash
source .venv-grounded/bin/activate
python -m robot101.data.preprocess \
  --annotation-repo-ids "$ANNOTATION_REPO" \
  --annotation-stage captioning --annotation-text caption \
  --episodes "$EPISODES" \
  --dst-repo-id "$GROUNDING_REPO" \
  --tapir-frame-batch-size 256 \
  --resume --push-to-hub
```

Using explicit `caption` requires completed original captions. Those captions
supply the grounding object text and destination frame tasks. The adapter loads the pinned source
videos and uses the selected annotation boundaries, preserving their node IDs.
Gripper events subdivide those intervals for grounding points only; the annotation
cut and full episode video remain unchanged. Use a fresh grounding destination
when switching from the earlier one-interval-per-caption behavior. Videos remain clean; grounding
points, masks, annotation provenance, and review status are sidecars.

For a metadata-only compatibility check first, run the same command with
`--dry-run` instead of `--resume --push-to-hub`. This does not validate GPU
inference or grounding quality. The full annotation-to-grounding GPU path still
needs an initial inspected test run before committing the whole batch.

On the same VM, `--annotation-input-dirs "$ANNOTATION_DIR"` can replace
`--annotation-repo-ids "$ANNOTATION_REPO"` to use local caption artifacts.
Do not run annotation and grounding simultaneously on the same GPU.

## 5. Aggregate selected subtasks with Claude

Run after grounding finishes, or later on your local machine from the uploaded
annotation repository. Supply your actual API key securely:

```bash
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY
echo
```

```bash
source .venv-annotation/bin/activate
python -m robot101.data.aggregate_annotations \
  --input-dir "$ANNOTATION_DIR" \
  --episodes "$EPISODES" \
  --output-dir "$AGGREGATION_DIR" \
  --model claude-haiku-5-5 \
  --node-scope selected \
  --rounds 3 --concurrency 2 --max-tokens 4096 \
  --resume --push-to-hub --dst-repo-id "$AGGREGATION_REPO"
```

This processes selected subtasks and keeps three refinement rounds. Stillness
measurements and original episode task prompting are off. To opt into stillness
guidance for the first/last selected subtasks, add
`--motion-first-last-only --motion-stillness-threshold 0.03`; the threshold is
0.03 source XYZ units (3 cm only if the coordinates are in meters). To intentionally skip refinement,
use `--no-review` with a new aggregation output directory/configuration.

Inspect `failures.json`, `review.json`, and `annotations_subtasks.jsonl` under
`$AGGREGATION_DIR`. Rerun the same command after resolving failures. Review flags
require inspection; successful API requests do not certify the actions.

Aggregation can instead run on your local machine using `--src-repo-id
"$ANNOTATION_REPO"` in place of `--input-dir "$ANNOTATION_DIR"`, with the
aggregation dependencies installed. No GPU is needed for that step.

## 6. Collect the results

There will be four annotation repos, four aggregation repos, and four grounded
repos, named by the variables above. Preserve local output/cache directories
until their uploads and failure/review reports have been checked.

Claude preserves the segmentation and node IDs, so its refined text can be
associated with grounding by original source episode ID and annotation node ID.
Aggregation does not automatically rewrite the grounded dataset's frame tasks or
sidecars: those retain the original captions used during grounding. Keep the
aggregation output alongside it for downstream training assembly. Grounding
diagnostics are not automatically sent to Claude by these commands.

For a later grounding run that reads multiple annotation worker repos, pass
their names together after `--annotation-repo-ids`, and use the same multi-range
`--episodes` syntax. They must have disjoint episode IDs and the same source SHA.
See [grounded_training_data.md](grounded_training_data.md) for this interface.
