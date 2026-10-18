# Combine all worker datasets

To replace old point tracks while keeping existing annotation/Claude results,
follow [Rerun grounding](rerun_grounding.md). The combiner supports
`--base-manifest existing/manifest.json` with repeatable `--replace-repo OLD=NEW`.
Unreplaced inputs retain their pinned revisions; replacements resolve once.

Run on your laptop after the worker uploads finish. No GPU or API inference is
used. This downloads and copies the complete input datasets, including videos;
allow disk space for both the Hub cache and the assembled output.

```bash
source .venv-viewer/bin/activate

REPOS=()
for WORKER in 1 2 3 4; do
  REPOS+=("kdaterao/so101_spread73_v1_annotations_vm${WORKER}")
  REPOS+=("kdaterao/so101_spread73_v1_aggregated_vm${WORKER}")
  REPOS+=("kdaterao/so101_spread73_v1_grounded_vm${WORKER}")
done

python -m robot101.data.combine_datasets \
  --repos "${REPOS[@]}" \
  --output-dir outputs/so101_spread73_v1_unified \
  --download-workers 4 \
  --rate-limit-wait-minutes 5 \
  --resume --push-to-hub \
  --dst-repo-id kdaterao/so101_spread73_v1_unified
```

Every listed repo must exist and be accessible. Omit a repository from the list
if that stage has not been published; missing stages are recorded for review.
Additional repos, such as already-computed episodes 3–5, can be appended if they
use the same artifact format. Duplicate episodes within the same stage are
rejected. Different source revisions remain distinct identities.

## Contents

### Quick lightweight test

Build a small worker-2 test from the published datasets, skipping image/video
bulk downloads. Use a fresh directory separate from the full download:

```bash
.venv-viewer/bin/python -m robot101.data.combine_datasets \
  --repos kdaterao/so101_spread73_v1_annotations_vm2 \
          kdaterao/so101_spread73_v1_aggregated_vm2 \
          kdaterao/so101_spread73_v1_grounded_vm2 \
  --output-dir outputs/so101_unified_test_vm2 \
  --lightweight --download-workers 1 --rate-limit-wait-minutes 5 --resume
```

Then launch the Streamlit viewer below with
`-- --root outputs/so101_unified_test_vm2`. Videos and masks download only when selected,
from the original pinned worker repos. These source repos must remain accessible.
This mode still downloads episode JSON and metadata, so normal Hub limits apply.
To upload this optional test bundle, add `--push-to-hub --dst-repo-id
kdaterao/so101_unified_test_vm2` to the same combine command. Lightweight mode
does not change an existing full-download manifest; use a separate output folder.

- `subtasks.jsonl`: one joined row per selected subtask, keyed by source dataset,
  source revision, original episode ID, and node ID. Includes original captions,
  refined Claude captions/actions, annotation JSON, grounding JSON, and paths.
- `artifacts/inputNN/`: complete pinned input snapshots. This includes clean or
  debug videos as originally published, state/action parquet, masks, full caption
  trees, review records and model provenance. The manifest maps each path to its repo.
- `review.json`: missing/failed stages and existing material review flags.
- `manifest.json`: pinned input revisions and resume configuration.
- `summary.json`: episode/subtask/review counts.

Annotation and aggregation cuts must match exactly. Grounding must contain
annotation node IDs with matching parent bounds. Multiple grounding-only
intervals can cover one parent caption contiguously; the joined table still has
one row per original caption, with `grounding_json.segments` for multiple intervals.
Old raw gripper-segmented outputs
cannot be joined automatically. Grounding's inclusive end frames are converted
to half-open bounds for matching. Missing episodes/stages are retained rather
than silently dropped. The tool does not resolve bad annotations or grounding.

**This is one indexed Hub dataset containing all worker artifacts, not a single
reindexed LeRobot dataset.** Worker-local episode IDs, video offsets and relative
mask paths stay valid under their original artifact directory. Use
`grounding_dataset_path` and `grounding_destination_episode` to locate robot
data, and source identity/node ID to join refined labels. Videos and existing
frame task strings are not rewritten with Claude's text.

The Hub `subtasks` configuration can be loaded with:

```python
from datasets import load_dataset

rows = load_dataset("kdaterao/so101_spread73_v1_unified", "subtasks", split="train")
```

Rerun the same command to resume interrupted downloads/copy/upload. Input SHAs
are pinned on the first run; `--resume` does not pick up newer worker commits.
Use a new output directory for refreshed inputs or a changed repository list.
HTTP 429 errors automatically pause for five minutes and then retry the same
operation using cached downloads. `--rate-limit-wait-minutes N` changes that
pause; server-provided reset/retry delays take precedence if longer. The SDK may
perform its own retries before surfacing an error. Repository resolution,
downloads and uploads use this handling. Countdown messages show when it will
resume. The default `--rate-limit-retries 0` keeps retrying rate limits until
completion or Ctrl+C; set a positive value to cap retries per operation. Other
errors still stop immediately. After Ctrl+C, rerun with `--resume`. Retry settings
and download worker counts can change without invalidating the output manifest.
Remove `--push-to-hub` to assemble locally. Source licensing remains with the
original artifacts; the combiner does not assign a replacement license.

## Streamlit viewer

```bash
.venv-viewer/bin/python scripts/install_requirements.py --profile viewer
.venv-viewer/bin/python -m streamlit run scripts/view_combined_dataset.py
```

The page contains only a **Source episode** selector and the top/wrist playback
videos with Claude's revised caption above each. Every source episode is listed
by default; no pagination or review filter hides episodes. The selector loads
joined JSON metadata, not all videos. Only the selected episode's videos and
masks are fetched. Playback runs across the whole episode, updating the caption
at subtask boundaries. Masks and visible grounding dots are rendered in cached
H.264 previews; stored videos stay unchanged. Camera players are independent.

The default uses `outputs/so101_spread73_v1_unified` if it contains
`subtasks.jsonl`; otherwise it uses `kdaterao/so101_spread73_v1_unified` on the Hub.
Choose another source through launch arguments, without adding controls to the page:

```bash
.venv-viewer/bin/python -m streamlit run scripts/view_combined_dataset.py -- \
  --root outputs/so101_unified_test_vm2
```

```bash
.venv-viewer/bin/python -m streamlit run scripts/view_combined_dataset.py -- \
  --repo-id kdaterao/so101_spread73_v1_unified
```

Optional `--revision SHA` pins the Hub input. For private repositories use
`hf auth login` in the viewer environment. Autoplay is muted and browsers may
require clicking Play. Missing videos/captions are reported without fabricated
replacements. Restart Streamlit to reload a modified local dataset immediately.
