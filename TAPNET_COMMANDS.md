# TAPIR / RoboTAP commands

Run from the repo root (`robot101/`). Use your venv Python if needed:

```powershell
.\.venv\Scripts\python.exe src\...
```

or, with the venv activated:

```powershell
python src\...
```

Camera intrinsics (`CAM_*`) live at the top of `src/tapnetCreate.py` and `src/tapnetGrab.py` (1080p wide-angle estimate by default). Edit those if you recalibrate.

**Undistort toggle:** set `UNDISTORT = True/False` at the top of both files, or pass `--undistort` / `--no-undistort` to either script. Create and grab must use the same setting (goals are stored in pixels). The task records which one it used, and tapnetGrab warns if they don't match. Recordings are always saved raw.

Pipeline (RoboTAP paper): demos are split into **stages** at grasp / release. Each stage servos a set of active points to where they ended in the demos, then runs the stage's primitive (close until clamp / open).

---

## 1. Record demos

Teleop with SO-101 leader, save wrist-cam clips under `data/demo_XXX/`.

```powershell
python src/tapnetRecord.py --out-dir data --camera 0 --port COM3 --leader-port COM4
```

Defaults are already `1920x1080`, `COM3` / `COM4`.

**Controls**

- Start episode when prompted (`y`)
- `q` — end episode
- `p` — print joints
- `space` or `e` — end the current stage (manual boundary; prints `[STAGE END]`); use this for approach poses where clamp detection does not apply
- Keep/discard prompt after each episode (`y` / `N`)

**Grasp detection:** when you squeeze an object with the leader and the follower gripper stalls on it, the terminal prints

```
demo_009 [CLAMP] frame=143 grip=31.2 cmd=12.0
```

and opening again prints `[RELEASE] ...`. These (plus `space`/`e`) are saved to `data/demo_XXX/events.json` and become stage boundaries. If you don't see `[CLAMP]` when you grab the object, press `space` or `e` at that moment instead.

Every demo should have the same grasp/release sequence (e.g. one grasp then one release → 3 stages).

---

## 2. Create a task (offline)

Shared TAPIR queries are tracked through all demos, then each stage's active points are chosen (moving + visible at the stage end + same endpoint across demos + motion-cluster voting).

```powershell
python src/tapnetCreate.py --data-dir data --out tasks/pick_place_task.npz
```

Faster smoke run:

```powershell
python src/tapnetCreate.py --demo data/demo_001 --demo data/demo_002 --out tasks/test.npz --num-sample-points 128 --num-active 32
```

Useful options:

| flag | default | meaning |
|---|---|---|
| `--viz` | off | save `<out>_viz.png` with each stage's goal points |
| `--num-sample-points` | 512 | total shared query points |
| `--num-active` | 128 | active points per stage |
| `--n-clusters` | 6 | motion clusters per stage |
| `--static-thresh` | 0.03 | below this motion (fraction of image diagonal) a point is static (gripper / held object) |
| `--funnel-quantile` | 0.4 | fraction of moving points kept as voters, by lowest endpoint spread across demos |
| `--goal-tail-frames` | 5 | stage endpoint = median of last N visible frames |
| `--segment episode` | events | ignore `events.json`, one stage |

Demos without `events.json` (older recordings) are a single stage. Old task files (single goal) must be rebuilt.

---

## 3. Run — trajectory following (`tapnetGrab`)

Follows nearest-demo-frame targets (then mean goal near stage end). Modes: `viz`, `servo_print`, `ik_print`, `robot`.

```powershell
python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode viz --camera 0 --port COM3
python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode servo_print --camera 0 --port COM3
python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode robot --camera 0 --port COM3
```

**Keys:** `q` quit / e-stop, `n` force next stage.

Stage ends when **inlier** mean pixel error ≤ `--converge-px` (default **40**) for `--converge-frames` (default **1**), then runs the primitive. Servo uses the same inlier subset (`--inlier-frac`, `--inlier-min-cos`); green = used, orange = rejected. Demo progress only switches to the mean goal (`--end-frac`); it does **not** finish the stage unless you set `--advance-progress` > 0.

**Test descend:** after stage 0 hits that pixel checkpoint, optionally drop straight down in world `-Z` before the primitive:

```powershell
python src/tapnetGrab.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --test-down-cm 3 --freeze-after-first
```

Overlay: stage, `demo d fT` or `MEAN goal`, pixel error, gripper state.

---

## 3b. Run — direct goal IBVS (`tapnetGrabGoal`)

Same task.npz and stages, but **only** servos toward each stage's fixed `goal_mean` with an image Jacobian + IK. No demo-trajectory matching. Prefer this when goals look right but demo-following motion feels odd.

```powershell
python src/tapnetGrabGoal.py --task tasks/pick_place_task_steps.npz --mode servo_print --undistort
python src/tapnetGrabGoal.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --freeze-after-first
python src/tapnetGrabGoal.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --first-primitive release --depth 0.20
```

**Keys:** same (`q`, `n`). Once **inlier** mean error to `goal_mean` is ≤ `--converge-px` for `--converge-frames` (default **1**), the servo step ends and the stage primitive runs (or the next stage if `none`). IBVS uses only a consistent subset of points (`--inlier-frac`, `--inlier-min-cos`) so background tracks do not pull into a local valley. Overlay: green = inliers used, orange = visible but rejected. `--depth` is assumed point depth Z (m) for the Jacobian (default 0.20).

---

## 3c. Run — hybrid GrabGoal → greedy (`tapnetGrabGreedy`)

Same task.npz / stages / gripper primitives as GrabGoal. **Default:** analytical GrabGoal IBVS (assumed `--depth`) until inlier error ≤ `--analytical-until-px` (default **80**), then latches to probe + discrete greedy ±axes for fine approach. Set `--analytical-until-px 0` for pure greedy from the start.

```powershell
python src/tapnetGrabGreedy.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --freeze-after-first
python src/tapnetGrabGreedy.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --analytical-until-px 80 --depth 0.20
python src/tapnetGrabGreedy.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --analytical-until-px 0
```

**Keys / converge / inliers / freeze:** same as GrabGoal. Greedy probing needs `--mode ik_print` or `robot`. Terminal: `[SWITCH] ... → probe+greedy`, `[PROBE]`, `[GREEDY]`.

---

## 3d. Run — IBVS + gripper orientation (`tapnetGrabPose`)

Like GrabGoal for pixels, but also aligns the gripper quaternion to the **demo end-of-stage joint pose** (FK in the SO-101 / MuJoCo base frame = up / left / right). A stage finishes only when inlier error ≤ `--converge-px` **and** orient error ≤ `--converge-orient-deg`.

Requires a task rebuilt with current `tapnetCreate` (stores `stage_*_goal_joints`).

```powershell
# rebuild task so goal_joints are saved
python src/tapnetCreate.py --data-dir data --out tasks/pick_place_task_steps.npz --viz

python src/tapnetGrabPose.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --freeze-after-first
python src/tapnetGrabPose.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --converge-orient-deg 15 --orient-gain 0.35
```

**Vs the other grab scripts**

| script | control |
|---|---|
| `tapnetGrab` | nearest demo frame + RoboTAP servo |
| `tapnetGrabGoal` | analytical image Jacobian + assumed `--depth` |
| `tapnetGrabGreedy` | hybrid: GrabGoal until `--analytical-until-px`, then probed `J` + greedy ±axes |
| `tapnetGrabPose` | GrabGoal IBVS + slerp gripper quat to demo end pose |

**What to look for in the terminal** (grab scripts)

- `[STAGE s/S] ... then 'grasp'|'release'`
- `[CONVERGED] ... -> primitive '...'`
- `[CLAMP] confirmed...` / `[RELEASE] opened to ...` / `[DONE]` / `[FREEZE]`

**Tuning**

| flag | default | meaning |
|---|---|---|
| `--first-primitive` | task | `grasp` = first gripper action closes, `release` = opens first, `task` = use the npz as built |
| `--freeze-after-first` | off | test: after stage 0 (servo + primitive) finishes, hold that joint pose and skip later stages |
| `--mount-roll` | 180 | real wrist camera roll vs MuJoCo `wrist_cam` (`CAM_MOUNT_ROLL_DEG`); 180 = image upside down vs model, fitted from demo data |
| `--servo-sign -1` | 1 | flip translation if the arm moves away from the goal |
| `--rot-sign -1` | 1 | flip wrist roll direction |
| `--gain-trans` / `--gain-z` / `--gain-rot` | 0.15 / 0.10 / 1.0 | servo gains |
| `--max-trans` / `--max-rot` | 0.01 m / 0.05 rad | per-tick clamps |
| `--smooth` | 0.3 | velocity low-pass (1 = off, lower = smoother but laggier) |
| `--deadband-px` | 8 | hold still when the error is below this |
| `--ik-min-move` | 0.006 m | re-solve IK only after the target moves this far; otherwise resend the last pose |
| `--roll-min-deg` | 1.5 | re-solve once accumulated wrist roll reaches this |
| `--max-joint-step` | 4 deg | per-joint cap on each tick's command |

## HF SmolVLA preprocessing with Vesta goals

`src/hf_preprocess_smolvla_vesta.py` is a separate preprocessing entrypoint; the
original `src/hf_preprocess_smolvla.py` remains available. The new script clusters
wrist candidates over only the last 30 frames of each stage, then tracks the
selected points backward through the full stage. Vesta supplies task-conditioned
goal points for each available top/side frame at `stage.end`; TAPIR tracks those
points backward through that stage before they are written as per-frame heatmaps.

```bash
python src/hf_preprocess_smolvla_vesta.py \
  --vesta-provider my_vesta_adapter:create_vesta_provider \
  --cluster-tail-frames 30 \
  --episodes 0-2 --dry-run

python src/hf_preprocess_smolvla_vesta.py \
  --vesta-provider my_vesta_adapter:create_vesta_provider \
  --cluster-tail-frames 30 \
  --episodes 0-99 --resume
```

The provider module must expose a factory (default name
`create_vesta_provider`) returning an object with
`goal_points(frame_rgb, task) -> N×2 pixel coordinates`. `frame_rgb` is a
`uint8` RGB NumPy array and `task` is the dataset's task text for the stage;
`--vesta-task` overrides it. Out-of-frame and non-finite points are discarded.
Stages without a task, valid Vesta goals, or a real stage-end camera frame are
reported and receive no third-person goal overlay. `--cluster-tail-frames` applies
per stage; shorter stages use their full available length.

## Episode video preprocessing with Molmo2 + spaCy

`src/hf_preprocess_smolvla_grounded.py` is the point-grounding data-generation
entrypoint. It extracts task noun phrases with spaCy, adds `robot gripper`, and
uses Molmo2 with the SO-101 connector on each available top/side camera at every subtask start. TAPIR
tracks candidate points forward at `--third-person-tracking-fps` (default 1 FPS),
including both stage endpoints; coordinates are interpolated back to the source
frame rate and sampled visibility is held between samples. The nearest noun
entity to the gripper is selected when normalized distance is at most `0.08`.
Otherwise it freezes the gripper coordinates at the stage endpoint as a static
goal for the whole stage. If tracking loses the gripper, it asks Molmo for a
fresh gripper snapshot on the exact subtask endpoint frame. Only a visible
tracked endpoint or a valid Molmo point from that same frame can supply the
fallback. Missing endpoint frames remain unresolved; coordinates from nearby
frames, stage starts, or later episode frames are not substituted. Diagnostics
record `snapshot_frame` and `snapshot_source`.
If no valid snapshot exists, it records an unresolved fallback and omits the
overlay. When no nouns are grounded it skips candidate tracking and directly
grounds the endpoint gripper. Successful semantic objects reuse their candidate
trajectories. POV tracking is unaffected by this third-person sampling rate.

Set `SO101_THIRD_PERSON_TRACKING_FPS=5` on the launcher or pass
`--third-person-tracking-fps 5` to adjust it. The output video's frame rate is
unchanged. Sparse tracking reduces temporal detail; faster-moving targets may
need a higher sampling rate.

POV processing follows `src/old_tapnet/tapnetCreate.py`: it extracts one shared
bank of TAPIR appearance features from multiple frames across the demos, then
tracks those exact descriptors in every demo. Equal candidate IDs therefore
refer to the same source descriptors across demos. Seeds come from each stage's
final `--cluster-tail-frames` frames (default 30), with
`--query-frames-per-stage 5`. Candidate tracking stays within these windows.
RoboTAP motion clustering, static-motion filtering, cluster voting, and
cross-demo endpoint funneling select the shared stage points; selected endpoints
are then tracked backward through each full stage. Extra stages are selected
per episode when episode stage counts differ. `point_tracks/pov_query_bank.json`
records the source episode, frame, and pixel coordinate of each shared query.

The existing lightweight defaults remain 128 sampled points and 16 selected
points; use `--num-sample-points 512 --num-poi-points 128` for the reference
script's density. As in the reference, at least 8 points are sampled per source
frame, so the total can exceed the requested point budget. Use a fresh output
destination when switching from the previous independently seeded POV method.

On a fresh Ubuntu RTX A6000 VM, install the inference/video packages and run:

```bash
bash packages-grounded-preprocess.sh
source .venv-grounded/bin/activate
hf auth login
bash scripts/preprocess_so101_grounded.sh
```

The launcher processes **episode 0 only** by default from
`felsager/community_dataset_v3_ee_smolVLA` and uploads the resulting LeRobot
videos/data to `kdaterao/community_v3_ee_smolvla_molmo_grounded`.
It downloads the TAPIR checkpoint, the official `allenai/Molmo2-4B` HF base,
and `so101_connector.pt` from `kdaterao/so101-molmo2-4b-gripper` as needed.
The base download is cached locally and is separate from the native SFT archive
used for training. Your full fine-tuned model does not need to be exported or
uploaded. This entrypoint runs inference only.

Choose another episode slice and output dataset:

```bash
SO101_EPISODES=3-5 SO101_DST_REPO=kdaterao/so101_grounded_test \
  bash scripts/preprocess_so101_grounded.sh --viz-dir outputs/grounded_viz
```

Pass `--resume` to continue an existing destination, or
`--molmo-connector /absolute/path/so101_connector.pt` to use your local connector.
The connector loader accepts the trusted single-GPU PyTorch checkpoint from our
training script, removes training wrappers, and rejects missing/extra tensors or
shape mismatches. It applies all connector weights, including new token embeddings,
once when loading the model. Molmo2 output coordinates are decoded at scale 1000
and converted to source-image pixels before TAPIR tracking.

Use `--dry-run` to inspect gripper-based stage boundaries without loading models
or writing a dataset. To run locally without uploading, call the Python entrypoint
without `--push-to-hub`:

```bash
python src/hf_preprocess_smolvla_grounded.py --episodes 0-2 \
  --dst-repo-id kdaterao/grounded_local --molmo-dtype bf16
```

The legacy MolmoPoint backend is still available with
`--molmo-backend molmopoint --molmo-model allenai/MolmoPoint-8B`.
`scripts/prepare_molmo2_so101_data.sh` prepares labeled images for VLM training;
`scripts/preprocess_so101_grounded.sh` processes robot episode videos.

Per-subtask records, including raw Molmo responses, selected entity, distance,
tracks, visibility, and failure reasons, are written to
`<dataset-cache>/point_tracks/epNNNNNN.json`. The same selected tracks are
rendered into wrist and third-person heatmaps in the LeRobot output dataset.

## Fine-tune Molmo on SO-101 gripper points

On a fresh Ubuntu Thunder Compute RTX A6000 VM, clone this repository and run
the package setup once:

```bash
bash packages-molmo2-so101.sh
source .venv-molmo2/bin/activate
```

It creates an isolated Python 3.12 environment, installs CUDA 13.0 PyTorch,
TorchCodec, and the pinned Molmo2 training/data dependencies, then checks the
GPU. This only installs packages. Log in with `hf auth login` before publishing
the processed dataset. The training command below starts a new fine-tune; it
does not load the saved connector for inference.

For the current run, use the Molmo2-4B SFT checkpoint and the official Molmo2
training repo. The launcher clones a pinned Molmo2 revision into its work
directory, installs the SO-101 point dataset adapter, and runs a one-GPU
connector fine-tune. On a Linux NVIDIA compute machine, run:

```bash
bash scripts/train_molmo2_so101.sh
```

The command downloads the label maker's default dataset,
`kdaterao/so101_locate_gripper`, and the Molmo2-4B SFT checkpoint. It uses
`point_labels.jsonl` and its referenced images, makes an episode-disjoint 90/10
train/validation split, and trains on those points only. It does not download
unrelated Molmo2 or PixMo datasets, and does not reinterpret boxes in
`labels.jsonl`. Training freezes the language model and vision encoder, updates
the vision-language connector, uses batch size 1 with reduced image crops and
a 1,024-token sequence limit, disables video and multi-image preprocessing, and
saves only tuned connector tensors to
`molmo2_so101_run/checkpoints/so101_molmo2_4b/so101_connector.pt`. Keep the
matching Molmo2-4B SFT checkpoint alongside that adapter for inference.

To publish a complete Transformers model with the connector merged into the
base weights, run after training:

```bash
python3 scripts/export_molmo2_so101.py
hf upload kdaterao/so101-molmo2-4b-gripper molmo2_so101_run/hf_model .
```

The exporter reads an original `model.pt` with memory mapping or streams
parameters from the native distributed `model_and_optim/.metadata` checkpoint,
merges all
tuned tensors (including additional token embeddings), and writes BF16
safetensors in approximately 2 GB shards. It downloads only the official
Hugging Face tokenizer, processor, configs, and custom Python code. The full
export is approximately 9 GB; allow that much additional disk space. It runs
on CPU without loading the model into GPU memory. The original checkpoint and
connector remain available. Use `--base-checkpoint`, `--connector`, and
`--output-dir` for nondefault paths; the output directory must be empty.
If export failed before writing weight shards, add `--resume` to reuse its
metadata-only output directory. Legacy distributed checkpoint storage records
are adapted in memory for newer PyTorch readers.
`--repo-id` can upload directly after export. After an interrupted upload,
rerun `hf upload` using the finished export directory.

This is connector fine-tuning, not full-parameter tuning. It saves the adapter
once at the end and skips full-model and optimizer checkpoints. Start with 500
steps; set `SO101_MAX_DURATION` to change that. If the dataset is private, set
`HF_TOKEN` before running.

To publish processed examples as a Hugging Face dataset with embedded images
and `train`/`validation` splits, authenticate with a write token and run:

```bash
bash scripts/prepare_molmo2_so101_data.sh
```

By default it creates the private dataset
`kdaterao/so101_molmo2_gripper_preprocessed`. Set
`SO101_PREPROCESSED_DATASET_REPO=OWNER/NAME` to choose another destination, or
set `SO101_PREPROCESSED_DATASET_PUBLIC=1` to create a public dataset. The local
JSONL files remain the inputs used by the training adapter.

Useful overrides:

```bash
SO101_DATASET_REPO=owner/dataset SO101_MAX_DURATION=1000 \
SO101_MOLMO2_WORKDIR=/scratch/molmo2 bash scripts/train_molmo2_so101.sh
```

The source labels stay grouped by episode during splitting so frames from one
episode cannot leak across train and validation.

The legacy `scripts/train_molmo_so101.sh` launcher remains available for the
original Molmo 0924 checkpoints. MolmoE-1B requires Megablocks; Molmo-7B-D does
not. The Molmo2-4B launcher avoids that MoE dependency.
| `--lookahead` / `--end-frac` | … | **tapnetGrab only** (demo trajectory following) |
| `--advance-progress` | 0 | **tapnetGrab** — optional progress escape (0 = stop on pixel error only) |
| `--depth` | 0.20 m | assumed Z for analytical Jacobian (`tapnetGrabGoal` / Pose / Greedy hybrid) |
| `--converge-orient-deg` / `--orient-gain` | 15 / 0.15 | **tapnetGrabPose** — quat align threshold (deg) and per-tick slerp |
| `--orient-after-px` | 50 | **tapnetGrabPose** — only rotate to goal quat after inlier error is this low (avoids yanking away) |
| `--analytical-until-px` | 80 | **tapnetGrabGreedy** — GrabGoal until this error, then greedy; `0` = pure greedy |
| `--probe-step` / `--probe-rot` | 0.005 m / ~2° | **tapnetGrabGreedy** — finite-diff probe magnitudes |
| `--probe-every` / `--probe-settle` | 45 / 2 | re-probe period (ticks); settle frames after each probe move |
| `--action-step` / `--action-rot` | 0.008 m / 0.04 rad | greedy ±axis candidate magnitudes |
| `--converge-px` / `--converge-frames` | 40 / 1 | finish when mean pixel error is below this for this many frames |
| `--grip-step` / `--grip-timeout` | 2.0 / 90 | grasp close / release open speed, primitive timeout (ticks) |
| `--release-open` | 95 | minimum open command for release (demo targets are floored to this) |

---

## Typical pipeline

```powershell
# 1) record a few demos (watch for [CLAMP] / [RELEASE])
python src/tapnetRecord.py --out-dir data --camera 0 --port COM3 --leader-port COM4

# 2) build task
python src/tapnetCreate.py --data-dir data --out tasks/pick_place_task.npz --viz

# 3) check tracking + directions, then move
#    trajectory-following:
python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode servo_print --camera 0 --port COM3
#    or direct goal IBVS (analytical J + depth):
python src/tapnetGrabGoal.py --task tasks/pick_place_task.npz --mode robot --undistort --camera 0 --port COM3
#    or hybrid GrabGoal → greedy refine:
python src/tapnetGrabGreedy.py --task tasks/pick_place_task.npz --mode robot --undistort --camera 0 --port COM3
```

## View uploaded Molmo point training data

Browse `kdaterao/so101_molmo2_gripper_preprocessed` with its human-labeled point
on each image. Filter by split, camera, and episode; use previous/next, arrow keys,
random sampling, or a sample number. Toggle the overlay to inspect the raw image.
The viewer displays normalized and pixel coordinates and never edits the dataset.

```bash
python3 -m pip install -r requirements-dataset-viewer.txt
hf auth login  # required if the dataset is private
python3 scripts/view_molmo_so101_dataset.py --open-browser
```

Open `http://127.0.0.1:8765`. The first run downloads the dataset to the normal
Hugging Face cache; images are decoded on demand. No GPU or model is needed.
Point conversion uses the same image width/height as the dataset preparation
script, and the image and overlay resize together to avoid display offsets.

For a headless VM, start the viewer there without `--open-browser`. On your own
computer, forward the port using your usual SSH host/address:

```bash
ssh -L 8765:127.0.0.1:8765 ubuntu@YOUR_VM_HOST
```

Then open `http://127.0.0.1:8765` on your own computer. Optional flags:
`--repo-id USER/DATASET`, `--revision REVISION`, `--config NAME`, and `--port 8766`.
This viewer is for the labeled-image dataset, not the clustered LeRobot videos.

### Random episode/subtask video viewer (local)

```bash
.venv/bin/python scripts/view_grounded_subtasks.py --open-browser
```

Defaults to the larger source dataset,
`felsager/community_dataset_v3_ee_smolVLA`, and samples **3 distinct random
episodes**. Gripper boundaries are recomputed automatically for raw source
episodes; no saved tracking reports are required. Wrist/top/side videos play
together, using LeRobot's per-camera episode timestamp offsets. Select an
episode/subtask, or press Play with automatic advancement enabled to move
through every subtask and then the next selected episode. For a preprocessed
dataset it can instead use recorded `point_tracks/ep*.json` boundaries and show
goal/fallback diagnostics. The viewer loads at most **3 episodes by default**.
It searches ordered metadata shards for selected IDs and downloads only the
frame-data files needed for those episodes. Saved reports are downloaded only
when viewing a preprocessed dataset, not for the raw source.
Videos download and cache on demand when their episode is selected. No GPU or model
weights are needed. Change the dataset with `--repo-id OWNER/DATASET`, or use a
local dataset directory with `--root /path/to/dataset`. Use the repo's existing
`.venv` for gripper preview; viewing saved boundaries uses only
`huggingface-hub`, `pandas`, and `pyarrow`.

Use `--random-episodes X` to pick X distinct episodes, and `--seed 42` to repeat
the same selection. Omit the seed to draw a new sample each run. Use
`--max-episodes 1` for one random source episode. On preprocessed datasets,
`--episode-offset 3 --max-episodes 3` selects the next three saved reports in
sorted source-episode order. Automatic advancement
stops at the end of the loaded selection. LeRobot may pack several episodes
into one video file; downloading that shared file can include additional footage,
but the viewer still limits playback to the selected episodes.

To test five random source episodes with repeatable selection:

```bash
.venv/bin/python scripts/view_grounded_subtasks.py --open-browser \
  --random-episodes 5 --seed 42 --gripper-min-change-frac 0.25
```

This downloads only the frame-data Parquet files containing the selected
episodes, reads their gripper signals, and uses the same detector options as
the inspector and grounded preprocessing to recompute playback boundaries.
Shared Parquet files can contain other episodes, but only selected rows are
used for detection. Tune with `--gripper-min-change-frac`,
`--gripper-min-change-abs`, `--gripper-smooth-window`, and
`--gripper-min-dwell-frames`. Saved reports and videos are not modified;
embedded heatmaps remain tied to the old splits. Apply the same options when
rerunning preprocessing to regenerate goals for the new splits.

To view the earlier preprocessed test dataset, pass
`--repo-id kdaterao/so101_pov_clustering_test_v2`. Add `--recompute-stages`
to preview new boundaries on those saved videos.

New preprocessing reports record the destination episode explicitly. Older
reports, including earlier runs, are mapped in sorted source-episode order when
their count matches the destination episode count; the viewer displays this
assumption. Private repositories require a saved HF login or `HF_TOKEN`.


## Faster episode preprocessing

### Inspect gripper stage splits

Use the preprocessing environment to inspect episode segmentation before a GPU run:

```bash
source .venv-grounded/bin/activate
python scripts/inspect_gripper_stages.py --episode 0
```

Open `outputs/gripper_stages/ep000000.html` in a browser (download it from the VM
first). The interactive plot shows state/action gripper signals, detected
grasp/release events, detector thresholds when events exist, and colored stage
ranges. Click a stage to zoom and hover to inspect frame values. JSON and CSV
reports are saved alongside the HTML. Adjacent stages share an inclusive
boundary frame, exactly as in preprocessing. Detection uses the same functions,
defaults, gripper index, and maximum stage truncation as the grounded pipeline;
it loads no model weights and decodes no videos. A fresh source may still require
LeRobot to download its dataset files.

To inspect a different source or detector configuration:

```bash
python scripts/inspect_gripper_stages.py --episode 1 \
  --gripper-source action --gripper-closed-frac 0.15 --gripper-open-frac 0.85
```

Pass the same gripper/stage flags to preprocessing when comparing results.

Grounded preprocessing and the inspector now default to
`--gripper-event-mode movement`: an opening or closing movement must change
position by at least `--gripper-min-change-frac 0.25` of the smoothed episode
range, then settle for the configured dwell time. Opening emits `release`;
closing emits `grasp`, including partial grasps that stop outside the old closed
band. Small movements below this displacement threshold do not split a stage.
The minimum stage gap defaults to one frame so quick release/grasp reversals
are not discarded by the former five-frame stage filter. Movement size and
dwell requirements still apply.

Use `--gripper-min-change-abs VALUE` to set an additional minimum displacement
in the dataset's gripper units; this is useful for episodes whose entire range
is small jitter. Both the relative and absolute limits must be met. If a real
quick reversal is suppressed by smoothing or the three-frame settling period,
inspect with `--gripper-smooth-window 1 --gripper-min-dwell-frames 1` and apply
the same settings to preprocessing. Shorter dwell is more sensitive to noise.
Use `--gripper-event-mode bands` to compare the earlier threshold detector.
The original and Vesta entrypoints retain their existing detector defaults.

```bash
.venv/bin/python scripts/inspect_gripper_stages.py --episode 0 \
  --gripper-min-change-frac 0.25
```

The grounded preprocessing entrypoint reads state/actions/task text directly
from the episode table. For the shared query bank and clustering it decodes only wrist tail
windows (plus frame 0 for the resolution check), rather than every frame of all
three cameras. Tail windows are decoded once for feature extraction and again
for tracking so decoded images do not accumulate across demos. The rendering
pass decodes full camera streams in uint8 batches,
with one seek per batch rather than per frame. Episode timestamp offsets,
padding masks, and the existing full-stage tracking behavior are retained.

Third-person selection reuses the chosen candidate trajectories instead of
running TAPIR through the same stage again. The script logs metadata, camera
decoding, Molmo query, stage, and episode durations. `--dry-run` reads metadata
without decoding camera videos or loading models.

```bash
bash scripts/preprocess_so101_grounded.sh --decode-batch-size 64
```

`--decode-batch-size` defaults to 64. Larger batches reduce seeks but consume
more CPU RAM while decoding. On the RTX A6000, optional TF32 acceleration can
speed up CUDA FP32 operations; it can slightly change tracks:

```bash
bash scripts/preprocess_so101_grounded.sh --tapir-tf32
```

For an interrupted destination, add `--resume`; already-written episodes are
skipped in the rendering pass. Tail clustering still includes all selected
episodes to preserve cross-demo selection. Downloaded source files and models
use the Hugging Face cache. First-run metadata downloading and rate-limit
backoffs are separate from GPU tracking performance.

Regression checks for sparse/batched decoding and selected-track reuse:

```bash
python tests/test_grounded_preprocess_speed.py
```

Source fields such as `source_dataset_index` are retained in every output frame
with their declared dtype and shape. LeRobot generates new frame/episode indices
for the destination. Required source fields are checked while loading metadata,
before tracking begins, to avoid a late feature mismatch during video writing.


## Measure CPU/GPU usage before adding workers

The profiler samples one PID and its descendants, including encoder children.
It records process CPU percentage and RSS RAM, process VRAM, process SM/memory/
encoder/decoder activity when NVIDIA exposes those counters, and separate
whole-GPU utilization and VRAM. Whole-GPU values include unrelated GPU jobs.
The implementation uses `nvidia-smi pmon` for process activity; `-`/unsupported
counters are recorded as null, not zero or a device-wide substitute.
See [NVIDIA's process monitoring documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#process-monitoring).

Attach to the preprocessing run already in progress, from another VM terminal:

```bash
source .venv-grounded/bin/activate
python -m pip install psutil
pgrep -af '[h]f_preprocess_smolvla_grounded.py'
# Replace 12345 with the Python preprocessing PID printed above.
python scripts/profile_preprocessing.py --pid 12345 --duration 120
```

The command above observes the process for two minutes and leaves it running.
For a complete run including startup, launch preprocessing through the profiler:

```bash
SO101_DST_REPO=kdaterao/so101_profile_test \
  python scripts/profile_preprocessing.py -- \
  bash scripts/preprocess_so101_grounded.sh --tapir-tf32
```

Profiles are saved under `outputs/profiles/<UTC timestamp>/`:

- `summary.json`: mean/peak CPU, peak RAM, mean/peak process SM activity, peak
  process VRAM, and whole-GPU measurements.
- `processes.csv`: samples by PID/GPU, convenient for plotting.
- `samples.jsonl`: detailed process and host/GPU samples.

CPU 100% means one logical core; 400% means roughly four cores of work. RSS
includes shared pages, so adding it across workers may overstate physical RAM.
GPU memory-activity percentage measures activity rather than capacity; VRAM
usage is reported in MiB separately. Sampling defaults to two seconds;
`--interval 1` increases resolution, while short spikes/processes can still be
missed. Use a fresh `--out-dir PATH` to choose the output directory.

Measure one-worker throughput and peak VRAM during full-stage tracking. If
GPU utilization is already high, adding workers to the same GPU may increase
contention. If GPU activity is low while CPU use is high, parallel video decoding
is a better first candidate. Test two workers and compare combined throughput
before scaling further; worker count cannot be chosen from VRAM alone. Separate
preprocessing jobs need distinct output destinations. Splitting episodes between
jobs also changes the cross-demo clustering group, so retain the intended group
when designing worker scheduling.


## Skip incompatible episodes and collect telemetry

The VM launcher now processes episode 0 by default and records resource usage
for the Python process and its encoder children. On an existing environment,
install the small profiler dependency once:

```bash
source .venv-grounded/bin/activate
python -m pip install psutil
SO101_DST_REPO=kdaterao/so101_grounded_single_test \
  bash scripts/preprocess_so101_grounded.sh --tapir-tf32
```

Telemetry is written to `outputs/profiles/<UTC timestamp>/summary.json`,
`processes.csv`, and `samples.jsonl`. Use `SO101_PROFILE_DIR=/path/to/new-directory`
to choose the profile directory, or `SO101_TELEMETRY=0` to disable profiling.
Set `SO101_EPISODES=0-9` (or pass `--episodes 0-9`) to request more episodes.

State/action and camera dimension mismatches are checked before tracking/writing.
Incompatible episodes are skipped and logged to
`<destination-cache>/point_tracks/skipped_episodes.jsonl`. They are not marked
successfully processed. The tests cover a malformed metadata episode, a valid
episode, and a wrong-resolution episode in the same run. Sparse wrist-tail
loading is retained. If all episodes are skipped, no empty dataset is uploaded.

The LeRobot statistics validator is adapted within this preprocessing process
so numeric padding masks are validated by their dtype, rather than the substring
`image` in their field name. This fixes the second-episode aggregation failure
`Shape of quantile 'min' ... (1,)`. The vendored LeRobot checkout is unchanged.
After the earlier crash in `save_episode`, use a fresh output destination:
metadata may already have been partly written before the aggregation exception.

### Heatmap confidence gaps and gripper recovery

Grounded preprocessing interpolates only bounded TAPIR visibility gaps: up to
0.5 seconds for POV and 2 seconds for sparse third-person tracks. Missing camera
frames and leading/trailing invisible spans remain unmarked. Set
`--pov-visibility-gap-seconds 0 --third-person-visibility-gap-seconds 0` to disable.
Tracking sidecars retain `raw_visibility` alongside the repaired visibility.

The gripper fallback uses only the exact subtask endpoint: either a visible
tracked gripper point there, or a fresh Molmo grounding on that same frame.
If the endpoint is missing or unresolved, the overlay is omitted. Points from
other frames are not substituted because the gripper may have moved.

### POV continuity across subtasks

By default each stage renders its own selected POV points together with points
from the immediately preceding stage. Previous points are seeded at their actual
visible endpoint and TAPIR tracks them forward through the next stage. Current
points still use tail clustering and full-stage backward tracking. No older
stages accumulate, and third-person goals are unaffected.

The `pov.previous_stage` sidecar records carried trajectories, visibility, and
failures. This adds a forward tracking pass for stages after the first, so it
can increase preprocessing time. `--no-pov-track-previous-stage` disables it.
Occluded or unresolved points can still disappear; carryover does not invent
positions or force visibility.

### Batched TAPIR preprocessing

The grounded entrypoint defaults to `--tapir-frame-batch-size 16`. This is
temporal chunking: ordered frames share one GPU upload, feature-extraction call,
and trajectory-estimation call, with causal state carried into the next chunk.
Outputs are copied back to CPU once per chunk. Tail candidates, selected-point
backtracking, previous-stage carryover, and third-person tracking all use it.
Short final chunks are processed without padding or dropping frames. Existing
callers of `BootsTAPIR` keep a one-frame default unless configured otherwise.

```bash
SO101_TAPIR_FRAME_BATCH_SIZE=16 \
SO101_DST_REPO=kdaterao/so101_grounded_batched_test_v1 \
bash scripts/preprocess_so101_grounded.sh
```

Compare the same episode with `SO101_TAPIR_FRAME_BATCH_SIZE=1`. Smaller chunks
use less VRAM; try 8 if 16 runs out of memory. Batching does not change source
FPS or remove tracking passes. Tracking consistency, GPU memory, and throughput
still need measurement on the GPU VM; no speedup factor is assumed.

### Hugging Face download concurrency

Downloads default to one concurrent file. Set `SO101_HF_DOWNLOAD_WORKERS=4`
when launching preprocessing to allow four concurrent file downloads. This
controls snapshot file downloads, not video decoding, tracking, or the Hub's
internal per-file transfer threads. Existing 429 retries remain enabled; reduce
the worker count if rate-limit waits increase. Restart the process to apply it.

```bash
SO101_HF_DOWNLOAD_WORKERS=4 bash scripts/preprocess_so101_grounded.sh
```
