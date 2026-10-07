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

## HF SmolVLA preprocessing with MolmoPoint + spaCy

`src/hf_preprocess_smolvla_grounded.py` is the point-grounding data-generation
entrypoint. It extracts task noun phrases with spaCy, adds `robot gripper`, and
uses MolmoPoint on each available top/side camera at every subtask start. TAPIR
tracks all candidate points forward to the subtask endpoint; the nearest noun
entity to the gripper is selected when normalized distance is at most `0.08`.
Otherwise it uses the gripper points as a fallback. The selected start-frame
points are tracked forward for the final third-person goal tracks.

POV processing samples one shared set of candidate points, tracks them only
through each stage's final 30 frames, selects points across the shared stage
prefix with RoboTAP funneling/clustering, then tracks selected endpoint points
backward through each full stage. Extra stages are selected per episode when
episode stage counts differ.

Install the extras and spaCy English model:

```bash
pip install -r requirements-molmo-grounding.txt
python -m spacy download en_core_web_sm
```

Run a small dataset slice:

```bash
python src/hf_preprocess_smolvla_grounded.py \
  --episodes 0-2 --dst-repo-id kdaterao/molmo_grounded_smoke \
  --cluster-tail-frames 30 --viz-dir outputs/molmo_grounded_viz
```

Use `--dry-run` to inspect gripper-based stage boundaries without loading
MolmoPoint or TAPIR. The default model is `allenai/MolmoPoint-8B`; pass
`--molmo-model <local-or-HF-checkpoint>` to use a converted custom checkpoint.
Per-subtask records, including raw Molmo responses, selected entity, distance,
tracks, visibility, and failure reasons, are written to
`<dataset-cache>/point_tracks/epNNNNNN.json`. The same selected tracks are
rendered into wrist and third-person heatmaps in the LeRobot output dataset.

## Fine-tune Molmo on SO-101 gripper points

The workspace includes the original AllenAI Molmo training repository as the
`molmo/` Git submodule. The launcher initializes it and installs the SO-101
dataset adapter from this repository automatically. On a Linux NVIDIA compute
machine, run:

```bash
bash scripts/train_molmo_so101.sh
```

The command uses the label maker's default dataset, `kdaterao/so101_locate_gripper`.
It downloads `point_labels.jsonl` and its referenced images, makes an
episode-disjoint 90/10 train/validation split, downloads Molmo-7B-D-0924 and
general PixMo point examples, and starts the native Molmo trainer with a
50/50 general/SO-101 pointing mixture. It consumes user-clicked point labels;
it does not reinterpret the bounding boxes in `labels.jsonl`. Checkpoints go
under `molmo_so101_run/checkpoints/so101_molmo/` by default.

This is configured for one GPU and updates the vision-language connector while
freezing the 7B language model and vision encoder to fit a more modest GPU. It
is a Molmo fine-tuning run, but not full-parameter tuning. The launcher prints
the available GPU and warns below 32 GiB VRAM; actual memory needs depend on
the card and CUDA/PyTorch build. Start with the default 500 steps, then increase
`SO101_MAX_DURATION` after inspecting the run. Set `HF_TOKEN` if the dataset is
private.

Useful overrides:

```bash
SO101_DATASET_REPO=owner/dataset SO101_MAX_DURATION=1000 \
SO101_MOLMO_WORKDIR=/scratch/molmo bash scripts/train_molmo_so101.sh
```

The source labels stay grouped by episode during splitting so frames from one
episode cannot leak across train and validation.
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
