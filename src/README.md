# Source layout

The application code is the `robot101` Python package. Run entrypoints as modules
from the repository root, using the environment for that workflow:

```bash
source .venv-grounded/bin/activate
python -m robot101.data.preprocess --episodes 0 --dst-repo-id kdaterao/so101_grounded_test
```

The setup scripts install the package in editable mode. To update an existing
environment after this reorganization:

```bash
python -m pip install --no-deps -e .
```

This registers the package; it does not install the GPU dependencies. See
[environment setup](../docs/README.md) and the individual run guides.

```text
src/
├── README.md
└── robot101/
    ├── data/
    │   └── utilities/
    ├── perception/
    ├── robot/
    ├── policies/
    ├── calibration/
    ├── legacy/
    └── paths.py
```

## data — training-data generation

- `preprocess.py`: current episode pipeline: wrist tail-window clustering, backward
  tracking, Molmo2 third-person goals, rendered heatmaps and optional Hub upload.
- `collect_points.py`: interactive gripper point labeling.
- `upload_labels.py`: upload the collected points and images to Hugging Face.

Shared modules live under `data/utilities/`:

- `preprocess_config.py`: arguments and defaults shared with the inspection tools.
- `episode_helpers.py`: camera keys, image conversion, dataset loading and progress.
- `episode_io.py`: episode metadata and batched camera decoding.
- `stages.py`: gripper motion events and inclusive subtask boundaries.
- `destination.py`: destination resume/recovery handling.
- `validation.py`: episode shape checks, skip reports and statistics validation.
- `hub_downloads.py`: download concurrency/retry and Windows cache helpers.

Commands and settings: [Preprocessing](../docs/preprocessing.md) and
[Molmo gripper point data](../docs/molmo_gripper_point_data.md).

## perception — model inference and point tracking

- `molmo2.py`: Molmo2 inference, including batched grounding requests.
- `connector.py`: load and apply the separately saved fine-tuned connector.
- `tracking.py`: TAPIR inference, batching, tracking and heatmap helpers.
- `motion_plan.py`: motion-cluster selection and RoboTAP planning/servo helpers.
- `view_tapir.py`: live camera viewer; click to seed tracked points.

```bash
python -m robot101.perception.view_tapir --help
```

## robot — physical robot control

Controllers, leader-arm and Xbox interfaces, gripper clamp detection, camera
previews, pose helpers, recording and teleoperation. `common.py` re-exports the
shared helpers from `helpers.py` and `display.py`; the former `utility.py` /
`utility/` naming collision is removed.

```bash
python -m robot101.robot.teleoperate
python -m robot101.robot.record
```

Configure ports, camera IDs and recording destinations in these modules before
running them with the physical robot.

## policies — action-policy training and inference

`train.py` contains the existing SmolVLA training entrypoint. `inference.py`
contains the existing robot policy inference loop. Configure their dataset,
checkpoint and hardware settings before running:

```bash
python -m robot101.policies.train
python -m robot101.policies.inference
```

X-VLA training is not implemented in this section yet.

## calibration — geometry and camera preparation

`transforms.py` provides coordinate/rotation conversions without importing robot
control. `capture_photos.py` and `calibrate_iphone.py` prepare camera calibration.
`blender_lily.py` is an authoring tool run inside Blender.

`inspect_apriltags.py` is a historical diagnostic. It still requires the existing
external `old.apriltag_calib_analysis` helper, which is absent from this checkout.

## legacy — earlier approaches and experiments

- `preprocess_wrist.py`: older wrist preprocessing entrypoint.
- `preprocess_vesta.py`: earlier external Vesta-provider preprocessing pipeline.
- `molmo_point.py`: older MolmoPoint inference backend.
- `tapnet/`: previous TAPIR task creation and visual servoing scripts.
- `simulation/`: older MuJoCo scene/control, joint conversion, AprilTag/Lily
  object tracking and viewer tools. SO-101 models and meshes are in
  `simulation/assets/so101/`; generated Lily assets use
  `simulation/assets/lily_objects/`. These tools require MuJoCo, GLFW, Trimesh
  and AprilTag detection libraries.
- Camera, dataset, Lily-relative and standalone TAPIR experiments.
- `robot_utilities.py`: historical monolithic utility implementation.

These remain available for reference. The current data pipeline uses the shared
modules above; it does not import an old preprocessing entrypoint for helpers.
The MolmoPoint backend is loaded only when selected explicitly.

## paths.py — shared locations

Defines the repository root, package root, SO-101 asset directory and TAPIR
checkpoint location. Use these constants when adding code that needs repository
files; do not infer the repository root from the depth of an individual module.

## scripts/ versus src/

Keep reusable application code and job entrypoints in `src/robot101/`.
Keep environment setup, profiling, browser dataset viewers and adapters for the
upstream Molmo2 training checkout in the repository's `scripts/` directory.
