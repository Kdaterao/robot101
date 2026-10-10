# robot101 — SO101 robot & LeRobot training/inference utilities

A small project that bundles:
- URDF / MuJoCo (MJCF) robot descriptions for the SO101 manipulator (used for simulation / planning).
- Python tooling built on top of the LeRobot ecosystem to train and run SmolVLA policies, control a physical SO100/101-style follower arm, teleoperate, and record datasets.

This repo contains model files and convenience scripts used when developing and running perception-to-action policies for the SO101 robot.


## RESULTS

A video demonstration of this robot can be found on my linkedin(https://lnkd.in/p/ekKawXb3)
## Stack
- Language(s): Python (requires Python 3.12 — 3.13 compatible)
- Framework / runtime: torch (PyTorch), LeRobot / SmolVLA policy
- Notable libraries: lerobot (robot + dataset abstractions), pygame (input/display), OpenCV, Hugging Face Hub utilities


## Source code

See [src/README.md](src/README.md) for the source layout and each section's role.

- `src/robot101/data/`: episode preprocessing, segmentation, point labeling and uploads.
- `src/robot101/perception/`: Molmo2 connector inference, TAPIR tracking and clustering.
- `src/robot101/robot/`: controllers, recording, teleoperation and shared robot helpers.
- `src/robot101/policies/`: SmolVLA training and physical robot inference.
- `src/robot101/calibration/`: camera preparation and geometry utilities.
- `src/robot101/legacy/`: older pipelines and experiments.
- `scripts/`: setup, profiling, dataset viewers and upstream training adapters.
- `requirements.txt`: dependency profiles; `pyproject.toml`: package metadata.

## Setup and run guides

- [All workflows](docs/README.md): package installers and run guide index.
- [Episode preprocessing](docs/preprocessing.md): Ubuntu/A6000 setup, clustering, Molmo2 inference, telemetry and dataset upload.
- [Molmo2 training](docs/molmo2_training.md): prepare point labels, tune the connector and upload weights.
- [Local viewers](docs/viewing.md): inspect episodes, subtasks and point labels.

Package installation scripts live in `scripts/setup/`. Python commands for
processing and training are documented separately in `docs/`.

For the robot control environment on Ubuntu:

```bash
bash scripts/setup/linux_setup.sh --profile robot
source .venv/bin/activate
hf auth login
```

Run the offline SmolVLA training entrypoint after configuring its dataset and
output settings:

```bash
python -m robot101.policies.train
```

See the source files above for robot ports, camera IDs and policy configuration
before running live inference or recording.
