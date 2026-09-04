# robot101 — SO101 robot & LeRobot training/inference utilities

A small project that bundles:
- URDF / MuJoCo (MJCF) robot descriptions for the SO101 manipulator (used for simulation / planning).
- Python tooling built on top of the LeRobot ecosystem to train and run SmolVLA policies, control a physical SO100/101-style follower arm, teleoperate, and record datasets.

This repo contains model files and convenience scripts used when developing and running perception-to-action policies for the SO101 robot.

## Stack
- Language(s): Python (requires Python 3.12 — 3.13 compatible)
- Framework / runtime: torch (PyTorch), LeRobot / SmolVLA policy
- Notable libraries: lerobot (robot + dataset abstractions), pygame (input/display), OpenCV, Hugging Face Hub utilities

## What’s in this repo (top-level)
- pyproject.toml / requirements.txt — project metadata & pinned dependency list
- packages.sh — helper for installing system-level packages
- src/ — Python source and main scripts
  - train.py — offline training script (SmolVLA + LeRobot dataset)
  - inference.py — live inference loop for running a policy on the real robot
  - teleoperate.py — teleoperation utilities (Xbox controller / UI)
  - record.py — dataset recording utilities
  - utility.py — camera helpers, joint/pose constants, GUI helpers, motion easing, and safety helpers (go_to_rest, ease_to_position, etc.)
  - controllers.py, resetDB.py, testDB.py, iphoneTest.py, xboxController.py, SO101LeaderController.py — supporting controllers & tests
- src/SO101/ — robot description & simulation files (URDF / MJCF)
  - robot.urdf, so101_new_calib.urdf, so101_old_calib.urdf, so101_new_calib.xml, so101_old_calib.xml, scene.xml, joints_properties.xml, assets/...
- README.md (this file) and a small notes.md / thundercompute.md used during development

## Files of interest (quick)
- src/train.py
  - Offline training loop. Uses LeRobotDataset and SmolVLAPolicy. Default dataset and model repo strings are defined inside (see code comments).
  - Looks for local checkpoint in outputs/train/smolvla_desk; otherwise loads from hub repo (kdaterao/smolVLA_desk2 in the current script).
- src/inference.py
  - Live inference + robot loop. Builds an inference frame from robot sensors, runs the SmolVLA model, postprocesses to robot action, and sends the action to the robot.
  - Example robot port and camera indices are inside; model defaults to `lerobot/smolvla_base`.
- src/utility.py
  - Contains FPS, REST_POSE, NEUTRAL_POS, camera preview helpers, joint-print helpers, go_to_rest, ease_to_position, and a small pygame prompt helper.
- src/SO101/*
  - URDF and MuJoCo files for the SO101 robot. Two calibration variants are provided: new_calib (virtual zero = middle of range) and old_calib (virtual zero = fully-extended horizontal). scene.xml uses one by default.

## Quickstart — install & run

1) Python / environment
- Python 3.12 (pyproject: requires-python >=3.12,<3.14)
- Recommended: create a virtualenv

2) Install dependencies
```bash
python -m pip install -r requirements.txt
# or use your preferred environment tool (poetry/uv/venv)
