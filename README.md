# Robot101

SO-101 teleoperation, demonstration recording, policy training, inference, and robot controllers.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/Kdaterao/robot101.git
cd robot101
uv sync
```

If `lerobot` is not initialized in an existing checkout:

```bash
git submodule update --init lerobot
uv sync
```

## Training and inference

- Train a SmolVLA policy: `uv run python -m robot101.policies.train --help`
- Run policy inference: `uv run python -m robot101.policies.inference --help`

Set the dataset, checkpoint, robot port, and camera IDs for your setup before running a hardware workflow.

## Teleoperation, recording, and controllers

- Teleoperate the follower: `uv run python -m robot101.robot.teleoperate`
- Record demonstrations with [`record.py`](src/robot101/robot/record.py): `uv run python -m robot101.robot.record --help`

The robot package contains follower and leader control, Xbox input, gripper handling, camera display, and shared robot controllers.

See [docs/README.md](docs/README.md) for the short workflow guide.
