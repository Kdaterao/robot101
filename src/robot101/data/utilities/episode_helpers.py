"""Shared dataset loading, image conversion, camera keys and progress helpers."""

from __future__ import annotations

from robot101.paths import REPO_ROOT

import json
import sys
from pathlib import Path

import numpy as np

from robot101.data.utilities.stages import Stage

DEFAULT_SRC = "felsager/community_dataset_v3_ee_smolVLA"
DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_tapnet"
CAMERAS_3P = ("top", "side")
CAMERA_WRIST = "wrist"
GRIPPER_INDEX = 7

_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))


def _load_lerobot():
    try:
        from robot101.data.utilities.hub_downloads import install_hf_download_guards
        from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

        install_hf_download_guards(max_workers=1)
    except ImportError as e:
        raise SystemExit(
            "lerobot is required. Install it or use the vendored copy at "
            f"{_LEROBOT_SRC}. Original error: {e}"
        ) from e
    return LeRobotDataset, LeRobotDatasetMetadata, DEFAULT_FEATURES, HF_LEROBOT_HOME


def _cam_key(name: str) -> str:
    return f"observation.images.{name}"


def _mask_key(name: str) -> str:
    return f"observation.images.{name}_padding_mask"


def _parse_episodes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds, separator, raw_step = part.partition(":")
            a, b = bounds.split("-", 1)
            step = int(raw_step) if separator else 1
            if step < 1:
                raise ValueError(f"Episode range step must be positive: {part!r}")
            out.extend(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return sorted(set(out))


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        # Tensor inputs occur when reading through LeRobot.
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_is_real(item: dict, camera: str) -> bool:
    key = _mask_key(camera)
    if key not in item:
        return True
    val = item[key]
    arr = _as_numpy(val).reshape(-1)
    return bool(arr[0])


def _tensor_to_rgb(img) -> np.ndarray:
    """LeRobot CHW float [0,1] (or HWC) -> RGB uint8."""
    arr = _as_numpy(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if float(arr.max()) <= 1.5 else arr, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _rgb_to_dataset(img_rgb: np.ndarray) -> np.ndarray:
    return np.asarray(img_rgb, dtype=np.uint8)


def _load_progress(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(int(x) for x in data.get("episodes", []))
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def _save_progress(path: Path, done: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episodes": sorted(done)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_episode_arrays(
    ds,
    cameras: list[str],
) -> dict:
    """Load the (already episode-filtered) dataset into memory."""
    n = ds.num_frames
    if n <= 0:
        raise ValueError("empty episode")
    gripper = np.zeros(n, dtype=np.float32)
    action_g = np.zeros(n, dtype=np.float32)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    tasks: list[str] = []
    frames: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
    masks: dict[str, list[bool]] = {c: [] for c in cameras}

    for i in range(n):
        item = ds[i]
        state = _as_numpy(item["observation.state"]).astype(np.float32).reshape(-1)
        action = _as_numpy(item["action"]).astype(np.float32).reshape(-1)
        states.append(state)
        actions.append(action)
        gripper[i] = float(state[GRIPPER_INDEX]) if state.shape[0] > GRIPPER_INDEX else 0.0
        action_g[i] = float(action[GRIPPER_INDEX]) if action.shape[0] > GRIPPER_INDEX else 0.0
        tasks.append(str(item.get("task", "")))
        for c in cameras:
            key = _cam_key(c)
            present = key in item
            if present:
                frames[c].append(_tensor_to_rgb(item[key]))
            else:
                frames[c].append(np.zeros((480, 640, 3), dtype=np.uint8))
            masks[c].append(_mask_is_real(item, c) if present else False)

    return {
        "n": n,
        "gripper": gripper,
        "action_gripper": action_g,
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "tasks": tasks,
        "frames": {c: frames[c] for c in cameras},
        "masks": {c: np.asarray(masks[c], dtype=bool) for c in cameras},
    }


def _owning_stage(stages: list[Stage], t: int) -> Stage:
    for st in stages:
        if st.start <= t <= st.end:
            return st
    return stages[-1]


def _clone_features(meta_features: dict, default_features: dict) -> dict:
    out = {}
    for k, v in meta_features.items():
        if k in default_features:
            continue
        # deep-ish copy of nested dicts
        out[k] = json.loads(json.dumps(v))
    return out

