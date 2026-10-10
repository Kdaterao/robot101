"""Shared Lily test-script helpers (object pick, camera quat CLI)."""

from __future__ import annotations

import sys

import numpy as np

from robot101.calibration.transforms import rpy_deg_to_quat_wxyz


def select_objects(mgr, names: list[str] | None) -> list[dict]:
    available = mgr.list_obj()
    if not available:
        print("No Lily objects yet. Create one with LilyObjectManager.create_obj first.")
        sys.exit(1)
    if names:
        missing = [n for n in names if n not in available]
        if missing:
            print(f"Unknown object(s): {', '.join(missing)}")
            sys.exit(1)
        chosen = names
    else:
        chosen = available
    return [mgr.load_obj(name) for name in chosen]


def parse_cam_quat(rpy, quat, default_rpy_deg: tuple[float, float, float]) -> np.ndarray:
    if quat is not None and rpy is not None:
        print("Use either --rpy or --quat, not both.")
        sys.exit(1)
    if quat is not None:
        q = np.array([float(v) for v in quat], dtype=float)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            print("--quat is too small to normalize.")
            sys.exit(1)
        return q / n
    roll, pitch, yaw = (float(v) for v in (rpy or default_rpy_deg))
    return rpy_deg_to_quat_wxyz(roll, pitch, yaw)
