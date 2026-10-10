"""Relative-pose teleop: gripper pose readout, printing, hard clamp."""

from __future__ import annotations

import time

import numpy as np
from lerobot.utils.robot_utils import precise_sleep

from robot101.robot.helpers import FPS
from robot101.calibration.transforms import R_to_quat_wxyz

_GRIPPER_CLOSE_SPEED = 40.0
_GRIPPER_STALL_THRESHOLD = 3.0
_GRIPPER_HARD_MAX_THRESHOLD = 12.0
_GRIPPER_STALL_FRAMES = 8


def gripper_pose(
    data, gripper_site_id: int, gripper_body_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Gripperframe (or gripper body) world pose: pos, quat wxyz."""
    if gripper_site_id >= 0:
        pos = np.asarray(data.site_xpos[gripper_site_id], dtype=float).copy()
        R = np.asarray(data.site_xmat[gripper_site_id], dtype=float).reshape(3, 3)
        return pos, R_to_quat_wxyz(R)
    if gripper_body_id >= 0:
        pos = np.asarray(data.xpos[gripper_body_id], dtype=float).copy()
        quat = np.asarray(data.xquat[gripper_body_id], dtype=float).copy()
        n = float(np.linalg.norm(quat))
        if n < 1e-12:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat = quat / n
        return pos, quat
    return None


def fmt_xyz(p) -> str:
    p = np.asarray(p, dtype=float).reshape(3)
    return f"{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}"


def fmt_quat(q) -> str:
    q = np.asarray(q, dtype=float).reshape(4)
    return f"{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}"


def fmt_action(action: dict) -> str:
    return "  ".join(f"{k}={float(v):+.2f}" for k, v in action.items())


def fmt_clamp(value: float) -> str:
    v = float(np.clip(value, 0.0, 100.0))
    return f"{v:.1f} (0 closed … 100 open)"


def obs_gripper_clamp(obs) -> float:
    return float(np.clip(obs["gripper.pos"], 0.0, 100.0))


def clamp_gripper_hard(robot) -> float:
    """Close until stall, then a bounded extra squeeze. Arm pose is held."""
    obs = robot.get_observation()
    actual = obs_gripper_clamp(obs)
    target = actual
    threshold = _GRIPPER_STALL_THRESHOLD
    stalled_frames = 0
    announced = False
    t0 = time.perf_counter()
    prev = t0
    while time.perf_counter() - t0 < 3.5:
        now = time.perf_counter()
        dt = min(max(now - prev, 0.0), 0.1)
        prev = now
        obs = robot.get_observation()
        actual = obs_gripper_clamp(obs)
        close_error = actual - target
        if close_error > threshold:
            target = actual - threshold
            stalled_frames += 1
            if stalled_frames >= _GRIPPER_STALL_FRAMES:
                if not announced:
                    print(
                        f"  gripper stalled at {actual:.1f}; "
                        "holding close-error "
                        f"{threshold:.1f} (same stall guard as Xbox LB)."
                    )
                    announced = True
                threshold = min(
                    _GRIPPER_HARD_MAX_THRESHOLD,
                    threshold + 0.05,
                )
                target = max(0.0, actual - threshold)
        else:
            stalled_frames = 0
            target -= _GRIPPER_CLOSE_SPEED * dt
            target = max(0.0, actual - threshold, target)
        action = {
            key: float(obs[key])
            for key in obs
            if isinstance(key, str) and key.endswith(".pos")
        }
        action["gripper.pos"] = float(np.clip(target, 0.0, 100.0))
        robot.send_action(action)
        precise_sleep(1.0 / FPS)
        if announced and threshold >= _GRIPPER_HARD_MAX_THRESHOLD:
            break
    hold = dict(action)
    for _ in range(int(0.4 * FPS)):
        obs = robot.get_observation()
        actual = obs_gripper_clamp(obs)
        hold["gripper.pos"] = float(np.clip(actual - threshold, 0.0, 100.0))
        robot.send_action(hold)
        precise_sleep(1.0 / FPS)
    final = obs_gripper_clamp(robot.get_observation())
    print(
        f"  clamp done: jaw {final:.1f}  "
        f"command {hold['gripper.pos']:.1f}  "
        f"stall threshold {threshold:.1f}"
    )
    return final
