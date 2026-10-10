"""Rigid transforms and camera geometry (MuJoCo wxyz quaternions)."""

from __future__ import annotations

import numpy as np

# OpenCV camera (Z forward, Y down) -> MuJoCo camera (Z back, Y up)
T_CV_TO_MJ = np.diag([1.0, -1.0, -1.0, 1.0])


def normalize_quat_wxyz(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_wxyz_to_R(quat) -> np.ndarray:
    w, x, y, z = (float(q) for q in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=float)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return normalize_quat_wxyz([w, x, y, z])


def rpy_deg_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX yaw-pitch-roll in degrees -> MuJoCo wxyz quaternion."""
    r, p, y = (np.deg2rad(float(a)) for a in (roll, pitch, yaw))
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    return normalize_quat_wxyz(quat)


def quat_angular_distance_deg(q_a, q_b) -> float:
    dot = abs(
        float(np.dot(np.asarray(q_a, dtype=float), np.asarray(q_b, dtype=float)))
    )
    return float(np.rad2deg(2.0 * np.arccos(min(1.0, dot))))


def quat_slerp_wxyz(q0, q1, t: float) -> np.ndarray:
    """Spherical linear interpolation of MuJoCo wxyz quaternions. ``t`` in [0,1]."""
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    t = float(np.clip(t, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + t * (q1 - q0))
    theta_0 = float(np.arccos(min(1.0, dot)))
    sin_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    s0 = float(np.sin(theta_0 - theta) / sin_0)
    s1 = float(np.sin(theta) / sin_0)
    return normalize_quat_wxyz(s0 * q0 + s1 * q1)


def T_from_R_t(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def T_from_pose(pos, quat_wxyz) -> np.ndarray:
    return T_from_R_t(quat_wxyz_to_R(quat_wxyz), pos)


def T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), R_to_quat_wxyz(T[:3, :3])


def T_inv(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(T, dtype=float))


def T_relative(T_world_a: np.ndarray, T_world_b: np.ndarray) -> np.ndarray:
    """T_a_b = inv(T_world_a) @ T_world_b."""
    return T_inv(T_world_a) @ T_world_b


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def intrinsics_from_K(K: np.ndarray) -> list[float]:
    K = np.asarray(K, dtype=float)
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def rot_error(R_cur, R_des) -> np.ndarray:
    """World-frame orientation error for the geometric Jacobian (vee of skew)."""
    R_err = np.asarray(R_des, dtype=float) @ np.asarray(R_cur, dtype=float).T
    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=float,
    )


def rigid_origin(transform: np.ndarray) -> np.ndarray:
    """Keep translation + rotation; bake non-uniform scale into vertices/tag offsets."""
    T = np.array(transform, dtype=float)
    R = T[:3, :3].copy()
    axes = []
    for i in range(3):
        length = float(np.linalg.norm(R[:, i]))
        if length < 1e-12:
            return np.eye(4)
        axes.append(R[:, i] / length)
    rot = np.column_stack(axes)
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = T[:3, 3]
    return out
