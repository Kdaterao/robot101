"""Live camera-pose nudging (MuJoCo / pygame / OpenCV)."""

from __future__ import annotations

import cv2
import glfw
import mujoco
import numpy as np
import pygame
import sys
from robot101.calibration.transforms import intrinsics_from_K


def setup_camera(
    tracker,
    width,
    height,
    k,
    cam_dist,
    frame_width,
    frame_height,
):
    cap = cv2.VideoCapture(tracker.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Could not open camera {tracker.camera_index}")
        sys.exit(1)

    probe = None
    for _ in range(8):
        ok, probe = cap.read()
        if ok and probe is not None and probe.size:
            break

    if probe is None or not probe.size:
        print("Could not read a camera frame to check resolution.")
        sys.exit(1)

    native_h, native_w = int(probe.shape[0]), int(probe.shape[1])

    k_for_frame = True

    if (native_w, native_h) != (frame_width, frame_height):
        sx = native_w / float(frame_width)
        sy = native_h / float(frame_height)

        if abs(sx - sy) < 0.02:
            print(f"Scaling K by sx={sx:.4f} sy={sy:.4f}.")

            k = k.copy()
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy
        else:
            k_for_frame = False
            print(
                f"WARNING: camera is {native_w}x{native_h}, "
                f"calibration is {frame_width}x{frame_height}."
            )

    dist_coeffs = np.asarray(cam_dist, dtype=np.float64).reshape(-1, 1)

    undistort_maps = None

    if k_for_frame and float(np.linalg.norm(dist_coeffs)) > 1e-12:
        undistort_maps = cv2.initUndistortRectifyMap(
            k,
            dist_coeffs,
            None,
            k,
            (native_w, native_h),
            cv2.CV_16SC2,
        )
        print("Undistort ON: remap keeps the calibrated K.")
    else:
        print("Undistort off.")

    tracker.camera_params = intrinsics_from_K(k)

    return cap, k, undistort_maps















def mocap_id(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"Body '{body_name}' is not a mocap body")
    return mocap_id


def drain_ui_events() -> None:
    """Drop keys typed into the y/N prompt so they don't nudge the camera."""
    pygame.event.clear()
    cv2.waitKey(1)


def held(window, glfw_key: int, pygame_keys, pygame_key: int) -> bool:
    if window is not None and glfw.get_key(window, glfw_key) == glfw.PRESS:
        return True
    return bool(pygame_keys[pygame_key])


def cam_hold_nudge(
    window,
    pygame_keys,
    dt: float,
    scale: float,
    *,
    nudge_mps: float,
    nudge_dps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-key camera pose delta: world meters, ZYX rpy degrees."""
    step_m = nudge_mps * float(dt) * float(scale)
    step_d = nudge_dps * float(dt) * float(scale)
    dxyz = np.zeros(3, dtype=float)
    drpy = np.zeros(3, dtype=float)
    if held(window, glfw.KEY_I, pygame_keys, pygame.K_i):
        dxyz[0] += step_m
    if held(window, glfw.KEY_K, pygame_keys, pygame.K_k):
        dxyz[0] -= step_m
    if held(window, glfw.KEY_L, pygame_keys, pygame.K_l):
        dxyz[1] += step_m
    if held(window, glfw.KEY_J, pygame_keys, pygame.K_j):
        dxyz[1] -= step_m
    if held(window, glfw.KEY_U, pygame_keys, pygame.K_u):
        dxyz[2] += step_m
    if held(window, glfw.KEY_O, pygame_keys, pygame.K_o):
        dxyz[2] -= step_m
    if held(window, glfw.KEY_RIGHT, pygame_keys, pygame.K_RIGHT):
        dxyz[1] += step_m
    if held(window, glfw.KEY_LEFT, pygame_keys, pygame.K_LEFT):
        dxyz[1] -= step_m
    if held(window, glfw.KEY_UP, pygame_keys, pygame.K_UP):
        dxyz[0] += step_m
    if held(window, glfw.KEY_DOWN, pygame_keys, pygame.K_DOWN):
        dxyz[0] -= step_m
    if held(window, glfw.KEY_PAGE_UP, pygame_keys, pygame.K_PAGEUP):
        dxyz[2] += step_m
    if held(window, glfw.KEY_PAGE_DOWN, pygame_keys, pygame.K_PAGEDOWN):
        dxyz[2] -= step_m
    if held(window, glfw.KEY_T, pygame_keys, pygame.K_t):
        drpy[0] += step_d
    if held(window, glfw.KEY_SEMICOLON, pygame_keys, pygame.K_SEMICOLON):
        drpy[0] -= step_d
    if held(window, glfw.KEY_Y, pygame_keys, pygame.K_y):
        drpy[1] += step_d
    if held(window, glfw.KEY_H, pygame_keys, pygame.K_h):
        drpy[1] -= step_d
    if held(window, glfw.KEY_N, pygame_keys, pygame.K_n):
        drpy[2] += step_d
    if held(window, glfw.KEY_M, pygame_keys, pygame.K_m):
        drpy[2] -= step_d
    return dxyz, drpy


def cam_tap_nudge(
    key: int,
    scale: float,
    *,
    tap_m: float,
    tap_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot OpenCV key nudge."""
    step_m = tap_m * float(scale)
    step_d = tap_deg * float(scale)
    dxyz = np.zeros(3, dtype=float)
    drpy = np.zeros(3, dtype=float)
    mapping = {
        ord("i"): (0, step_m, True),
        ord("k"): (0, -step_m, True),
        ord("l"): (1, step_m, True),
        ord("j"): (1, -step_m, True),
        ord("u"): (2, step_m, True),
        ord("o"): (2, -step_m, True),
        ord("t"): (0, step_d, False),
        ord(";"): (0, -step_d, False),
        ord("y"): (1, step_d, False),
        ord("h"): (1, -step_d, False),
        ord("n"): (2, step_d, False),
        ord("m"): (2, -step_d, False),
    }
    hit = mapping.get(key)
    if hit is None:
        return dxyz, drpy
    axis, delta, is_pos = hit
    if is_pos:
        dxyz[axis] = delta
    else:
        drpy[axis] = delta
    return dxyz, drpy


def format_cam_cli(xyz, rpy, quat) -> str:
    pos = " ".join(f"{float(v):.4f}" for v in xyz)
    if rpy is not None:
        ori = " ".join(f"{float(v):.2f}" for v in rpy)
        return f"--displacement {pos} --rpy {ori}"
    ori = " ".join(f"{float(v):.4f}" for v in quat)
    return f"--displacement {pos} --quat {ori}"
