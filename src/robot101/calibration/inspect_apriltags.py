"""Diagnose AprilTag → PnP → camera-to-world error with tags 4, 6, 7, 8, 9.

Uses the same AprilTagRosTracker pipeline as test_lily_external.py
(detect → bundle PnP → T_world_cam). Does not change that pipeline and
does not apply a correction map. Measure first.

Physical layout (SO-101 base is the world origin):

                     FRONT (+X)
                       ↑

             Tag 7          Tag 8
                ●------------●
                ←-- 15 cm --→
                     ●
                   Tag 9
                ●------------●
             Tag 6          Tag 4

                     ↑
                     │ 23.79 cm to tag 9
                  SO-101 base

    uv run python -m robot101.calibration.inspect_apriltags
    uv run python -m robot101.calibration.inspect_apriltags --camera 1 --tag-size 0.024

Press q to quit. Press R to recapture after a report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np



from robot101.legacy.simulation.aprilRosTracker import AprilTagRosTracker
from old.apriltag_calib_analysis import (
    mean_std as _mean_std,
    known_world_positions as _known_world_positions,
    print_report,
    standalone_tag_objects as _standalone_tag_objects,
)
from robot101.legacy.simulation.lily_shared import parse_cam_quat
from robot101.calibration.transforms import (
    T_from_pose as _pose_T,
    camera_matrix as _camera_matrix,
    intrinsics_from_K as _params_from_K,
    T_to_pose as _T_to_pose,
)



def _fmt_m(p) -> str:
    p = np.asarray(p, dtype=float).reshape(3)
    return f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m"


def _fmt_cm(p) -> str:
    p = np.asarray(p, dtype=float).reshape(3) * 100.0
    return f"[{p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}] cm"

def _pair_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# From tags 7/8/9: geometry was correct, world Z was +13.88 cm high.
CAM_XYZ = (0.28105 + 0.0012, 0.0237,0.3901 + 0.0004) 
CAM_RPY_DEG = (180.0, 0.0, 90.0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAM_FX = 822.317
CAM_FY = 822.317
CAM_CX = 319.495
CAM_CY = 242.502
CAM_DIST = (
    -0.0449369,
    1.17277,
    0,
    0,
    -3.63244,
    0,
    0,
    0,
)
UNDISTORT = True
TAG_IDS = (4, 6, 7, 8, 9)
MIN_SAMPLES = 30
MAX_SAMPLES = 50
COLLECT_SECONDS = 2.0


def _parse_cam_quat(rpy, quat) -> np.ndarray:
    return parse_cam_quat(rpy, quat, CAM_RPY_DEG)


def _setup_capture(args) -> tuple:
    cam_xyz = np.array([float(v) for v in args.displacement], dtype=float)
    cam_rpy = (
        None
        if args.quat is not None
        else np.array([float(v) for v in (args.rpy or CAM_RPY_DEG)], dtype=float)
    )
    cam_quat = _parse_cam_quat(args.rpy, args.quat)
    T_world_cam = _pose_T(cam_xyz, cam_quat)
    dist_coeffs = np.asarray(
        args.dist if args.dist is not None else CAM_DIST,
        dtype=np.float64,
    ).reshape(-1, 1)
    K_distorted = _camera_matrix(
        float(args.fx), float(args.fy), float(args.cx), float(args.cy)
    )
    want_undistort = bool(args.undistort) and float(np.linalg.norm(dist_coeffs)) > 1e-12

    objects = _standalone_tag_objects(TAG_IDS, float(args.tag_size))
    tracker = AprilTagRosTracker(
        family=args.family,
        tag_size=args.tag_size,
        camera_index=args.camera,
        camera_params=_params_from_K(K_distorted),
    )
    tracker.obj_tags(objects)

    cap = cv2.VideoCapture(tracker.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
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
    native_size = (native_w, native_h)
    calib_size = (int(FRAME_WIDTH), int(FRAME_HEIGHT))
    print(
        f"Camera native frame.shape={probe.shape} "
        f"({native_w}x{native_h}). "
        f"Requested {args.width}x{args.height}. "
        f"K is for {FRAME_WIDTH}x{FRAME_HEIGHT}."
    )
    k_for_frame = True
    if native_size != calib_size:
        sx = native_w / float(FRAME_WIDTH)
        sy = native_h / float(FRAME_HEIGHT)
        if abs(sx - sy) < 0.02:
            print(
                f"Same aspect as calibration; scaling K by "
                f"sx={sx:.4f} sy={sy:.4f}. Not stretching the image."
            )
            K_distorted = K_distorted.copy()
            K_distorted[0, 0] *= sx
            K_distorted[0, 2] *= sx
            K_distorted[1, 1] *= sy
            K_distorted[1, 2] *= sy
        else:
            k_for_frame = False
            print(
                f"WARNING: camera is {native_w}x{native_h} "
                f"({native_w / max(native_h, 1):.3f}:1), calibration is "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} "
                f"({FRAME_WIDTH / max(FRAME_HEIGHT, 1):.3f}:1). "
                "Not resizing (that stretches pixels and invalidates K). "
                "Set the camera to the calibration resolution."
            )
    frame_size = native_size
    undistort = want_undistort
    if undistort and not k_for_frame:
        print("WARNING: undistort skipped; native size does not match K.")
        undistort = False

    undistort_maps: tuple[np.ndarray, np.ndarray] | None = None
    if undistort:
        undistort_maps = cv2.initUndistortRectifyMap(
            K_distorted,
            dist_coeffs,
            None,
            K_distorted,
            frame_size,
            cv2.CV_16SC2,
        )
        track_K = K_distorted
        axes_dist = np.zeros((5, 1), dtype=np.float64)
        print(
            "Undistort ON: remap keeps the calibrated K "
            "(no getOptimalNewCameraMatrix / new_K)."
        )
    else:
        track_K = K_distorted
        axes_dist = dist_coeffs if k_for_frame else np.zeros((5, 1), dtype=np.float64)
        print("Undistort off: tracker uses the calibrated K on native frames.")

    cam_k_params = _params_from_K(track_K)
    tracker.camera_params = cam_k_params
    return (
        cap,
        tracker,
        track_K,
        axes_dist,
        undistort_maps,
        cam_k_params,
        cam_xyz,
        cam_rpy,
        cam_quat,
        T_world_cam,
        undistort,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw AprilTag PnP world-frame error with tags 4/6/7/8/9"
    )
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=0.024)
    parser.add_argument("--fx", type=float, default=CAM_FX)
    parser.add_argument("--fy", type=float, default=CAM_FY)
    parser.add_argument("--cx", type=float, default=CAM_CX)
    parser.add_argument("--cy", type=float, default=CAM_CY)
    parser.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=UNDISTORT,
        help="Undistort each frame with OpenCV before AprilTag tracking "
        "(uses CAM_DIST / --dist). Keeps fx/fy/cx/cy; does not compute a new_K.",
    )
    parser.add_argument(
        "--dist",
        nargs="+",
        type=float,
        default=None,
        metavar="K",
        help="OpenCV distortion coefficients (default: CAM_DIST at top of file).",
    )
    parser.add_argument(
        "--displacement",
        nargs=3,
        type=float,
        default=list(CAM_XYZ),
        metavar=("X", "Y", "Z"),
        help="External camera XYZ in meters from the SO101 / world origin.",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Camera orientation in degrees (ZYX yaw-pitch-roll).",
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Camera orientation as a MuJoCo wxyz quaternion (instead of --rpy).",
    )
    parser.add_argument(
        "--swap-lr",
        action="store_true",
        help="Swap left/right: 7/6 with 8/4 if +Y is not robot-left.",
    )
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument(
        "--seconds",
        type=float,
        default=COLLECT_SECONDS,
        help="Keep collecting at least this long after the first valid tag.",
    )
    args = parser.parse_args()

    actual = _known_world_positions(bool(args.swap_lr))
    (
        cap,
        tracker,
        cam_k,
        axes_dist,
        undistort_maps,
        cam_k_params,
        cam_xyz,
        cam_rpy,
        cam_quat,
        T_world_cam,
        undistort,
    ) = _setup_capture(args)

    print("World axes: +X forward from SO-101 base, +Y left, +Z up, table z=0.")
    if args.swap_lr:
        print("Left/right swapped: tags 7/6 are -Y, tags 8/4 are +Y.")
    for tag_id in TAG_IDS:
        print(f"  known tag {tag_id}: {_fmt_m(actual[tag_id])}  "
              f"({_fmt_cm(actual[tag_id])})")
    print(
        "  spokes to 9: "
        + "  ".join(
            f"{a}→{b} {_pair_dist(actual[a], actual[b]) * 100.0:.2f} cm"
            for a, b in ((6, 9), (4, 9), (7, 9), (8, 9))
        )
    )
    print(
        "  15 cm: "
        + "  ".join(
            f"{a}→{b} {_pair_dist(actual[a], actual[b]) * 100.0:.2f} cm"
            for a, b in ((7, 8), (6, 4), (6, 7), (4, 8))
        )
    )
    print(
        f"SO101 is origin. Camera at "
        f"{cam_xyz[0]:+.3f} {cam_xyz[1]:+.3f} {cam_xyz[2]:+.3f} m "
        f"quat {cam_quat[0]:+.3f} {cam_quat[1]:+.3f} "
        f"{cam_quat[2]:+.3f} {cam_quat[3]:+.3f}. "
        "T_world_obj = T_world_cam @ T_cam_obj."
    )
    print(
        f"External camera {tracker.camera_index}. "
        f"tracker K "
        f"fx={cam_k_params[0]:.1f} fy={cam_k_params[1]:.1f} "
        f"cx={cam_k_params[2]:.1f} cy={cam_k_params[3]:.1f}. "
        f"undistort={'ON' if undistort else 'off'}. "
        "apriltag_ros bundle PnP per tag. No correction map."
    )
    print(
        f"Collecting up to {args.max_samples} samples/tag "
        f"(min {args.min_samples}, ≥{args.seconds:.1f}s). "
        "q quit, R recapture."
    )

    samples_cam: dict[int, list[np.ndarray]] = {i: [] for i in TAG_IDS}
    samples_world: dict[int, list[np.ndarray]] = {i: [] for i in TAG_IDS}
    collect_t0: float | None = None
    report_done = False
    diagnosis = ""

    cv2.namedWindow("AprilTag calib (4/6/7/8/9)", cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed.")
                break
            if undistort_maps is not None:
                frame = cv2.remap(
                    frame,
                    undistort_maps[0],
                    undistort_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                )

            tracker.camera_params = list(cam_k_params)
            detections = tracker.detect(frame)
            localized = tracker.poses_from_bundles(detections)

            now = time.perf_counter()
            collecting = not report_done
            if collecting:
                for tag_id in TAG_IDS:
                    hit = localized.get(f"tag_{tag_id}")
                    if hit is None:
                        continue
                    if len(samples_world[tag_id]) >= args.max_samples:
                        continue
                    pos, quat, _primary = hit
                    wpos, _wquat = _T_to_pose(T_world_cam @ _pose_T(pos, quat))
                    samples_cam[tag_id].append(np.asarray(pos, dtype=float).reshape(3))
                    samples_world[tag_id].append(np.asarray(wpos, dtype=float).reshape(3))
                    if collect_t0 is None:
                        collect_t0 = now

                counts = [len(samples_world[i]) for i in TAG_IDS]
                elapsed = 0.0 if collect_t0 is None else (now - collect_t0)
                if (
                    collect_t0 is not None
                    and elapsed >= float(args.seconds)
                    and all(n >= int(args.min_samples) for n in counts)
                ):
                    est_world = {}
                    est_cam = {}
                    std_world = {}
                    n_samples = {}
                    missing = [i for i in TAG_IDS if len(samples_world[i]) == 0]
                    if missing:
                        print(f"No samples for tags {missing}; keep the tags in view.")
                    else:
                        for tag_id in TAG_IDS:
                            mean_w, std_w = _mean_std(samples_world[tag_id])
                            mean_c, _std_c = _mean_std(samples_cam[tag_id])
                            est_world[tag_id] = mean_w
                            est_cam[tag_id] = mean_c
                            std_world[tag_id] = std_w
                            n_samples[tag_id] = len(samples_world[tag_id])
                        diagnosis = print_report(
                            actual,
                            est_world,
                            est_cam,
                            std_world,
                            n_samples,
                            T_world_cam,
                            cam_xyz,
                            cam_rpy,
                            tag_size=float(args.tag_size),
                            fx=float(cam_k_params[0]),
                        )
                        report_done = True

            overlay = [
                f"seen: {sorted(int(k) for k in detections) or 'none'}",
                "q quit  R recapture  no correction map",
                f"undistort: {'ON' if undistort else 'off'}",
            ]
            if report_done:
                overlay.append(f"DONE  diagnosis: {diagnosis}")
            else:
                elapsed = 0.0 if collect_t0 is None else (now - collect_t0)
                overlay.append(
                    f"collecting {elapsed:.1f}s / {args.seconds:.1f}s"
                )
            for tag_id in TAG_IDS:
                n = len(samples_world[tag_id])
                overlay.append(f"tag {tag_id}: {n}/{args.min_samples} samples")
                hit = localized.get(f"tag_{tag_id}")
                if hit is None:
                    overlay.append(f"  tag {tag_id}: waiting")
                    continue
                pos, quat, _primary = hit
                wpos, _wquat = _T_to_pose(T_world_cam @ _pose_T(pos, quat))
                err = actual[tag_id] - wpos
                overlay.append(
                    f"  world {_fmt_cm(wpos)}  err {_fmt_cm(err)}  "
                    f"|e|={float(np.linalg.norm(err)) * 100.0:.1f}cm"
                )
                overlay.append(
                    f"  cam dx={pos[0]:+.3f} dy={pos[1]:+.3f} "
                    f"dz={pos[2]:+.3f}  dist={float(np.linalg.norm(pos)):.3f}m"
                )

            for seen_id, (R, t) in tracker.last_tag_poses.items():
                rvec, _ = cv2.Rodrigues(R)
                cv2.drawFrameAxes(
                    frame, cam_k, axes_dist, rvec, t.reshape(3, 1), args.tag_size
                )
                corners = tracker.last_corners.get(seen_id)
                if corners is not None:
                    pts = corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx, cy = tracker.last_centers[seen_id]
                cv2.putText(
                    frame,
                    str(seen_id),
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            y = 24
            for line in overlay:
                waiting = "waiting" in line
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 255) if waiting else (0, 255, 0),
                    2,
                )
                y += 20
            cv2.imshow("AprilTag calib (4/6/7/8/9)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                samples_cam = {i: [] for i in TAG_IDS}
                samples_world = {i: [] for i in TAG_IDS}
                collect_t0 = None
                report_done = False
                diagnosis = ""
                print("Recapture: samples cleared.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
