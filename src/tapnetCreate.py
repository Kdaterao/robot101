"""Offline RoboTAP-style task builder for SO-101 wrist-cam demos.

Usage:
  uv run python src/tapnetCreate.py --data-dir data --calib calib/wrist_cam.json \\
      --out tasks/pick_place_task.npz

No AprilTag / LilyTags. Goals = TAPIR endpoints at gripper-open (segment end).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tapnet_utils import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    aggregate_goals_median,
    bgr_to_rgb,
    cluster_motion_tracks,
    features_to_numpy,
    goals_from_tracks,
    load_demo_folder,
    load_segment_bounds,
    load_wrist_calib,
    pick_object_cluster,
    resolve_torch_device,
    sample_query_points,
    save_task_npz,
    segment_gripper_close_open,
    select_active_indices,
    select_feature_subset,
    undistort_bgr,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a TAPIR visual-servoing task from demos")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing demo_*/ folders",
    )
    p.add_argument(
        "--demo",
        type=Path,
        action="append",
        default=None,
        help="Explicit demo folder (repeatable). Overrides --data-dir discovery.",
    )
    p.add_argument(
        "--calib",
        type=Path,
        required=True,
        help="Wrist camera calib JSON (or dir with camera_matrix.npy)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tasks/pick_place_task.npz"),
        help="Output task npz",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for TAPIR + clustering: auto|cuda|cpu|mps "
        "(auto prefers CUDA, else CPU).",
    )
    p.add_argument(
        "--segment",
        choices=("episode", "gripper"),
        default="episode",
        help="episode: full recording (keypress end in tapnetRecord). "
        "gripper: legacy close->open detection.",
    )
    p.add_argument("--close-threshold", type=float, default=30.0)
    p.add_argument("--num-sample-points", type=int, default=256)
    p.add_argument("--num-active", type=int, default=12)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--viz",
        action="store_true",
        help="Show overlay of active points + goals for each demo",
    )
    return p.parse_args()


def _discover_demos(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    demos = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and (d / "video.mp4").is_file()
    )
    if not demos:
        raise FileNotFoundError(f"no demo_*/video.mp4 under {data_dir}")
    return demos


def _process_demo(
    demo_path: Path,
    calib,
    maps,
    tapir: BootsTAPIR,
    args: argparse.Namespace,
) -> dict:
    demo = load_demo_folder(demo_path)
    n = len(demo["frames_bgr"])
    if args.segment == "gripper":
        start, end = segment_gripper_close_open(
            demo["gripper"], close_threshold=args.close_threshold
        )
        how = "gripper close->open"
    else:
        start, end = load_segment_bounds(demo_path, n)
        how = "full episode (record until q)"
    print(f"[{demo['name']}] segment frames [{start}, {end}]  ({how})")

    # Undistort segment frames
    seg_bgr = []
    for fr in demo["frames_bgr"][start : end + 1]:
        if fr.shape[1] != calib.width or fr.shape[0] != calib.height:
            raise ValueError(
                f"{demo['name']}: frame {fr.shape[1]}x{fr.shape[0]} != "
                f"calib {calib.width}x{calib.height}"
            )
        seg_bgr.append(undistort_bgr(fr, maps))
    seg_rgb = [bgr_to_rgb(f) for f in seg_bgr]

    h, w = seg_rgb[0].shape[:2]
    rng = np.random.default_rng(args.seed)
    query_xy = sample_query_points(w, h, args.num_sample_points, rng=rng)

    print(f"[{demo['name']}] tracking {len(query_xy)} points with TAPIR...")
    tracks, visibility, features, query_tyx = tapir.track_video(
        seg_rgb, query_xy, query_frame_index=0
    )

    labels = cluster_motion_tracks(
        tracks,
        visibility,
        n_clusters=args.n_clusters,
        device=args.device,
    )
    obj_c = pick_object_cluster(tracks, visibility, labels)
    active = select_active_indices(
        tracks,
        visibility,
        labels,
        obj_c,
        max_points=args.num_active,
    )
    goals = goals_from_tracks(tracks, visibility, active)
    active_feats = select_feature_subset(features, active)

    print(
        f"[{demo['name']}] object cluster={obj_c}  "
        f"active={len(active)}  "
        f"goal median=({goals[:, 0].mean():.1f}, {goals[:, 1].mean():.1f})"
    )

    if args.viz:
        canvas = seg_bgr[-1].copy()
        for g in goals:
            cv2.circle(canvas, (int(g[0]), int(g[1])), 6, (0, 255, 255), -1)
        for i in active:
            p = tracks[i, -1]
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        cv2.imshow(f"tapnetCreate:{demo['name']}", canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(f"tapnetCreate:{demo['name']}")

    return {
        "name": demo["name"],
        "segment_start": start,
        "segment_end": end,
        "query_xy": query_xy[active],
        "query_tyx": query_tyx[active],
        "tracks": tracks[active],
        "visibility": visibility[active],
        "labels": labels[active],
        "goals": goals,
        "features": active_feats,
        "cluster_id": obj_c,
        "width": w,
        "height": h,
    }


def main() -> None:
    args = _parse_args()
    demo_paths = args.demo if args.demo else _discover_demos(args.data_dir)
    calib = load_wrist_calib(args.calib)
    maps = calib.maps
    print(
        f"Calib {calib.width}x{calib.height}  "
        f"undistort={'ON' if maps is not None else 'off (zero dist)'}"
    )

    device = resolve_torch_device(args.device)
    print(f"tapnetCreate using device={device}", flush=True)
    tapir = BootsTAPIR(checkpoint=args.checkpoint, device=device)
    results = [_process_demo(p, calib, maps, tapir, args) for p in demo_paths]

    # v1: single-demo goals, or median when multi-demo with same active count
    # and order (same seed / first demo defines the active set).
    primary = results[0]
    if len(results) == 1:
        goal_points = primary["goals"]
    else:
        # Multi-demo: only aggregate when point counts match (same seed + pipeline).
        # True descriptor correspondence is future work; warn and use median of
        # equally sized goal arrays, else fall back to first demo.
        sizes = {len(r["goals"]) for r in results}
        if len(sizes) == 1:
            goal_points = aggregate_goals_median([r["goals"] for r in results])
            print(f"Aggregated goals across {len(results)} demos (median).")
        else:
            print(
                "WARNING: active-point counts differ across demos; "
                "using first demo goals only."
            )
            goal_points = primary["goals"]

    feat_arrays = features_to_numpy(primary["features"])
    payload = {
        "active_point_ids": np.arange(len(goal_points), dtype=np.int32),
        "query_points": primary["query_xy"].astype(np.float32),
        "query_points_tyx": primary["query_tyx"].astype(np.float32),
        "goal_points": goal_points.astype(np.float32),
        "cluster_ids": primary["labels"].astype(np.int32),
        "segment_start": np.array([primary["segment_start"]], dtype=np.int32),
        "segment_end": np.array([primary["segment_end"]], dtype=np.int32),
        "camera_width": np.array([primary["width"]], dtype=np.int32),
        "camera_height": np.array([primary["height"]], dtype=np.int32),
        "camera_matrix": calib.K.astype(np.float64),
        "dist_coeffs": calib.dist.astype(np.float64),
        "close_threshold": np.array([args.close_threshold], dtype=np.float32),
        "demo_names": np.asarray([r["name"] for r in results]),
        **feat_arrays,
    }
    save_task_npz(args.out, **payload)
    print(f"Saved task -> {args.out.resolve()}")
    print(
        f"  active={len(goal_points)}  "
        f"resolution={primary['width']}x{primary['height']}"
    )


if __name__ == "__main__":
    main()
