"""Offline RoboTAP motion-plan builder for SO-101 wrist-cam demos.

1. Split each demo into stages at grasp / release events (gripper clamp
   detection saved by tapnetRecord in events.json).
2. Sample query points on frames spread across every stage of every demo, extract
   their TAPIR features once, and track that shared query set through all
   demos (point i is the same physical point everywhere).
3. Per stage: keep points that move, are visible at the end, and end in the
   same place across demos (funneling); they vote for a motion cluster; the
   winning cluster's points (up to --num-active) are the stage's active points.
4. Save a multi-stage task: shared query features, and per stage the demo
   trajectories of the active points (servo targets), mean goal, and the
   gripper primitive to run when the stage converges.

Usage:
  python -m robot101.legacy.tapnet.tapnetCreate --data-dir data --out tasks/pick_place_task.npz
  # optional: --calib path/to.json  (overrides CAM_* below)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Wrist camera intrinsics — edit these (same as test_lily_relative.py)
# Calibrated at 1280x720, scaled x1.5 for 1920x1080. Use the full 8-param
# rational model (5-param alone warps badly). Rebuild tasks after changing.
# ---------------------------------------------------------------------------
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAM_FX = 1262.480288175
CAM_FY = 1264.71879012
CAM_CX = 913.586772795
CAM_CY = 609.63948621
CAM_DIST = (
    3.83407503,  # k1
    -2.80771278,  # k2
    -0.00124016194,  # p1
    0.00296903720,  # p2
    -0.755297301,  # k3
    4.27744077,  # k4
    -1.18548628,  # k5
    -2.35292973,  # k6
)
# False = use raw camera frames. Must match UNDISTORT in tapnetGrab.py
# (or --undistort / --no-undistort) since goals are stored in pixels.
UNDISTORT = True

from robot101.perception.motion_plan import (
    MotionPlan,
    StagePlan,
    align_stage_counts,
    load_demo_events,
    pack_plan,
    select_stage_active,
    stages_from_events,
)
from robot101.perception.tracking import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    _gripper_from_states,
    bgr_to_rgb,
    concat_query_features,
    default_wrist_calib,
    features_to_numpy,
    load_wrist_calib,
    resolve_torch_device,
    sample_query_points,
    save_task_npz,
    select_feature_subset,
    undistort_bgr,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a RoboTAP motion plan from demos")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
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
        default=None,
        help="Optional calib JSON/dir. Default: CAM_* constants at top of this file.",
    )
    p.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=UNDISTORT,
        help=f"Undistort frames with CAM_* (default from UNDISTORT={UNDISTORT})",
    )
    p.add_argument("--out", type=Path, default=Path("tasks/pick_place_task.npz"))
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps")
    p.add_argument(
        "--segment",
        choices=("events", "episode"),
        default="events",
        help="events: stages from events.json (clamp/release). episode: one stage.",
    )
    p.add_argument(
        "--first-primitive",
        choices=("grasp", "release"),
        default="grasp",
        help="For manual space/e events: first stage ends with close (grasp) or open (release)",
    )
    p.add_argument(
        "--num-sample-points",
        type=int,
        default=512,
        help="Total shared query points (split across demos x stages x query frames)",
    )
    p.add_argument(
        "--query-frames-per-stage",
        type=int,
        default=5,
        help="Frames per stage (start .. end) on which query points are sampled",
    )
    p.add_argument("--num-active", type=int, default=128, help="Active points per stage")
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument(
        "--static-thresh",
        type=float,
        default=0.03,
        help="Points moving less than this fraction of the image diagonal are static",
    )
    p.add_argument(
        "--funnel-quantile",
        type=float,
        default=0.4,
        help="Keep the lowest-endpoint-spread fraction of moving points as voters",
    )
    p.add_argument(
        "--goal-tail-frames",
        type=int,
        default=5,
        help="Stage endpoint = median of the last N (visible) frames",
    )
    p.add_argument(
        "--gripper-settle-frames",
        type=int,
        default=15,
        help="Gripper target is read this many frames after each boundary",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--viz",
        action="store_true",
        help="Save <out>_viz.png with each stage's mean goals on the first demo",
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


def _iter_frames_rgb(video: Path, calib, wanted: set[int] | None = None):
    """Yield (index, RGB frame). calib=None → raw frames; else lily-style P=K
    remap at native size."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    maps = None
    first = True
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if first and calib is not None:
                _, maps = calib.maps_for_frame(fr.shape[1], fr.shape[0])
                first = False
            if wanted is None or i in wanted:
                yield i, bgr_to_rgb(undistort_bgr(fr, maps))
            i += 1
    finally:
        cap.release()


def _count_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main() -> None:
    args = _parse_args()
    demo_paths = args.demo if args.demo else _discover_demos(args.data_dir)
    names = [p.name for p in demo_paths]
    if args.calib is not None:
        calib = load_wrist_calib(args.calib)
        print(f"Calib from file: {args.calib}")
    else:
        calib = default_wrist_calib(
            fx=CAM_FX, fy=CAM_FY, cx=CAM_CX, cy=CAM_CY,
            dist=CAM_DIST, width=FRAME_WIDTH, height=FRAME_HEIGHT,
        )
        print("Calib from CAM_* constants at top of tapnetCreate.py")
    print(
        f"  {calib.width}x{calib.height}  "
        f"fx={calib.K[0, 0]:.3f} fy={calib.K[1, 1]:.3f}  "
        f"cx={calib.K[0, 2]:.3f} cy={calib.K[1, 2]:.3f}  "
        f"dist={tuple(float(x) for x in calib.dist.ravel())}"
    )
    print(f"Undistort: {'ON' if args.undistort else 'OFF (raw frames)'}")
    frame_calib = calib if args.undistort else None

    device = resolve_torch_device(args.device)
    print(f"tapnetCreate using device={device}", flush=True)
    tapir = BootsTAPIR(checkpoint=args.checkpoint, device=device)

    # ---- stages per demo ----
    grippers: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    stage_lists = []
    n_frames: list[int] = []
    for path in demo_paths:
        states = np.load(path / "robot_states.npy")
        n = min(_count_frames(path / "video.mp4"), len(states))
        n_frames.append(n)
        all_states.append(np.asarray(states[:n], dtype=np.float32))
        grippers.append(_gripper_from_states(states)[:n])
        events = load_demo_events(path) if args.segment == "events" else None
        if args.segment == "events" and events is None:
            print(f"[{path.name}] no events.json -> single stage")
        stage_lists.append(
            stages_from_events(events, n, first_primitive=args.first_primitive)
        )
    n_stages = align_stage_counts(stage_lists, names)
    for name, sl in zip(names, stage_lists):
        desc = "  ".join(f"[{s.start},{s.end}]->{s.primitive}" for s in sl)
        print(f"[{name}] {len(sl)} stage(s): {desc}")

    # ---- shared queries: sampled on frames spread across every stage of every demo
    # (wrist-cam points sampled at the stage start are mostly out of view by its end)
    D = len(demo_paths)
    K = max(1, args.query_frames_per_stage)
    n_per = max(8, args.num_sample_points // max(1, D * n_stages * K))
    rng = np.random.default_rng(args.seed)
    feats_list = []
    query_tyx_list = []
    query_src = []
    width = height = None
    for d, path in enumerate(demo_paths):
        q_frames: set[int] = set()
        for st in stage_lists[d]:
            hi = max(st.start, st.end - args.goal_tail_frames // 2)
            q_frames.update(int(round(f)) for f in np.linspace(st.start, hi, K))
        for idx, rgb in _iter_frames_rgb(path / "video.mp4", frame_calib, q_frames):
            h, w = rgb.shape[:2]
            if width is None:
                width, height = w, h
            elif (w, h) != (width, height):
                raise ValueError(f"{path.name}: {w}x{h} != {width}x{height}")
            xy = sample_query_points(w, h, n_per, rng=rng)
            tyx = np.stack([np.zeros(len(xy)), xy[:, 1], xy[:, 0]], axis=1).astype(np.float32)
            feats_list.append(tapir.init_features(rgb, tyx))
            query_tyx_list.append(tyx)
            query_src += [(d, idx)] * len(xy)
    features = concat_query_features(feats_list)
    query_tyx = np.concatenate(query_tyx_list, axis=0)
    N = len(query_tyx)
    print(
        f"Shared queries: {N} points ({n_per} per query frame, {K} frames per stage, "
        f"{D} demos x {n_stages} stages)"
    )

    # ---- track the shared queries through every demo ----
    tracks: list[np.ndarray] = []
    vis: list[np.ndarray] = []
    viz_frame = None
    for d, path in enumerate(demo_paths):
        last = [None]

        def stream(path=path, n=n_frames[d], last=last):
            for i, rgb in _iter_frames_rgb(path / "video.mp4", frame_calib):
                if i >= n:
                    break
                last[0] = rgb
                yield rgb

        tr, vi = tapir.track_with_features(
            stream(), features, label=path.name, num_frames=n_frames[d]
        )
        if d == 0 and last[0] is not None:
            viz_frame = cv2.cvtColor(last[0], cv2.COLOR_RGB2BGR)
        tracks.append(tr)
        vis.append(vi)
        print(f"[{path.name}] mean visibility={vi.mean():.2f}")

    # ---- active points per stage ----
    selections = []
    for s in range(n_stages):
        st_tr = [tracks[d][:, stage_lists[d][s].start : stage_lists[d][s].end + 1] for d in range(D)]
        st_vi = [vis[d][:, stage_lists[d][s].start : stage_lists[d][s].end + 1] for d in range(D)]
        frames_desc = ", ".join(
            f"{names[d]}[{stage_lists[d][s].start},{stage_lists[d][s].end}]" for d in range(D)
        )
        print(f"Stage {s} ({stage_lists[0][s].primitive} after): {frames_desc}")
        sel = select_stage_active(
            st_tr, st_vi, width, height,
            num_active=args.num_active,
            n_clusters=args.n_clusters,
            static_thresh=args.static_thresh,
            funnel_quantile=args.funnel_quantile,
            goal_tail_frames=args.goal_tail_frames,
            device=device,
            seed=args.seed + s,
            label=f"  stage {s}:",
        )
        selections.append((sel, st_tr, st_vi))

    # Runtime tracks the union of all stages' points (one causal state).
    union = np.unique(np.concatenate([sel["point_ids"] for sel, _, _ in selections]))
    remap = {int(g): i for i, g in enumerate(union)}

    stage_plans = []
    for s, (sel, st_tr, st_vi) in enumerate(selections):
        ids = sel["point_ids"]
        ends = sel["ends"][:, ids]  # [D,M,2]
        ev = sel["end_vis"][:, ids]  # [D,M]
        w = ev.astype(np.float32)[..., None]
        goal_mean = np.where(
            ev.any(axis=0)[:, None],
            (ends * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1.0),
            ends.mean(axis=0),
        )
        g_vals = []
        joint_ends = []
        for d in range(D):
            st = stage_lists[d][s]
            nxt_end = stage_lists[d][s + 1].end if s + 1 < n_stages else st.end
            g_vals.append(grippers[d][min(st.end + args.gripper_settle_frames, nxt_end)])
            # End-of-stage arm+gripper pose in SO-101 base / LeRobot degrees.
            joint_ends.append(all_states[d][st.end, :6])
        goal_joints = np.median(np.asarray(joint_ends, dtype=np.float32), axis=0)
        stage_plans.append(
            StagePlan(
                point_ids=np.array([remap[int(i)] for i in ids], dtype=np.int64),
                demo_tracks=[t[ids] for t in st_tr],
                demo_vis=[v[ids] for v in st_vi],
                goal_mean=goal_mean.astype(np.float32),
                goal_valid=ev.any(axis=0),
                primitive=stage_lists[0][s].primitive,
                gripper_target=float(np.median(g_vals)),
                goal_joints=goal_joints.astype(np.float32),
            )
        )
        print(
            f"Stage {s}: {len(ids)} active points, primitive={stage_lists[0][s].primitive}, "
            f"gripper target={np.median(g_vals):.1f}, "
            f"goal_joints={np.array2string(goal_joints, precision=1, suppress_small=True)}"
        )

    plan = MotionPlan(stages=stage_plans, demo_names=names)

    if args.viz and viz_frame is not None:
        colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
        canvas = viz_frame.copy()
        for s, st in enumerate(stage_plans):
            for g in st.goal_mean[st.goal_valid]:
                cv2.circle(canvas, (int(g[0]), int(g[1])), 6, colors[s % len(colors)], 2)
        viz_path = args.out.with_name(args.out.stem + "_viz.png")
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_path), canvas)
        print(f"Saved viz -> {viz_path.resolve()}")

    payload = {
        "query_points_tyx": query_tyx[union].astype(np.float32),
        "query_points": query_tyx[union][:, [2, 1]].astype(np.float32),
        "query_source": np.asarray(query_src, dtype=np.int32)[union],
        "camera_width": np.array([width], dtype=np.int32),
        "camera_height": np.array([height], dtype=np.int32),
        "camera_matrix": calib.K.astype(np.float64),
        "dist_coeffs": calib.dist.astype(np.float64),
        "undistorted": np.array([bool(args.undistort)]),
        **features_to_numpy(select_feature_subset(features, union)),
        **pack_plan(plan),
    }
    save_task_npz(args.out, **payload)
    print(f"Saved task -> {args.out.resolve()}")
    print(
        f"  stages={n_stages}  tracked points={len(union)}  "
        f"demos={D}  resolution={width}x{height}  "
        f"undistort={'on' if args.undistort else 'off'}"
    )


if __name__ == "__main__":
    main()
