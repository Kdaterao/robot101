"""Preprocess felsager SmolVLA EE data: gripper stages, TapNet wrist heatmaps.

Writes a new LeRobot v3 dataset with heatmap-filtered camera videos ready for training.
"""

from __future__ import annotations

from robot101.paths import REPO_ROOT

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from robot101.perception.motion_plan import (
    Stage,
    events_from_gripper_thresholds,
    stages_from_events,
)
from robot101.perception.tracking import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    _kmeans_torch,
    points_heatmap,
    sample_query_points,
)

DEFAULT_SRC = "felsager/community_dataset_v3_ee_smolVLA"
DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_tapnet"
CAMERAS_3P = ("top", "side")
CAMERA_WRIST = "wrist"
GRIPPER_INDEX = 7

_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))


from robot101.data.utilities.episode_helpers import (
    _load_lerobot,
    _cam_key,
    _mask_key,
    _parse_episodes,
    _as_numpy,
    _mask_is_real,
    _tensor_to_rgb,
    _rgb_to_dataset,
    _load_progress,
    _save_progress,
    _load_episode_arrays,
    _owning_stage,
    _clone_features,
    DEFAULT_SRC,
    DEFAULT_DST,
    CAMERAS_3P,
    CAMERA_WRIST,
    GRIPPER_INDEX
)

def _wrist_poi_at_end(
    tapir: BootsTAPIR,
    frames_rgb: list[np.ndarray],
    *,
    num_sample: int,
    num_poi: int,
    n_clusters: int,
    static_thresh_px: float,
    rng: np.random.Generator,
    label: str,
) -> np.ndarray:
    """Forward-track random queries; return up to num_poi endpoints as POI seeds."""
    if len(frames_rgb) < 2:
        h, w = frames_rgb[0].shape[:2]
        return sample_query_points(w, h, num_poi, rng=rng)

    h, w = frames_rgb[0].shape[:2]
    xy = sample_query_points(w, h, num_sample, rng=rng)
    tracks, vis, _, _ = tapir.track_video(frames_rgb, xy, query_frame_index=0)
    # tracks [N,T,2]
    starts = tracks[:, 0]
    ends = tracks[:, -1]
    motion = np.linalg.norm(ends - starts, axis=1)
    vis_end = vis[:, -1]
    ok = vis_end & (motion >= float(static_thresh_px))
    if int(ok.sum()) < max(2, n_clusters):
        # fall back to top-k motion among visible-at-end
        score = motion.copy()
        score[~vis_end] = -1.0
        order = np.argsort(-score)
        pick = order[: min(num_poi, len(order))]
        return ends[pick].astype(np.float32)

    feats = ends[ok]
    k = int(min(n_clusters, len(feats)))
    scale = np.array([w, h], dtype=np.float32)
    labels = _kmeans_torch(feats / scale, k, seed=int(rng.integers(0, 1_000_000)))
    seeds: list[np.ndarray] = []
    for c in range(k):
        members = feats[labels == c]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        i = int(np.argmin(np.linalg.norm(members - centroid, axis=1)))
        seeds.append(members[i])
    if not seeds:
        return ends[ok][:num_poi].astype(np.float32)
    seeds_arr = np.stack(seeds, axis=0).astype(np.float32)
    if len(seeds_arr) > num_poi:
        # keep most-moving cluster reps: approximate by distance from start of matched pts
        return seeds_arr[:num_poi]
    if len(seeds_arr) < num_poi:
        extra = sample_query_points(w, h, num_poi - len(seeds_arr), rng=rng)
        seeds_arr = np.concatenate([seeds_arr, extra], axis=0)
    print(f"  {label}: {len(seeds_arr)} wrist POI seeds", flush=True)
    return seeds_arr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HF SmolVLA preprocess: gripper stages + TapNet wrist heatmaps -> new LeRobot dataset"
    )
    p.add_argument("--src-repo-id", default=DEFAULT_SRC)
    p.add_argument("--dst-repo-id", default=DEFAULT_DST)
    p.add_argument("--episodes", default=None, help="e.g. '0-99' or '0,1,5'")
    p.add_argument("--device", default=None)
    p.add_argument("--tapnet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dry-run", action="store_true", help="Only compute stages; no TapNet/write")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--first-primitive", default="grasp", choices=["grasp", "release"])
    p.add_argument("--max-stages", type=int, default=32)
    p.add_argument("--min-stage-frames", type=int, default=5)
    p.add_argument("--gripper-source", default="state", choices=["state", "action"])
    p.add_argument("--gripper-closed-frac", type=float, default=0.15)
    p.add_argument("--gripper-open-frac", type=float, default=0.85)
    p.add_argument("--gripper-min-dwell-frames", type=int, default=3)
    p.add_argument("--gripper-vel-stall", type=float, default=0.02)
    p.add_argument("--gripper-vel-min", type=float, default=0.05)
    p.add_argument("--gripper-smooth-window", type=int, default=5)
    p.add_argument("--num-sample-points", type=int, default=128)
    p.add_argument("--num-poi-points", type=int, default=16)
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--static-thresh-px", type=float, default=8.0)
    p.add_argument("--heatmap-sigma", type=float, default=40.0)
    p.add_argument("--heatmap-alpha", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Video decode backend (default pyav — avoids broken torchcodec/FFmpeg on Windows)",
    )
    p.add_argument("--viz-dir", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    LeRobotDataset, LeRobotDatasetMetadata, DEFAULT_FEATURES, HF_LEROBOT_HOME = _load_lerobot()

    ep_filter = _parse_episodes(args.episodes)
    src_root = HF_LEROBOT_HOME / args.src_repo_id
    meta_kwargs: dict = {"root": src_root}
    src_base_kwargs: dict = {
        "root": src_root,
        "video_backend": args.video_backend,
    }
    if (src_root / "meta" / "info.json").is_file():
        print(f"Loading local source at {src_root}")
    else:
        print(f"Downloading source to {src_root} from Hub: {args.src_repo_id}")
        print(
            "(Windows: Hub copies files instead of symlinks. "
            "If a prior download failed mid-way, delete this folder and retry.)"
        )

    # Metadata-only pass for fps / features / episode count
    src_meta = LeRobotDatasetMetadata(args.src_repo_id, **meta_kwargs)
    fps = float(src_meta.fps)
    if ep_filter is not None:
        episode_ids = ep_filter
    else:
        episode_ids = list(range(int(src_meta.total_episodes)))

    print(f"Source episodes to process: {len(episode_ids)} (fps={fps})")

    dst = None
    progress_path = HF_LEROBOT_HOME / args.dst_repo_id / "_preprocess_done.json"
    done = _load_progress(progress_path) if args.resume else set()

    if not args.dry_run:
        dst_root = HF_LEROBOT_HOME / args.dst_repo_id
        features = _clone_features(dict(src_meta.features), DEFAULT_FEATURES)
        if args.resume and (dst_root / "meta").exists():
            print(f"Resuming destination at {dst_root}")
            dst = LeRobotDataset.resume(repo_id=args.dst_repo_id, root=dst_root)
        else:
            if dst_root.exists() and not args.resume:
                raise SystemExit(
                    f"Destination {dst_root} already exists. Pass --resume or delete it."
                )
            dst = LeRobotDataset.create(
                repo_id=args.dst_repo_id,
                root=dst_root,
                fps=int(round(fps)),
                robot_type=getattr(src_meta, "robot_type", None) or "so100_so101_ee",
                features=features,
                use_videos=True,
            )

    tapir = None
    if not args.dry_run:
        device = args.device
        print(f"Loading TapNet from {args.tapnet_checkpoint}...")
        tapir = BootsTAPIR(checkpoint=args.tapnet_checkpoint, device=device)

    rng = np.random.default_rng(args.seed)
    cameras = [CAMERA_WRIST, *CAMERAS_3P]

    for ep_i, ep_idx in enumerate(episode_ids):
        if ep_idx in done:
            print(f"[ep {ep_idx}] already done, skip")
            continue

        print(f"\n=== Episode {ep_idx} ({ep_i + 1}/{len(episode_ids)}) ===", flush=True)
        # One-episode load keeps HF video decode scoped and indexing correct.
        src_ep = LeRobotDataset(
            args.src_repo_id, episodes=[ep_idx], **src_base_kwargs
        )
        data = _load_episode_arrays(src_ep, cameras)
        grip = data["gripper"] if args.gripper_source == "state" else data["action_gripper"]
        events = events_from_gripper_thresholds(
            grip,
            fps=fps,
            closed_frac=args.gripper_closed_frac,
            open_frac=args.gripper_open_frac,
            min_dwell_frames=args.gripper_min_dwell_frames,
            vel_stall=args.gripper_vel_stall,
            vel_min=args.gripper_vel_min,
            smooth_window=args.gripper_smooth_window,
        )
        stages = stages_from_events(
            events,
            data["n"],
            min_stage_frames=args.min_stage_frames,
            first_primitive=args.first_primitive,
        )
        if len(stages) > args.max_stages:
            print(
                f"  truncating {len(stages)} stages -> {args.max_stages}",
                flush=True,
            )
            stages = stages[: args.max_stages]
            stages[-1] = Stage(stages[-1].start, data["n"] - 1, "none")

        desc = "  ".join(f"[{s.start},{s.end}]->{s.primitive}" for s in stages)
        print(f"  events={len(events)} stages={len(stages)}: {desc}", flush=True)
        if events:
            e0 = events[0]
            print(
                f"  gripper range [{e0['g_min']:.3f},{e0['g_max']:.3f}] "
                f"closed<={e0['closed_thresh']:.3f} open>={e0['open_thresh']:.3f}",
                flush=True,
            )

        if args.dry_run:
            done.add(ep_idx)
            _save_progress(progress_path, done)
            continue

        assert tapir is not None and dst is not None

        # --- Stage 2: wrist backward tracks per stage ---
        wrist_tracks = np.zeros((data["n"], args.num_poi_points, 2), dtype=np.float32)
        wrist_vis = np.zeros((data["n"], args.num_poi_points), dtype=bool)
        wrist_frames = data["frames"][CAMERA_WRIST]
        wrist_ok = bool(data["masks"][CAMERA_WRIST].any())

        for si, st in enumerate(reversed(stages)):
            if not wrist_ok:
                break
            clip = wrist_frames[st.start : st.end + 1]
            if len(clip) < 2:
                continue
            # only use frames where mask is true if possible
            seeds = _wrist_poi_at_end(
                tapir,
                clip,
                num_sample=args.num_sample_points,
                num_poi=args.num_poi_points,
                n_clusters=args.n_clusters,
                static_thresh_px=args.static_thresh_px,
                rng=rng,
                label=f"ep{ep_idx}/stage{len(stages) - 1 - si}",
            )
            tr, vi = tapir.track_segment_backward(
                clip,
                seeds,
                label=f"ep{ep_idx}/s{len(stages) - 1 - si}-back",
            )
            # tr [N,T,2] — pad/truncate to num_poi_points
            n_pts = min(tr.shape[0], args.num_poi_points)
            T = tr.shape[1]
            wrist_tracks[st.start : st.start + T, :n_pts] = np.transpose(tr[:n_pts], (1, 0, 2))
            wrist_vis[st.start : st.start + T, :n_pts] = np.transpose(vi[:n_pts], (1, 0))

        if args.viz_dir is not None:
            viz = Path(args.viz_dir) / f"ep{ep_idx:06d}"
            viz.mkdir(parents=True, exist_ok=True)
            for st in stages:
                t = st.end
                bgr = cv2.cvtColor(wrist_frames[t], cv2.COLOR_RGB2BGR)
                heat = points_heatmap(
                    bgr,
                    wrist_tracks[t],
                    wrist_vis[t],
                    sigma=args.heatmap_sigma,
                    alpha=args.heatmap_alpha,
                )
                cv2.imwrite(str(viz / f"wrist_t{t:05d}.jpg"), heat)

        # --- write episode ---
        for t in range(data["n"]):
            st = _owning_stage(stages, t)
            frame: dict = {
                "observation.state": data["states"][t],
                "action": data["actions"][t],
                "task": data["tasks"][t] or "tapnet_preprocess",
            }
            # wrist heatmap
            if _cam_key(CAMERA_WRIST) in dst.meta.features:
                bgr = cv2.cvtColor(wrist_frames[t], cv2.COLOR_RGB2BGR)
                if wrist_ok and data["masks"][CAMERA_WRIST][t]:
                    bgr = points_heatmap(
                        bgr,
                        wrist_tracks[t],
                        wrist_vis[t],
                        sigma=args.heatmap_sigma,
                        alpha=args.heatmap_alpha,
                    )
                frame[_cam_key(CAMERA_WRIST)] = _rgb_to_dataset(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
            # Copy third-person views; grounded goals use the Molmo2 entrypoint.
            for cam in CAMERAS_3P:
                key = _cam_key(cam)
                if key not in dst.meta.features:
                    continue
                frame[key] = _rgb_to_dataset(data["frames"][cam][t])
                mkey = _mask_key(cam)
                if mkey in dst.meta.features:
                    frame[mkey] = np.array([data["masks"][cam][t]], dtype=bool)
            wmask = _mask_key(CAMERA_WRIST)
            if wmask in dst.meta.features:
                frame[wmask] = np.array([data["masks"][CAMERA_WRIST][t]], dtype=bool)

            # copy any other non-image features already handled; skip unknown
            dst.add_frame(frame)

        dst.save_episode()
        done.add(ep_idx)
        _save_progress(progress_path, done)
        print(f"  saved episode {ep_idx}", flush=True)

    if dst is not None:
        print("Finalizing destination dataset...")
        dst.finalize()
        if args.push_to_hub:
            print(f"Pushing {args.dst_repo_id} to Hub...")
            dst.push_to_hub(
                branch="main",
                tags=["robotics", "smolvla", "tapnet", "so101"],
                license="apache-2.0",
                push_videos=True,
            )
    print("Done.")


if __name__ == "__main__":
    main()
