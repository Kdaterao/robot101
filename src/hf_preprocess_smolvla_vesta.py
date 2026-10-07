"""Preprocess SmolVLA EE data with short-window wrist clustering and Vesta goals.

Candidate wrist points are clustered over only the final frames of each stage, then
the selected points are tracked backward through that stage. Vesta supplies
third-person goal points, which are also tracked backward through their stages.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_SRC = "felsager/community_dataset_v3_ee_smolVLA"
DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_vesta"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "tapnet"
    / "checkpoints"
    / "causal_bootstapir_checkpoint.pt"
)
CAMERAS_3P = ("top", "side")
CAMERA_WRIST = "wrist"
GRIPPER_INDEX = 7

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))


def _load_lerobot():
    try:
        from hf_hub_windows import install_hf_download_guards
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
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
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
            if key in item:
                frames[c].append(_tensor_to_rgb(item[key]))
            else:
                frames[c].append(np.zeros((480, 640, 3), dtype=np.uint8))
            masks[c].append(_mask_is_real(item, c))

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


def _load_vesta_provider(spec: str):
    """Load ``module[:factory]``; factory returns an object with goal_points()."""
    module_name, separator, factory_name = spec.partition(":")
    factory_name = factory_name if separator else "create_vesta_provider"
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        provider = factory()
    except (ImportError, AttributeError, TypeError) as e:
        raise SystemExit(
            f"Could not load Vesta provider {spec!r}: {e}. Expected "
            "module:create_vesta_provider returning an object with "
            "goal_points(frame_rgb, task)."
        ) from e
    if not callable(getattr(provider, "goal_points", None)):
        raise SystemExit(
            f"Vesta provider {spec!r} must return an object with "
            "goal_points(frame_rgb, task)."
        )
    return provider


def _valid_goal_points(raw, width: int, height: int) -> np.ndarray:
    """Validate Vesta's pixel-coordinate result and discard invalid points."""
    if raw is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(raw, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an N×2 array of pixel coordinates, got {points.shape}")
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return points[valid].astype(np.float32, copy=False)


def _clone_features(meta_features: dict, default_features: dict) -> dict:
    out = {}
    for k, v in meta_features.items():
        if k in default_features:
            continue
        # deep-ish copy of nested dicts
        out[k] = json.loads(json.dumps(v))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HF SmolVLA preprocess: tail-window wrist clustering + Vesta goals"
    )
    p.add_argument("--src-repo-id", default=DEFAULT_SRC)
    p.add_argument("--dst-repo-id", default=DEFAULT_DST)
    p.add_argument("--episodes", default=None, help="e.g. '0-99' or '0,1,5'")
    p.add_argument("--device", default=None)
    p.add_argument("--tapnet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--vesta-provider",
        default=None,
        help="Python provider as module[:factory]; factory defaults to create_vesta_provider",
    )
    p.add_argument(
        "--vesta-task",
        default=None,
        help="Task text override passed to Vesta (otherwise use the episode task text)",
    )
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
    p.add_argument(
        "--cluster-tail-frames",
        type=int,
        default=30,
        help="Candidate-point clustering window at the end of each stage",
    )
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
    if args.cluster_tail_frames < 1:
        raise SystemExit("--cluster-tail-frames must be at least 1")
    # Keep --help usable even when optional TapNet dependencies are unavailable.
    global BootsTAPIR, _kmeans_torch, points_heatmap, sample_query_points
    from robotap import Stage, events_from_gripper_thresholds, stages_from_events
    from tapnet_utils import BootsTAPIR, _kmeans_torch, points_heatmap, sample_query_points

    vesta = None
    if not args.dry_run:
        if not args.vesta_provider:
            raise SystemExit(
                "--vesta-provider is required unless --dry-run is used. See "
                "TAPNET_COMMANDS.md for the provider contract."
            )
        vesta = _load_vesta_provider(args.vesta_provider)
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
            if not data["masks"][CAMERA_WRIST][st.end]:
                print(
                    f"  wrist clustering skipped stage[{st.start},{st.end}]: "
                    "no real stage-end frame",
                    flush=True,
                )
                continue
            clip = wrist_frames[st.start : st.end + 1]
            if not clip:
                continue
            # Track the full candidate set only over the stage's tail window.
            tail_len = min(len(clip), max(1, args.cluster_tail_frames))
            tail_clip = clip[-tail_len:]
            seeds = _wrist_poi_at_end(
                tapir,
                tail_clip,
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

        # --- Third-person Vesta goals, propagated backward through each stage ---
        stage_goal_tracks: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            c: {} for c in CAMERAS_3P
        }
        for st in stages:
            episode_task = next(
                (str(task).strip() for task in data["tasks"][st.start : st.end + 1] if str(task).strip()),
                "",
            )
            task_text = args.vesta_task.strip() if args.vesta_task else episode_task
            if not task_text:
                print(
                    f"  Vesta unresolved stage[{st.start},{st.end}]: no task text",
                    flush=True,
                )
                continue
            for cam in CAMERAS_3P:
                if not data["masks"][cam][st.end]:
                    print(
                        f"  Vesta unresolved {cam} stage[{st.start},{st.end}]: "
                        "no real stage-end frame",
                        flush=True,
                    )
                    continue
                frame = data["frames"][cam][st.end]
                height, width = frame.shape[:2]
                try:
                    raw_points = vesta.goal_points(frame, task_text)
                    points = _valid_goal_points(raw_points, width, height)
                except Exception as e:
                    print(
                        f"  Vesta unresolved {cam} stage[{st.start},{st.end}]: {e}",
                        flush=True,
                    )
                    continue
                if len(points) == 0:
                    print(
                        f"  Vesta unresolved {cam} stage[{st.start},{st.end}]: "
                        "no valid goal points",
                        flush=True,
                    )
                    continue
                clip = data["frames"][cam][st.start : st.end + 1]
                try:
                    tracks, visible = tapir.track_segment_backward(
                        clip,
                        points,
                        label=f"ep{ep_idx}/{cam}/s{st.start}-{st.end}-back",
                    )
                except Exception as e:
                    print(
                        f"  Vesta goal tracking failed {cam} stage[{st.start},{st.end}]: {e}",
                        flush=True,
                    )
                    continue
                stage_goal_tracks[cam][id(st)] = (tracks, visible)
                print(
                    f"  {cam} stage[{st.start},{st.end}] Vesta goals={len(points)} "
                    f"tracked={tracks.shape[1]} frames",
                    flush=True,
                )

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
            # Third-person Vesta goal tracks
            for cam in CAMERAS_3P:
                key = _cam_key(cam)
                if key not in dst.meta.features:
                    continue
                bgr = cv2.cvtColor(data["frames"][cam][t], cv2.COLOR_RGB2BGR)
                tracked = stage_goal_tracks[cam].get(id(st))
                if tracked is not None and data["masks"][cam][t]:
                    tracks, vis = tracked
                    local_t = t - st.start
                    local_t = min(max(local_t, 0), tracks.shape[1] - 1)
                    bgr = points_heatmap(
                        bgr,
                        tracks[:, local_t],
                        vis[:, local_t],
                        sigma=args.heatmap_sigma,
                        alpha=args.heatmap_alpha,
                    )
                frame[key] = _rgb_to_dataset(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
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
