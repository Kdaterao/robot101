"""Build POV-clustered and third-person Molmo-grounded SmolVLA data.

POV points are selected across demonstrations from their stage-tail tracks and
then tracked backward through each full stage. Third-person points are grounded
from task noun phrases plus the gripper, associated at the stage endpoint, and
tracked forward from the selected stage-start points. Per-stage diagnostics are
written beside the LeRobot output in ``point_tracks/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from hf_preprocess_smolvla import (
    CAMERAS_3P,
    CAMERA_WRIST,
    DEFAULT_SRC,
    _cam_key,
    _clone_features,
    _load_lerobot,
    _load_progress,
    _mask_key,
    _mask_is_real,
    _parse_episodes,
    _rgb_to_dataset,
    _save_progress,
)
from molmo_point_worker import MolmoPointWorker
from molmo2_worker import Molmo2Worker
from grounded_episode_io import load_episode_metadata, decode_episode_cameras
from grounded_destination import recover_empty_destination, check_resume_metadata
from grounded_validation import (EpisodeShapeError, validate_episode_shapes,
                                 record_skipped_episode, install_statistics_validation)
from robotap import (
    Stage,
    events_from_gripper_thresholds,
    select_stage_active,
    stages_from_events,
)
from tapnet_utils import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    points_heatmap,
    sample_query_points,
    track_tail_endpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_molmo_grounded"
DEFAULT_MOLMO = "allenai/Molmo2-4B"
GRIPPER_QUERY = "robot gripper"


def _task_text(tasks: list[str]) -> str:
    nonempty = [str(t).strip() for t in tasks if str(t).strip()]
    return Counter(nonempty).most_common(1)[0][0] if nonempty else ""


def _extract_noun_phrases(task: str, nlp, max_objects: int) -> list[str]:
    """Extract deduplicated, task-relevant noun chunks using spaCy."""
    if not task.strip():
        return []
    doc = nlp(task)
    ignored = {"it", "this", "that", "them", "thing", "object", "robot", "gripper"}
    phrases: list[str] = []
    seen: set[str] = set()
    for chunk in doc.noun_chunks:
        root = chunk.root
        phrase = " ".join(token.text for token in chunk if not token.is_punct).strip()
        norm = " ".join(phrase.lower().split())
        if not norm or root.pos_ == "PRON" or norm in ignored or norm in seen:
            continue
        # Drop noun chunks that are only determiners/pronouns or task boilerplate.
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in chunk):
            continue
        seen.add(norm)
        phrases.append(phrase)
        if len(phrases) >= max(1, int(max_objects)):
            break
    return phrases


def _valid_points(raw: dict, width: int, height: int) -> np.ndarray:
    pts = np.asarray(raw.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
    keep = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )
    return pts[keep]


def _json_points(points: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in point]
        for point in np.asarray(points).reshape(-1, 2)
    ]


def _last_endpoint(tracks: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return last visible coordinates and whether each point reached the end."""
    n = tracks.shape[0]
    ends = np.zeros((n, 2), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        ids = np.flatnonzero(visible[i])
        if len(ids):
            ends[i] = tracks[i, ids[-1]]
            valid[i] = bool(visible[i, -1])
    return ends, valid


def _draw_points(frame_rgb: np.ndarray, points: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(np.asarray(points).reshape(-1, 2)):
        cv2.circle(bgr, (int(round(x)), int(round(y))), 7, (0, 255, 0), 2)
        cv2.putText(bgr, str(i), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if label:
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(bgr, label[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return bgr


def _stage_cache_entry(
    tapir: BootsTAPIR,
    frames: list[np.ndarray],
    masks: np.ndarray,
    stage: Stage,
    query_xy: np.ndarray,
    tail_frames: int,
    label: str,
) -> dict:
    length = min(max(1, int(tail_frames)), stage.end - stage.start + 1)
    tail_start = stage.end - length + 1
    clip = frames[tail_start : stage.end + 1]
    real = masks[tail_start : stage.end + 1]
    if len(clip) < 1 or not bool(real.any()):
        return {
            "tracks": np.zeros((len(query_xy), max(1, len(clip)), 2), dtype=np.float32),
            "visible": np.zeros((len(query_xy), max(1, len(clip))), dtype=bool),
            "tail_start": tail_start,
            "length": length,
        }
    first_real = int(np.flatnonzero(real)[0])
    clip = clip[first_real:]
    real = real[first_real:]
    tail_start += first_real
    tracks, visible, _, _ = tapir.track_video(clip, query_xy, query_frame_index=0)
    # Ignore padding at each camera's invalid frames.
    visible[:, ~real] = False
    return {"tracks": tracks, "visible": visible, "tail_start": tail_start, "length": length}


def _select_pov_points(
    stage_cache: list[list[dict]],
    stage_lists: list[list[Stage]],
    args: argparse.Namespace,
    device,
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    """Choose shared candidate ids per aligned stage; local-select excess stages."""
    counts = [len(stages) for stages in stage_lists]
    n_aligned = min(counts, default=0)
    if len(set(counts)) > 1:
        print(
            f"WARNING: episode stage counts differ {counts}; clustering the shared "
            f"prefix of {n_aligned} stages, then selecting extra stages per episode."
        )
    selected: dict[tuple[int, int], np.ndarray] = {}
    if not stage_lists:
        return selected, 0

    def select(entries: list[dict], label: str, seed: int) -> np.ndarray:
        tracks = [entry["tracks"] for entry in entries]
        vis = [entry["visible"] for entry in entries]
        h = int(entries[0].get("height", 480))
        w = int(entries[0].get("width", 640))
        result = select_stage_active(
            tracks,
            vis,
            width=w,
            height=h,
            num_active=args.num_poi_points,
            n_clusters=args.n_clusters,
            static_thresh=args.static_thresh,
            funnel_quantile=args.funnel_quantile,
            goal_tail_frames=args.goal_tail_frames,
            device=device,
            seed=seed,
            label=label,
        )
        return np.asarray(result["point_ids"], dtype=np.int64)

    for si in range(n_aligned):
        entries = [stage_cache[ei][si] for ei in range(len(stage_cache))]
        ids = select(entries, f"shared stage {si}", args.seed + si)
        for ei in range(len(stage_lists)):
            selected[(ei, si)] = ids

    for ei, stages in enumerate(stage_lists):
        for si in range(n_aligned, len(stages)):
            selected[(ei, si)] = select(
                [stage_cache[ei][si]], f"episode {ei} local stage {si}", args.seed + ei * 997 + si
            )
    return selected, n_aligned


def _ground_and_track_camera(
    tapir: BootsTAPIR,
    molmo: MolmoPointWorker | Molmo2Worker,
    frames: list[np.ndarray],
    masks: np.ndarray,
    stage: Stage,
    entities: list[str],
    threshold: float,
    ambiguity_margin: float,
    label: str,
) -> dict:
    """Ground candidates on the stage start, track, associate at stage end, retrack POI."""
    clip = frames[stage.start : stage.end + 1]
    if not clip:
        return {"status": "failed", "reason": "empty_stage", "tracks": [], "visibility": []}
    start_real = bool(masks[stage.start])
    if not start_real:
        return {"status": "failed", "reason": "missing_start_frame", "tracks": [], "visibility": []}
    if not bool(masks[stage.end]):
        return {"status": "failed", "reason": "missing_end_frame", "tracks": [], "visibility": []}
    frame = clip[0]
    h, w = frame.shape[:2]
    groundings: dict[str, dict] = {}
    point_parts: list[np.ndarray] = []
    ranges: dict[str, tuple[int, int]] = {}
    failures: list[str] = []
    for entity in entities:
        prompt = f"Point to the {entity}."
        started = perf_counter()
        print(f"  Molmo {label}: {entity}", flush=True)
        try:
            result = molmo.ground(frame, prompt)
            print(f"    grounding finished in {perf_counter()-started:.1f}s", flush=True)
            raw_points = np.asarray(result.get("points_xy", []), dtype=np.float32).reshape(-1, 2)
            points = _valid_points(result, w, h)
            if len(points) < len(raw_points):
                failures.append(f"invalid_coordinates:{entity}:{len(raw_points) - len(points)}")
            groundings[entity] = {
                "prompt": prompt,
                "raw_response": str(result.get("raw_response", "")),
                "raw_points_xy": _json_points(raw_points),
                "points_xy": points.tolist(),
                "points_xy_norm": (points / np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)).tolist(),
            }
            if len(points):
                lo = sum(len(part) for part in point_parts)
                point_parts.append(points)
                ranges[entity] = (lo, lo + len(points))
            else:
                failures.append(f"{entity}_grounding_failed")
        except Exception as exc:  # record model failures per stage/camera
            groundings[entity] = {"prompt": prompt, "raw_response": "", "points_xy": [], "error": str(exc)}
            failures.append(f"{entity}_grounding_failed")

    if GRIPPER_QUERY not in ranges:
        return {
            "status": "failed",
            "reason": "gripper_grounding_failed",
            "failures": sorted(set(failures + ["gripper_grounding_failed"])),
            "groundings": groundings,
            "tracks": [],
            "visibility": [],
            "source": "none",
        }

    all_points = np.concatenate(point_parts, axis=0).astype(np.float32)
    tracking_started = perf_counter()
    try:
        candidates, candidate_vis, _, _ = tapir.track_video(clip, all_points, query_frame_index=0)
        print(f"  TAPIR {label}: {len(clip)} frames / {len(all_points)} points in {perf_counter()-tracking_started:.1f}s", flush=True)
    except Exception as exc:
        return {
            "status": "failed", "reason": f"candidate_tracking_failed:{exc}",
            "failures": sorted(set(failures + ["candidate_tracking_failed"])),
            "groundings": groundings, "tracks": [], "visibility": [], "source": "none",
        }
    stage_masks = masks[stage.start : stage.end + 1]
    candidate_vis[:, ~stage_masks] = False
    end_points, reaches_end = _last_endpoint(candidates, candidate_vis)
    gi0, gi1 = ranges[GRIPPER_QUERY]
    gripper_ids = np.arange(gi0, gi1)
    live_gripper = gripper_ids[reaches_end[gripper_ids]]
    if not len(live_gripper):
        return {
            "status": "failed",
            "reason": "tracker_lost_gripper",
            "failures": sorted(set(failures + ["tracker_lost_gripper"])),
            "groundings": groundings,
            "tracks": [],
            "visibility": [],
            "source": "none",
        }

    def norm(p):
        return np.asarray(p, dtype=np.float32) / np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)

    semantic_distances: list[tuple[float, str, np.ndarray]] = []
    for entity in entities:
        if entity == GRIPPER_QUERY or entity not in ranges:
            continue
        o0, o1 = ranges[entity]
        object_ids = np.arange(o0, o1)
        object_ids = object_ids[reaches_end[object_ids]]
        if not len(object_ids):
            failures.append(f"tracker_lost_object:{entity}")
            continue
        dmat = np.linalg.norm(norm(end_points[live_gripper])[:, None] - norm(end_points[object_ids])[None], axis=2)
        semantic_distances.append((float(dmat.min()), entity, object_ids))

    semantic_distances.sort(key=lambda item: item[0])
    close = [x for x in semantic_distances if x[0] <= threshold]
    ambiguous = len(close) > 1 and (close[1][0] - close[0][0]) <= ambiguity_margin
    if ambiguous:
        source, entity, distance = "gripper_fallback", None, None
        failures.append("ambiguous_object_match")
        status, reason = "failed", "ambiguous_object_match"
        chosen_ids = live_gripper
    elif close:
        distance, entity, chosen_ids = close[0]
        source, status, reason = "semantic_object", "ok", None
    else:
        source, entity, distance = "gripper_fallback", None, None
        status, reason = "ok", "no_semantic_object_near_gripper"
        chosen_ids = live_gripper

    initial_points = all_points[chosen_ids]
    # Each query has an independent causal state. Reuse the chosen trajectories
    # from the candidate pass instead of tracking the same video again.
    final_tracks = candidates[chosen_ids].copy()
    final_vis = candidate_vis[chosen_ids].copy()
    keep = final_vis[:, -1]
    if bool(keep.any()) and not bool(keep.all()):
        failures.append(f"tracker_lost_selected_points:{int((~keep).sum())}")
    if not bool(keep.any()):
        failures.append("tracker_lost_selected_goal")
        status, reason = "failed", "tracker_lost_selected_goal"
        final_tracks = np.zeros((0, len(clip), 2), dtype=np.float32)
        final_vis = np.zeros((0, len(clip)), dtype=bool)
        initial_points = np.zeros((0, 2), dtype=np.float32)
    else:
        final_tracks = final_tracks[keep]
        final_vis = final_vis[keep]
        initial_points = initial_points[keep]

    return {
        "status": status,
        "reason": reason,
        "failures": sorted(set(failures)),
        "source": source,
        "entity": entity,
        "distance_to_gripper": distance,
        "threshold": float(threshold),
        "image_size": [int(w), int(h)],
        "initial_points": initial_points.tolist(),
        "initial_points_norm": (initial_points / np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)).tolist(),
        "tracks": final_tracks.tolist(),
        "tracks_normalized": (
            final_tracks / np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)
        ).tolist(),
        "visibility": final_vis.tolist(),
        "groundings": groundings,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-repo-id", default=DEFAULT_SRC)
    p.add_argument("--dst-repo-id", default=DEFAULT_DST)
    p.add_argument("--episodes", default="0", help="Episode selection (default: 0), e.g. 0-9 or 0,1,5")
    p.add_argument("--molmo-model", default=DEFAULT_MOLMO, help="Base HF model ID or local Transformers checkpoint")
    p.add_argument("--molmo-backend", choices=["molmo2", "molmopoint"], default="molmo2")
    p.add_argument("--molmo-connector", type=Path, help="Local trusted so101_connector.pt; overrides Hub download")
    p.add_argument("--molmo-connector-repo", default="kdaterao/so101-molmo2-4b-gripper")
    p.add_argument("--molmo-connector-revision", default="main")
    p.add_argument("--molmo-dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--max-task-objects", type=int, default=8)
    p.add_argument("--object-proximity-threshold", type=float, default=0.08)
    p.add_argument("--ambiguity-margin", type=float, default=0.01)
    p.add_argument("--tapnet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dry-run", action="store_true", help="Only detect gripper stages; do not load models or write")
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
    p.add_argument("--cluster-tail-frames", type=int, default=30)
    p.add_argument("--num-sample-points", type=int, default=128)
    p.add_argument("--num-poi-points", type=int, default=16)
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--static-thresh", type=float, default=0.03)
    p.add_argument("--funnel-quantile", type=float, default=0.4)
    p.add_argument("--goal-tail-frames", type=int, default=5)
    p.add_argument("--heatmap-sigma", type=float, default=40.0)
    p.add_argument("--heatmap-alpha", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tapir-tf32", action="store_true", help="Allow faster TF32 FP32 operations on Ampere GPUs; may slightly change tracks")
    p.add_argument("--decode-batch-size", type=int, default=64, help="Frames per video seek; increase with available CPU RAM")
    p.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    p.add_argument("--viz-dir", type=Path, default=None)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cluster_tail_frames < 1:
        raise SystemExit("--cluster-tail-frames must be positive")
    if args.decode_batch_size < 1:
        raise SystemExit("--decode-batch-size must be positive")
    if args.object_proximity_threshold < 0 or args.ambiguity_margin < 0:
        raise SystemExit("distance threshold and ambiguity margin must be non-negative")
    LeRobotDataset, LeRobotDatasetMetadata, DEFAULT_FEATURES, HF_LEROBOT_HOME = _load_lerobot()
    ep_filter = _parse_episodes(args.episodes)
    src_root = HF_LEROBOT_HOME / args.src_repo_id
    src_meta = LeRobotDatasetMetadata(args.src_repo_id, root=src_root)
    fps = float(src_meta.fps)
    episode_ids = ep_filter if ep_filter is not None else list(range(int(src_meta.total_episodes)))
    print(f"Source episodes={len(episode_ids)} fps={fps}")
    cameras = [CAMERA_WRIST, *CAMERAS_3P]
    progress_path = HF_LEROBOT_HOME / args.dst_repo_id / "_preprocess_done.json"
    done = _load_progress(progress_path) if args.resume else set()
    dst_root = HF_LEROBOT_HOME / args.dst_repo_id
    skip_path = dst_root / "point_tracks" / "skipped_episodes.jsonl"
    dst = None
    if not args.dry_run:
        features = _clone_features(dict(src_meta.features), DEFAULT_FEATURES)
        if args.resume and dst_root.exists():
            if recover_empty_destination(dst_root) is not None:
                done = set()
        if args.resume and (dst_root / "meta").exists():
            try:
                check_resume_metadata(dst_root)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            dst = LeRobotDataset.resume(repo_id=args.dst_repo_id, root=dst_root)
        else:
            if dst_root.exists():
                raise SystemExit(f"Destination {dst_root} already exists; use --resume or choose another destination")
            dst = LeRobotDataset.create(
                repo_id=args.dst_repo_id,
                root=dst_root,
                fps=int(round(fps)),
                robot_type=getattr(src_meta, "robot_type", None) or "so100_so101_ee",
                features=features,
                use_videos=True,
            )
        install_statistics_validation(dst.meta.features)

    if args.dry_run:
        for ep_idx in episode_ids:
            source = LeRobotDataset(args.src_repo_id, episodes=[ep_idx], root=src_root, video_backend=args.video_backend)
            try:
                data = load_episode_metadata(source, cameras)
                validate_episode_shapes(data, src_meta.features, cameras)
            except EpisodeShapeError as exc:
                print(f"[skip ep {ep_idx}] metadata shape: {exc}", flush=True)
                continue
            grip = data["gripper"] if args.gripper_source == "state" else data["action_gripper"]
            events = events_from_gripper_thresholds(
                grip, fps=fps, closed_frac=args.gripper_closed_frac, open_frac=args.gripper_open_frac,
                min_dwell_frames=args.gripper_min_dwell_frames, vel_stall=args.gripper_vel_stall,
                vel_min=args.gripper_vel_min, smooth_window=args.gripper_smooth_window,
            )
            stages = stages_from_events(events, data["n"], args.min_stage_frames, args.first_primitive)
            print(f"episode {ep_idx}: {len(events)} gripper events, stages=" + ", ".join(f"[{s.start},{s.end}]" for s in stages))
        return

    try:
        import spacy
        nlp = spacy.load(args.spacy_model)
    except ImportError as exc:
        raise SystemExit("Install spaCy and an English model, e.g. python -m spacy download en_core_web_sm") from exc
    except OSError as exc:
        raise SystemExit(f"spaCy model {args.spacy_model!r} not installed. Run: python -m spacy download {args.spacy_model}") from exc

    if args.tapir_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA FP32 operations", flush=True)
    tapir = BootsTAPIR(checkpoint=args.tapnet_checkpoint, device=args.device)
    if args.molmo_backend == "molmo2":
        molmo = Molmo2Worker(
            args.molmo_model, device=args.device, dtype=args.molmo_dtype,
            connector_path=args.molmo_connector, connector_repo=args.molmo_connector_repo,
            connector_revision=args.molmo_connector_revision,
        )
    else:
        molmo = MolmoPointWorker(args.molmo_model, device=args.device, dtype=args.molmo_dtype)

    # Pass 1: track shared POV queries only through each stage's short tail.
    stage_lists: list[list[Stage]] = []
    tail_cache: list[list[dict]] = []
    eligible_episode_ids = []
    query_uv = None
    shared_resolution = None
    for epi, ep_idx in enumerate(episode_ids):
        print(f"[tail pass] episode {ep_idx} ({epi + 1}/{len(episode_ids)})", flush=True)
        source = LeRobotDataset(args.src_repo_id, episodes=[ep_idx], root=src_root, video_backend=args.video_backend)
        data = None
        try:
            data = load_episode_metadata(source, cameras)
            grip = data["gripper"] if args.gripper_source == "state" else data["action_gripper"]
            events = events_from_gripper_thresholds(
                grip, fps=fps, closed_frac=args.gripper_closed_frac, open_frac=args.gripper_open_frac,
                min_dwell_frames=args.gripper_min_dwell_frames, vel_stall=args.gripper_vel_stall,
                vel_min=args.gripper_vel_min, smooth_window=args.gripper_smooth_window,
            )
            stages = stages_from_events(events, data["n"], args.min_stage_frames, args.first_primitive)
            if len(stages) > args.max_stages:
                stages = stages[: args.max_stages]
                stages[-1] = Stage(stages[-1].start, data["n"] - 1, "none")
            tail_indices = {0}
            for stage in stages:
                tail_indices.update(range(max(stage.start, stage.end - args.cluster_tail_frames + 1), stage.end + 1))
            decode_episode_cameras(source, data, [CAMERA_WRIST], indices=tail_indices,
                                   batch_size=args.decode_batch_size, backend=args.video_backend)
            validate_episode_shapes(data, src_meta.features, [CAMERA_WRIST])
            wrist = data["frames"][CAMERA_WRIST]
            h, w = wrist[0].shape[:2]
            if shared_resolution is None:
                shared_resolution = (w, h)
                query = sample_query_points(w, h, args.num_sample_points, rng=np.random.default_rng(args.seed))
                query_uv = query / np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)
            elif (w, h) != shared_resolution:
                raise EpisodeShapeError(
                    f"Wrist camera resolution changed between episodes: {shared_resolution} vs {(w, h)}. "
                    "Cross-demo funneling requires consistent camera resolution."
                )
            query_xy = query_uv * np.array([max(1, w - 1), max(1, h - 1)], dtype=np.float32)
            episode_cache = []
            for si, st in enumerate(stages):
                entry = _stage_cache_entry(
                    tapir, wrist, data["masks"][CAMERA_WRIST], st, query_xy,
                    args.cluster_tail_frames, f"ep{ep_idx}/stage{si}/tail",
                )
                entry["height"], entry["width"] = wrist[0].shape[:2]
                episode_cache.append(entry)
            stage_lists.append(stages)
            tail_cache.append(episode_cache)
            eligible_episode_ids.append(ep_idx)
        except EpisodeShapeError as exc:
            record_skipped_episode(skip_path, ep_idx, "tail_pass", exc)
        del data, source

    if not eligible_episode_ids:
        print(f"No compatible episodes; see {skip_path}", flush=True)
    selected_ids, n_aligned = _select_pov_points(tail_cache, stage_lists, args, tapir.device)
    print(f"POV cross-demo alignment: {n_aligned} shared stage indices")

    # Pass 2: full-stage POV backtracking and third-person semantic tracks.
    for epi, ep_idx in enumerate(eligible_episode_ids):
        if ep_idx in done:
            print(f"[ep {ep_idx}] already written, skip", flush=True)
            continue
        print(f"\n=== Grounded preprocessing episode {ep_idx} ===", flush=True)
        source = LeRobotDataset(args.src_repo_id, episodes=[ep_idx], root=src_root, video_backend=args.video_backend)
        episode_started = perf_counter()
        data = None
        try:
            data = load_episode_metadata(source, cameras)
            decode_episode_cameras(source, data, cameras, batch_size=args.decode_batch_size,
                                   backend=args.video_backend)
            validate_episode_shapes(data, dst.meta.features, cameras)
        except EpisodeShapeError as exc:
            record_skipped_episode(skip_path, ep_idx, "full_episode", exc)
            del data, source
            continue
        stages = stage_lists[epi]
        n = data["n"]
        wrist_tracks = np.zeros((n, args.num_poi_points, 2), dtype=np.float32)
        wrist_vis = np.zeros((n, args.num_poi_points), dtype=bool)
        wrist_frames = data["frames"][CAMERA_WRIST]
        episode_records = []
        task = _task_text(data["tasks"])
        try:
            nouns = _extract_noun_phrases(task, nlp, args.max_task_objects)
        except Exception as exc:
            nouns = []
            noun_failure = f"noun_extraction_failed:{exc}"
        else:
            noun_failure = None if nouns else ("task_text_missing" if not task else "no_task_noun_phrases")
        entities = [GRIPPER_QUERY, *nouns]
        print(f"task={task!r} entities={entities}", flush=True)

        for si, st in enumerate(stages):
            stage_started = perf_counter()
            print(f"  stage {si}: {st.end-st.start+1} frames", flush=True)
            cache = tail_cache[epi][si]
            ids = selected_ids.get((epi, si), np.zeros(0, dtype=np.int64))
            pov_failure = None
            if len(ids):
                tail_tr = cache["tracks"][ids]
                tail_vi = cache["visible"][ids]
                end_points = track_tail_endpoint(tail_tr, tail_vi, n_tail=args.goal_tail_frames)
                full_clip = wrist_frames[st.start : st.end + 1]
                back_tracks = np.zeros((len(end_points), len(full_clip), 2), dtype=np.float32)
                back_vis = np.zeros((len(end_points), len(full_clip)), dtype=bool)
                if len(full_clip) and bool(data["masks"][CAMERA_WRIST][st.end]):
                    try:
                        back_tracks, back_vis = tapir.track_segment_backward(
                            full_clip, end_points, label=f"ep{ep_idx}/stage{si}/pov-back"
                        )
                        back_vis[:, ~data["masks"][CAMERA_WRIST][st.start : st.end + 1]] = False
                        print(f"  wrist backtracking finished in {perf_counter()-stage_started:.1f}s", flush=True)
                    except Exception as exc:
                        pov_failure = f"pov_tracking_failed:{exc}"
                else:
                    pov_failure = "missing_wrist_end_frame"
                count = min(args.num_poi_points, len(end_points))
                length = min(back_tracks.shape[1], st.end - st.start + 1)
                wrist_tracks[st.start : st.start + length, :count] = np.transpose(back_tracks[:count, :length], (1, 0, 2))
                wrist_vis[st.start : st.start + length, :count] = np.transpose(back_vis[:count, :length], (1, 0))
            else:
                end_points = np.zeros((0, 2), dtype=np.float32)
                back_tracks = np.zeros((0, st.end - st.start + 1, 2), dtype=np.float32)
                back_vis = np.zeros((0, st.end - st.start + 1), dtype=bool)
                pov_failure = "no_active_points_selected"

            third = {}
            for cam in CAMERAS_3P:
                if _cam_key(cam) not in src_meta.features:
                    continue
                result = _ground_and_track_camera(
                    tapir, molmo, data["frames"][cam], data["masks"][cam], st,
                    entities, args.object_proximity_threshold, args.ambiguity_margin,
                    f"ep{ep_idx}/stage{si}/{cam}",
                )
                if noun_failure:
                    result["status"] = "failed"
                    result["reason"] = noun_failure
                    result["failures"] = sorted(set(result.get("failures", []) + [noun_failure]))
                third[cam] = result
                print(
                    f"  {cam} stage {si}: source={result.get('source')} entity={result.get('entity')} "
                    f"distance={result.get('distance_to_gripper')} status={result.get('status')} reason={result.get('reason')}",
                    flush=True,
                )

            stage_record = {
                "episode": int(ep_idx), "subtask": int(si), "start_frame": int(st.start), "end_frame": int(st.end),
                "primitive": st.primitive,
                "pov": {
                    "source": "cross_demo_tail_cluster" if si < n_aligned else "local_tail_cluster",
                    "status": "failed" if pov_failure else "ok",
                    "reason": pov_failure,
                    "active_points_end": end_points.tolist(),
                    "tracks": back_tracks.tolist(), "visibility": back_vis.tolist(),
                },
                "grounding_model": {
                    "backend": args.molmo_backend, "base": args.molmo_model,
                    "connector": str(args.molmo_connector or args.molmo_connector_repo)
                    if args.molmo_backend == "molmo2" else None,
                    "connector_revision": args.molmo_connector_revision
                    if args.molmo_backend == "molmo2" and not args.molmo_connector else None,
                },
                "third_person": third,
            }
            episode_records.append(stage_record)
            print(f"  stage {si} finished in {perf_counter()-stage_started:.1f}s", flush=True)
            if args.viz_dir:
                viz_dir = args.viz_dir / f"ep{ep_idx:06d}"
                viz_dir.mkdir(parents=True, exist_ok=True)
                t = st.end
                cv2.imwrite(
                    str(viz_dir / f"stage{si:02d}_pov_end.jpg"),
                    points_heatmap(cv2.cvtColor(wrist_frames[t], cv2.COLOR_RGB2BGR), end_points, np.ones(len(end_points), bool), args.heatmap_sigma, args.heatmap_alpha),
                )
                for cam, result in third.items():
                    initial = np.asarray(result.get("initial_points", []), dtype=np.float32).reshape(-1, 2)
                    title = f"{cam}: {result.get('source')} {result.get('entity')} d={result.get('distance_to_gripper')}"
                    cv2.imwrite(
                        str(viz_dir / f"stage{si:02d}_{cam}_start.jpg"),
                        _draw_points(data["frames"][cam][st.start], initial, title),
                    )
                    tr = np.asarray(result.get("tracks", []), dtype=np.float32)
                    vis = np.asarray(result.get("visibility", []), dtype=bool)
                    if tr.size and tr.shape[1] == st.end - st.start + 1:
                        cv2.imwrite(
                            str(viz_dir / f"stage{si:02d}_{cam}_end.jpg"),
                            points_heatmap(cv2.cvtColor(data["frames"][cam][st.end], cv2.COLOR_RGB2BGR), tr[:, -1], vis[:, -1], args.heatmap_sigma, args.heatmap_alpha),
                        )

        sidecar_root = args.sidecar_dir or (dst_root / "point_tracks")
        sidecar_root.mkdir(parents=True, exist_ok=True)
        sidecar = sidecar_root / f"ep{ep_idx:06d}.json"
        sidecar.write_text(json.dumps({"episode": ep_idx, "task": task, "entities": entities, "subtasks": episode_records}, indent=2) + "\n", encoding="utf-8")

        # Render selected tracks into the camera streams used by the dataset.
        goals_by_stage = {}
        for si, record in enumerate(episode_records):
            goals_by_stage[si] = {}
            for cam, result in record["third_person"].items():
                tr = np.asarray(result.get("tracks", []), dtype=np.float32)
                vi = np.asarray(result.get("visibility", []), dtype=bool)
                goals_by_stage[si][cam] = (tr, vi)
        for t in range(n):
            si = next((i for i, st in enumerate(stages) if st.start <= t <= st.end), max(0, len(stages) - 1))
            frame = {
                **data["extras"][t],
                "observation.state": data["states"][t],
                "action": data["actions"][t],
                "task": data["tasks"][t] or task or "molmo_point_grounded_preprocess",
            }
            if _cam_key(CAMERA_WRIST) in dst.meta.features:
                rgb = wrist_frames[t]
                if data["masks"][CAMERA_WRIST][t]:
                    rgb = cv2.cvtColor(
                        points_heatmap(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), wrist_tracks[t], wrist_vis[t], args.heatmap_sigma, args.heatmap_alpha),
                        cv2.COLOR_BGR2RGB,
                    )
                frame[_cam_key(CAMERA_WRIST)] = _rgb_to_dataset(rgb)
                if _mask_key(CAMERA_WRIST) in dst.meta.features:
                    frame[_mask_key(CAMERA_WRIST)] = np.asarray([data["masks"][CAMERA_WRIST][t]], dtype=bool)
            for cam in CAMERAS_3P:
                key = _cam_key(cam)
                if key not in dst.meta.features:
                    continue
                rgb = data["frames"][cam][t]
                tr, vi = goals_by_stage.get(si, {}).get(cam, (np.zeros((0, 0, 2), np.float32), np.zeros((0, 0), bool)))
                local_t = t - stages[si].start
                if tr.ndim == 3 and tr.shape[1] > local_t and data["masks"][cam][t]:
                    rgb = cv2.cvtColor(
                        points_heatmap(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), tr[:, local_t], vi[:, local_t], args.heatmap_sigma, args.heatmap_alpha),
                        cv2.COLOR_BGR2RGB,
                    )
                frame[key] = _rgb_to_dataset(rgb)
                if _mask_key(cam) in dst.meta.features:
                    frame[_mask_key(cam)] = np.asarray([data["masks"][cam][t]], dtype=bool)
            dst.add_frame(frame)
        dst.save_episode()
        done.add(ep_idx)
        _save_progress(progress_path, done)
        print(f"  wrote {sidecar} and episode {ep_idx} in {perf_counter()-episode_started:.1f}s", flush=True)
        del data, source

    if dst is not None:
        dst.finalize()
        if int(dst.meta.total_episodes) == 0:
            print(f"No episodes written; no dataset upload. Skip report: {skip_path}", flush=True)
        if args.push_to_hub and int(dst.meta.total_episodes) > 0:
            dst.push_to_hub(branch="main", tags=["robotics", "smolvla", "tapnet", args.molmo_backend], license="apache-2.0", push_videos=True)
            sidecar_root = args.sidecar_dir or (dst_root / "point_tracks")
            if sidecar_root.is_dir():
                from huggingface_hub import HfApi

                HfApi().upload_folder(
                    folder_path=str(sidecar_root),
                    path_in_repo="point_tracks",
                    repo_id=args.dst_repo_id,
                    repo_type="dataset",
                    allow_patterns=["*.json"],
                    commit_message="Upload per-subtask point tracks and grounding diagnostics",
                )
    print("Done.")


if __name__ == "__main__":
    main()
