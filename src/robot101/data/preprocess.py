"""Build POV-clustered and third-person Molmo-grounded SmolVLA data.

POV uses tapnetCreate.py's shared appearance-feature bank and motion clustering.
Query descriptors sampled from multiple stage-tail frames keep the same identity
across demonstrations. Points are selected from their stage-tail tracks and
then tracked backward through each full stage. Third-person points are grounded
from task noun phrases plus the gripper, associated at the stage endpoint, and
tracked forward at a configurable sparse rate. Unresolved objects use a static
endpoint gripper snapshot. Per-stage diagnostics are
written beside the LeRobot output in ``point_tracks/``.
"""

from __future__ import annotations

from robot101.paths import REPO_ROOT

import argparse
import json
import sys
from time import perf_counter
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from robot101.data.utilities.episode_helpers import (
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
from robot101.perception.molmo2 import Molmo2Worker
from robot101.data.utilities.episode_io import load_episode_metadata, decode_episode_cameras
from robot101.data.utilities.destination import recover_empty_destination, check_resume_metadata
from robot101.data.utilities.validation import (EpisodeShapeError, validate_episode_shapes,
                                 record_skipped_episode, install_statistics_validation)
from robot101.perception.motion_plan import (
    Stage,
    events_from_gripper_thresholds,
    select_stage_active,
    stages_from_events,
)
from robot101.perception.tracking import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    concat_query_features,
    points_heatmap,
    sample_query_points,
    track_tail_endpoint,
)

LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_molmo_grounded"
DEFAULT_MOLMO = "allenai/Molmo2-4B"
GRIPPER_QUERY = "robot gripper"


def _task_text(tasks: list[str]) -> str:
    nonempty = [str(t).strip() for t in tasks if str(t).strip()]
    return Counter(nonempty).most_common(1)[0][0] if nonempty else ""


def _load_source_episode(LeRobotDataset, repo_id: str, ep_idx: int, root: Path, video_backend: str):
    """Load one episode, syncing its referenced shards when the local cache is partial.

    LeRobot can have valid episode metadata locally while the parquet shard for
    that episode is absent. In that state, filtering the cached parquet raises
    ``Instruction 'train' corresponds to no data!`` before its normal missing
    file check can download the episode. Retry once with a forced metadata sync;
    LeRobot then downloads only the selected episode's data and video files.
    """
    kwargs = {
        "episodes": [ep_idx],
        "root": root,
        "video_backend": video_backend,
    }
    try:
        return LeRobotDataset(repo_id, **kwargs)
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        print(
            f"  Episode {ep_idx} is missing from the local parquet cache; "
            "syncing its data and camera shards from the Hub...",
            flush=True,
        )
        return LeRobotDataset(repo_id, force_cache_sync=True, **kwargs)


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


def _tail_query_frames(stages: list[Stage], tail_frames: int, query_frames: int,
                       goal_tail_frames: int, masks: np.ndarray) -> list[int]:
    """Choose distinct, real seed frames inside each stage's clustering window."""
    indices = set()
    for stage in stages:
        lo = max(stage.start, stage.end - tail_frames + 1)
        hi = max(lo, stage.end - goal_tail_frames // 2)
        indices.update(int(round(f)) for f in np.linspace(lo, hi, query_frames))
    return sorted(i for i in indices if masks[i])


def _stage_cache_entry(
    tapir: BootsTAPIR,
    frames: list[np.ndarray],
    masks: np.ndarray,
    stage: Stage,
    shared_features,
    num_queries: int,
    tail_frames: int,
    label: str,
) -> dict:
    length = min(max(1, int(tail_frames)), stage.end - stage.start + 1)
    tail_start = stage.end - length + 1
    clip = frames[tail_start : stage.end + 1]
    real = masks[tail_start : stage.end + 1]
    if len(clip) < 1 or not bool(real.any()):
        return {
            "tracks": np.zeros((num_queries, max(1, len(clip)), 2), dtype=np.float32),
            "visible": np.zeros((num_queries, max(1, len(clip))), dtype=bool),
            "tail_start": tail_start,
            "length": length,
        }
    first_real = int(np.flatnonzero(real)[0])
    clip = clip[first_real:]
    real = real[first_real:]
    tail_start += first_real
    tracks, visible = tapir.track_with_features(
        clip, shared_features, label=label, num_frames=len(clip))
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


def _third_person_sample_indices(length: int, fps: float, tracking_fps: float,
                                 masks: np.ndarray) -> np.ndarray:
    """Always include endpoints; only track actual camera frames."""
    if fps <= 0 or tracking_fps <= 0:
        raise ValueError("Source and third-person tracking FPS must be positive")
    stride = max(1.0, fps / tracking_fps)
    indices = np.unique(np.r_[np.rint(np.arange(0, length, stride)).astype(int), 0, length - 1])
    indices = indices[(indices >= 0) & (indices < length)]
    return indices[np.asarray(masks, bool)[indices]]


def _expand_sparse_tracks(tracks, visible, indices, masks):
    """Interpolate coordinates between visible samples; hold sampled visibility."""
    frames = np.arange(len(masks))
    previous = np.clip(np.searchsorted(indices, frames, side="right") - 1, 0, len(indices) - 1)
    full = np.zeros((len(tracks), len(frames), 2), np.float32)
    full_visible = visible[:, previous].copy()
    for i in range(len(tracks)):
        valid = visible[i] & np.isfinite(tracks[i]).all(axis=1)
        if not valid.any():
            full_visible[i] = False
            continue
        for axis in range(2):
            full[i, :, axis] = np.interp(frames, indices[valid], tracks[i, valid, axis])
        full_visible[i] &= np.isfinite(tracks[i, previous]).all(axis=1)
    full_visible[:, ~np.asarray(masks, bool)] = False
    return full, full_visible


def _track_transition_points(
    tapir: BootsTAPIR,
    frames: list[np.ndarray],
    masks: np.ndarray,
    seed_frame: int,
    next_stage_start: int,
    persist_frames: int,
    points: np.ndarray,
    *,
    fps: float,
    tracking_fps: float | None,
    visibility_gap_seconds: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track previous-subtask endpoints into the next subtask's opening window."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    end_frame = min(len(frames) - 1, next_stage_start + persist_frames - 1)
    length = max(0, end_frame - next_stage_start + 1)
    empty_tracks = np.zeros((0, length, 2), dtype=np.float32)
    empty_vis = np.zeros((0, length), dtype=bool)
    if (
        not len(points) or length == 0 or seed_frame < 0 or seed_frame > next_stage_start
        or not masks[seed_frame]
    ):
        return empty_tracks, empty_vis, points[:0], empty_vis.copy()
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return empty_tracks, empty_vis, points, empty_vis.copy()

    clip = frames[seed_frame : end_frame + 1]
    clip_masks = np.asarray(masks[seed_frame : end_frame + 1], dtype=bool)
    if tracking_fps is None:
        sample_indices = np.flatnonzero(clip_masks)
    else:
        sample_indices = _third_person_sample_indices(len(clip), fps, tracking_fps, clip_masks)
    if len(sample_indices) == 0 or sample_indices[0] != 0:
        return empty_tracks, empty_vis, points[:0], empty_vis.copy()

    sparse_tracks, sparse_vis, _, _ = tapir.track_video(
        [clip[i] for i in sample_indices], points, query_frame_index=0
    )
    full_tracks, full_vis = _expand_sparse_tracks(sparse_tracks, sparse_vis, sample_indices, clip_masks)
    offset = next_stage_start - seed_frame
    full_tracks = full_tracks[:, offset : offset + length]
    full_vis = full_vis[:, offset : offset + length]
    full_vis[:, ~np.asarray(masks[next_stage_start : end_frame + 1], dtype=bool)] = False
    raw_vis = full_vis.copy()
    full_tracks, full_vis = _bridge_visibility_gaps(
        full_tracks, full_vis, masks[next_stage_start : end_frame + 1],
        round(fps * visibility_gap_seconds),
    )
    return full_tracks, full_vis, points, raw_vis


def _bridge_visibility_gaps(tracks, visible, masks, max_gap_frames):
    """Interpolate bounded confidence gaps; never fill missing camera frames."""
    tracks = np.asarray(tracks, dtype=np.float32).copy()
    visible = np.asarray(visible, dtype=bool).copy()
    masks = np.asarray(masks, dtype=bool)
    visible &= masks[None, :] & np.isfinite(tracks).all(axis=-1)
    for point in range(len(tracks)):
        anchors = np.flatnonzero(visible[point])
        for left, right in zip(anchors[:-1], anchors[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap_frames and masks[left:right + 1].all():
                weight = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
                tracks[point, left + 1:right] = (
                    tracks[point, left] * (1 - weight[:, None])
                    + tracks[point, right] * weight[:, None]
                )
                visible[point, left + 1:right] = True
    return tracks, visible


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
    fps: float = 30.0,
    tracking_fps: float = 1.0,
    episode_gripper_cache: dict | None = None,
    visibility_gap_seconds: float = 2.0,
) -> dict:
    """Associate sparse semantic tracks; otherwise freeze an endpoint gripper snapshot."""
    clip = frames[stage.start : stage.end + 1]
    if not clip:
        return {"status": "failed", "reason": "empty_stage", "tracks": [], "visibility": []}
    stage_masks = np.asarray(masks[stage.start : stage.end + 1], dtype=bool)
    groundings: dict[str, dict] = {}
    failures: list[str] = []
    sample_indices = None
    h, w = clip[0].shape[:2]
    episode_gripper_cache = {} if episode_gripper_cache is None else episode_gripper_cache

    def fallback(reason, tracked_points=None, tracked_vis=None):
        """Freeze gripper coordinates, never the moving gripper trajectory."""
        failures.append(reason)
        snapshot = len(clip) - 1 if len(clip) and stage_masks[-1] else None
        snapshot_source = "subtask_end"
        points = np.zeros((0, 2), np.float32)
        if snapshot is not None and tracked_points is not None:
            points = _valid_points({"points_xy": tracked_points[tracked_vis[:, snapshot], snapshot]}, w, h)
        if snapshot is not None and not len(points):
            try:
                print(f"  Molmo {label}: fallback gripper snapshot at stage-end frame {stage.end}", flush=True)
                raw = molmo.ground(clip[snapshot], f"Point to the {GRIPPER_QUERY}.")
                points = _valid_points(raw, w, h)
                groundings["fallback_gripper_snapshot"] = {
                    "frame": stage.end, "points_xy": points.tolist(),
                    "raw_response": str(raw.get("raw_response", "")),
                }
            except Exception as exc:
                failures.append(f"fallback_gripper_grounding_failed:{exc}")
        # Coordinates from any other time can represent a different goal.
        # Leave this stage unresolved if its exact endpoint has no valid point.
        tracks = np.repeat(points[:, None, :], len(clip), axis=1)
        visible = np.repeat(stage_masks[None, :], len(points), axis=0)
        return {
            "status": "ok" if len(points) else "failed", "reason": reason if len(points) else "fallback_gripper_unresolved",
            "failures": sorted(set(failures)), "source": "gripper_fallback" if len(points) else "none",
            "fallback_mode": "static_snapshot", "snapshot_frame": stage.start + snapshot if len(points) else None,
            "snapshot_source": snapshot_source if len(points) else None,
            "entity": None, "distance_to_gripper": None, "image_size": [w, h],
            "initial_points": points.tolist(),
            "initial_points_norm": (points / np.array([max(1, w - 1), max(1, h - 1)])).tolist(),
            "tracks": tracks.tolist(), "visibility": visible.tolist(),
            "tracks_normalized": (tracks / np.array([max(1, w - 1), max(1, h - 1)])).tolist(),
            "groundings": groundings, "tracking_fps": tracking_fps,
        }

    start_real = bool(masks[stage.start])
    if not start_real:
        return fallback("missing_start_frame")
    if not bool(masks[stage.end]):
        return fallback("missing_end_frame")
    frame = clip[0]
    point_parts: list[np.ndarray] = []
    ranges: dict[str, tuple[int, int]] = {}
    batched_results = None
    if hasattr(molmo, "ground_batch"):
        started = perf_counter()
        print(f"  Molmo {label}: grounding {len(entities)} independent requests", flush=True)
        batched_results = molmo.ground_batch([(frame, f"Point to the {entity}.") for entity in entities])
        if len(batched_results) != len(entities):
            raise ValueError("Molmo grounding batch length mismatch")
        print(f"    batched grounding finished in {perf_counter()-started:.1f}s", flush=True)
    for entity_index, entity in enumerate(entities):
        prompt = f"Point to the {entity}."
        started = perf_counter()
        if batched_results is None:
            print(f"  Molmo {label}: {entity}", flush=True)
        try:
            result = batched_results[entity_index] if batched_results is not None else molmo.ground(frame, prompt)
            if result.get("error"):
                raise RuntimeError(result["error"])
            if batched_results is None:
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
        return fallback("gripper_grounding_failed")
    if len(ranges) == 1:
        return fallback("no_semantic_object_grounded")

    all_points = np.concatenate(point_parts, axis=0).astype(np.float32)
    tracking_started = perf_counter()
    sample_indices = _third_person_sample_indices(len(clip), fps, tracking_fps, stage_masks)
    try:
        candidates, candidate_vis, _, _ = tapir.track_video(
            [clip[i] for i in sample_indices], all_points, query_frame_index=0)
        candidates, candidate_vis = _expand_sparse_tracks(candidates, candidate_vis, sample_indices, stage_masks)
        print(f"  TAPIR {label}: {len(sample_indices)}/{len(clip)} frames at {tracking_fps:g} FPS / "
              f"{len(all_points)} points in {perf_counter()-tracking_started:.1f}s", flush=True)
    except Exception as exc:
        return fallback(f"candidate_tracking_failed:{exc}")
    end_points, reaches_end = _last_endpoint(candidates, candidate_vis)
    gi0, gi1 = ranges[GRIPPER_QUERY]
    gripper_ids = np.arange(gi0, gi1)
    live_gripper = gripper_ids[reaches_end[gripper_ids]]
    if not len(live_gripper):
        return fallback("tracker_lost_gripper", candidates[gripper_ids], candidate_vis[gripper_ids])

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
        return fallback("ambiguous_object_match", candidates[gripper_ids], candidate_vis[gripper_ids])
    elif close:
        distance, entity, chosen_ids = close[0]
        source, status, reason = "semantic_object", "ok", None
    else:
        return fallback("no_semantic_object_near_gripper", candidates[gripper_ids], candidate_vis[gripper_ids])

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

    raw_final_vis = final_vis.copy()
    final_tracks, final_vis = _bridge_visibility_gaps(
        final_tracks, final_vis, stage_masks, round(fps * visibility_gap_seconds)
    )

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
        "raw_visibility": raw_final_vis.tolist(),
        "visibility_gap_seconds": visibility_gap_seconds,
        "groundings": groundings,
        "tracking_fps": tracking_fps,
        "tracking_sample_frames": (sample_indices + stage.start).tolist(),
    }


from robot101.data.utilities.preprocess_config import build_parser


def main() -> None:
    args = build_parser().parse_args()
    if any(not np.isfinite(v) or v < 0 for v in (args.pov_visibility_gap_seconds, args.third_person_visibility_gap_seconds)):
        raise SystemExit("Visibility gap seconds must be finite and nonnegative")
    if not np.isfinite(args.transition_persist_seconds) or args.transition_persist_seconds < 0:
        raise SystemExit("--transition-persist-seconds must be finite and nonnegative")
    if args.molmo_batch_size < 1:
        raise SystemExit("--molmo-batch-size must be positive")
    if args.query_frames_per_stage < 1 or args.num_sample_points < 1:
        raise SystemExit("Query frame and sample point counts must be positive")
    if args.cluster_tail_frames < 1:
        raise SystemExit("--cluster-tail-frames must be positive")
    if args.tapir_frame_batch_size < 1:
        raise SystemExit("--tapir-frame-batch-size must be positive")
    if args.decode_batch_size < 1:
        raise SystemExit("--decode-batch-size must be positive")
    if args.object_proximity_threshold < 0 or args.ambiguity_margin < 0:
        raise SystemExit("distance threshold and ambiguity margin must be non-negative")
    if not np.isfinite(args.third_person_tracking_fps) or args.third_person_tracking_fps <= 0:
        raise SystemExit("--third-person-tracking-fps must be positive and finite")
    if not 0 < args.gripper_min_change_frac <= 1 or not np.isfinite(args.gripper_min_change_abs) or args.gripper_min_change_abs < 0:
        raise SystemExit("Gripper change fraction must be in (0, 1]; absolute change must be finite and nonnegative")
    if args.min_stage_frames < 1 or args.max_stages < 1:
        raise SystemExit("Stage frame/count limits must be positive")
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
            source = _load_source_episode(
                LeRobotDataset, args.src_repo_id, ep_idx, src_root, args.video_backend
            )
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
                min_change_frac=args.gripper_min_change_frac if args.gripper_event_mode == "movement" else None,
                min_change_abs=args.gripper_min_change_abs,
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
    print(f"TAPIR temporal frame batch: {args.tapir_frame_batch_size}", flush=True)
    tapir = BootsTAPIR(checkpoint=args.tapnet_checkpoint, device=args.device,
                      frame_batch_size=args.tapir_frame_batch_size)
    if args.molmo_backend == "molmo2":
        molmo = Molmo2Worker(
            args.molmo_model, device=args.device, dtype=args.molmo_dtype,
            connector_path=args.molmo_connector, connector_repo=args.molmo_connector_repo,
            connector_revision=args.molmo_connector_revision, batch_size=args.molmo_batch_size,
        )
    else:
        from robot101.legacy.molmo_point import MolmoPointWorker
        molmo = MolmoPointWorker(args.molmo_model, device=args.device, dtype=args.molmo_dtype)

    # Build one shared appearance-feature bank, as in tapnetCreate.py.
    # Candidate tracking remains restricted to stage tails.
    stage_lists: list[list[Stage]] = []
    tail_cache: list[list[dict]] = []
    eligible_episode_ids = []
    feature_parts = []
    query_source_parts = []
    query_xy_parts = []
    query_rng = np.random.default_rng(args.seed)
    shared_resolution = None
    for epi, ep_idx in enumerate(episode_ids):
        print(f"[query bank] episode {ep_idx} ({epi + 1}/{len(episode_ids)})", flush=True)
        source = _load_source_episode(
            LeRobotDataset, args.src_repo_id, ep_idx, src_root, args.video_backend
        )
        data = None
        try:
            data = load_episode_metadata(source, cameras)
            grip = data["gripper"] if args.gripper_source == "state" else data["action_gripper"]
            events = events_from_gripper_thresholds(
                grip, fps=fps, closed_frac=args.gripper_closed_frac, open_frac=args.gripper_open_frac,
                min_dwell_frames=args.gripper_min_dwell_frames, vel_stall=args.gripper_vel_stall,
                vel_min=args.gripper_vel_min, smooth_window=args.gripper_smooth_window,
                min_change_frac=args.gripper_min_change_frac if args.gripper_event_mode == "movement" else None,
                min_change_abs=args.gripper_min_change_abs,
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
            elif (w, h) != shared_resolution:
                raise EpisodeShapeError(
                    f"Wrist camera resolution changed between episodes: {shared_resolution} vs {(w, h)}. "
                    "Cross-demo funneling requires consistent camera resolution."
                )
            # Spread seed frames across every stage tail, and extract each
            # seed's descriptors once. They retain the same identity in all demos.
            query_frames = _tail_query_frames(stages, args.cluster_tail_frames,
                                             args.query_frames_per_stage, args.goal_tail_frames,
                                             data["masks"][CAMERA_WRIST])
            n_per = max(8, args.num_sample_points // max(1, len(episode_ids) * len(stages) * args.query_frames_per_stage))
            for index in query_frames:
                xy = sample_query_points(w, h, n_per, rng=query_rng)
                tyx = np.stack([np.zeros(len(xy)), xy[:, 1], xy[:, 0]], axis=1).astype(np.float32)
                feature_parts.append(tapir.init_features(wrist[index], tyx))
                query_source_parts.append(np.tile([ep_idx, index], (len(xy), 1)))
                query_xy_parts.append(xy)
            stage_lists.append(stages)
            eligible_episode_ids.append(ep_idx)
        except EpisodeShapeError as exc:
            record_skipped_episode(skip_path, ep_idx, "tail_pass", exc)
        del data, source

    if not eligible_episode_ids or not feature_parts:
        print(f"No valid shared wrist queries; see {skip_path}", flush=True)
        dst.finalize()
        return
    shared_features = concat_query_features(feature_parts)
    query_sources = np.concatenate(query_source_parts).astype(np.int64)
    query_source_xy = np.concatenate(query_xy_parts).astype(np.float32)
    num_queries = len(query_sources)
    del feature_parts
    bank_root = args.sidecar_dir or (dst_root / "point_tracks")
    bank_root.mkdir(parents=True, exist_ok=True)
    (bank_root / "pov_query_bank.json").write_text(json.dumps({
        "method": "tapnetCreate_shared_appearance_features",
        "source_episode_frame": query_sources.tolist(),
        "source_points_xy": query_source_xy.tolist(),
        "tail_frames": args.cluster_tail_frames,
        "query_frames_per_stage": args.query_frames_per_stage,
    }) + "\n", encoding="utf-8")
    print(f"Shared wrist feature bank: {num_queries} queries; same descriptors in every episode", flush=True)

    # Track this exact feature bank through each stage tail in every demo.
    for epi, ep_idx in enumerate(eligible_episode_ids):
        source = _load_source_episode(
            LeRobotDataset, args.src_repo_id, ep_idx, src_root, args.video_backend
        )
        data = load_episode_metadata(source, cameras)
        stages = stage_lists[epi]
        tail_indices = {0}
        for stage in stages:
            tail_indices.update(range(max(stage.start, stage.end - args.cluster_tail_frames + 1), stage.end + 1))
        decode_episode_cameras(source, data, [CAMERA_WRIST], indices=tail_indices,
                               batch_size=args.decode_batch_size, backend=args.video_backend)
        episode_cache = []
        for si, stage in enumerate(stages):
            entry = _stage_cache_entry(
                tapir, data["frames"][CAMERA_WRIST], data["masks"][CAMERA_WRIST], stage,
                shared_features, num_queries, args.cluster_tail_frames, f"ep{ep_idx}/stage{si}/tail")
            entry["height"], entry["width"] = data["frames"][CAMERA_WRIST][0].shape[:2]
            episode_cache.append(entry)
        tail_cache.append(episode_cache)
        del data, source
    del shared_features

    selected_ids, n_aligned = _select_pov_points(tail_cache, stage_lists, args, tapir.device)
    print(f"POV cross-demo alignment: {n_aligned} shared stage indices")

    # Pass 2: full-stage POV backtracking and third-person semantic tracks.
    for epi, ep_idx in enumerate(eligible_episode_ids):
        if ep_idx in done:
            print(f"[ep {ep_idx}] already written, skip", flush=True)
            continue
        print(f"\n=== Grounded preprocessing episode {ep_idx} ===", flush=True)
        source = _load_source_episode(
            LeRobotDataset, args.src_repo_id, ep_idx, src_root, args.video_backend
        )
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
        pov_slots = args.num_poi_points * (2 if args.pov_track_previous_stage else 1)
        wrist_tracks = np.zeros((n, pov_slots, 2), dtype=np.float32)
        wrist_vis = np.zeros((n, pov_slots), dtype=bool)
        previous_pov = None
        wrist_frames = data["frames"][CAMERA_WRIST]
        episode_records = []
        episode_gripper_caches = {cam: {} for cam in CAMERAS_3P}
        previous_camera_points = {cam: None for cam in CAMERAS_3P}
        transition_frames = int(round(fps * args.transition_persist_seconds))
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
            # Inclusive stage boundaries may share a frame; reset slots so
            # an older stage cannot leak into the new current/previous pair.
            wrist_tracks[st.start:st.end + 1] = 0
            wrist_vis[st.start:st.end + 1] = False
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
                raw_back_vis = back_vis.copy()
                if len(full_clip) and bool(data["masks"][CAMERA_WRIST][st.end]):
                    try:
                        back_tracks, back_vis = tapir.track_segment_backward(
                            full_clip, end_points, label=f"ep{ep_idx}/stage{si}/pov-back"
                        )
                        back_vis[:, ~data["masks"][CAMERA_WRIST][st.start : st.end + 1]] = False
                        raw_back_vis = back_vis.copy()
                        back_tracks, back_vis = _bridge_visibility_gaps(
                            back_tracks, back_vis, data["masks"][CAMERA_WRIST][st.start : st.end + 1],
                            round(fps * args.pov_visibility_gap_seconds),
                        )
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
                raw_back_vis = back_vis.copy()
                pov_failure = "no_active_points_selected"

            # Continue only the immediately preceding stage's current points.
            # Seed on their actual endpoint frame, then track forward; never
            # freeze old coordinates or recursively accumulate older stages.
            previous_record = {
                "source_subtask": si - 1 if si else None,
                "status": "disabled" if not args.pov_track_previous_stage else "unavailable",
                "tracks": [], "visibility": [], "raw_visibility": [],
            }
            if args.pov_track_previous_stage and transition_frames > 0 and previous_pov is not None:
                seed_frame, seed_tracks, seed_vis = previous_pov
                seeds = seed_tracks[seed_vis & np.isfinite(seed_tracks).all(axis=1)][:args.num_poi_points]
                if len(seeds) and data["masks"][CAMERA_WRIST][seed_frame]:
                    try:
                        carry_length = min(st.end - st.start + 1, transition_frames)
                        carry_end = st.start + carry_length - 1
                        print(
                            f"  POV stage {si}: continuing {len(seeds)} previous-stage points "
                            f"for {carry_length / fps:.2f}s",
                            flush=True,
                        )
                        carry_tracks, carry_vis, _, _ = tapir.track_video(
                            wrist_frames[seed_frame:carry_end + 1], seeds, query_frame_index=0,
                        )
                        offset = st.start - seed_frame
                        carry_tracks = carry_tracks[:, offset:offset + carry_length]
                        carry_vis = carry_vis[:, offset:offset + carry_length]
                        stage_masks = data["masks"][CAMERA_WRIST][st.start:carry_end + 1]
                        carry_vis[:, ~stage_masks] = False
                        raw_carry_vis = carry_vis.copy()
                        carry_tracks, carry_vis = _bridge_visibility_gaps(
                            carry_tracks, carry_vis, stage_masks,
                            round(fps * args.pov_visibility_gap_seconds),
                        )
                        count = len(seeds)
                        slots = slice(args.num_poi_points, args.num_poi_points + count)
                        wrist_tracks[st.start:carry_end + 1, slots] = carry_tracks.transpose(1, 0, 2)
                        wrist_vis[st.start:carry_end + 1, slots] = carry_vis.T
                        previous_record.update(
                            status="ok", seed_frame=seed_frame, seed_points=seeds.tolist(),
                            tracks=carry_tracks.tolist(), visibility=carry_vis.tolist(),
                            raw_visibility=raw_carry_vis.tolist(),
                            persist_seconds=args.transition_persist_seconds,
                            persist_frames=carry_length,
                        )
                    except Exception as exc:
                        previous_record.update(status="failed", reason=f"previous_pov_tracking_failed:{exc}")
                        print(f"  POV carryover failed: {exc}", flush=True)
            previous_pov = (st.end, back_tracks[:, -1].copy(), back_vis[:, -1].copy())

            third = {}
            for cam in CAMERAS_3P:
                if _cam_key(cam) not in src_meta.features:
                    continue
                result = _ground_and_track_camera(
                    tapir, molmo, data["frames"][cam], data["masks"][cam], st,
                    entities, args.object_proximity_threshold, args.ambiguity_margin,
                    f"ep{ep_idx}/stage{si}/{cam}", fps=fps, tracking_fps=args.third_person_tracking_fps,
                    episode_gripper_cache=episode_gripper_caches[cam],
                    visibility_gap_seconds=args.third_person_visibility_gap_seconds,
                )
                transition = {
                    "source_subtask": si - 1 if si else None,
                    "status": "disabled" if transition_frames == 0 else "unavailable",
                    "tracks": [], "visibility": [], "raw_visibility": [],
                }
                prior = previous_camera_points[cam]
                if transition_frames > 0 and prior is not None:
                    try:
                        carry_tracks, carry_vis, carry_points, raw_carry_vis = _track_transition_points(
                            tapir, data["frames"][cam], data["masks"][cam],
                            prior["frame"], st.start, transition_frames, prior["points"],
                            fps=fps, tracking_fps=args.third_person_tracking_fps,
                            visibility_gap_seconds=args.third_person_visibility_gap_seconds,
                            label=f"ep{ep_idx}/stage{si}/{cam}/transition",
                        )
                        transition.update(
                            status="ok" if bool(carry_vis.any()) else "tracker_lost",
                            seed_points=carry_points.tolist(), tracks=carry_tracks.tolist(),
                            visibility=carry_vis.tolist(), raw_visibility=raw_carry_vis.tolist(),
                            persist_seconds=args.transition_persist_seconds,
                            persist_frames=int(carry_tracks.shape[1]),
                        )
                    except Exception as exc:
                        transition.update(status="failed", reason=f"transition_tracking_failed:{exc}")
                result["transition_from_previous"] = transition
                result_tracks = np.asarray(result.get("tracks", []), dtype=np.float32)
                result_vis = np.asarray(result.get("visibility", []), dtype=bool)
                if result_tracks.ndim == 3 and result_tracks.shape[1] and result_vis.shape == result_tracks.shape[:2]:
                    valid = result_vis[:, -1] & np.isfinite(result_tracks[:, -1]).all(axis=1)
                    previous_camera_points[cam] = {
                        "frame": st.end,
                        "points": result_tracks[valid, -1].copy(),
                    }
                else:
                    previous_camera_points[cam] = None
                if noun_failure:
                    result["failures"] = sorted(set(result.get("failures", []) + [noun_failure]))
                    if not result.get("tracks"):
                        result["status"] = "failed"
                        result["reason"] = noun_failure
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
                    "query_ids": ids.tolist(),
                    "query_sources_episode_frame": query_sources[ids].tolist(),
                    "tracks": back_tracks.tolist(), "visibility": back_vis.tolist(),
                    "raw_visibility": raw_back_vis.tolist(),
                    "visibility_gap_seconds": args.pov_visibility_gap_seconds,
                    "previous_stage": previous_record,
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
        sidecar.write_text(json.dumps({"episode": ep_idx, "destination_episode": int(dst.meta.total_episodes),
                                      "task": task, "entities": entities,
                                      "heatmap": {"sigma": args.heatmap_sigma, "alpha": args.heatmap_alpha},
                                      "subtasks": episode_records}, indent=2) + "\n", encoding="utf-8")

        # Keep dataset videos clean. Tracked coordinates and visibility are in
        # the episode sidecar; --viz-dir can render separate review images.
        for t in range(n):
            frame = {
                **data["extras"][t],
                "observation.state": data["states"][t],
                "action": data["actions"][t],
                "task": data["tasks"][t] or task or "molmo_point_grounded_preprocess",
            }
            if _cam_key(CAMERA_WRIST) in dst.meta.features:
                frame[_cam_key(CAMERA_WRIST)] = _rgb_to_dataset(wrist_frames[t])
                if _mask_key(CAMERA_WRIST) in dst.meta.features:
                    frame[_mask_key(CAMERA_WRIST)] = np.asarray([data["masks"][CAMERA_WRIST][t]], dtype=bool)
            for cam in CAMERAS_3P:
                key = _cam_key(cam)
                if key not in dst.meta.features:
                    continue
                frame[key] = _rgb_to_dataset(data["frames"][cam][t])
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
