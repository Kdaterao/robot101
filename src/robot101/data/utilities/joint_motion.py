"""Estimate low motion from measured state coordinates, without visual models."""
from __future__ import annotations

import re

import numpy as np

from robot101.data.utilities.episode_helpers import _parse_episodes


def estimate_stillness(states, timestamps, scales, *, window_seconds=1.0,
                       speed_threshold=0.01, excursion_threshold=0.01, max_gap_seconds=0.1):
    """Return per-frame heuristic scores; unknown windows have null scores.

    Scales are fixed per-coordinate ranges, never estimated from this episode.
    A score of 0.5 corresponds to either configured low-motion threshold.
    """
    q, t, scale = np.asarray(states, float), np.asarray(timestamps, float), np.asarray(scales, float)
    if q.ndim != 2 or not q.shape[1] or len(q) < 2 or t.shape != (len(q),):
        raise ValueError("Need at least two timestamped state rows and one selected coordinate")
    if scale.shape != (q.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Each selected coordinate requires a finite positive normalization scale")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    for value in (window_seconds, speed_threshold, excursion_threshold, max_gap_seconds):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Window, thresholds and maximum timestamp gap must be finite and positive")
    normalized = q / scale
    dt = np.diff(t)
    speeds = np.abs(np.diff(normalized, axis=0)) / dt[:, None]
    rows = []
    for frame, stamp in enumerate(t):
        lo = int(np.searchsorted(t, stamp - window_seconds / 2, side="left"))
        hi = int(np.searchsorted(t, stamp + window_seconds / 2, side="right"))
        row = dict(frame=frame, timestamp=float(stamp), window_start_frame=lo,
                   window_end_frame=hi, stillness_score=None, likely_still=None)
        if hi - lo < 2:
            row["reason"] = "insufficient_window_samples"
        elif not np.isfinite(normalized[lo:hi]).all():
            row["reason"] = "invalid_state_samples"
        elif np.any(dt[lo:hi - 1] > max_gap_seconds):
            row["reason"] = "timestamp_gap"
        else:
            # Speed is robust to isolated derivative spikes. Excursion still
            # detects short jumps and slow accumulated movement in the window.
            speed = float(np.max(np.quantile(speeds[lo:hi - 1], 0.9, axis=0)))
            excursion = float(np.max(np.ptp(normalized[lo:hi], axis=0)))
            ratio = max(speed / speed_threshold, excursion / excursion_threshold)
            score = float(np.exp2(-min(ratio, 64.0) ** 2))
            row.update(stillness_score=score, likely_still=score >= 0.5,
                       normalized_speed_per_second=speed, normalized_excursion=excursion)
        rows.append(row)
    known = [r for r in rows if r["stillness_score"] is not None]
    summary = dict(frames=len(rows), known_frames=len(known), unknown_frames=len(rows) - len(known),
                   mean_stillness_score=float(np.mean([r["stillness_score"] for r in known])) if known else None,
                   likely_still_fraction=float(np.mean([r["likely_still"] for r in known])) if known else None,
                   window_seconds=window_seconds, speed_threshold=speed_threshold,
                   excursion_threshold=excursion_threshold, max_gap_seconds=max_gap_seconds,
                   meaning="Heuristic low-motion score, not a calibrated probability or grasp/contact detector",
                   averaging="Sample-weighted; unknown windows excluded")
    return dict(summary=summary, frames=rows)


def select_coordinates(feature, width, indices, include_gripper):
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors", names.get("axes"))
    if not isinstance(names, list) or len(names) != width:
        if indices is None:
            raise ValueError("State coordinate names unavailable; explicitly select --joint-indices (e.g. 0-5)")
        names = [f"coordinate_{i}" for i in range(width)]
    names = [str(n) for n in names]
    selected = list(range(width)) if indices is None else _parse_episodes(indices)
    if not selected or min(selected) < 0 or max(selected) >= width:
        raise ValueError("Joint indices are empty or outside the state vector")
    excluded = []
    if not include_gripper:
        excluded = [i for i in selected if re.search(r"gripp|finger|jaw", names[i], re.I)]
        selected = [i for i in selected if i not in excluded]
    if not selected:
        raise ValueError("No arm coordinates remain after excluding gripper/finger coordinates")
    selected_names = [names[i] for i in selected]
    labels = " ".join(selected_names).lower()
    if re.search(r"shoulder|elbow|wrist|joint|motor", labels):
        kind = "joint_state"
    elif re.search(r"end.effector|eef|quaternion|position|rotation", labels) or any(
            n.lower() in {"x", "y", "z", "qx", "qy", "qz", "qw", "rx", "ry", "rz"} for n in selected_names):
        kind = "pose_state_coordinates"
    else:
        kind = "state_coordinates_unverified"
    return selected, selected_names, [names[i] for i in excluded], kind


def normalization_scales(metadata, key, selected, explicit):
    if explicit:
        scale = np.array([float(v) for v in explicit.split(",")])
        if len(scale) == 1:
            scale = np.repeat(scale, len(selected))
        return scale, "explicit_raw_unit_ranges"
    stats = metadata.stats.get(key, {})
    if "min" not in stats or "max" not in stats:
        raise ValueError("Dataset normalization ranges unavailable; supply --joint-scales in raw position units")
    low, high = np.asarray(stats["min"], float).reshape(-1), np.asarray(stats["max"], float).reshape(-1)
    scale = (high - low)[selected]
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Selected dataset ranges contain constant/invalid coordinates; select other indices or supply --joint-scales")
    return scale, "dataset_wide_min_max"



def node_motion_context(states, timestamps, metadata, start, end, *, indices=None,
                        explicit_scales=None, window_seconds=1.0,
                        speed_threshold=0.01, excursion_threshold=0.01,
                        max_gap_seconds=0.1):
    """Compact caption evidence, computed strictly inside half-open node bounds.

    Missing metadata/data remains unknown. Never infer normalization from the
    current clip or use commanded actions as measured robot motion.
    """
    context = dict(start_frame=int(start), end_frame=int(end),
                   state_key="observation.state", status="unavailable")
    try:
        states = np.asarray(states, float)
        feature = metadata.features.get("observation.state", {})
        selected, names, excluded, kind = select_coordinates(feature, states.shape[1], indices, False)
        scales, source = normalization_scales(metadata, "observation.state", selected, explicit_scales)
        result = estimate_stillness(states[start:end, selected], timestamps[start:end], scales,
            window_seconds=window_seconds, speed_threshold=speed_threshold,
            excursion_threshold=excursion_threshold, max_gap_seconds=max_gap_seconds)
        summary = result["summary"]
        context.update(status="ok" if summary["known_frames"] else "unavailable",
                       signal_kind=kind, coordinate_indices=selected, coordinate_names=names,
                       excluded_gripper_coordinates=excluded, scales=scales.tolist(),
                       scale_source=source, **summary)
        if not summary["known_frames"]:
            context["reason"] = "no_valid_motion_windows"
        # Include temporal runs rather than an episode-wide average alone.
        # Unknown windows stay explicit and do not join separate idle spans.
        spans = []
        for row in result["frames"]:
            label = "unknown" if row["likely_still"] is None else (
                "low_motion" if row["likely_still"] else "above_low_motion_threshold")
            frame = start + row["frame"]
            if spans and spans[-1]["state"] == label:
                spans[-1]["end_frame"] = frame + 1
            else:
                spans.append(dict(start_frame=frame, end_frame=frame + 1, state=label))
        # Avoid spending a caption's context budget on noisy threshold crossings.
        context["motion_spans"] = spans if len(spans) <= 16 else []
        context["motion_spans_omitted"] = len(spans) > 16
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        context["reason"] = str(exc)
    return context
