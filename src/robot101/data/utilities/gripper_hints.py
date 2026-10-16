"""Episode-relative gripper sensor context for sparse-frame captioning."""
from __future__ import annotations

import numpy as np


def build_gripper_hints(record, node, frame_ids, *, kind, open_direction="increasing"):
    """Likely positions, not calibrated jaw widths or successful-grasp labels."""
    if open_direction not in {"increasing", "decreasing"}:
        raise ValueError("Invalid gripper opening direction")
    frames = [int(i) for i in frame_ids]
    fps = float(record["fps"])
    origin = frames[0] if frames else node["start_frame"]
    context = dict(version="episode-relative-gripper-v1", source="observation.state gripper",
                   open_direction=open_direction, available=False,
                   node_bounds=[node["start_frame"], node["end_frame"]],
                   time_reference="source episode; clip times are relative to the first sampled frame",
                   clip_origin_frame=origin,
                   warning="Episode-relative position hints only; not proof of contact, grasp, release of an object, or task success.",
                   states=[], transitions=[])
    raw = record.get("gripper")
    if raw is None:
        context["reason"] = "missing_gripper_signal"
        return context
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        context["reason"] = "invalid_gripper_signal"
        return context
    if values.ndim != 1 or len(values) != record["n_frames"]:
        context["reason"] = "invalid_gripper_signal_shape"
        return context
    if any(i < 0 or i >= len(values) for i in frames):
        raise ValueError("Gripper sample frame outside episode")
    if len(values) > 1:
        smooth = np.convolve(np.pad(values, (2, 2), mode="reflect"), np.ones(5) / 5, mode="valid")
    else:
        smooth = values.copy()
    finite = smooth[np.isfinite(smooth)]
    if not len(finite) or np.ptp(finite) < 1e-3:
        context["reason"] = "missing_or_insufficient_gripper_range"
        return context
    low, high = float(finite.min()), float(finite.max())
    context.update(available=True, episode_range=[low, high], closed_fraction=0.15, open_fraction=0.85)
    for frame in frames:
        value = smooth[frame]
        if np.isfinite(value):
            opening = (value - low) / (high - low)
            if open_direction == "decreasing":
                opening = 1 - opening
            state = "likely_closed" if opening <= 0.15 else "likely_open" if opening >= 0.85 else "intermediate"
        else:
            state = "unknown"
        # Compress repeated sampled states while retaining the source bounds.
        if context["states"] and context["states"][-1]["state"] == state:
            context["states"][-1].update(last_frame=frame, last_time_seconds=round(frame / fps, 3),
                                          last_clip_seconds=round((frame - origin) / fps, 3))
        else:
            context["states"].append(dict(state=state, first_frame=frame, last_frame=frame,
                first_time_seconds=round(frame / fps, 3), last_time_seconds=round(frame / fps, 3),
                first_clip_seconds=round((frame - origin) / fps, 3), last_clip_seconds=round((frame - origin) / fps, 3)))
    if kind == "video":
        for event in record.get("gripper_events", []):
            frame = event.get("frame")
            if type(frame) is not int or not node["start_frame"] <= frame < node["end_frame"]:
                continue
            event_type = event.get("type")
            if event_type not in {"grasp", "release"}:
                continue
            closing = event_type == "grasp"
            if open_direction == "decreasing":
                closing = not closing
            transition = dict(frame=frame, time_seconds=round(frame / fps, 3),
                              clip_seconds=round((frame - origin) / fps, 3),
                              state="likely_closing_transition" if closing else "likely_opening_transition")
            start = event.get("movement_start_frame")
            if type(start) is int:
                transition["movement_start_frame"] = max(node["start_frame"], start)
            context["transitions"].append(transition)
    return context
