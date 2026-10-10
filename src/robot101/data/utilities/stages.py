"""Gripper events and inclusive episode subtask boundaries."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Stage:
    start: int
    end: int
    primitive: str  # what to do after servoing this stage


def load_demo_events(demo_dir: str | Path) -> list[dict] | None:
    path = Path(demo_dir) / "events.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", data) if isinstance(data, dict) else data
    return sorted(events, key=lambda e: int(e["frame"]))


def stages_from_events(
    events: list[dict] | None,
    num_frames: int,
    min_stage_frames: int = 5,
    first_primitive: str = "grasp",
) -> list[Stage]:
    """Boundaries at event frames.

    Typed clamp/release events keep their type. Manual events (space/e)
    alternate starting with ``first_primitive`` (``grasp`` or ``release``).
    """
    if first_primitive not in ("grasp", "release"):
        raise ValueError(f"first_primitive must be grasp|release, got {first_primitive!r}")
    last = int(num_frames) - 1
    if not events:
        return [Stage(0, last, "none")]
    stages: list[Stage] = []
    start = 0
    next_manual = first_primitive
    for e in events:
        f = int(e["frame"])
        if f - start < min_stage_frames or f >= last:
            continue
        kind = e.get("type", "manual")
        if kind == "manual":
            kind = next_manual
            next_manual = "release" if next_manual == "grasp" else "grasp"
        stages.append(Stage(start, f, kind))
        start = f
    stages.append(Stage(start, last, "none"))
    return stages


def events_from_gripper_thresholds(
    gripper: np.ndarray,
    fps: float = 30.0,
    *,
    closed_frac: float = 0.15,
    open_frac: float = 0.85,
    min_dwell_frames: int = 3,
    vel_stall: float = 0.02,
    vel_min: float = 0.05,
    smooth_window: int = 5,
    lookback_frames: int = 15,
    range_eps: float = 1e-3,
    min_change_frac: float | None = None,
    min_change_abs: float = 0.0,
) -> list[dict]:
    """Detect grasp/release from per-episode gripper min/max + gripper velocity.

    Open/closed bands are episode-local:
      closed_thresh = g_min + closed_frac * (g_max - g_min)
      open_thresh   = g_min + open_frac   * (g_max - g_min)

    An event fires only after the signal stays in the target band with
    ``|v| <= vel_stall`` for ``min_dwell_frames``, and only if ``|v|`` recently
    exceeded ``vel_min`` (filters idle chatter). Returns events.json-compatible
    dicts with type ``grasp`` (settled closed) or ``release`` (settled open).
    If ``min_change_frac`` is supplied, use settled directional movements
    instead of absolute bands. The required displacement is the larger of
    that fraction of the smoothed episode range and ``min_change_abs``.
    This detects partial grasps and releases away from the range extremes.
    """
    g_raw = np.asarray(gripper, dtype=np.float64).reshape(-1)
    T = int(g_raw.shape[0])
    if T == 0:
        return []
    if min_change_frac is not None and not (0 < min_change_frac <= 1):
        raise ValueError("min_change_frac must be in (0, 1]")
    if not np.isfinite(min_change_abs) or min_change_abs < 0:
        raise ValueError("min_change_abs must be finite and nonnegative")
    if not (0.0 <= closed_frac < open_frac <= 1.0):
        raise ValueError(
            f"need 0 <= closed_frac < open_frac <= 1, got "
            f"closed_frac={closed_frac}, open_frac={open_frac}"
        )

    w = max(1, int(smooth_window))
    if w == 1:
        g = g_raw.copy()
    else:
        # Reflect pad so episode edges don't invent a fake min/max span.
        pad = w // 2
        padded = np.pad(g_raw, (pad, pad), mode="reflect")
        kernel = np.ones(w, dtype=np.float64) / w
        g = np.convolve(padded, kernel, mode="valid")
        if len(g) > T:
            g = g[:T]
        elif len(g) < T:
            # odd/even window mismatch — fall back to centered same-length slice
            g = np.convolve(g_raw, kernel, mode="same")

    g_min = float(np.min(g))
    g_max = float(np.max(g))
    span = g_max - g_min
    if span < float(range_eps):
        return []

    closed_thresh = g_min + float(closed_frac) * span
    open_thresh = g_min + float(open_frac) * span

    fps = float(max(fps, 1e-6))
    v = np.zeros(T, dtype=np.float64)
    if T > 1:
        v[1:] = (g[1:] - g[:-1]) * fps

    lookback = max(1, int(lookback_frames))
    dwell_need = max(1, int(min_dwell_frames))
    events: list[dict] = []

    last_emitted: str | None = None
    dwell_band: str | None = None
    dwell_count = 0

    def recent_motion(t: int) -> bool:
        t0 = max(0, t - lookback + 1)
        return bool(np.max(np.abs(v[t0 : t + 1])) >= float(vel_min))

    if min_change_frac is not None:
        required = max(float(min_change_frac) * span, float(min_change_abs))
        direction, origin, origin_frame = 0, float(g[0]), 0
        for t in range(1, T):
            # Reversals begin a new movement at the previous local extremum.
            # Ignore stationary jitter when deciding movement direction.
            if abs(v[t]) >= float(vel_min) and abs(v[t]) > float(vel_stall):
                sign = 1 if v[t] > 0 else -1
                if sign != direction:
                    direction, origin, origin_frame = sign, float(g[t - 1]), t - 1
                    dwell_count = 0
            change = direction * (float(g[t]) - origin)
            if not direction or change < required or abs(v[t]) > float(vel_stall):
                dwell_count = 0
                continue
            dwell_count += 1
            kind = "release" if direction > 0 else "grasp"
            if dwell_count < dwell_need or kind == last_emitted or not recent_motion(t):
                continue
            events.append({
                "frame": int(t), "type": kind, "gripper": float(g_raw[t]),
                "g_smooth": float(g[t]), "closed_thresh": closed_thresh,
                "open_thresh": open_thresh, "g_min": g_min, "g_max": g_max,
                "detector": "position_change", "movement_start_frame": origin_frame,
                "position_change": float(change), "required_position_change": required,
            })
            last_emitted = kind
            direction, origin, origin_frame = 0, float(g[t]), t
            dwell_count = 0
        return events

    for t in range(T):
        if g[t] <= closed_thresh:
            band = "closed"
        elif g[t] >= open_thresh:
            band = "open"
        else:
            dwell_band = None
            dwell_count = 0
            continue

        stalled = abs(v[t]) <= float(vel_stall)
        if band == dwell_band and stalled:
            dwell_count += 1
        elif stalled:
            dwell_band = band
            dwell_count = 1
        else:
            dwell_band = None
            dwell_count = 0
            continue

        if dwell_count < dwell_need:
            continue
        if band == last_emitted:
            continue
        if not recent_motion(t):
            continue

        kind = "grasp" if band == "closed" else "release"
        events.append(
            {
                "frame": int(t),
                "type": kind,
                "gripper": float(g_raw[t]),
                "g_smooth": float(g[t]),
                "closed_thresh": closed_thresh,
                "open_thresh": open_thresh,
                "g_min": g_min,
                "g_max": g_max,
            }
        )
        last_emitted = band
        dwell_band = None
        dwell_count = 0

    return events


