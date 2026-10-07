"""RoboTAP motion-plan building (offline) and stage execution helpers (online).

Follows Vecerik et al., "RoboTAP" (ICRA 2024):
  - demos are split into stages at grasp / release (here: gripper clamp
    detection from tapnetRecord's events.json),
  - one shared set of query features is tracked through every demo, so point i
    is the same physical point in all demos,
  - active points per stage = points whose stage endpoints agree across demos
    (funneling), that move, and that are visible at the end; they vote for a
    motion cluster, and the winning cluster's points become the stage's queries,
  - at runtime the target is a frame slightly ahead of the nearest demo frame,
    switching to the across-demo average near the end of the stage,
  - 4-DoF servo: mean translation first (Gram-Schmidt), rotation from the
    centred residual, z from spread (variance) matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tapnet_utils import _kmeans_torch, resolve_torch_device, track_tail_endpoint

PRIMITIVES = ("none", "grasp", "release")
PRIMITIVE_CODE = {name: i for i, name in enumerate(PRIMITIVES)}


# ---------------------------------------------------------------------------
# Stage segmentation
# ---------------------------------------------------------------------------


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
) -> list[dict]:
    """Detect grasp/release from per-episode gripper min/max + gripper velocity.

    Open/closed bands are episode-local:
      closed_thresh = g_min + closed_frac * (g_max - g_min)
      open_thresh   = g_min + open_frac   * (g_max - g_min)

    An event fires only after the signal stays in the target band with
    ``|v| <= vel_stall`` for ``min_dwell_frames``, and only if ``|v|`` recently
    exceeded ``vel_min`` (filters idle chatter). Returns events.json-compatible
    dicts with type ``grasp`` (settled closed) or ``release`` (settled open).
    """
    g_raw = np.asarray(gripper, dtype=np.float64).reshape(-1)
    T = int(g_raw.shape[0])
    if T == 0:
        return []
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


def remap_plan_first_primitive(plan: MotionPlan, first: str) -> None:
    """Rewrite grasp/release primitives so the first gripper action is ``first``.

    ``none`` stages are left alone. Remaining gripper stages alternate
    grasp ↔ release starting with ``first``. Gripper targets are reassigned
    by action type so a remapped ``release`` keeps an open target (not the
    closed value that belonged to the old ``grasp`` stage).
    """
    if first not in ("grasp", "release"):
        raise ValueError(f"first must be grasp|release, got {first!r}")
    idxs = [i for i, st in enumerate(plan.stages) if st.primitive in ("grasp", "release")]
    pools: dict[str, list[float]] = {"grasp": [], "release": []}
    for i in idxs:
        pools[plan.stages[i].primitive].append(float(plan.stages[i].gripper_target))
    next_p = first
    for i in idxs:
        st = plan.stages[i]
        st.primitive = next_p
        pool = pools[next_p]
        if pool:
            st.gripper_target = pool.pop(0)
        else:
            # no stored target of this type — leave as-is; runtime may floor open
            pass
        next_p = "release" if next_p == "grasp" else "grasp"


def align_stage_counts(stage_lists: list[list[Stage]], names: list[str]) -> int:
    counts = [len(s) for s in stage_lists]
    n = min(counts)
    if len(set(counts)) > 1:
        print(
            f"WARNING: demos have different stage counts "
            f"{dict(zip(names, counts))}; using the first {n} stages "
            "(last kept stage runs to the end of each demo)."
        )
        for sl in stage_lists:
            if len(sl) > n:
                sl[n - 1] = Stage(sl[n - 1].start, sl[-1].end, "none")
                del sl[n:]
    return n


# ---------------------------------------------------------------------------
# Active point selection per stage
# ---------------------------------------------------------------------------


def _first_visible(tr: np.ndarray, vis: np.ndarray, n_head: int = 5) -> np.ndarray:
    """Median of the first n_head visible frames (per point) → [N,2]."""
    return track_tail_endpoint(tr[:, ::-1], vis[:, ::-1], n_tail=n_head)


def _resample(tr: np.ndarray, n: int) -> np.ndarray:
    T = tr.shape[1]
    idx = np.linspace(0, T - 1, n)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    a = (idx - i0).astype(np.float32)[None, :, None]
    return (1.0 - a) * tr[:, i0] + a * tr[:, i1]


def select_stage_active(
    stage_tracks: list[np.ndarray],
    stage_vis: list[np.ndarray],
    width: int,
    height: int,
    num_active: int = 128,
    n_clusters: int = 6,
    static_thresh: float = 0.03,
    funnel_quantile: float = 0.4,
    goal_tail_frames: int = 5,
    merge_frac: float = 0.5,
    min_vis_frac: float = 0.3,
    device=None,
    seed: int = 42,
    label: str = "",
) -> dict[str, Any]:
    """Pick active points for one stage.

    stage_tracks[d]: [N, T_d, 2] for the shared query set in demo d (stage slice).
    Returns point_ids plus diagnostics.
    """
    D = len(stage_tracks)
    N = stage_tracks[0].shape[0]
    diag = float(np.hypot(width, height))
    scale = np.array([width, height], dtype=np.float32)

    ends = np.stack(
        [track_tail_endpoint(t, v, n_tail=goal_tail_frames) for t, v in zip(stage_tracks, stage_vis)],
        axis=0,
    )  # [D,N,2]
    starts = np.stack(
        [_first_visible(t, v) for t, v in zip(stage_tracks, stage_vis)], axis=0
    )
    end_vis = np.stack(
        [v[:, -goal_tail_frames:].any(axis=1) for v in stage_vis], axis=0
    )  # [D,N]
    vis_frac = np.stack([v.mean(axis=1) for v in stage_vis], axis=0)  # [D,N]

    motion = np.linalg.norm(ends - starts, axis=2) / diag  # [D,N]
    med_motion = np.median(motion, axis=0)
    moving = med_motion >= static_thresh

    need_vis = max(1, int(np.ceil(D / 2)))
    visible_end = end_vis.sum(axis=0) >= need_vis

    # Funneling: active points end in the same place (relative to the camera)
    # across demos even if they start in different places.
    if D >= 2:
        ends_n = ends / scale
        spread = np.zeros(N, dtype=np.float64)
        for i in range(N):
            m = end_vis[:, i]
            pts = ends_n[m, i] if m.sum() >= 2 else ends_n[:, i]
            spread[i] = float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
        ok = moving & visible_end
        thresh = (
            float(np.quantile(spread[ok], funnel_quantile)) if ok.any() else np.inf
        )
        funnel = spread <= thresh
    else:
        spread = np.zeros(N, dtype=np.float64)
        funnel = np.ones(N, dtype=bool)

    candidates = moving & visible_end & funnel

    # Motion clustering on trajectories concatenated across demos.
    feats = np.concatenate(
        [(_resample(t, 16) / scale).reshape(N, -1) for t in stage_tracks]
        + [(ends / scale).transpose(1, 0, 2).reshape(N, -1)],
        axis=1,
    ).astype(np.float32)
    feats = (feats - feats.mean(axis=0)) / np.maximum(feats.std(axis=0), 1e-6)
    k = int(min(n_clusters, N))
    labels = _kmeans_torch(feats, k, seed=seed, device=resolve_torch_device(device))

    votes = np.array([int(np.sum(candidates & (labels == c))) for c in range(k)])
    cl_motion = np.array(
        [float(med_motion[labels == c].mean()) if np.any(labels == c) else 0.0 for c in range(k)]
    )
    # Clusters that barely move (gripper, held object) cannot win.
    votes_eff = np.where(cl_motion >= static_thresh, votes, 0)
    if votes_eff.max() <= 0:
        best = int(np.argmax(cl_motion))
        winners = [best]
        print(f"{label} WARNING: no active-point votes; using most-moving cluster {best}")
    else:
        best = int(np.argmax(votes_eff))
        winners = [
            c for c in range(k)
            if votes_eff[c] > 0 and votes_eff[c] >= merge_frac * votes_eff[best]
        ]

    members = np.flatnonzero(np.isin(labels, winners))
    good_vis = vis_frac.mean(axis=0) >= min_vis_frac
    members_vis = members[good_vis[members]]
    if len(members_vis) >= 4:
        members = members_vis

    rng = np.random.default_rng(seed)
    if len(members) > num_active:
        # prefer funneling candidates, then fill randomly
        cand_m = members[candidates[members]]
        rest = members[~candidates[members]]
        if len(cand_m) >= num_active:
            point_ids = rng.choice(cand_m, size=num_active, replace=False)
        else:
            fill = rng.choice(rest, size=num_active - len(cand_m), replace=False)
            point_ids = np.concatenate([cand_m, fill])
    else:
        point_ids = members
    point_ids = np.sort(point_ids.astype(np.int64))

    print(
        f"{label} candidates={int(candidates.sum())}/{N} "
        f"(moving={int(moving.sum())} visible_end={int(visible_end.sum())} "
        f"funnel={int(funnel.sum())})"
    )
    print(
        f"{label} cluster votes={votes.tolist()} motion="
        f"{[round(m, 3) for m in cl_motion.tolist()]} winners={winners}"
    )
    med_spread = float(np.median(spread[point_ids])) if len(point_ids) else 0.0
    print(
        f"{label} active={len(point_ids)}  endpoint spread median={med_spread:.4f} "
        "(fraction of image size)"
    )
    return {
        "point_ids": point_ids,
        "labels": labels,
        "votes": votes,
        "winners": winners,
        "ends": ends,
        "end_vis": end_vis,
    }


# ---------------------------------------------------------------------------
# Task (motion plan) packing
# ---------------------------------------------------------------------------


@dataclass
class StagePlan:
    point_ids: np.ndarray  # indices into the task's (union) query set
    demo_tracks: list[np.ndarray]  # per demo [M, T_d, 2]
    demo_vis: list[np.ndarray]  # per demo [M, T_d]
    goal_mean: np.ndarray  # [M, 2]
    goal_valid: np.ndarray  # [M]
    primitive: str
    gripper_target: float
    # Mean demo end-of-stage joint pose (deg, LeRobot order: 5 arm + gripper).
    # Used by tapnetGrabPose to FK a goal gripper quaternion in the SO-101 base frame.
    goal_joints: np.ndarray | None = None


@dataclass
class MotionPlan:
    stages: list[StagePlan]
    demo_names: list[str]
    extras: dict[str, Any] = field(default_factory=dict)


def pack_plan(plan: MotionPlan) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "task_version": np.array([2], dtype=np.int32),
        "n_stages": np.array([len(plan.stages)], dtype=np.int32),
        "n_demos": np.array([len(plan.demo_names)], dtype=np.int32),
        "demo_names": np.asarray(plan.demo_names),
    }
    for s, st in enumerate(plan.stages):
        out[f"stage_{s}_point_ids"] = st.point_ids.astype(np.int32)
        out[f"stage_{s}_goal_mean"] = st.goal_mean.astype(np.float32)
        out[f"stage_{s}_goal_valid"] = st.goal_valid.astype(bool)
        out[f"stage_{s}_primitive"] = np.array([PRIMITIVE_CODE[st.primitive]], dtype=np.int32)
        out[f"stage_{s}_gripper_target"] = np.array([st.gripper_target], dtype=np.float32)
        if st.goal_joints is not None:
            out[f"stage_{s}_goal_joints"] = np.asarray(st.goal_joints, dtype=np.float32).reshape(6)
        for d, (tr, vis) in enumerate(zip(st.demo_tracks, st.demo_vis)):
            out[f"stage_{s}_demo{d}_tracks"] = tr.astype(np.float32)
            out[f"stage_{s}_demo{d}_vis"] = vis.astype(bool)
    return out


def unpack_plan(task: dict[str, Any]) -> MotionPlan:
    if int(np.asarray(task.get("task_version", [1])).reshape(-1)[0]) < 2:
        raise ValueError(
            "task.npz is the old single-goal format; rebuild it with tapnetCreate."
        )
    n_stages = int(np.asarray(task["n_stages"]).reshape(-1)[0])
    n_demos = int(np.asarray(task["n_demos"]).reshape(-1)[0])
    stages = []
    for s in range(n_stages):
        gj = None
        key = f"stage_{s}_goal_joints"
        if key in task:
            gj = np.asarray(task[key], dtype=np.float32).reshape(6)
        stages.append(
            StagePlan(
                point_ids=np.asarray(task[f"stage_{s}_point_ids"], dtype=np.int64),
                demo_tracks=[np.asarray(task[f"stage_{s}_demo{d}_tracks"]) for d in range(n_demos)],
                demo_vis=[np.asarray(task[f"stage_{s}_demo{d}_vis"], dtype=bool) for d in range(n_demos)],
                goal_mean=np.asarray(task[f"stage_{s}_goal_mean"], dtype=np.float32),
                goal_valid=np.asarray(task[f"stage_{s}_goal_valid"], dtype=bool),
                primitive=PRIMITIVES[int(np.asarray(task[f"stage_{s}_primitive"]).reshape(-1)[0])],
                gripper_target=float(np.asarray(task[f"stage_{s}_gripper_target"]).reshape(-1)[0]),
                goal_joints=gj,
            )
        )
    return MotionPlan(stages=stages, demo_names=[str(x) for x in np.asarray(task["demo_names"])])


def joints_to_obs(joints6: np.ndarray) -> dict[str, float]:
    """LeRobot-style obs dict from a 6-vector (arm deg + gripper 0..100)."""
    j = np.asarray(joints6, dtype=float).reshape(6)
    keys = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    return {k: float(j[i]) for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Runtime: target selection + 4-DoF servo
# ---------------------------------------------------------------------------


@dataclass
class TargetChoice:
    goals: np.ndarray  # [M,2]
    valid: np.ndarray  # [M] usable (visible live and at target)
    demo: int  # -1 when using the across-demo mean
    frame: int
    progress: float  # 0..1 within the stage
    using_mean: bool


def choose_target(
    stage: StagePlan,
    points: np.ndarray,
    visible: np.ndarray,
    lookahead: int = 5,
    end_frac: float = 0.85,
    min_points: int = 3,
    mean_switch_px: float | None = None,
) -> TargetChoice | None:
    """Nearest demo frame → target a few frames ahead; mean goal near the end."""
    best = None
    for d, (tr, vis) in enumerate(zip(stage.demo_tracks, stage.demo_vis)):
        both = vis & visible[:, None]  # [M,T]
        cnt = both.sum(axis=0)
        dist = np.linalg.norm(tr - points[:, None, :], axis=2)
        cost = np.where(both, dist, 0.0).sum(axis=0) / np.maximum(cnt, 1)
        cost = np.where(cnt >= min_points, cost, np.inf)
        t = int(np.argmin(cost))
        if np.isfinite(cost[t]) and (best is None or cost[t] < best[0]):
            best = (float(cost[t]), d, t)
    if best is None:
        return None
    cost, d, t = best
    T = stage.demo_tracks[d].shape[1]
    progress = t / max(T - 1, 1)
    use_mean = progress >= end_frac or (
        mean_switch_px is not None and cost <= mean_switch_px and progress >= 0.5
    )
    if use_mean:
        valid = visible & stage.goal_valid
        if valid.sum() >= min_points:
            return TargetChoice(stage.goal_mean, valid, -1, t, progress, True)
    t_goal = min(t + int(lookahead), T - 1)
    goals = stage.demo_tracks[d][:, t_goal]
    valid = visible & stage.demo_vis[d][:, t_goal]
    return TargetChoice(goals, valid, d, t_goal, progress, False)


def solve_robotap_servo(
    points_uv: np.ndarray,
    goals_uv: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Camera-frame velocity [vx, vy, vz, wz] (OpenCV axes) that moves the
    image points toward their goals.

    Normalized coords: u=(x - W/2)/(H/2), v=(y - H/2)/(H/2) (vertical in [-1,1]).
    Translation explains the mean error first (Gram-Schmidt); rotation and
    scale are fitted on the centred residual; z uses spread matching so noise
    does not bias the camera backwards.
    """
    p = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(goals_uv, dtype=np.float64).reshape(-1, 2)
    half_h = max(height, 1) / 2.0
    c = np.array([width / 2.0, height / 2.0])
    pn = (p - c) / half_h
    gn = (g - c) / half_h

    mean_err = (gn - pn).mean(axis=0)
    pc = pn - pn.mean(axis=0)
    gc = gn - gn.mean(axis=0)

    if len(p) >= 2:
        cross = float(np.sum(pc[:, 0] * gc[:, 1] - pc[:, 1] * gc[:, 0]))
        dot = float(np.sum(pc * gc))
        theta = float(np.arctan2(cross, dot)) if (abs(cross) + abs(dot)) > 1e-12 else 0.0
        sp = float(np.sqrt((pc ** 2).sum(axis=1).mean()))
        sg = float(np.sqrt((gc ** 2).sum(axis=1).mean()))
        log_scale = float(np.log(sg / sp)) if sp > 1e-6 and sg > 1e-6 else 0.0
    else:
        theta = 0.0
        log_scale = 0.0

    # Points must move by +mean_err → camera moves the opposite way.
    # Goal spread larger than current → camera moves forward (+Z).
    # Points must rotate by +theta (u right, v down) → camera rolls by -theta.
    return np.array([-mean_err[0], -mean_err[1], log_scale, -theta], dtype=np.float64)


def select_servo_inliers(
    points_uv: np.ndarray,
    goals_uv: np.ndarray,
    valid: np.ndarray,
    keep_frac: float = 0.45,
    min_points: int = 3,
    min_cos: float = 0.25,
    goal_radius_frac: float = 0.12,
) -> np.ndarray:
    """Mask of points to use for IBVS (drops conflicting / background pulls).

    Outliers often *outnumber* object tracks, so a global median of error
    vectors fails. Instead:

    1. Prefer points whose **goals** sit in the densest image region
       (object goals cluster; background goals scatter).
    2. Within that set, keep errors that agree with the *local* median pull
       (cosine ≥ ``min_cos``).
    3. Trim to the lowest-error ``keep_frac`` of that consistent set.
    """
    points_uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    goals_uv = np.asarray(goals_uv, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    out = np.zeros(len(valid), dtype=bool)
    idx = np.flatnonzero(valid)
    if len(idx) < min_points:
        out[idx] = True
        return out

    g = goals_uv[idx]
    # Goal density: mean distance to 4 nearest other goals (lower = denser).
    if len(idx) >= 5:
        dmat = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        knn = np.sort(dmat, axis=1)[:, : min(4, len(idx) - 1)]
        dens = knn.mean(axis=1)
        # keep the densest half (or enough for min_points)
        dens_thr = float(np.percentile(dens, 50))
        tight = dens <= dens_thr
        if int(tight.sum()) < min_points:
            order = np.argsort(dens)
            tight = np.zeros(len(idx), dtype=bool)
            tight[order[: max(min_points, len(idx) // 2)]] = True
        # also require goals near the densest mode center
        seed = idx[int(np.argmin(dens))]
        span = float(np.hypot(*(g.max(axis=0) - g.min(axis=0))))
        rad = max(goal_radius_frac * max(span, 1.0), 20.0)
        near_mode = np.linalg.norm(g - goals_uv[seed], axis=1) <= rad
        cluster = tight & near_mode
        if int(cluster.sum()) < min_points:
            cluster = tight
    else:
        cluster = np.ones(len(idx), dtype=bool)

    cidx = idx[cluster]
    err = goals_uv[cidx] - points_uv[cidx]
    mag = np.linalg.norm(err, axis=1)
    med = np.median(err, axis=0)
    med_n = float(np.linalg.norm(med))
    if med_n < 1e-6:
        order = np.argsort(mag)
        n_keep = max(min_points, int(np.ceil(keep_frac * len(cidx))))
        out[cidx[order[: min(n_keep, len(cidx))]]] = True
        return out

    cos = (err @ med) / np.maximum(mag * med_n, 1e-9)
    agree = cos >= float(min_cos)
    if int(agree.sum()) < min_points:
        agree = np.ones(len(cidx), dtype=bool)
    cand = cidx[agree]
    mag_c = mag[agree]
    n_keep = max(min_points, int(np.ceil(keep_frac * len(cand))))
    n_keep = min(n_keep, len(cand))
    order = np.argsort(mag_c)
    out[cand[order[:n_keep]]] = True
    # safety: never return fewer than min_points if we have them
    if int(out.sum()) < min_points:
        order = np.argsort(np.linalg.norm(goals_uv[idx] - points_uv[idx], axis=1))
        out[:] = False
        out[idx[order[:min_points]]] = True
    return out


def solve_jacobian_servo(
    points_uv: np.ndarray,
    goals_uv: np.ndarray,
    width: int,
    height: int,
    z: float = 0.20,
    robust_iters: int = 2,
) -> np.ndarray:
    """Eye-in-hand image Jacobian LS → camera velocity [vx, vy, vz, wz].

    Normalized coords: u=(x - W/2)/(H/2), v=(y - H/2)/(H/2). Per point L for
    (vx, vy, vz, wz) with constant depth Z (OpenCV axes)::

        [ -1/Z   0    u/Z   v ]
        [  0   -1/Z   v/Z  -u ]

    Solves L ξ = (g_n - p_n). With ``robust_iters`` > 1, re-solves after
    dropping the worst residual half of points (avoids local valleys from
    irrelevant tracks).
    """
    p = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(goals_uv, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(4, dtype=np.float64)
    half_h = max(height, 1) / 2.0
    c = np.array([width / 2.0, height / 2.0])
    pn = (p - c) / half_h
    gn = (g - c) / half_h
    z = float(max(z, 1e-3))
    keep = np.ones(len(p), dtype=bool)
    xi = np.zeros(4, dtype=np.float64)
    for _ in range(max(1, int(robust_iters))):
        rows = []
        errs = []
        for (u, v), (ug, vg) in zip(pn[keep], gn[keep]):
            rows.append([-1.0 / z, 0.0, u / z, v])
            rows.append([0.0, -1.0 / z, v / z, -u])
            errs.append(ug - u)
            errs.append(vg - v)
        L = np.asarray(rows, dtype=np.float64)
        e = np.asarray(errs, dtype=np.float64)
        xi, *_ = np.linalg.lstsq(L, e, rcond=None)
        xi = np.asarray(xi, dtype=np.float64).reshape(4)
        # per-point residual in normalized coords
        res = np.zeros(len(p), dtype=np.float64)
        for i, ((u, v), (ug, vg)) in enumerate(zip(pn, gn)):
            Li = np.array(
                [[-1.0 / z, 0.0, u / z, v], [0.0, -1.0 / z, v / z, -u]],
                dtype=np.float64,
            )
            ei = np.array([ug - u, vg - v], dtype=np.float64)
            res[i] = float(np.linalg.norm(Li @ xi - ei))
        active = np.flatnonzero(keep)
        if len(active) <= 4:
            break
        thr = float(np.median(res[active])) * 2.0
        if thr < 1e-9:
            break
        new_keep = keep & (res <= thr)
        if int(new_keep.sum()) < 3:
            # keep the best half by residual
            order = np.argsort(res[active])
            new_keep = np.zeros(len(p), dtype=bool)
            new_keep[active[order[: max(3, len(active) // 2)]]] = True
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return xi


def mean_pixel_error(points: np.ndarray, goals: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() == 0:
        return float("inf")
    return float(np.linalg.norm(points[valid] - goals[valid], axis=1).mean())


def mean_normalized_uv(
    points_uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Mean of valid points in normalized image coords (u,v). None if empty."""
    m = np.asarray(valid, dtype=bool).reshape(-1)
    if not np.any(m):
        return None
    p = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)[m]
    half_h = max(height, 1) / 2.0
    c = np.array([width / 2.0, height / 2.0])
    return ((p - c) / half_h).mean(axis=0)


def normalized_error(
    points_uv: np.ndarray,
    goals_uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Mean (goal - point) in normalized coords over valid mask."""
    m = np.asarray(valid, dtype=bool).reshape(-1)
    if not np.any(m):
        return None
    p = mean_normalized_uv(points_uv, m, width, height)
    g = mean_normalized_uv(goals_uv, m, width, height)
    if p is None or g is None:
        return None
    return g - p


class EmpiricalImageJacobian:
    """Online finite-difference J: Δ(u,v)_norm ≈ J @ Δ(x,y,z,rz)_cam.

    Columns are camera-frame OpenCV axes (before mount-roll). Used by the
    greedy discrete ±axis controller in tapnetGrabGreedy.
    """

    AXIS_NAMES = ("dx", "dy", "dz", "drz")

    def __init__(self) -> None:
        self.J = np.zeros((2, 4), dtype=np.float64)
        self.filled = np.zeros(4, dtype=bool)

    def reset(self) -> None:
        self.J[:] = 0.0
        self.filled[:] = False

    @property
    def ready(self) -> bool:
        return bool(self.filled.all())

    def update_column(self, axis: int, dp: np.ndarray, da: float) -> None:
        """Set column ``axis`` from observed feature change ``dp`` and step ``da``."""
        axis = int(axis)
        da = float(da)
        if axis < 0 or axis > 3 or abs(da) < 1e-12:
            return
        self.J[:, axis] = np.asarray(dp, dtype=np.float64).reshape(2) / da
        self.filled[axis] = True

    def predict_error(self, e: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Predicted error after taking camera action ``a`` (4,)."""
        e = np.asarray(e, dtype=np.float64).reshape(2)
        a = np.asarray(a, dtype=np.float64).reshape(4)
        return e + self.J @ a

    def best_axis_action(
        self,
        e: np.ndarray,
        step: float,
        rstep: float,
        min_improve: float = 1e-4,
    ) -> tuple[np.ndarray | None, float, float]:
        """Pick among 8 ±axis moves the one that most reduces ||e||.

        Returns (action or None, current_E, predicted_E).
        """
        e = np.asarray(e, dtype=np.float64).reshape(2)
        E0 = float(np.linalg.norm(e))
        if not self.ready:
            return None, E0, E0
        best_a = None
        best_E = E0
        for i, mag in enumerate((step, step, step, rstep)):
            for sign in (+1.0, -1.0):
                a = np.zeros(4, dtype=np.float64)
                a[i] = sign * mag
                Ep = float(np.linalg.norm(self.predict_error(e, a)))
                if Ep < best_E - float(min_improve):
                    best_E = Ep
                    best_a = a
        return best_a, E0, best_E
