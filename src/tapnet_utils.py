"""Shared TAPIR tracking, wrist-cam undistort, and visual-servoing helpers.

No AprilTag / LilyTags. Eye-in-hand (wrist) camera assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tree
from tapnet.torch import tapir_model
from tapnet.torch.tapir_model import QueryFeatures

# Import transforms module directly to avoid utility/__init__.py side effects
# (lerobot camera imports) when only offline create is needed.
import importlib.util

_TRANSFORMS_PATH = Path(__file__).resolve().parent / "utility" / "transforms.py"
_spec = importlib.util.spec_from_file_location("_tapnet_transforms", _TRANSFORMS_PATH)
_transforms = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_transforms)
T_CV_TO_MJ = _transforms.T_CV_TO_MJ
R_to_quat_wxyz = _transforms.R_to_quat_wxyz
quat_wxyz_to_R = _transforms.quat_wxyz_to_R

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "tapnet" / "checkpoints" / "causal_bootstapir_checkpoint.pt"
)


# ---------------------------------------------------------------------------
# Calibration / undistort (P=K, no getOptimalNewCameraMatrix)
# ---------------------------------------------------------------------------


@dataclass
class WristCalib:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def maps(self) -> tuple[np.ndarray, np.ndarray] | None:
        if float(np.linalg.norm(self.dist)) <= 1e-12:
            return None
        return cv2.initUndistortRectifyMap(
            self.K,
            self.dist,
            None,
            self.K,
            (self.width, self.height),
            cv2.CV_16SC2,
        )


def load_wrist_calib(path: str | Path) -> WristCalib:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        width = int(data.get("width", data.get("image_width", 0)))
        height = int(data.get("height", data.get("image_height", 0)))
        if width <= 0 or height <= 0:
            raise ValueError(f"calib JSON missing width/height: {path}")
        return WristCalib(K=K, dist=dist, width=width, height=height)

    # Directory with camera_matrix.npy + dist_coeffs.npy (+ optional size in json)
    K = np.load(path / "camera_matrix.npy").astype(np.float64).reshape(3, 3)
    dist = np.load(path / "dist_coeffs.npy").astype(np.float64).reshape(-1, 1)
    meta = path / "calibration.json"
    if meta.is_file():
        with meta.open("r", encoding="utf-8") as f:
            data = json.load(f)
        width = int(data.get("width", data.get("image_width", 640)))
        height = int(data.get("height", data.get("image_height", 480)))
    else:
        width, height = 640, 480
    return WristCalib(K=K, dist=dist, width=width, height=height)


def undistort_bgr(
    frame_bgr: np.ndarray, maps: tuple[np.ndarray, np.ndarray] | None
) -> np.ndarray:
    if maps is None:
        return frame_bgr
    return cv2.remap(frame_bgr, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Demo I/O + gripper segmentation
# ---------------------------------------------------------------------------


def load_demo_folder(demo_dir: str | Path) -> dict[str, Any]:
    demo_dir = Path(demo_dir)
    video_path = demo_dir / "video.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video: {video_path}")

    states = np.load(demo_dir / "robot_states.npy")
    gripper = _gripper_from_states(states)

    ts_path = demo_dir / "timestamps.npy"
    if ts_path.is_file():
        timestamps = np.load(ts_path)
    else:
        timestamps = np.arange(len(gripper), dtype=np.float64)

    frames_bgr = _read_video_bgr(video_path)
    n = min(len(frames_bgr), len(gripper), len(timestamps))
    return {
        "name": demo_dir.name,
        "frames_bgr": frames_bgr[:n],
        "gripper": np.asarray(gripper[:n], dtype=np.float64),
        "timestamps": np.asarray(timestamps[:n], dtype=np.float64),
        "robot_states": np.asarray(states[:n]),
    }


def _gripper_from_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    if states.ndim == 1:
        return states.astype(np.float64)
    if states.ndim == 2 and states.shape[1] >= 1:
        # JOINT_POS_KEYS order: gripper is last of 6
        return states[:, -1].astype(np.float64)
    raise ValueError(f"unsupported robot_states shape: {states.shape}")


def _read_video_bgr(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    return frames


def segment_full_episode(num_frames: int) -> tuple[int, int]:
    """Whole recording is the segment (start at first frame, end at last).

    tapnetRecord ends the episode on keypress (q); create uses that full clip.
    """
    n = int(num_frames)
    if n < 2:
        raise ValueError(f"need at least 2 frames, got {n}")
    return 0, n - 1


def load_segment_bounds(demo_dir: str | Path, num_frames: int) -> tuple[int, int]:
    """Load segment.npy if present, else use the full episode."""
    demo_dir = Path(demo_dir)
    path = demo_dir / "segment.npy"
    if path.is_file():
        seg = np.load(path)
        start, end = int(seg[0]), int(seg[1])
        end = min(end, num_frames - 1)
        if end <= start:
            raise ValueError(f"invalid saved segment [{start}, {end}] in {path}")
        return start, end
    return segment_full_episode(num_frames)


def segment_gripper_close_open(
    gripper: np.ndarray,
    close_threshold: float = 30.0,
) -> tuple[int, int]:
    """Return (segment_start, segment_end) = first close edge → next open edge.

    Gripper convention: 0 closed … 100 open. Optional legacy mode.
    """
    g = np.asarray(gripper, dtype=np.float64)
    closed = g < close_threshold
    start = None
    for t in range(1, len(closed)):
        if closed[t] and not closed[t - 1]:
            start = t
            break
    if start is None:
        raise ValueError("no gripper-close transition found")
    end = None
    for t in range(start + 1, len(closed)):
        if (not closed[t]) and closed[t - 1]:
            end = t
            break
    if end is None:
        end = len(g) - 1
    if end <= start:
        raise ValueError(f"invalid segment [{start}, {end}]")
    return int(start), int(end)


# ---------------------------------------------------------------------------
# TAPIR (causal BootsTAPIR, PyTorch)
# ---------------------------------------------------------------------------


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BootsTAPIR:
    """Thin wrapper around causal BootsTAPIR for offline + online tracking."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ):
        self.device = device or _default_device()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TAPIR checkpoint not found: {checkpoint}")
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    @staticmethod
    def preprocess(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        return frames / 255.0 * 2.0 - 1.0

    def init_features(
        self, frame_rgb: np.ndarray, query_points_tyx: np.ndarray
    ) -> QueryFeatures:
        """frame_rgb: HxWx3 uint8. query_points_tyx: [N,3] (t,y,x) pixel."""
        frame = torch.tensor(frame_rgb, device=self.device)
        pts = torch.tensor(query_points_tyx, dtype=torch.float32, device=self.device)
        frames = self.preprocess(frame[None, None])
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        return self.model.get_query_features(
            frames,
            is_training=False,
            query_points=pts[None, :],
            feature_grids=feature_grids,
        )

    def initial_causal_state(self, num_points: int, features: QueryFeatures):
        state = self.model.construct_initial_causal_state(
            num_points, len(features.resolutions) - 1
        )
        return tree.map_structure(lambda x: x.to(self.device), state)

    def predict(
        self,
        frame_rgb: np.ndarray,
        features: QueryFeatures,
        causal_state,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Returns tracks [N,2], visibles [N], updated causal_state."""
        frame = torch.tensor(frame_rgb, device=self.device)
        frames = self.preprocess(frame[None, None])
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_state,
            get_causal_context=True,
        )
        causal_state = trajectories["causal_context"]
        del trajectories["causal_context"]
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(uncertainty)) > 0.5
        # tracks: [1, N, 1, 2] → [N, 2]
        track_np = tracks[0, :, 0, :].detach().cpu().numpy().astype(np.float32)
        vis_np = visibles[0, :, 0].detach().cpu().numpy().astype(bool)
        return track_np, vis_np, causal_state

    def track_video(
        self,
        frames_rgb: list[np.ndarray] | np.ndarray,
        query_xy: np.ndarray,
        query_frame_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, QueryFeatures, np.ndarray]:
        """Track points through a clip.

        query_xy: [N,2] in (u,v)=(x,y) pixels on query_frame_index.
        Returns tracks [N,T,2], visibility [N,T], features, query_points_tyx [N,3].
        """
        frames_rgb = list(frames_rgb)
        t0 = int(query_frame_index)
        h, w = frames_rgb[t0].shape[:2]
        query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
        # TAPIR query format: (t, y, x)
        query_tyx = np.stack(
            [
                np.zeros(len(query_xy), dtype=np.float32),
                query_xy[:, 1],
                query_xy[:, 0],
            ],
            axis=1,
        )
        features = self.init_features(frames_rgb[t0], query_tyx)
        causal = self.initial_causal_state(len(query_xy), features)

        T = len(frames_rgb)
        N = len(query_xy)
        tracks = np.zeros((N, T, 2), dtype=np.float32)
        visibility = np.zeros((N, T), dtype=bool)

        for t, frame in enumerate(frames_rgb):
            pts, vis, causal = self.predict(frame, features, causal)
            tracks[:, t] = pts
            visibility[:, t] = vis

        # Ensure query frame matches query (TAPIR may drift on t0)
        tracks[:, t0, 0] = np.clip(query_xy[:, 0], 0, w - 1)
        tracks[:, t0, 1] = np.clip(query_xy[:, 1], 0, h - 1)
        visibility[:, t0] = True
        return tracks, visibility, features, query_tyx


def sample_query_points(
    width: int,
    height: int,
    n: int,
    margin: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform samples in (u,v) pixels inside a margin box. Shape [N,2]."""
    rng = rng or np.random.default_rng(42)
    u0, u1 = margin * width, (1.0 - margin) * width
    v0, v1 = margin * height, (1.0 - margin) * height
    u = rng.uniform(u0, u1, size=n)
    v = rng.uniform(v0, v1, size=n)
    return np.stack([u, v], axis=1).astype(np.float32)


def features_to_numpy(features: QueryFeatures) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, t in enumerate(features.lowres):
        out[f"lowres_{i}"] = t.detach().cpu().numpy()
    for i, t in enumerate(features.hires):
        out[f"hires_{i}"] = t.detach().cpu().numpy()
    out["resolutions"] = np.asarray(features.resolutions, dtype=np.int32)
    out["n_lowres"] = np.array([len(features.lowres)], dtype=np.int32)
    out["n_hires"] = np.array([len(features.hires)], dtype=np.int32)
    return out


def features_from_numpy(data: dict[str, Any], device: torch.device) -> QueryFeatures:
    n_low = int(np.asarray(data["n_lowres"]).reshape(-1)[0])
    n_hi = int(np.asarray(data["n_hires"]).reshape(-1)[0])
    lowres = [
        torch.tensor(np.asarray(data[f"lowres_{i}"]), device=device)
        for i in range(n_low)
    ]
    hires = [
        torch.tensor(np.asarray(data[f"hires_{i}"]), device=device)
        for i in range(n_hi)
    ]
    resolutions = [tuple(map(int, r)) for r in np.asarray(data["resolutions"])]
    return QueryFeatures(lowres=lowres, hires=hires, resolutions=resolutions)


def select_feature_subset(features: QueryFeatures, indices: np.ndarray) -> QueryFeatures:
    idx = np.asarray(indices, dtype=np.int64)
    lowres = [t[:, idx] for t in features.lowres]
    hires = [t[:, idx] for t in features.hires]
    return QueryFeatures(
        lowres=lowres, hires=hires, resolutions=features.resolutions
    )


# ---------------------------------------------------------------------------
# Motion clustering + active points + goals
# ---------------------------------------------------------------------------


def _kmeans_numpy(
    feats: np.ndarray, k: int, n_iter: int = 40, seed: int = 42
) -> np.ndarray:
    """Minimal k-means (avoids sklearn dependency)."""
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    k = int(min(k, n))
    centers = feats[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1).astype(np.int32)
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = feats[m].mean(axis=0)
    return labels


def cluster_motion_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    n_clusters: int = 4,
    min_visibility: float = 0.3,
    min_path_px: float = 8.0,
) -> np.ndarray:
    """Simple trajectory clustering (RoboTAP heuristics, not JAX EM).

    Returns labels [N] with -1 for rejected (low visibility / static).
    """
    N, T, _ = tracks.shape
    del T
    labels = np.full(N, -1, dtype=np.int32)
    vis_frac = visibility.mean(axis=1)
    disp = tracks[:, -1] - tracks[:, 0]
    path = np.linalg.norm(np.diff(tracks, axis=1), axis=2).sum(axis=1)
    path = np.where(visibility[:, 1:].sum(axis=1) > 0, path, 0.0)

    keep = (vis_frac >= min_visibility) & (path >= min_path_px)
    if keep.sum() < max(n_clusters, 2):
        keep = path >= (min_path_px * 0.25)
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return labels

    h = max(float(tracks[:, :, 1].max()), 1.0)
    w = max(float(tracks[:, :, 0].max()), 1.0)
    feats = np.concatenate(
        [
            disp[idx] / np.array([w, h], dtype=np.float32),
            tracks[idx, 0] / np.array([w, h], dtype=np.float32),
            tracks[idx, -1] / np.array([w, h], dtype=np.float32),
            (path[idx] / max(w, h))[:, None],
        ],
        axis=1,
    ).astype(np.float64)

    k = int(min(n_clusters, len(idx)))
    sub = _kmeans_numpy(feats, k)
    labels[idx] = sub.astype(np.int32)
    return labels


def pick_object_cluster(
    tracks: np.ndarray,
    visibility: np.ndarray,
    labels: np.ndarray,
) -> int:
    """Choose cluster with largest total path among non-rejected points."""
    path = np.linalg.norm(np.diff(tracks, axis=1), axis=2).sum(axis=1)
    best_c, best_score = -1, -1.0
    for c in sorted(set(labels.tolist())):
        if c < 0:
            continue
        m = labels == c
        score = float(path[m].sum() * visibility[m].mean())
        if score > best_score:
            best_score = score
            best_c = c
    if best_c < 0:
        raise ValueError("no valid motion cluster found")
    return best_c


def select_active_indices(
    tracks: np.ndarray,
    visibility: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    max_points: int = 12,
    min_visibility: float = 0.4,
) -> np.ndarray:
    mask = (labels == cluster_id) & (visibility.mean(axis=1) >= min_visibility)
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        cand = np.flatnonzero(labels == cluster_id)
    if len(cand) == 0:
        raise ValueError("no candidates in object cluster")

    # Prefer points visible at the end
    end_vis = visibility[cand, -1]
    order = np.argsort(-end_vis.astype(np.float32))
    cand = cand[order]

    # Greedy farthest-point on start positions for spatial diversity
    selected: list[int] = []
    start_uv = tracks[cand, 0]
    remaining = list(range(len(cand)))
    # seed: most visible at end
    selected.append(remaining.pop(0))
    while remaining and len(selected) < max_points:
        sel_uv = start_uv[selected]
        best_j, best_d = remaining[0], -1.0
        for j in remaining:
            d = float(np.linalg.norm(start_uv[j] - sel_uv, axis=1).min())
            if d > best_d:
                best_d = d
                best_j = j
        remaining.remove(best_j)
        selected.append(best_j)
    return cand[np.asarray(selected, dtype=np.int64)]


def goals_from_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """goal_i = last visible position near segment end. Shape [M,2]."""
    goals = np.zeros((len(active), 2), dtype=np.float32)
    T = tracks.shape[1]
    for j, i in enumerate(active):
        vis = visibility[i]
        # Prefer true last frame; else last visible
        if vis[-1]:
            goals[j] = tracks[i, -1]
        else:
            vis_idx = np.flatnonzero(vis)
            if len(vis_idx) == 0:
                goals[j] = tracks[i, T - 1]
            else:
                goals[j] = tracks[i, int(vis_idx[-1])]
    return goals


def aggregate_goals_median(goal_list: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(goal_list, axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Visual servoing (4-DoF) + eye-in-hand mapping
# ---------------------------------------------------------------------------


def normalize_uv(points_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel (u,v) → roughly centered normalized coords for the Jacobian."""
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    u = (pts[:, 0] - 0.5 * width) / max(width, 1)
    v = (pts[:, 1] - 0.5 * height) / max(height, 1)
    return np.stack([u, v], axis=1)


def build_jacobians(uv_norm: np.ndarray) -> np.ndarray:
    """Stack 2x4 Jacobians for N points → [2N, 4]."""
    uv = np.asarray(uv_norm, dtype=np.float64).reshape(-1, 2)
    blocks = []
    for u, v in uv:
        blocks.append(
            np.array(
                [
                    [1.0, 0.0, -u, -v],
                    [0.0, 1.0, -v, u],
                ],
                dtype=np.float64,
            )
        )
    return np.vstack(blocks)


def solve_visual_servo(
    points_uv: np.ndarray,
    goals_uv: np.ndarray,
    width: int,
    height: int,
    gain: float = 0.5,
) -> np.ndarray:
    """Return xi = [X_dot, Y_dot, Z_dot, Rz_dot] in camera (OpenCV) frame units."""
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    goals = np.asarray(goals_uv, dtype=np.float64).reshape(-1, 2)
    err = gain * (goals - pts)
    # Normalize error into same units as Jacobian coords
    dp = np.empty(err.size, dtype=np.float64)
    dp[0::2] = err[:, 0] / max(width, 1)
    dp[1::2] = err[:, 1] / max(height, 1)
    uv = normalize_uv(pts, width, height)
    J = build_jacobians(uv)
    xi, *_ = np.linalg.lstsq(J, dp, rcond=None)
    return xi.astype(np.float64)


def clamp_delta(
    delta: np.ndarray,
    max_trans: float = 0.01,
    max_rot: float = 0.05,
) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64).reshape(4).copy()
    d[:3] = np.clip(d[:3], -max_trans, max_trans)
    d[3] = float(np.clip(d[3], -max_rot, max_rot))
    return d


def camera_delta_to_world(
    delta_cam_cv: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_R_mj: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Map OpenCV-camera Δxyz + ΔRz to world-frame translation and Rz about cam Z.

    cam_R_mj: MuJoCo camera rotation (columns +X right, +Y up, +Z back).
    Returns (d_pos_world [3], d_rz_cam).
    """
    d = np.asarray(delta_cam_cv, dtype=np.float64).reshape(4)
    d_cv = np.array([d[0], d[1], d[2], 1.0], dtype=np.float64)
    d_mj_h = T_CV_TO_MJ @ d_cv
    d_mj = d_mj_h[:3]
    R = np.asarray(cam_R_mj, dtype=np.float64).reshape(3, 3)
    d_world = R @ d_mj
    del cam_pos_world  # unused; caller applies to current EE
    return d_world, float(d[3])


def apply_ee_delta(
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    d_pos_world: np.ndarray,
    d_rz: float,
    cam_R_mj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate EE in world; rotate about camera optical axis (MuJoCo -Z ≈ CV +Z)."""
    new_pos = np.asarray(ee_pos, dtype=float).reshape(3) + np.asarray(
        d_pos_world, dtype=float
    ).reshape(3)
    # OpenCV +Z forward corresponds to MuJoCo camera -Z (via T_CV_TO_MJ)
    z_mj = np.asarray(cam_R_mj, dtype=float).reshape(3, 3)[:, 2]
    axis = -z_mj  # CV forward in world
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    # Rodrigues
    k = axis
    theta = float(d_rz)
    K = np.array(
        [
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ],
        dtype=float,
    )
    R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    R_ee = quat_wxyz_to_R(ee_quat_wxyz)
    new_quat = R_to_quat_wxyz(R_delta @ R_ee)
    return new_pos, new_quat


def save_task_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_task_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
