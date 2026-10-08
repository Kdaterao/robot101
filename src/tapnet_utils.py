"""Shared TAPIR tracking, wrist-cam undistort, and visual-servoing helpers.

No AprilTag / LilyTags. Eye-in-hand (wrist) camera assumed.
"""

from __future__ import annotations

import json
from itertools import islice
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
# Wrist camera intrinsics (edit these like test_lily_relative.py)
# Calibrated at 1280x720, scaled x1.5 for 1920x1080. Remap with P=K.
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


def camera_matrix_from_params(
    fx: float = CAM_FX,
    fy: float = CAM_FY,
    cx: float = CAM_CX,
    cy: float = CAM_CY,
) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def default_wrist_calib(
    fx: float = CAM_FX,
    fy: float = CAM_FY,
    cx: float = CAM_CX,
    cy: float = CAM_CY,
    dist: tuple | list | np.ndarray = CAM_DIST,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> "WristCalib":
    return WristCalib(
        K=camera_matrix_from_params(fx, fy, cx, cy),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        width=int(width),
        height=int(height),
    )


# ---------------------------------------------------------------------------
# Calibration / undistort (same pattern as test_lily_relative.py:
# remap with P=K, no getOptimalNewCameraMatrix, never stretch-resize frames)
# ---------------------------------------------------------------------------


@dataclass
class WristCalib:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    def maps_for_frame(
        self, frame_width: int, frame_height: int
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Lily-style maps for a native frame size.

        Returns (K_used, maps_or_None). Scales K if aspect matches calib;
        skips undistort (maps=None) if aspect differs. Never resizes the image.
        """
        return prepare_undistort_maps(
            self.K,
            self.dist,
            calib_width=self.width,
            calib_height=self.height,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @property
    def maps(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Maps for the calibrated resolution (create path)."""
        _, maps = self.maps_for_frame(self.width, self.height)
        return maps


def prepare_undistort_maps(
    K: np.ndarray,
    dist: np.ndarray,
    calib_width: int,
    calib_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Build initUndistortRectifyMap exactly like test_lily_relative.py.

    - If native size == calib size: remap with P=K
    - If same aspect: scale K by sx/sy, remap at native size (no image stretch)
    - If aspect differs: skip undistort (maps=None), keep K as-is
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    native = (int(frame_width), int(frame_height))
    calib = (int(calib_width), int(calib_height))

    k_for_frame = True
    if native != calib:
        sx = native[0] / float(calib[0])
        sy = native[1] / float(calib[1])
        if abs(sx - sy) < 0.02:
            print(
                f"Same aspect as calibration; scaling K by "
                f"sx={sx:.4f} sy={sy:.4f}. Not stretching the image."
            )
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        else:
            k_for_frame = False
            print(
                f"WARNING: camera is {native[0]}x{native[1]} "
                f"({native[0] / max(native[1], 1):.3f}:1), calibration is "
                f"{calib[0]}x{calib[1]} "
                f"({calib[0] / max(calib[1], 1):.3f}:1). "
                "Not resizing (that stretches pixels and invalidates K). "
                "Set the camera to the calibration resolution."
            )

    if (not k_for_frame) or float(np.linalg.norm(dist)) <= 1e-12:
        if float(np.linalg.norm(dist)) > 1e-12 and not k_for_frame:
            print("WARNING: undistort skipped; native size does not match K.")
        return K, None

    maps = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        K,  # P=K — keep calibrated fx/fy/cx/cy
        native,
        cv2.CV_16SC2,
    )
    print(
        "Undistort ON: remap keeps the calibrated K "
        "(no getOptimalNewCameraMatrix / new_K)."
    )
    return K, maps


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
    """cv2.remap with lily's INTER_LINEAR; no-op if maps is None."""
    if maps is None:
        return frame_bgr
    return cv2.remap(
        frame_bgr,
        maps[0],
        maps[1],
        interpolation=cv2.INTER_LINEAR,
    )


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Demo I/O
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


# ---------------------------------------------------------------------------
# TAPIR (causal BootsTAPIR, PyTorch)
# ---------------------------------------------------------------------------


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve 'cuda' / 'cpu' / 'mps' / None → torch.device (prefer CUDA)."""
    if device is None or device == "" or device == "auto":
        return _default_device()
    if isinstance(device, torch.device):
        return device
    d = str(device).lower()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see packages-tapnet-gpu.sh)."
            )
        return torch.device("cuda")
    if d == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if d == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {device}")


class BootsTAPIR:
    """Thin wrapper around causal BootsTAPIR for offline + online tracking."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
        track_size: tuple[int, int] | None = (256, 256),
        frame_batch_size: int = 1,
    ):
        # Frames are resized to track_size (w, h) for TAPIR; all point
        # coordinates in/out stay in the caller's full-resolution pixels.
        if frame_batch_size < 1:
            raise ValueError("TAPIR frame_batch_size must be positive")
        self.frame_batch_size = int(frame_batch_size)
        self.track_size = track_size
        self.device = resolve_torch_device(device)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TAPIR checkpoint not found: {checkpoint}")
        print(f"BootsTAPIR device: {self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(self.device)}", flush=True)
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

    def _resize(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return (resized frame, sx, sy) where resized = full * (sx, sy)."""
        h, w = frame_rgb.shape[:2]
        if self.track_size is None or (w, h) == tuple(self.track_size):
            return frame_rgb, 1.0, 1.0
        tw, th = int(self.track_size[0]), int(self.track_size[1])
        small = cv2.resize(frame_rgb, (tw, th), interpolation=cv2.INTER_AREA)
        return small, tw / float(w), th / float(h)

    def init_features(
        self, frame_rgb: np.ndarray, query_points_tyx: np.ndarray
    ) -> QueryFeatures:
        """frame_rgb: HxWx3 uint8. query_points_tyx: [N,3] (t,y,x) full-res pixels."""
        small, sx, sy = self._resize(frame_rgb)
        q = np.asarray(query_points_tyx, dtype=np.float32).reshape(-1, 3).copy()
        q[:, 0] = 0.0
        q[:, 1] *= sy
        q[:, 2] *= sx
        frame = torch.tensor(small, device=self.device)
        pts = torch.tensor(q, dtype=torch.float32, device=self.device)
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
        """Returns tracks [N,2] (full-res px), visibles [N], updated causal_state."""
        tracks, visible, causal_state = self.predict_batch([frame_rgb], features, causal_state)
        return tracks[:, 0], visible[:, 0], causal_state

    def predict_batch(
        self,
        frames_rgb,
        features: QueryFeatures,
        causal_state,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Process a temporal chunk, returning [N,T,2], [N,T], and causal state.

        Time stays in the model's temporal dimension, not the independent-video
        batch dimension. The returned state must seed the next ordered chunk.
        """
        resized = [self._resize(frame) for frame in frames_rgb]
        if not resized:
            raise ValueError("TAPIR predict_batch requires at least one frame")
        small, sx, sy = resized[0]
        if any(frame.shape != small.shape or x != sx or y != sy for frame, x, y in resized):
            raise ValueError("TAPIR temporal chunks require consistent frame dimensions")
        # One upload and one feature-extraction call for the whole chunk.
        video = torch.as_tensor(np.stack([frame for frame, _, _ in resized]), device=self.device)
        frames = self.preprocess(video[None])
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
        next_state = trajectories["causal_context"]
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(uncertainty)) > 0.2
        # Copy each output once per chunk instead of once per frame.
        track_np = tracks[0].detach().cpu().numpy().astype(np.float32)
        track_np[..., 0] /= sx
        track_np[..., 1] /= sy
        vis_np = visibles[0].detach().cpu().numpy().astype(bool)
        return track_np, vis_np, next_state

    def track_with_features(
        self,
        frames_rgb,
        features: QueryFeatures,
        label: str = "",
        num_frames: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track precomputed query features through a clip.

        Lets the same queries (e.g. sampled in another demo) be tracked in
        every demo, so point i corresponds across demos. frames_rgb may be a
        generator (frames are not kept). Returns tracks [N,T,2], visibility [N,T].
        """
        if num_frames is None and hasattr(frames_rgb, "__len__"):
            num_frames = len(frames_rgb)
        N = int(features.lowres[0].shape[1])
        causal = self.initial_causal_state(N, features)
        tracks: list[np.ndarray] = []
        visibility: list[np.ndarray] = []
        iterator = iter(frames_rgb)
        processed = 0
        log_every = max(1, (num_frames or 100) // 10)
        next_log = 1
        while True:
            chunk = list(islice(iterator, self.frame_batch_size))
            if not chunk:
                break
            if self.frame_batch_size == 1:
                pts, vis, causal = self.predict(chunk[0], features, causal)
                pts, vis = pts[:, None, :], vis[:, None]
            else:
                pts, vis, causal = self.predict_batch(chunk, features, causal)
            tracks.append(pts)
            visibility.append(vis)
            processed += len(chunk)
            if processed >= next_log or (num_frames is not None and processed == num_frames):
                print(
                    f"  TAPIR {label} [{self.device}] frame {processed}/{num_frames or '?'} "
                    f"(frame batch={self.frame_batch_size})",
                    flush=True,
                )
                next_log = processed + log_every
        if not tracks:
            return np.zeros((N, 0, 2), np.float32), np.zeros((N, 0), bool)
        return np.concatenate(tracks, axis=1), np.concatenate(visibility, axis=1)

    def track_segments(self, jobs, segment_batch_size=0):
        """Batch independent clips along B; each clip retains its own state.

        Jobs contain frames, query_xy, and label. Returns (tracks, visibility)
        or an exception per job. Padding is terminal only and is discarded.
        Zero segment_batch_size groups all jobs. OOM splits the group safely.
        """
        jobs = list(jobs)
        if not jobs:
            return []
        if segment_batch_size < 0:
            raise ValueError("segment_batch_size must be nonnegative")

        def run(group, frame_batch_size):
            features, counts, lengths, scales = [], [], [], []
            max_queries = max(len(job["query_xy"]) for job in group)
            if max_queries == 0 or any(not len(job["frames"]) or not len(job["query_xy"]) for job in group):
                raise ValueError("Segment jobs require nonempty clips and queries")
            for job in group:
                points = np.asarray(job["query_xy"], np.float32).reshape(-1, 2)
                counts.append(len(points))
                lengths.append(len(job["frames"]))
                points = np.concatenate([points, np.repeat(points[-1:], max_queries - len(points), axis=0)])
                query = np.column_stack([np.zeros(max_queries), points[:, 1], points[:, 0]])
                features.append(self.init_features(job["frames"][0], query))
                _, sx, sy = self._resize(job["frames"][0])
                scales.append((sx, sy))
            resolutions = features[0].resolutions
            if any(item.resolutions != resolutions for item in features):
                raise ValueError("Segment feature resolutions differ")
            queries = QueryFeatures(
                lowres=tuple(torch.cat([item.lowres[i] for item in features], dim=0) for i in range(len(features[0].lowres))),
                hires=tuple(torch.cat([item.hires[i] for item in features], dim=0) for i in range(len(features[0].hires))),
                resolutions=resolutions,
            )
            state = self.initial_causal_state(max_queries, queries)
            state = tree.map_structure(lambda tensor: tensor.repeat(len(group), *([1] * (tensor.ndim - 1))), state)
            outputs = [(np.zeros((n, t, 2), np.float32), np.zeros((n, t), bool)) for n, t in zip(counts, lengths)]
            max_length = max(lengths)
            for start in range(0, max_length, frame_batch_size):
                stop = min(max_length, start + frame_batch_size)
                videos = []
                for job, scale in zip(group, scales):
                    resized = [self._resize(job["frames"][min(t, len(job["frames"]) - 1)]) for t in range(start, stop)]
                    if any((sx, sy) != scale for _, sx, sy in resized):
                        raise ValueError("Frame dimensions changed within a segment")
                    videos.append(np.stack([frame for frame, _, _ in resized]))
                video = self.preprocess(torch.as_tensor(np.stack(videos), device=self.device))
                grids = self.model.get_feature_grids(video, is_training=False)
                # Model refinement reshapes context dictionaries internally.
                # Each batch item has independent tensors along dimension B.
                result = self.model.estimate_trajectories(
                    video.shape[-3:-1], is_training=False, feature_grids=grids,
                    query_features=queries, query_points_in_video=None, query_chunk_size=64,
                    causal_context=state, get_causal_context=True,
                )
                state = result["causal_context"]
                tracks = result["tracks"][-1].detach().cpu().numpy().astype(np.float32)
                visible = ((1 - F.sigmoid(result["occlusion"][-1])) *
                           (1 - F.sigmoid(result["expected_dist"][-1])) > 0.2).detach().cpu().numpy()
                for i, (n, length, (sx, sy)) in enumerate(zip(counts, lengths, scales)):
                    valid = max(0, min(stop, length) - start)
                    if valid:
                        tracks[i, :n, :valid, 0] /= sx
                        tracks[i, :n, :valid, 1] /= sy
                        outputs[i][0][:, start:start + valid] = tracks[i, :n, :valid]
                        outputs[i][1][:, start:start + valid] = visible[i, :n, :valid]
                print(f"  TAPIR segment batch [{self.device}] {len(group)} clips, frame {stop}/{max_length}", flush=True)
                del video, grids, result, tracks, visible
            return outputs

        def safe_run(group, frame_batch_size):
            try:
                return run(group, frame_batch_size)
            except Exception as exc:
                # Restart smaller groups with fresh state. A failed job cannot
                # poison successful clips or cause an entire episode to drop.
                message = str(exc)
                out_of_memory = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if out_of_memory and frame_batch_size > 1:
                smaller = max(1, frame_batch_size // 2)
                print(f"  TAPIR retrying segment batch with {smaller} frames after OOM", flush=True)
                return safe_run(group, smaller)
            if len(group) > 1:
                middle = len(group) // 2
                print(f"  TAPIR splitting {len(group)} segment jobs after: {message}", flush=True)
                return safe_run(group[:middle], frame_batch_size) + safe_run(group[middle:], frame_batch_size)
            return [RuntimeError(message)]

        size = segment_batch_size or len(jobs)
        results = []
        for start in range(0, len(jobs), size):
            results.extend(safe_run(jobs[start:start + size], self.frame_batch_size))
        return results

    def track_segment_backward(
        self,
        frames_rgb: list[np.ndarray] | np.ndarray,
        query_xy_at_end: np.ndarray,
        label: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track points seeded at the last frame back through a stage clip.

        ``frames_rgb`` is the forward-time stage clip [start..end]. Queries are
        defined on the last frame; the clip is reversed, tracked causally, then
        flipped back. Returns tracks [N,T,2], visibility [N,T] in forward time.
        """
        frames = list(frames_rgb)
        if not frames:
            q = np.asarray(query_xy_at_end, dtype=np.float32).reshape(-1, 2)
            return np.zeros((len(q), 0, 2), np.float32), np.zeros((len(q), 0), bool)

        query_xy = np.asarray(query_xy_at_end, dtype=np.float32).reshape(-1, 2)
        rev = frames[::-1]
        # query on reversed frame 0 == original last frame
        query_tyx = np.stack(
            [
                np.zeros(len(query_xy), dtype=np.float32),
                query_xy[:, 1],
                query_xy[:, 0],
            ],
            axis=1,
        )
        features = self.init_features(rev[0], query_tyx)
        tr_rev, vi_rev = self.track_with_features(
            rev, features, label=label or "backward", num_frames=len(rev)
        )
        # flip time axis back to forward order
        return tr_rev[:, ::-1].copy(), vi_rev[:, ::-1].copy()

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
        tracks, visibility = self.track_with_features(
            frames_rgb, features, num_frames=len(frames_rgb)
        )

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


def points_heatmap(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    visibles: np.ndarray | None = None,
    sigma: float = 40.0,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay a jet heatmap peaked at visible track points."""
    h, w = frame_bgr.shape[:2]
    if len(points) == 0 or sigma <= 0:
        return frame_bgr.copy()

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if visibles is None:
        vis = np.ones(len(pts), dtype=bool)
    else:
        vis = np.asarray(visibles, dtype=bool).reshape(-1)
        if len(vis) != len(pts):
            raise ValueError(f"visibles length {len(vis)} != points {len(pts)}")

    scale = 4
    small_h, small_w = max(1, h // scale), max(1, w // scale)
    heat_s = np.zeros((small_h, small_w), dtype=np.float32)
    sig_s = max(1.0, sigma / scale)
    radius = int(max(3, round(3.0 * sig_s)))

    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sig_s * sig_s)).astype(np.float32)

    for (x, y), is_vis in zip(pts, vis):
        if not is_vis:
            continue
        cx = int(round(float(x) / scale))
        cy = int(round(float(y) / scale))
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius + 1, cy + radius + 1
        kx0, ky0 = 0, 0
        kx1, ky1 = kernel.shape[1], kernel.shape[0]
        if x0 < 0:
            kx0 = -x0
            x0 = 0
        if y0 < 0:
            ky0 = -y0
            y0 = 0
        if x1 > small_w:
            kx1 -= x1 - small_w
            x1 = small_w
        if y1 > small_h:
            ky1 -= y1 - small_h
            y1 = small_h
        if x0 >= x1 or y0 >= y1:
            continue
        heat_s[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    if not np.any(heat_s):
        return frame_bgr.copy()

    heat = cv2.resize(heat_s, (w, h), interpolation=cv2.INTER_LINEAR)
    heat /= max(float(heat.max()), 1e-6)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_color, alpha, 0)


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


def concat_query_features(feature_list: list[QueryFeatures]) -> QueryFeatures:
    """Concatenate QueryFeatures along the point axis (dim 1)."""
    if not feature_list:
        raise ValueError("no features to concatenate")
    first = feature_list[0]
    lowres = [
        torch.cat([f.lowres[i] for f in feature_list], dim=1)
        for i in range(len(first.lowres))
    ]
    hires = [
        torch.cat([f.hires[i] for f in feature_list], dim=1)
        for i in range(len(first.hires))
    ]
    return QueryFeatures(lowres=lowres, hires=hires, resolutions=first.resolutions)


def select_feature_subset(features: QueryFeatures, indices: np.ndarray) -> QueryFeatures:
    idx = np.asarray(indices, dtype=np.int64)
    lowres = [t[:, idx] for t in features.lowres]
    hires = [t[:, idx] for t in features.hires]
    return QueryFeatures(
        lowres=lowres, hires=hires, resolutions=features.resolutions
    )


# ---------------------------------------------------------------------------
# Clustering / track helpers (used by robotap.py)
# ---------------------------------------------------------------------------


def _kmeans_torch(
    feats: np.ndarray,
    k: int,
    n_iter: int = 40,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """k-means on CUDA when available (falls back to CPU torch)."""
    device = resolve_torch_device(device)
    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    n = int(x.shape[0])
    k = int(min(k, n))
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx.to(device)].clone()
    labels = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(n_iter):
        d = torch.cdist(x, centers, p=2)
        labels = d.argmin(dim=1)
        for c in range(k):
            m = labels == c
            if bool(m.any()):
                centers[c] = x[m].mean(dim=0)
    return labels.detach().cpu().numpy().astype(np.int32)


def track_tail_endpoint(
    track_xy: np.ndarray,
    visibility: np.ndarray | None = None,
    n_tail: int = 5,
) -> np.ndarray:
    """Goal UV = median of the last ``n_tail`` frames (prefer visible).

    ``track_xy`` is [T,2] or a batch [N,T,2] (returns [N,2]).
    """
    arr = np.asarray(track_xy, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    N, T, _ = arr.shape
    n_tail = int(max(1, min(n_tail, T)))
    vis = None if visibility is None else np.asarray(visibility, dtype=bool)
    if vis is not None and vis.ndim == 1:
        vis = vis[None, ...]

    out = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        tail = arr[i, -n_tail:]
        if vis is not None:
            v = vis[i, -n_tail:]
            if np.any(v):
                pts = tail[v]
            else:
                # no visible in tail — use last visible in full track, else last frame
                full_v = np.flatnonzero(vis[i])
                if len(full_v):
                    pts = arr[i, int(full_v[-1])][None, :]
                else:
                    pts = tail[-1:]
        else:
            pts = tail
        out[i] = np.median(pts, axis=0)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Eye-in-hand mapping
# ---------------------------------------------------------------------------


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
