"""SAM 2 gripper masks and mask-aware POV candidate statistics."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile

import cv2
import numpy as np

BODY_PROMPT = (
    'Point to visible solid parts of the robot gripper body and each finger. '
    'Do not point to the empty space between the jaws or to the held object.'
)
NEGATIVE_PROMPT = (
    'Point inside visible non-robot objects touching or immediately beside the robot gripper, '
    'including any object held between the fingers. These are objects to EXCLUDE from a gripper mask. '
    'Do not point to the robot body, fingers, empty space or background. '
    'Return no points if no such object is visible.'
)


def valid_points(raw, width, height):
    if raw.get('error'):
        return np.empty((0, 2), np.float32)
    points = np.asarray(raw.get('points_xy', []), dtype=np.float32).reshape(-1, 2)
    keep = np.isfinite(points).all(1) & (points >= 0).all(1)
    keep &= (points[:, 0] < width) & (points[:, 1] < height)
    return points[keep]


class GripperMaskWorker:
    """One SAM video state per clustering clip, with CPU-offloaded frames."""
    def __init__(self, model_id='facebook/sam2.1-hiera-tiny', device=None):
        import torch
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as exc:
            raise RuntimeError('SAM 2 is missing; rerun Linux setup --profile grounded') from exc
        self.model_id = model_id
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.predictor = SAM2VideoPredictor.from_pretrained(model_id, device=self.device, apply_postprocessing=False)
        self.predictor.eval()

    def propagate(self, frames, seed_frame, points, labels=None):
        import torch
        points = np.asarray(points, np.float32).reshape(-1, 2)
        labels = np.ones(len(points), np.int32) if labels is None else np.asarray(labels, np.int32)
        if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
            raise ValueError('SAM prompts require positive points and matching 0/1 labels')
        h, w = frames[0].shape[:2]
        output = np.zeros((len(frames), h, w), bool)
        # SAM's public video API accepts a numbered JPEG directory. Keep it
        # temporary; source videos and dataset frames are never altered.
        with tempfile.TemporaryDirectory(prefix='robot101-sam2-') as directory:
            for i, frame in enumerate(frames):
                if frame.shape != (h, w, 3):
                    raise ValueError('SAM clip resolution changed')
                if not cv2.imwrite(str(Path(directory) / f'{i:06d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise OSError('Could not write SAM input frame')
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()
            with torch.inference_mode(), amp:
                state = self.predictor.init_state(directory, offload_video_to_cpu=True, offload_state_to_cpu=True)
                try:
                    _, _, logits = self.predictor.add_new_points_or_box(
                        state, frame_idx=seed_frame, obj_id=1, points=points, labels=labels)
                    output[seed_frame] = (logits[0, 0] > 0).cpu().numpy()
                    for reverse in (False, True):
                        if reverse and seed_frame == 0:
                            continue
                        for index, _, logits in self.predictor.propagate_in_video(
                            state, start_frame_idx=seed_frame, reverse=reverse):
                            output[index] = (logits[0, 0] > 0).cpu().numpy()
                finally:
                    self.predictor.reset_state(state)
        return output


def filter_candidates(tracks, visible, masks, max_weight=2.0, radius_fraction=0.05, goal_tail_frames=5):
    """Return eligibility, observation weights and tracks filled from valid data.

    Interpolation uses only eligible observations; coordinates underneath the
    gripper never contribute to motion features. Unseen endpoints are rejected.
    """
    from robot101.perception.tracking import track_tail_endpoint
    tracks = np.asarray(tracks, np.float32)
    visible = np.asarray(visible, bool).copy()
    masks = np.asarray(masks, bool)
    n, length = visible.shape
    if masks.ndim != 3 or masks.shape[0] != length or tracks.shape != (n, length, 2):
        raise ValueError('Mask and track frame dimensions differ')
    if not np.isfinite(max_weight) or max_weight < 1 or not np.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError('Mask weight must be >=1 and radius fraction >0')
    h, w = masks.shape[1:]
    weights = np.zeros((n, length), np.float32)
    radius = radius_fraction * np.hypot(w, h)
    for t, mask in enumerate(masks):
        if not mask.any():
            visible[:, t] = False
            continue
        xy = tracks[:, t]
        valid = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
        pixels = np.rint(np.nan_to_num(xy)).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, w-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, h-1)
        visible[:, t] &= valid & ~mask[pixels[:, 1], pixels[:, 0]]
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d = distance[pixels[:, 1], pixels[:, 0]]
        weights[:, t] = np.where(visible[:, t], 1 + (max_weight-1)*np.maximum(0, 1-d/radius), 0)
    endpoints = track_tail_endpoint(tracks, visible, n_tail=goal_tail_frames)
    eligible = visible[:, -1].copy()
    finite = np.isfinite(endpoints).all(1) & (endpoints >= 0).all(1) & (endpoints[:, 0] < w) & (endpoints[:, 1] < h)
    xy = np.rint(np.nan_to_num(endpoints)).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, w-1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h-1)
    eligible &= finite & ~masks[-1, xy[:, 1], xy[:, 0]]
    filled = np.zeros_like(tracks)
    for q in range(n):
        ix = np.flatnonzero(visible[q])
        if len(ix):
            for axis in range(2):
                filled[q, :, axis] = np.interp(np.arange(length), ix, tracks[q, ix, axis])
    return filled, visible, weights, eligible, endpoints
