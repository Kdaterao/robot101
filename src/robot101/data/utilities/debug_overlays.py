"""Render debug videos from saved tracks and clustering-window masks."""
import json
from pathlib import Path

import cv2
import numpy as np


def check_output_mode(root, debug):
    """Older destinations without a mode marker contain clean videos."""
    marker = Path(root) / "meta" / "grounding_output.json"
    existing = json.loads(marker.read_text()).get("debug_overlays", False) if marker.exists() else False
    if Path(root).exists() and bool(existing) != bool(debug):
        raise ValueError("Destination overlay mode differs; choose a separate --dst-repo-id")


def write_output_mode(root, debug):
    marker = Path(root) / "meta" / "grounding_output.json"
    marker.write_text(json.dumps({"debug_overlays": bool(debug),
                                 "video_content": "debug_masks_and_points" if debug else "clean"}, indent=2)+"\n")


class DebugOverlayRenderer:
    def __init__(self, records, sidecar_root):
        self.records = records
        self.masks = []
        for record in records:
            info = record["pov"].get("gripper_filter", {})
            packed = None
            if info.get("status") == "ok" and info.get("mask_file"):
                with np.load(Path(sidecar_root) / info["mask_file"], allow_pickle=False) as archive:
                    packed = (archive["packed_masks"].copy(), archive["shape"].copy(),
                              {int(frame): i for i, frame in enumerate(archive["frame_indices"])})
            self.masks.append((info, packed))

    @staticmethod
    def _dots(image, result, offset, color):
        tracks = np.asarray(result.get("tracks", []))
        visibility = np.asarray(result.get("visibility", []), dtype=bool)
        if tracks.ndim != 3 or visibility.shape != tracks.shape[:2] or not 0 <= offset < tracks.shape[1]:
            return
        h, w = image.shape[:2]
        for point in tracks[visibility[:, offset], offset]:
            if np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h:
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(image, xy, 5, (0, 0, 0), -1)
                cv2.circle(image, xy, 3, color, -1)

    def render(self, rgb, frame, camera):
        image = rgb.copy()
        current = next((r for r in self.records if r["start_frame"] <= frame <= r["end_frame"]), None)
        if current is None:
            return image
        offset = frame-current["start_frame"]
        status = ""
        if camera == "wrist":
            status = "mask: outside window"
            for info, packed in self.masks:
                window = info.get("window")
                if window is None or not window[0] <= frame <= window[1]:
                    continue
                if packed is None:
                    status = "mask: REVIEW required"
                    continue
                bits, shape, indices = packed
                if frame not in indices:
                    status = "mask: REVIEW required"
                    continue
                mask = np.unpackbits(bits[indices[frame]], axis=-1, count=int(shape[2])).astype(bool)
                if mask.shape != image.shape[:2]:
                    status = "mask: REVIEW required"
                    continue
                image[mask] = (0.6*image[mask]+0.4*np.array([255, 0, 255])).astype(np.uint8)
                status = "mask: SAM2"
            if current["pov"].get("gripper_filter", {}).get("status") == "disabled":
                status = "mask: disabled"
            result = current["pov"]
            previous = result.get("previous_stage", {})
        else:
            result = current.get("third_person", {}).get(camera, {})
            previous = result.get("transition_from_previous", {})
        self._dots(image, previous, offset, (0, 255, 255))
        self._dots(image, result, offset, (0, 255, 0))
        text = f"DEBUG subtask {current['subtask']} {result.get('source', 'none')} {status}"
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        return image
