"""apriltag_ros-style bundle pose for Lily objects.

ROS is not required. This is the pose path from AprilRobotics/apriltag_ros:
each object is a tag bundle, and one solvePnP uses every visible tag's
corners at once (instead of one PnP per tag).

Detection still uses AprilTag 3 via pupil_apriltags, the same library
apriltag_ros wraps.
"""

from __future__ import annotations

import cv2
import numpy as np
from pupil_apriltags import Detector
from robot101.calibration.transforms import R_to_quat_wxyz, quat_wxyz_to_R


def _up_from_tags(tags: dict) -> np.ndarray | None:
    top = tags.get(1) or tags.get("1")
    bottom = tags.get(6) or tags.get("6")
    if top is None or bottom is None:
        return None
    d = np.asarray(top["pos"], dtype=float) - np.asarray(bottom["pos"], dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return None
    return d / n


def _tag_to_T(tag: dict, up: np.ndarray | None = None) -> np.ndarray:
    """Tag pose in the object/bundle frame (AprilTag: +Z out, +Y down)."""
    p = np.asarray(tag["pos"], dtype=float).reshape(3)
    R = quat_wxyz_to_R(tag["quat"])
    x_stored = R[:, 0]
    z_stored = R[:, 2]

    if float(np.linalg.norm(p)) > 1e-8:
        z = p / np.linalg.norm(p)
    else:
        z = z_stored / (np.linalg.norm(z_stored) + 1e-12)

    x = None
    if up is not None:
        up = np.asarray(up, dtype=float).reshape(3)
        up_n = float(np.linalg.norm(up))
        if up_n > 1e-8:
            up = up / up_n
        up_plane = up - z * float(np.dot(up, z))
        if float(np.linalg.norm(up_plane)) > 0.2:
            y = -up_plane / np.linalg.norm(up_plane)
            x = np.cross(y, z)
            x = x / np.linalg.norm(x)
        else:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(z, helper))) > 0.9:
                helper = np.array([0.0, 0.0, 1.0])
            x = helper - z * float(np.dot(helper, z))
            x = x / np.linalg.norm(x)

    if x is None:
        x = x_stored - z * float(np.dot(x_stored, z))
        if float(np.linalg.norm(x)) < 1e-8:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(z, helper))) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            x = np.cross(helper, z)
        x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)

    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, z))
    T[:3, 3] = p
    return T


def _tag_corners_object(T_obj_tag: np.ndarray, tag_size: float) -> np.ndarray:
    """Four tag corners in the bundle frame.

    AprilTag / pupil_apriltags order, counterclockwise from bottom-left,
    in the tag frame with +Y down:

        0: (-s, +s, 0)  bottom-left
        1: (+s, +s, 0)  bottom-right
        2: (+s, -s, 0)  top-right
        3: (-s, -s, 0)  top-left
    """
    s = float(tag_size) * 0.5
    corners_tag = np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=float,
    )
    R = T_obj_tag[:3, :3]
    t = T_obj_tag[:3, 3]
    return corners_tag @ R.T + t


class AprilTagRosTracker:
    """Detect AprilTags, then solve each Lily object as an apriltag_ros bundle."""

    def __init__(
        self,
        family: str = "tag36h11",
        tag_size: float = 0.03,
        camera_index: int = 1,
        camera_params=(800.0, 800.0, 640.0, 360.0),
    ):
        self.family = family
        self.tag_size = tag_size
        self.camera_index = camera_index
        self.camera_params = [float(x) for x in camera_params]
        self.detector = Detector(
            families=family,
            nthreads=1,
            quad_decimate=1.0,
            refine_edges=True,
        )
        self.last_centers: dict[int, tuple[float, float]] = {}
        self.last_corners: dict[int, np.ndarray] = {}
        self.last_tag_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.by_tag: dict[int, tuple[dict, dict]] = {}
        self._obj_up: dict[str, np.ndarray] = {}
        self._T_obj_tag: dict[int, np.ndarray] = {}
        self._tag_size: dict[int, float] = {}
        self.last_bundle_counts: dict[str, int] = {}

    def obj_tags(self, objects: list[dict]) -> None:
        self.by_tag = {}
        self._obj_up = {}
        self._T_obj_tag = {}
        self._tag_size = {}
        for obj in objects:
            tags = obj.get("tags") or {}
            up = _up_from_tags(tags)
            if up is not None:
                self._obj_up[obj["name"]] = up
            for tag_id, tag in tags.items():
                tag_id = int(tag_id)
                if tag_id in self.by_tag:
                    raise ValueError(f"Duplicate tag ID: {tag_id}")
                self.by_tag[tag_id] = (obj, tag)
                self._T_obj_tag[tag_id] = _tag_to_T(tag, up)
                self._tag_size[tag_id] = float(tag.get("size", self.tag_size))

    def detect(self, frame) -> dict[int, np.ndarray]:
        """Returns {tag_id: 4x2 image corners} and fills last_* for overlays."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        detections: dict[int, np.ndarray] = {}
        self.last_centers = {}
        self.last_corners = {}
        self.last_tag_poses = {}
        for result in results:
            tag_id = int(result.tag_id)
            corners = np.asarray(result.corners, dtype=float).reshape(4, 2)
            detections[tag_id] = corners
            center = result.center
            self.last_centers[tag_id] = (float(center[0]), float(center[1]))
            self.last_corners[tag_id] = corners
            self.last_tag_poses[tag_id] = (
                np.asarray(result.pose_R, dtype=float),
                np.asarray(result.pose_t, dtype=float).reshape(3),
            )
        return detections

    def poses_from_bundles(
        self, detections: dict[int, np.ndarray]
    ) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
        """Object poses in the camera frame via one solvePnP per bundle."""
        fx, fy, cx, cy = (float(v) for v in self.camera_params)
        cam_k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)

        bundles: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {}
        for tag_id, corners in detections.items():
            hit = self.by_tag.get(int(tag_id))
            if hit is None:
                continue
            obj, _tag = hit
            T_obj_tag = self._T_obj_tag[int(tag_id)]
            size = self._tag_size[int(tag_id)]
            obj_pts = _tag_corners_object(T_obj_tag, size)
            img_pts = np.asarray(corners, dtype=float).reshape(4, 2)
            bundles.setdefault(obj["name"], []).append(
                (obj_pts, img_pts, int(tag_id))
            )

        poses: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self.last_bundle_counts = {}
        for name, parts in bundles.items():
            obj_pts = np.concatenate([p[0] for p in parts], axis=0)
            img_pts = np.concatenate([p[1] for p in parts], axis=0)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts.astype(np.float64),
                img_pts.astype(np.float64),
                cam_k,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            t = np.asarray(tvec, dtype=float).reshape(3)
            if (not np.all(np.isfinite(t))) or t[2] <= 0.0:
                continue
            R, _ = cv2.Rodrigues(rvec)
            R = np.asarray(R, dtype=float)
            if not np.all(np.isfinite(R)):
                continue
            primary = max(parts, key=lambda p: _quad_area(p[1]))[2]
            poses[name] = (t, R_to_quat_wxyz(R), primary)
            self.last_bundle_counts[name] = len(parts)
        return poses


def _quad_area(corners) -> float:
    pts = np.asarray(corners, dtype=float).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )
