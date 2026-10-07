"""Live camera TapNet click-to-track with point heatmap + covering shape."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pygame
import torch
import tree
from PIL import Image
from tapnet.torch.tapir_model import QueryFeatures

from florence2_worker import (
    DEFAULT_FLORENCE_MODEL,
    DEFAULT_LOCATE_BASE,
    DEFAULT_LOCATE_HF,
    boxes_from_ground_result,
    load_locator,
    resolve_locate_model,
)
from tapnet_utils import BootsTAPIR, DEFAULT_CHECKPOINT, points_heatmap

DEFAULT_LOCATOR = "locateanything"
DEFAULT_PROMPT = "SO-101 gripper"


def expand_polygon(poly: np.ndarray, pad: float) -> np.ndarray:
    """Push polygon vertices outward from the centroid by `pad` pixels."""
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(poly) == 0:
        return poly.astype(np.int32)
    if pad <= 0 or len(poly) == 1:
        return np.round(poly).astype(np.int32)
    center = poly.mean(axis=0)
    dirs = poly - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-3)
    expanded = center + dirs * ((norms + pad) / norms)
    return np.round(expanded).astype(np.int32)


def covering_polygon(
    points: np.ndarray,
    pad: float,
    mode: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Polygon covering all points: convex hull, or concave disk-union outline."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return None

    if mode == "concave":
        # Union of padded disks + edges between angle-sorted neighbors.
        # Contour can dent inward (concave) when points form that layout.
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(1, int(round(pad)))
        for x, y in pts:
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)
        if len(pts) >= 2:
            center = pts.mean(axis=0)
            order = np.argsort(
                np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            )
            ordered = pts[order]
            thickness = max(1, 2 * radius)
            for i in range(len(ordered)):
                a = (
                    int(round(ordered[i, 0])),
                    int(round(ordered[i, 1])),
                )
                b = (
                    int(round(ordered[(i + 1) % len(ordered), 0])),
                    int(round(ordered[(i + 1) % len(ordered), 1])),
                )
                cv2.line(mask, a, b, 255, thickness)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        return cnt.reshape(-1, 2).astype(np.int32)

    # Convex hull (default), with outward padding.
    if len(pts) == 1:
        x, y = pts[0]
        r = max(1, int(round(pad)))
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        circle = np.stack(
            [x + r * np.cos(angles), y + r * np.sin(angles)], axis=1
        )
        return np.round(circle).astype(np.int32)

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    poly = hull.reshape(-1, 2)
    return expand_polygon(poly, pad)


def fill_shape_from_surroundings(frame, polygon, inpaint_radius=5):
    """Inpaint the polygon interior so it blends with surrounding pixels."""
    if polygon is None or len(polygon) < 3:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 255)

    # Slightly erode so the green outline / border still has real context.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    if not np.any(mask):
        return frame

    return cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)


def render_frame(
    frame,
    polygon,
    points,
    visibles,
    heatmap_enabled=True,
    fill_enabled=True,
    heatmap_sigma=40.0,
    locate_boxes=None,
):
    """Optionally heatmap from TapNet points, inpaint shape, then overlay guides."""
    if heatmap_enabled and len(points) > 0:
        out = points_heatmap(frame, points, visibles, sigma=heatmap_sigma)
    else:
        out = frame.copy()

    if fill_enabled and polygon is not None and len(polygon) >= 3:
        out = fill_shape_from_surroundings(out, polygon)

    for (x, y), vis in zip(points, visibles):
        if not vis:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 5, (255, 0, 0), -1)

    if polygon is not None and len(polygon) >= 2:
        cv2.polylines(
            out,
            [polygon.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    if locate_boxes:
        for box in locate_boxes:
            pt1 = (int(round(box["x1"])), int(round(box["y1"])))
            pt2 = (int(round(box["x2"])), int(round(box["y2"])))
            cv2.rectangle(out, pt1, pt2, (0, 255, 255), 2)

    return out, polygon


def sample_points_in_box(
    box: dict,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform samples in (x, y) pixels inside a LocateAnything box. Shape [N,2]."""
    rng = rng or np.random.default_rng()
    x1, x2 = float(box["x1"]), float(box["x2"])
    y1, y2 = float(box["y1"]), float(box["y2"])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # Degenerate box: seed its center repeatedly.
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        return np.tile(np.array([[cx, cy]], dtype=np.float32), (n, 1))
    xs = rng.uniform(x1, x2, size=n)
    ys = rng.uniform(y1, y2, size=n)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def seed_from_locate(
    tracker: ClickTracker,
    frame_rgb: np.ndarray,
    worker,
    phrase: str,
    num_points: int,
    locator_name: str = "locateanything",
) -> list[dict]:
    """Clear tracker, ground phrase, sample points in the first box, seed TapNet."""
    h, w = frame_rgb.shape[:2]
    image = Image.fromarray(frame_rgb)
    result = worker.ground_single(image, phrase)
    answer = result.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)
    boxes = boxes_from_ground_result(worker, result, w, h)
    if not boxes:
        print(f"{locator_name}: no boxes for '{phrase}'")
        print(f"  raw answer: {answer[:200]!r}")
        return []

    # Prefer the largest box when multiple are returned.
    def _area(b: dict) -> float:
        return abs(b["x2"] - b["x1"]) * abs(b["y2"] - b["y1"])

    boxes = sorted(boxes, key=_area, reverse=True)
    seed_box = boxes[0]
    pts = sample_points_in_box(seed_box, num_points)
    tracker.clear()
    for x, y in pts:
        tracker.add_click(frame_rgb, float(x), float(y))
    print(
        f"{locator_name} seeded {len(pts)} points in box "
        f"({seed_box['x1']:.0f},{seed_box['y1']:.0f})-"
        f"({seed_box['x2']:.0f},{seed_box['y2']:.0f}) for '{phrase}'"
    )
    return boxes


class ClickTracker:
    """Fixed-slot TapNet tracker: click to add/replace a tracked point."""

    def __init__(
        self,
        tapir: BootsTAPIR,
        num_points: int = 8,
        shape_pad: int = 24,
        shape_mode: str = "convex",
    ):
        self.tapir = tapir
        self.num_points = num_points
        self.shape_pad = shape_pad
        self.shape_mode = shape_mode
        self.have_point = [False] * num_points
        self.next_idx = 0
        self.features = None
        self.causal = None
        self.tracks = np.zeros((num_points, 2), dtype=np.float32)
        self.vis = np.zeros(num_points, dtype=bool)

    def _ensure_init(self, frame_rgb: np.ndarray):
        if self.features is not None:
            return
        dummy = np.zeros((self.num_points, 3), dtype=np.float32)
        self.features = self.tapir.init_features(frame_rgb, dummy)
        self.causal = self.tapir.initial_causal_state(self.num_points, self.features)

    def add_click(self, frame_rgb: np.ndarray, x: float, y: float):
        """Same click-update path as tapnetex.py, with causal tensors on GPU."""
        self._ensure_init(frame_rgb)
        device = self.tapir.device
        query = np.array([[0.0, y, x]], dtype=np.float32)
        new_features = self.tapir.init_features(frame_rgb, query)
        idx_to_update = np.array([self.next_idx])

        def apply_update_idx(s1, s2):
            # Mirror tapnetex update, but keep s2 on the same device as s1.
            s1[:, idx_to_update] = s2.to(s1.device)
            return s1

        self.features = QueryFeatures(
            lowres=tree.map_structure(
                apply_update_idx, self.features.lowres, new_features.lowres
            ),
            hires=tree.map_structure(
                apply_update_idx, self.features.hires, new_features.hires
            ),
            resolutions=self.features.resolutions,
        )

        # construct_initial_causal_state is CPU; move like BootsTAPIR / tapnetex.
        init_causal = self.tapir.model.construct_initial_causal_state(
            len(idx_to_update), len(self.features.resolutions) - 1
        )
        init_causal = tree.map_structure(lambda t: t.to(device), init_causal)
        self.causal = tree.map_structure(apply_update_idx, self.causal, init_causal)

        self.have_point[self.next_idx] = True
        self.tracks[self.next_idx] = (x, y)
        self.vis[self.next_idx] = True
        self.next_idx = (self.next_idx + 1) % self.num_points

    def clear(self):
        self.have_point = [False] * self.num_points
        self.next_idx = 0
        self.features = None
        self.causal = None
        self.tracks[:] = 0
        self.vis[:] = False

    def step(self, frame_rgb: np.ndarray):
        if not any(self.have_point):
            return
        self._ensure_init(frame_rgb)
        self.tracks, self.vis, self.causal = self.tapir.predict(
            frame_rgb, self.features, self.causal
        )

    def active_points(self):
        pts = []
        vis = []
        for i, active in enumerate(self.have_point):
            if not active:
                continue
            pts.append(self.tracks[i])
            vis.append(bool(self.vis[i]))
        if not pts:
            return np.zeros((0, 2), np.float32), np.zeros((0,), bool)
        return np.asarray(pts, dtype=np.float32), np.asarray(vis, dtype=bool)

    def covering_shape(self, width: int, height: int):
        pts, vis = self.active_points()
        if len(pts) == 0:
            return None
        visible = pts[vis]
        if len(visible) == 0:
            return None
        return covering_polygon(
            visible,
            pad=float(self.shape_pad),
            mode=self.shape_mode,
            width=width,
            height=height,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera TapNet click-track with point heatmap"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="BootsTAPIR checkpoint path",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
        help="Max tracked click points (slots recycle)",
    )
    parser.add_argument(
        "--shape",
        choices=("convex", "concave"),
        default="convex",
        help="Covering shape around tracked points",
    )
    parser.add_argument(
        "--shape-pad",
        type=int,
        default=24,
        help="Padding around tracked points for the covering shape",
    )
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=40.0,
        help="Gaussian radius (px) for each TapNet point in the heatmap",
    )
    parser.add_argument(
        "--track-size",
        type=int,
        default=256,
        help="Square size TapNet runs at (lower = faster)",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Referring phrase (default: {DEFAULT_PROMPT!r}; L re-grounds)",
    )
    parser.add_argument(
        "--no-locate",
        action="store_true",
        help="Disable grounding; click-to-track only",
    )
    parser.add_argument(
        "--locator",
        default=DEFAULT_LOCATOR,
        choices=["locateanything", "florence2"],
        help="Grounding backend (LocateAnything kept as default; florence2 for A/B)",
    )
    parser.add_argument(
        "--locate-model",
        default=None,
        help=(
            "Model path/id for --locator. LocateAnything resolves: "
            f"local work_dirs → Hub {DEFAULT_LOCATE_HF} → {DEFAULT_LOCATE_BASE}; "
            f"Florence default {DEFAULT_FLORENCE_MODEL}"
        ),
    )
    parser.add_argument(
        "--locate-points",
        type=int,
        default=None,
        help="TapNet seed points sampled inside grounded box (default: --num-points)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    locate_points = (
        args.locate_points if args.locate_points is not None else args.num_points
    )
    locate_phrase = (args.prompt or "").strip() or None
    use_locate = (not args.no_locate) and bool(locate_phrase)
    if args.locator == "florence2":
        locate_model = args.locate_model or DEFAULT_FLORENCE_MODEL
    else:
        locate_model = resolve_locate_model(args.locate_model)

    print("Loading BootsTAPIR...")
    tapir = BootsTAPIR(
        checkpoint=args.checkpoint,
        track_size=(args.track_size, args.track_size),
    )
    tracker = ClickTracker(
        tapir,
        num_points=args.num_points,
        shape_pad=args.shape_pad,
        shape_mode=args.shape,
    )

    worker = None
    if use_locate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        worker = load_locator(args.locator, locate_model, device=device)

    pygame.init()
    help_keys = "F heatmap, I fill, H shape, L locate, C clear, Q quit"
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"TapNet heatmap ({args.shape}) | {help_keys}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    clock = pygame.time.Clock()
    heatmap_enabled = True
    fill_enabled = True
    locate_boxes: list[dict] = []
    need_initial_locate = bool(worker is not None and locate_phrase)
    print(
        "Click points to track. F=heatmap, I=fill, H=convex/concave, "
        "L=locate, C=clear, Q/Esc=quit"
    )
    if locate_phrase:
        print(f"{args.locator} phrase: {locate_phrase!r}")

    try:
        while True:
            clicks = []
            request_locate = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        heatmap_enabled = not heatmap_enabled
                        state = "ON" if heatmap_enabled else "OFF"
                        print(f"Heatmap {state}")
                    elif event.key == pygame.K_i:
                        fill_enabled = not fill_enabled
                        state = "ON" if fill_enabled else "OFF"
                        print(f"Shape fill {state}")
                    elif event.key == pygame.K_h:
                        tracker.shape_mode = (
                            "concave" if tracker.shape_mode == "convex" else "convex"
                        )
                        print(f"Shape mode: {tracker.shape_mode}")
                        pygame.display.set_caption(
                            f"TapNet heatmap ({tracker.shape_mode}) | {help_keys}"
                        )
                    elif event.key == pygame.K_l:
                        request_locate = True
                    elif event.key == pygame.K_c:
                        tracker.clear()
                        locate_boxes = []
                        print("Cleared tracked points")
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicks.append(event.pos)

            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame")
                break

            frame = cv2.resize(frame, (args.width, args.height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if worker is not None and (need_initial_locate or request_locate):
                if not locate_phrase:
                    print("No --prompt set; pass --prompt to use grounding")
                else:
                    locate_boxes = seed_from_locate(
                        tracker,
                        frame_rgb,
                        worker,
                        locate_phrase,
                        locate_points,
                        locator_name=args.locator,
                    )
                need_initial_locate = False

            for x, y in clicks:
                tracker.add_click(frame_rgb, float(x), float(y))
                print(f"Tracking point @ ({x}, {y})")

            tracker.step(frame_rgb)
            pts, vis = tracker.active_points()
            polygon = tracker.covering_shape(args.width, args.height)

            out, _ = render_frame(
                frame,
                polygon,
                pts,
                vis,
                heatmap_enabled,
                fill_enabled,
                heatmap_sigma=args.heatmap_sigma,
                locate_boxes=locate_boxes,
            )
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(out_rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    finally:
        cap.release()
        pygame.quit()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
