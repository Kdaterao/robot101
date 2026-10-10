#!/usr/bin/env python3
"""Manually correct third-person goal points in grounded dataset sidecars.

Left-click one or more object points on a stage frame, then press Enter to
track them through that stage and save the corrected sidecar. Press n to skip,
q to stop. No robot-gripper fallback is used for goal labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from robot101.data.utilities.episode_helpers import _load_lerobot, _parse_episodes
from robot101.data.utilities.episode_io import decode_episode_cameras, load_episode_metadata
from robot101.perception.tracking import BootsTAPIR, DEFAULT_CHECKPOINT


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _sidecars(
    repo_id: str, root: Path, local_only: bool,
    selected_episodes: set[int] | None, max_episodes: int,
) -> list[Path]:
    directory = root / "point_tracks"
    local = sorted(directory.glob("ep[0-9]*.json")) if directory.is_dir() else []
    if local:
        if selected_episodes is not None:
            local = [p for p in local if int(p.stem[2:]) in selected_episodes]
        else:
            local = local[:max_episodes]
        return local
    if local_only:
        return []

    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    names = sorted(
        name for name in files
        if name.startswith("point_tracks/ep") and name.endswith(".json")
        and Path(name).stem[2:].isdigit()
    )
    if selected_episodes is not None:
        names = [name for name in names if int(Path(name).stem[2:]) in selected_episodes]
    else:
        names = names[:max_episodes]
    for name in names:
        hf_hub_download(repo_id, name, repo_type="dataset", local_dir=root)
    return [root / name for name in names]


def _load_episode(LeRobotDataset, repo_id: str, episode: int, root: Path, backend: str):
    kwargs = {"episodes": [episode], "root": root, "video_backend": backend}
    try:
        return LeRobotDataset(repo_id, **kwargs)
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        print(f"Episode {episode} is missing from local cache; syncing its shards from the Hub.")
        return LeRobotDataset(repo_id, force_cache_sync=True, **kwargs)


def _is_resolved(result: dict) -> bool:
    if result.get("status") != "ok" or result.get("source") in {None, "none", "gripper_fallback"}:
        return False
    tracks = np.asarray(result.get("tracks", []), dtype=np.float32)
    visibility = np.asarray(result.get("visibility", []), dtype=bool)
    return tracks.ndim == 3 and visibility.shape == tracks.shape[:2] and bool(visibility.any())


class PointPicker:
    def __init__(self, window: str, max_width: int):
        self.max_width = max_width
        import tkinter as tk
        from tkinter import messagebox
        self.tk = tk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = window

    def pick(self, frames, valid_locals, title: str, old_result: dict):
        from PIL import Image, ImageDraw, ImageTk

        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title(self.window)
        current = [valid_locals[0]]
        clicks: list[tuple[float, float]] = []
        action = ["skip"]
        h, w = frames[0].shape[:2]
        scale = min(1.0, self.max_width / w)
        display_size = (round(w * scale), round(h * scale))
        canvas = tk.Canvas(window, width=display_size[0], height=display_size[1], highlightthickness=0)
        canvas.pack()
        status = tk.Label(window, anchor="w", justify="left")
        status.pack(fill="x")
        buttons = tk.Frame(window)
        buttons.pack(fill="x")
        photo = [None]

        def finish(which):
            action[0] = which
            window.destroy()

        def render():
            local = current[0]
            image = Image.fromarray(frames[local]).convert("RGB")
            if scale < 1.0:
                image = image.resize(display_size)
            draw = ImageDraw.Draw(image)
            for point in _point_at_frame(old_result, local):
                x, y = (float(point[0]) * scale, float(point[1]) * scale)
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="#00d7ff", width=3)
            for point in clicks:
                x, y = point[0] * scale, point[1] * scale
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#ff3030", width=3)
            photo[0] = ImageTk.PhotoImage(image)
            canvas.configure(width=image.width, height=image.height)
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo[0], anchor="nw")
            absolute = local
            status.configure(
                text=f"{title} | stage frame {absolute} | {len(clicks)} point(s)\n"
                     "Click goal point(s); Enter tracks and saves. Right click or Backspace undoes."
            )

        def add_point(event):
            clicks.append((event.x / scale, event.y / scale))
            render()

        def undo(event=None):
            if clicks:
                clicks.pop()
                render()

        def clear(event=None):
            clicks.clear()
            render()

        def step(delta):
            pos = valid_locals.index(current[0])
            current[0] = valid_locals[max(0, min(len(valid_locals) - 1, pos + delta))]
            render()

        def save(event=None):
            if not clicks:
                self.messagebox.showinfo("No points", "Click at least one goal point, or choose Skip.", parent=window)
                return
            finish("save")

        canvas.bind("<Button-1>", add_point)
        canvas.bind("<Button-3>", undo)
        window.bind("<BackSpace>", undo)
        window.bind("<Delete>", undo)
        window.bind("<KeyPress-a>", lambda event: step(-1))
        window.bind("<KeyPress-d>", lambda event: step(1))
        window.bind("<Left>", lambda event: step(-1))
        window.bind("<Right>", lambda event: step(1))
        window.bind("<KeyPress-c>", clear)
        window.bind("<Return>", save)
        window.bind("<KeyPress-n>", lambda event: finish("skip"))
        window.bind("<KeyPress-q>", lambda event: finish("quit"))
        tk.Button(buttons, text="Previous frame", command=lambda: step(-1)).pack(side="left")
        tk.Button(buttons, text="Next frame", command=lambda: step(1)).pack(side="left")
        tk.Button(buttons, text="Undo", command=undo).pack(side="left")
        tk.Button(buttons, text="Clear", command=clear).pack(side="left")
        tk.Button(buttons, text="Track and save", command=save).pack(side="right")
        tk.Button(buttons, text="Skip", command=lambda: finish("skip")).pack(side="right")
        tk.Button(buttons, text="Quit", command=lambda: finish("quit")).pack(side="right")
        window.protocol("WM_DELETE_WINDOW", lambda: finish("quit"))
        window.focus_force()
        canvas.focus_set()
        render()
        self.root.wait_window(window)
        return action[0], current[0], np.asarray(clicks, dtype=np.float32).reshape(-1, 2)

    def close(self):
        self.root.destroy()


def _track_from_seed(tapir, clip, seed_local: int, points: np.ndarray):
    """Track from a human-selected frame both backward and forward in the stage."""
    before, before_vis = tapir.track_segment_backward(
        clip[: seed_local + 1], points, label="manual-grounding-backward"
    )
    after, after_vis, _, _ = tapir.track_video(
        clip[seed_local:], points, query_frame_index=0
    )
    tracks = np.concatenate((before[:, :-1], after), axis=1)
    visibility = np.concatenate((before_vis[:, :-1], after_vis), axis=1)
    tracks[:, seed_local] = points
    visibility[:, seed_local] = True
    return tracks, visibility


def _point_at_frame(result: dict, local_frame: int) -> np.ndarray:
    tracks = np.asarray(result.get("tracks", []), dtype=np.float32)
    visible = np.asarray(result.get("visibility", []), dtype=bool)
    if tracks.ndim != 3 or visible.shape != tracks.shape[:2] or local_frame >= tracks.shape[1]:
        return np.zeros((0, 2), dtype=np.float32)
    valid = visible[:, local_frame] & np.isfinite(tracks[:, local_frame]).all(axis=1)
    return tracks[valid, local_frame]


def _set_next_stage_transition(
    tapir, source, data, report, stage_index, camera, tracks, visibility, fps,
    persist_seconds, backend,
):
    """Refresh the following stage's carryover after its predecessor is corrected."""
    stages = report.get("subtasks", [])
    if persist_seconds <= 0 or stage_index + 1 >= len(stages):
        return
    current_stage, next_stage = stages[stage_index], stages[stage_index + 1]
    seed_frame = int(current_stage["end_frame"])
    next_start, next_end = int(next_stage["start_frame"]), int(next_stage["end_frame"])
    length = min(max(1, round(fps * persist_seconds)), next_end - next_start + 1)
    carry_end = next_start + length - 1
    valid = visibility[:, -1] & np.isfinite(tracks[:, -1]).all(axis=1)
    points = tracks[valid, -1]
    next_result = next_stage.setdefault("third_person", {}).setdefault(camera, {})
    transition = {
        "source_subtask": stage_index, "source": "human_manual",
        "persist_seconds": persist_seconds, "persist_frames": length,
        "tracks": [], "visibility": [], "raw_visibility": [],
    }
    if not len(points) or seed_frame > next_start or not data["masks"][camera][seed_frame]:
        transition.update(status="failed", reason="no_visible_manual_endpoint")
        next_result["transition_from_previous"] = transition
        return

    decode_episode_cameras(source, data, [camera],
                           indices=range(seed_frame, carry_end + 1),
                           batch_size=64, backend=backend)
    clip = data["frames"][camera][seed_frame : carry_end + 1]
    carried, carried_vis, _, _ = tapir.track_video(clip, points, query_frame_index=0)
    offset = next_start - seed_frame
    carried = carried[:, offset : offset + length]
    carried_vis = carried_vis[:, offset : offset + length]
    carried_vis[:, ~np.asarray(data["masks"][camera][next_start : carry_end + 1], dtype=bool)] = False
    transition.update(
        status="ok" if bool(carried_vis.any()) else "tracker_lost",
        seed_points=points.tolist(), tracks=carried.tolist(),
        visibility=carried_vis.tolist(), raw_visibility=carried_vis.tolist(),
    )
    next_result["transition_from_previous"] = transition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="kdaterao/community_v3_ee_smolvla_molmo_grounded")
    parser.add_argument("--root", type=Path, help="Local LeRobot dataset root; defaults to the Hugging Face cache")
    parser.add_argument("--episodes", help="Source episode IDs, e.g. 105 or 105-120:5")
    parser.add_argument("--max-episodes", type=int, default=1, help="Maximum sidecar episodes to open by default")
    parser.add_argument("--camera", choices=["top", "side", "both"], default="both")
    parser.add_argument("--review-all", action="store_true", help="Allow replacing already resolved model tracks")
    parser.add_argument("--entity", help="Optional name for manually selected goal points")
    parser.add_argument("--device", help="TAPIR device (defaults to CUDA, MPS, then CPU)")
    parser.add_argument("--tapnet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tapir-frame-batch-size", type=int, default=256)
    parser.add_argument("--transition-persist-seconds", type=float, default=1.5)
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--local-only", action="store_true", help="Do not download sidecars from the Hub")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload corrected sidecars after labeling")
    args = parser.parse_args()

    LeRobotDataset, LeRobotDatasetMetadata, _, HF_LEROBOT_HOME = _load_lerobot()
    root = (args.root or (HF_LEROBOT_HOME / args.repo_id)).expanduser().resolve()
    selected_episodes = set(_parse_episodes(args.episodes)) if args.episodes else None
    reports = _sidecars(
        args.repo_id, root, args.local_only, selected_episodes, max(0, args.max_episodes)
    )
    if not reports:
        raise SystemExit(f"No point_tracks/ep*.json sidecars selected under {root}")
    if args.max_episodes < 1 or args.tapir_frame_batch_size < 1 or args.transition_persist_seconds < 0:
        raise SystemExit("Episode/batch sizes must be positive and transition persistence nonnegative")

    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    cameras = [name for name in (("top", "side") if args.camera == "both" else (args.camera,))
               if f"observation.images.{name}" in metadata.features]
    if not cameras:
        raise SystemExit("Dataset has no selected third-person camera streams")
    tapir = BootsTAPIR(checkpoint=args.tapnet_checkpoint, device=args.device,
                      frame_batch_size=args.tapir_frame_batch_size)
    picker = PointPicker("Manual third-person point correction", args.max_display_width)
    changed: list[Path] = []
    stop = False

    try:
        for sidecar in reports:
            report = json.loads(sidecar.read_text(encoding="utf-8"))
            source_episode = int(report["episode"])
            destination_episode = int(report.get("destination_episode", source_episode))
            source = _load_episode(LeRobotDataset, args.repo_id, destination_episode, root, args.video_backend)
            data = load_episode_metadata(source, cameras)
            for stage_index, stage in enumerate(report.get("subtasks", [])):
                start, end = int(stage["start_frame"]), int(stage["end_frame"])
                if start < 0 or end < start or end >= data["n"]:
                    print(f"Skipping ep{source_episode} stage{stage_index}: invalid frame range [{start}, {end}]")
                    continue
                for camera in cameras:
                    result = stage.setdefault("third_person", {}).setdefault(camera, {})
                    if not args.review_all and _is_resolved(result):
                        continue
                    indices = list(range(start, end + 1))
                    decode_episode_cameras(source, data, [camera], indices=indices,
                                           batch_size=64, backend=args.video_backend)
                    frames = data["frames"][camera]
                    masks = data["masks"][camera]
                    valid_locals = [i for i, frame in enumerate(frames[start:end + 1])
                                    if frame is not None and masks[start + i]]
                    if not valid_locals:
                        print(f"Skipping ep{source_episode} stage{stage_index} {camera}: no valid camera frame")
                        continue
                    label = (f"episode {source_episode} → output {destination_episode} | "
                             f"stage {stage_index} frames {start}-{end} | {camera} | "
                             f"{result.get('reason') or result.get('entity') or 'unresolved'}")
                    stage_frames = frames[start : end + 1]
                    action, current, point_xy = picker.pick(stage_frames, valid_locals, label, result)
                    if action == "quit":
                        stop = True
                        break
                    if action == "skip":
                        continue

                    tracks, visibility = _track_from_seed(tapir, stage_frames, current, point_xy)
                    visibility[:, ~np.asarray(masks[start : end + 1], dtype=bool)] = False
                    height, width = stage_frames[0].shape[:2]
                    result.update({
                        "status": "ok", "reason": None, "failures": [],
                        "source": "human_manual", "entity": args.entity or result.get("entity") or report.get("task", "manual goal"),
                        "distance_to_gripper": None, "image_size": [width, height],
                        "initial_points": point_xy.tolist(),
                        "initial_points_norm": (point_xy / np.array([max(1, width - 1), max(1, height - 1)])).tolist(),
                        "tracks": tracks.tolist(), "visibility": visibility.tolist(),
                        "raw_visibility": visibility.tolist(),
                        "tracks_normalized": (tracks / np.array([max(1, width - 1), max(1, height - 1)])).tolist(),
                        "tracking_fps": float(metadata.fps),
                        "tracking_sample_frames": list(range(start, end + 1)),
                        "manual_review_required": False,
                        "manual_correction": {"seed_frame": start + current, "seed_points": point_xy.tolist(),
                                              "previous_source": result.get("source"),
                                              "previous_reason": result.get("reason")},
                    })
                    _set_next_stage_transition(
                        tapir, source, data, report, stage_index, camera,
                        tracks, visibility, float(metadata.fps),
                        args.transition_persist_seconds, args.video_backend,
                    )
                    _atomic_write_json(sidecar, report)
                    if sidecar not in changed:
                        changed.append(sidecar)
                    print(f"Saved manual points: {sidecar} stage {stage_index} {camera}")
                if stop:
                    break
            del data, source
            if stop:
                break
    finally:
        picker.close()

    if args.push_to_hub and changed:
        from huggingface_hub import HfApi

        api = HfApi()
        for sidecar in changed:
            path_in_repo = sidecar.relative_to(root).as_posix()
            api.upload_file(
                path_or_fileobj=str(sidecar), path_in_repo=path_in_repo,
                repo_id=args.repo_id, repo_type="dataset",
                commit_message=f"Manually correct third-person points in {sidecar.stem}",
            )
            print(f"Uploaded {path_in_repo}")
    elif changed:
        print("Corrections are local. Add --push-to-hub to upload the modified sidecars.")


if __name__ == "__main__":
    main()
