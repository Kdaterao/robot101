"""Sample SO101 frames from a LeRobot HF dataset and point-label grippers.

Preferred workflow (small Hub pulls + review):
  python src/locate_collect_label.py batch --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_REPO = "felsager/community_dataset_v3_ee_smolVLA"
DEFAULT_HF_PUSH_REPO = "kdaterao/so101_locate_gripper"
DEFAULT_CAMERAS = ("top", "side")
DEFAULT_PHRASE = "SO-101 gripper"
POINT_LABELS_NAME = "point_labels.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "locate_gripper"
BATCH_STATE_NAME = "batch_state.json"

# Vendored LeRobot lives at <repo>/lerobot/src when not installed as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))


def _load_lerobot():
    try:
        from hf_hub_windows import install_hf_download_guards

        # Import lerobot first so we can patch its bound snapshot_download.
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME

        install_hf_download_guards(max_workers=1)
    except ImportError as e:
        raise SystemExit(
            "lerobot is required for `sample`/`batch`. Install it or use the vendored "
            f"copy at {_LEROBOT_SRC}. Original error: {e}"
        ) from e
    return LeRobotDataset, HF_LEROBOT_HOME


def _cam_key(name: str) -> str:
    return f"observation.images.{name}"


def _mask_key(name: str) -> str:
    return f"observation.images.{name}_padding_mask"


def _tensor_to_bgr(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert LeRobot CHW float [0,1] (or HWC) image to BGR uint8."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _mask_is_real(item: dict, camera: str) -> bool:
    key = _mask_key(camera)
    if key not in item:
        return True
    val = item[key]
    if isinstance(val, torch.Tensor):
        return bool(val.reshape(-1)[0].item())
    if isinstance(val, np.ndarray):
        return bool(np.asarray(val).reshape(-1)[0])
    return bool(val)


def _frame_name(episode: int, frame: int, camera: str) -> str:
    return f"ep{episode:06d}_f{frame:06d}_{camera}.jpg"


def _default_video_backend() -> str:
    # torchcodec needs FFmpeg full-shared DLLs on Windows and often fails here.
    # pyav is installed in this env and is the LeRobot-supported fallback.
    import os

    return os.environ.get("LEROBOT_VIDEO_BACKEND", "pyav")


def _load_dataset(repo_id: str, episodes: list[int] | None, video_backend: str | None = None):
    LeRobotDataset, HF_LEROBOT_HOME = _load_lerobot()
    root = HF_LEROBOT_HOME / repo_id
    backend = video_backend or _default_video_backend()
    # Always pin root so retries land in a stable folder and use local_dir copy
    # mode (safer on Windows without Developer Mode / symlink privilege).
    kwargs: dict = {"root": root, "video_backend": backend}
    if (root / "meta" / "info.json").is_file():
        print(f"Loading local dataset at {root} (video_backend={backend})")
    else:
        print(f"Downloading dataset to {root} from Hub: {repo_id}")
        print(
            "(First sync pulls meta; later batches only fetch episode videos. "
            "Hub copies files on Windows. Resume after 429 — do not delete cache.)"
        )
    if episodes is not None:
        kwargs["episodes"] = episodes

    try:
        return LeRobotDataset(repo_id, **kwargs)
    except ValueError as exc:
        # Local cache may only have early episodes (e.g. 0-49). Filtering to a
        # missing episode raises HF Datasets "Instruction train corresponds to
        # no data!" instead of triggering a Hub fetch. Force sync those files.
        msg = str(exc)
        if "corresponds to no data" not in msg:
            raise
        print(
            f"  Episode {episodes} not in local parquet cache — "
            "downloading missing data/videos from Hub..."
        )
        kwargs["force_cache_sync"] = True
        return LeRobotDataset(repo_id, **kwargs)


def _parse_episodes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _pick_episode_frame_ids(
    n_frames: int,
    num_timestamps: int,
    stride: int,
    *,
    shuffle: bool,
    seed: int,
    frame_offset: int = 0,
) -> list[int]:
    """Pick exactly ``num_timestamps`` distinct frame indices inside the episode.

    Does **not** always include frame 0. Positions are spread through the clip and
    shifted by ``frame_offset`` so later batches from the same / next episodes vary.
    """
    if n_frames <= 0 or num_timestamps <= 0:
        return []
    k = min(int(num_timestamps), int(n_frames))
    stride = max(1, int(stride))
    n = int(n_frames)

    if shuffle:
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), k=k))

    # Shift so consecutive batches don't always land on the opening pose.
    # Bias by ~1/3 episode so a fresh cursor (offset=0, seed=0) is not frame 0.
    shift = int(frame_offset + seed * 17 + max(1, n // 3)) % n

    if k == 1:
        fr = shift
        if stride > 1:
            fr = (fr // stride) * stride
        if fr <= 0 and n > 1:
            fr = min(n - 1, stride)
        return [min(n - 1, fr)]

    # k interior landmarks (e.g. k=2 -> ~33% and ~67%), then rotate by shift.
    ids: list[int] = []
    for i in range(k):
        # fractions 1/(k+1), 2/(k+1), ... avoid endpoints 0 and n-1 by default
        frac = (i + 1) / (k + 1)
        fr = int(round(frac * (n - 1)))
        fr = (fr + shift) % n
        ids.append(fr)

    # Deduplicate while preserving order; fill gaps if collision on short clips.
    out: list[int] = []
    seen: set[int] = set()
    for fr in ids:
        if fr not in seen:
            out.append(fr)
            seen.add(fr)
    probe = 0
    while len(out) < k and probe < n:
        cand = (shift + probe * max(1, n // (k + 1))) % n
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
        probe += 1
    return sorted(out)


def _opencv_read_frames(video_path: Path, abs_frame_ids: list[int]) -> dict[int, np.ndarray]:
    """Read specific absolute frame indices from an mp4.

    OpenCV ``CAP_PROP_POS_FRAMES`` seek is unreliable on many H.264 files (lands on
    the nearest keyframe / start). Seek a little before the first target, then
    decode forward so each requested index is the true frame.
    """
    if not abs_frame_ids:
        return {}
    wanted = sorted(set(int(i) for i in abs_frame_ids if int(i) >= 0))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: cannot open video {video_path}")
        return {}
    out: dict[int, np.ndarray] = {}
    want_set = set(wanted)
    first = wanted[0]
    # Back up ~1s so we land on/before a keyframe, then scan to targets.
    seek_to = max(0, first - 30)
    if seek_to > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        # After a flaky seek, trust sequential counting from wherever we landed
        # only if the reported position looks sane; otherwise restart from 0.
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if landed < 0 or abs(landed - seek_to) > 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            i = 0
        else:
            i = landed
    else:
        i = 0

    max_f = wanted[-1]
    while i <= max_f and len(out) < len(want_set):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want_set:
            out[i] = bgr
        i += 1

    # If seek skipped past a target, fall back to a full scan from 0.
    missing = [f for f in wanted if f not in out]
    if missing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        need = set(missing)
        while i <= max_f and need:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in need:
                out[i] = bgr
                need.discard(i)
            i += 1

    cap.release()
    missing = [f for f in wanted if f not in out]
    if missing:
        print(
            f"  Warning: missing frames {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} in {video_path.name}"
        )
    return out


def _episode_index_from_ds(ds) -> int:
    if getattr(ds, "episodes", None):
        return int(ds.episodes[0])
    row = ds.hf_dataset[0]
    ep = row["episode_index"]
    if isinstance(ep, torch.Tensor):
        return int(ep.item())
    return int(ep)


def _load_batch_state(path: Path) -> dict:
    if not path.is_file():
        return {
            "next_episode": 0,
            "total_images_saved": 0,
            "batches_completed": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "next_episode": 0,
            "total_images_saved": 0,
            "batches_completed": 0,
        }


def _save_batch_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def sample_from_dataset(
    ds,
    *,
    images_dir: Path,
    cameras: list[str],
    num_frames: int,
    stride: int,
    timestamps_per_episode: int = 1,
    frame_offset: int = 0,
    shuffle: bool = False,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Write up to ``num_frames`` JPEGs using OpenCV on a few chosen frame indices."""
    images_dir.mkdir(parents=True, exist_ok=True)
    n = int(ds.num_frames)
    if n <= 0:
        return {
            "saved": 0,
            "skipped_existing": 0,
            "skipped_pad": 0,
            "checked": 0,
            "last_episode": None,
        }

    ep_idx = _episode_index_from_ds(ds)
    ep_meta = ds.meta.episodes[ep_idx]
    fps = float(ds.meta.fps)
    n_ts = max(1, int(timestamps_per_episode))
    frame_ids = _pick_episode_frame_ids(
        n, n_ts, stride, shuffle=shuffle, seed=seed + ep_idx, frame_offset=frame_offset
    )

    print(
        f"Episode {ep_idx}: {len(frame_ids)} timestamp(s) -> up to "
        f"{len(frame_ids) * len(cameras)} images (cap {num_frames}), "
        f"frames={frame_ids}, cameras={cameras}"
    )

    saved = 0
    skipped_existing = 0
    skipped_pad = 0
    checked = 0
    prev_mean: float | None = None
    identical_warn = 0

    for cam in cameras:
        if saved >= num_frames:
            break
        key = _cam_key(cam)
        if key not in ds.meta.features:
            print(f"Warning: missing camera feature {key}")
            continue

        video_rel = ds.meta.get_video_file_path(ep_idx, key)
        video_path = Path(ds.root) / video_rel
        if not video_path.is_file():
            print(f"  Missing video file {video_path}, skipping camera {cam}")
            continue

        from_key = f"videos/{key}/from_timestamp"
        from_ts = float(ep_meta[from_key]) if from_key in ep_meta else 0.0
        start_f = int(round(from_ts * fps))
        abs_ids = [start_f + int(fr) for fr in frame_ids]
        frames_bgr = _opencv_read_frames(video_path, abs_ids)

        for fr, abs_f in zip(frame_ids, abs_ids):
            if saved >= num_frames:
                break
            checked += 1

            # Padding mask from parquet row (no video decode).
            try:
                row = ds.hf_dataset[int(fr)]
            except Exception:
                row = None
            if row is not None and not _mask_is_real(row, cam):
                skipped_pad += 1
                continue

            fname = _frame_name(ep_idx, int(fr), cam)
            path = images_dir / fname
            if path.exists() and not overwrite:
                skipped_existing += 1
                continue

            bgr = frames_bgr.get(abs_f)
            if bgr is None:
                continue

            mean = float(np.mean(bgr))
            if prev_mean is not None and abs(mean - prev_mean) < 0.15:
                identical_warn += 1
            prev_mean = mean

            cv2.imwrite(str(path), bgr)
            saved += 1
            if saved % 10 == 0 or saved == num_frames:
                print(f"  saved {saved}/{num_frames} ({cam} f={fr})")

    if identical_warn >= max(3, saved // 2) and saved > 0:
        print(
            "  Warning: many consecutive frames look nearly identical. "
            "Try a larger --stride or check the source video."
        )

    print(
        f"Done. saved={saved}, skipped_existing={skipped_existing}, "
        f"skipped_padded={skipped_pad}, checked_frames={checked}"
    )
    return {
        "saved": saved,
        "skipped_existing": skipped_existing,
        "skipped_pad": skipped_pad,
        "checked": checked,
        "last_episode": ep_idx,
    }


def sample_batch_episode_by_episode(
    *,
    repo_id: str,
    out_dir: Path,
    cameras: list[str],
    batch_size: int,
    stride: int,
    timestamps_per_episode: int,
    start_episode: int,
    max_episodes: int,
    shuffle: bool,
    seed: int,
    frame_offset: int = 0,
    overwrite: bool,
    video_backend: str | None = None,
) -> dict:
    """Pull at most ``batch_size`` new images, loading one HF episode at a time."""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_total = 0
    ep = int(start_episode)
    end_ep = int(start_episode) + max(1, int(max_episodes))
    episodes_touched: list[int] = []

    print(
        f"\n=== Pull batch: up to {batch_size} images, "
        f"episodes {start_episode}..{end_ep - 1} (one episode per Hub fetch) ==="
    )

    while saved_total < batch_size and ep < end_ep:
        remaining = batch_size - saved_total
        print(f"\n-- Episode {ep} (need {remaining} more images) --", flush=True)
        try:
            ds = _load_dataset(repo_id, [ep], video_backend=video_backend)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed to load episode {ep}: {exc}")
            ep += 1
            continue

        if ds.num_frames <= 0:
            print(f"  Episode {ep} has 0 frames, skipping")
            ep += 1
            continue

        stats = sample_from_dataset(
            ds,
            images_dir=images_dir,
            cameras=cameras,
            num_frames=remaining,
            stride=stride,
            timestamps_per_episode=timestamps_per_episode,
            frame_offset=frame_offset + ep * stride,
            shuffle=shuffle,
            seed=seed + ep,
            overwrite=overwrite,
        )
        if stats["saved"] > 0 or stats["skipped_existing"] > 0:
            episodes_touched.append(ep)
        saved_total += int(stats["saved"])
        # Always advance so we do not re-download the same episode next batch.
        ep += 1

    print(f"\nBatch pull finished: {saved_total} new images; next_episode={ep}")
    return {
        "saved": saved_total,
        "next_episode": ep,
        "episodes_touched": episodes_touched,
    }


def cmd_sample(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    episodes = _parse_episodes(args.episodes)

    if getattr(args, "episode_by_episode", False) or (
        episodes is not None and len(episodes) > 1 and args.num_frames <= 50
    ):
        # Prefer one-episode loads when sampling a small batch over a range.
        if episodes is None:
            start, count = 0, max(1, args.num_frames)  # fall back
            max_eps = count
        else:
            start = episodes[0]
            max_eps = len(episodes)
        stats = sample_batch_episode_by_episode(
            repo_id=args.repo_id,
            out_dir=out_dir,
            cameras=cameras,
            batch_size=args.num_frames,
            stride=args.stride,
            timestamps_per_episode=args.timestamps_per_episode,
            start_episode=start,
            max_episodes=max_eps,
            shuffle=args.shuffle,
            seed=args.seed,
            overwrite=args.overwrite,
            video_backend=args.video_backend,
        )
        print(f"Images: {out_dir / 'images'}")
        print(f"Next episode hint: {stats['next_episode']}")
        return

    ds = _load_dataset(args.repo_id, episodes, video_backend=args.video_backend)
    sample_from_dataset(
        ds,
        images_dir=out_dir / "images",
        cameras=cameras,
        num_frames=args.num_frames,
        stride=args.stride,
        timestamps_per_episode=args.timestamps_per_episode,
        shuffle=args.shuffle,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Images: {out_dir / 'images'}")


def _load_labeled_paths(labels_path: Path) -> set[str]:
    labeled: set[str] = set()
    if not labels_path.exists():
        return labeled
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            img = row.get("image")
            if img:
                labeled.add(img.replace("\\", "/"))
    return labeled


class PointLabeler:
    """OpenCV single-click point labeler."""

    def __init__(self, window: str = "locate_label"):
        self.window = window
        self.point: tuple[int, int] | None = None
        self.base: np.ndarray | None = None
        self.view: np.ndarray | None = None

    def set_image(self, bgr: np.ndarray) -> None:
        self.base = bgr.copy()
        self.point = None
        self._redraw()

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        assert self.base is not None
        h, w = self.base.shape[:2]
        # The label window is autosized to the native image dimensions, so HighGUI
        # callback coordinates already refer to source-image pixels.
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    def _redraw(self) -> None:
        assert self.base is not None
        vis = self.base.copy()
        if self.point is not None:
            x, y = self.point
            cv2.drawMarker(vis, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(vis, (x, y), 5, (0, 255, 0), 2)
        self.view = vis
        cv2.imshow(self.window, vis)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self.base is not None and event == cv2.EVENT_LBUTTONDOWN:
            self.point = self._clamp(x, y)
            self._redraw()

    def clear_point(self) -> None:
        self.point = None
        self._redraw()


def _parse_meta_from_name(name: str) -> tuple[int | None, int | None, str | None]:
    # ep000012_f000340_top.jpg
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        ep = int(parts[0][2:])
        fr = int(parts[1][1:])
        cam = "_".join(parts[2:])
        return ep, fr, cam
    except ValueError:
        return None, None, None


def run_label_ui(
    *,
    out_dir: Path,
    phrase: str,
    relabel_all: bool = False,
    only_unlabeled: bool = True,
) -> str:
    """OpenCV label loop. Returns 'done' | 'quit' | 'empty'."""
    images_dir = out_dir / "images"
    # Keep legacy box annotations intact; point annotations have their own file.
    labels_path = out_dir / POINT_LABELS_NAME
    if not images_dir.is_dir():
        raise SystemExit(f"No images directory at {images_dir}. Run `sample`/`batch` first.")

    images = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not images:
        return "empty"

    labeled = _load_labeled_paths(labels_path)
    if relabel_all:
        pending = images
        labeled = set()
    elif only_unlabeled:
        pending = [p for p in images if f"images/{p.name}" not in labeled]
    else:
        pending = images

    print(f"Images: {len(images)} total, {len(pending)} to label, phrase={phrase!r}")
    print("Click the center of the gripper jaws.")
    print("Controls: click=mark | s=save | n=skip | u=clear point | q=quit")

    if not pending:
        print("Nothing to label.")
        return "empty"

    labeler = PointLabeler()
    cv2.namedWindow(labeler.window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(labeler.window, labeler.on_mouse)

    saved = 0
    skipped = 0
    i = 0
    while i < len(pending):
        path = pending[i]
        rel = f"images/{path.name}"
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"Failed to read {path}, skipping")
            i += 1
            continue

        labeler.set_image(bgr)
        title = f"[{i + 1}/{len(pending)}] {path.name}"
        cv2.setWindowTitle(labeler.window, title)
        print(title)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Quit. saved={saved}, skipped={skipped}, remaining={len(pending) - i}")
                return "quit"
            if key == ord("n"):
                skipped += 1
                i += 1
                break
            if key == ord("u"):
                labeler.clear_point()
                continue
            if key == ord("s"):
                point = labeler.point
                if point is None:
                    print("  Click the gripper point before saving (s).")
                    continue
                ep, fr, cam = _parse_meta_from_name(path.name)
                h, w = bgr.shape[:2]
                x, y = point
                row = {
                    "image": rel.replace("\\", "/"),
                    "phrase": phrase,
                    "point_xy": [int(x), int(y)],
                    "point_xy_norm": [float(x / max(1, w - 1)), float(y / max(1, h - 1))],
                    "point_target": "center_of_gripper_jaws",
                    "width": w,
                    "height": h,
                    "episode": ep,
                    "frame": fr,
                    "camera": cam,
                }
                with labels_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                labeled.add(rel)
                saved += 1
                print(f"  saved point {row['point_xy']} -> {labels_path.name}")
                i += 1
                break

    cv2.destroyAllWindows()
    print(f"Done. saved={saved}, skipped={skipped}")
    print(f"Labels: {labels_path}")
    return "done"


def cmd_label(args: argparse.Namespace) -> None:
    status = run_label_ui(
        out_dir=Path(args.out),
        phrase=args.phrase,
        relabel_all=args.relabel_all,
    )
    if status == "empty" and not (Path(args.out) / "images").is_dir():
        raise SystemExit(f"No images directory at {Path(args.out) / 'images'}.")
    if getattr(args, "push_to_hub", False):
        from locate_push_hf import push_locate_gripper

        push_locate_gripper(
            Path(args.out),
            args.hf_repo_id,
            private=bool(args.private),
            include_sft=True,
        )


def cmd_push(args: argparse.Namespace) -> None:
    from locate_push_hf import push_locate_gripper

    push_locate_gripper(
        Path(args.out),
        args.hf_repo_id,
        private=bool(args.private),
        include_sft=not bool(getattr(args, "no_sft", False)),
        commit_message=args.commit_message,
    )


def _maybe_push_locate(args: argparse.Namespace, out_dir: Path) -> None:
    if not getattr(args, "push_to_hub", False):
        return
    from locate_push_hf import push_locate_gripper

    push_locate_gripper(
        out_dir,
        args.hf_repo_id,
        private=bool(getattr(args, "private", False)),
        include_sft=True,
    )


def cmd_batch(args: argparse.Namespace) -> None:
    """Pull a small batch of frames, label them, repeat until quit."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / BATCH_STATE_NAME
    state = _load_batch_state(state_path)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    if args.reset_cursor:
        state["next_episode"] = int(args.start_episode)
        print(f"Reset next_episode -> {state['next_episode']}")

    next_ep = int(state.get("next_episode", args.start_episode))
    if args.start_episode and not args.reset_cursor and state.get("next_episode") is None:
        next_ep = int(args.start_episode)

    batch_idx = int(state.get("batches_completed", 0))
    frame_offset = int(state.get("frame_offset", batch_idx * int(args.stride)))
    print(
        f"Batch labeling | batch_size={args.batch_size} images | "
        f"timestamps/ep={args.timestamps_per_episode} | stride={args.stride} | "
        f"start episode={next_ep}"
    )
    print(
        "Each round: one episode, a few frame indices (not the whole clip), "
        "then label UI.\n"
    )

    while True:
        batch_idx += 1
        print(f"\n########## BATCH {batch_idx} ##########")
        pull = sample_batch_episode_by_episode(
            repo_id=args.repo_id,
            out_dir=out_dir,
            cameras=cameras,
            batch_size=args.batch_size,
            stride=args.stride,
            timestamps_per_episode=args.timestamps_per_episode,
            start_episode=next_ep,
            max_episodes=args.max_episodes_per_batch,
            shuffle=args.shuffle,
            seed=args.seed + batch_idx,
            frame_offset=frame_offset,
            overwrite=False,
            video_backend=args.video_backend,
        )
        frame_offset += args.stride
        next_ep = int(pull["next_episode"])
        state.update(
            {
                "repo_id": args.repo_id,
                "next_episode": next_ep,
                "frame_offset": frame_offset,
                "total_images_saved": int(state.get("total_images_saved", 0))
                + int(pull["saved"]),
                "batches_completed": batch_idx,
                "cameras": args.cameras,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "timestamps_per_episode": args.timestamps_per_episode,
                "last_episodes": pull["episodes_touched"],
            }
        )
        _save_batch_state(state_path, state)

        if pull["saved"] == 0:
            print(
                "No new images this pull (load/download may have failed, or "
                "files already existed). Label UI will only show older unlabeled "
                f"leftovers — not episode {args.start_episode} unless files were saved."
            )

        status = run_label_ui(out_dir=out_dir, phrase=args.phrase)
        if status == "quit":
            print(f"Stopped. Resume later with the same command (next_episode={next_ep}).")
            print(f"State: {state_path}")
            _maybe_push_locate(args, out_dir)
            return

        labeled_n = len(_load_labeled_paths(out_dir / POINT_LABELS_NAME))
        print(
            f"\nBatch {batch_idx} review done. "
            f"labels={labeled_n}, next_episode={next_ep}, state={state_path}"
        )

        if args.auto_continue:
            continue

        try:
            ans = input("Pull next batch and review? [Y/n/q] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("n", "no", "q", "quit"):
            print(f"Stopped. Resume with: python src/locate_collect_label.py batch --out {out_dir}")
            _maybe_push_locate(args, out_dir)
            return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample HF teleop frames and click-label SO-101 gripper points."
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sample", help="Sample frames from a LeRobot HF dataset into data/locate_gripper/images")
    sp.add_argument("--repo-id", default=DEFAULT_REPO, help="Hugging Face LeRobot dataset id")
    sp.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root directory")
    sp.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS), help="Comma-separated camera names")
    sp.add_argument("--num-frames", type=int, default=4, help="Max images to write (keep small)")
    sp.add_argument(
        "--timestamps-per-episode",
        type=int,
        default=1,
        help="Distinct moments per episode (each moment × each camera = images)",
    )
    sp.add_argument(
        "--stride",
        type=int,
        default=30,
        help="When timestamps=1, which frame index to pick (offset advances per batch)",
    )
    sp.add_argument(
        "--episodes",
        default=None,
        help="Optional episode filter, e.g. '0,1,5' or '0-20'",
    )
    sp.add_argument(
        "--episode-by-episode",
        action="store_true",
        help="Load one episode at a time (smaller Hub video pulls)",
    )
    sp.add_argument("--shuffle", action="store_true", help="Shuffle candidate indices before sampling")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--overwrite", action="store_true", help="Rewrite existing JPEGs")
    sp.add_argument(
        "--video-backend",
        default=_default_video_backend(),
        choices=["pyav", "torchcodec"],
        help="Video decode backend (default pyav — avoids broken torchcodec/FFmpeg on Windows)",
    )
    sp.set_defaults(func=cmd_sample)

    lp = sub.add_parser("label", help="OpenCV single-click UI to write point_labels.jsonl")
    lp.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root with images/")
    lp.add_argument("--phrase", default=DEFAULT_PHRASE, help="Description associated with each labeled point")
    lp.add_argument("--relabel-all", action="store_true", help="Ignore existing point_labels.jsonl entries")
    lp.add_argument("--push-to-hub", action="store_true", help="Upload locate_gripper after labeling")
    lp.add_argument("--hf-repo-id", default=DEFAULT_HF_PUSH_REPO, help="Destination HF dataset repo")
    lp.add_argument("--private", action="store_true", help="Private HF dataset")
    lp.set_defaults(func=cmd_label)

    pp = sub.add_parser("push", help="Upload data/locate_gripper to Hugging Face Hub")
    pp.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pp.add_argument("--hf-repo-id", default=DEFAULT_HF_PUSH_REPO, help="Destination HF dataset repo")
    pp.add_argument("--private", action="store_true")
    pp.add_argument("--no-sft", action="store_true", help="Skip locate_sft.jsonl / recipe.json")
    pp.add_argument("--commit-message", default=None)
    pp.set_defaults(func=cmd_push)

    bp = sub.add_parser(
        "batch",
        help="Pull N images (episode-by-episode), label them, repeat until quit",
    )
    bp.add_argument("--repo-id", default=DEFAULT_REPO, help="Source LeRobot HF dataset to sample from")
    bp.add_argument("--out", type=Path, default=DEFAULT_OUT)
    bp.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    bp.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Max new images per round (default: 1 timestamp × top+side = 2)",
    )
    bp.add_argument(
        "--timestamps-per-episode",
        type=int,
        default=1,
        help="Moments to sample per episode (1 = one still, two cameras -> 2 JPEGs)",
    )
    bp.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Frame index step when picking the single timestamp (varies by batch cursor)",
    )
    bp.add_argument(
        "--max-episodes-per-batch",
        type=int,
        default=1,
        help="Episodes to open per round (1 = one Hub video fetch per batch)",
    )
    bp.add_argument("--start-episode", type=int, default=0, help="Episode cursor if no state file")
    bp.add_argument("--reset-cursor", action="store_true", help="Reset batch_state next_episode")
    bp.add_argument("--phrase", default=DEFAULT_PHRASE)
    bp.add_argument("--shuffle", action="store_true")
    bp.add_argument("--seed", type=int, default=0)
    bp.add_argument(
        "--video-backend",
        default=_default_video_backend(),
        choices=["pyav", "torchcodec"],
        help="Video decode backend (default pyav — avoids broken torchcodec/FFmpeg on Windows)",
    )
    bp.add_argument(
        "--auto-continue",
        action="store_true",
        help="Do not prompt between batches (Ctrl+C or q in label UI to stop)",
    )
    bp.add_argument(
        "--push-to-hub",
        action="store_true",
        help="When you quit batch mode, upload locate_gripper to Hugging Face",
    )
    bp.add_argument(
        "--hf-repo-id",
        default=DEFAULT_HF_PUSH_REPO,
        help="Destination HF dataset repo for --push-to-hub",
    )
    bp.add_argument("--private", action="store_true", help="Private HF dataset")
    bp.set_defaults(func=cmd_batch)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
