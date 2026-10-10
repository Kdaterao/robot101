"""Push local ``data/locate_gripper`` labels + images to a Hugging Face dataset repo.

Examples:
  python -m robot101.data.upload_labels
  python -m robot101.data.upload_labels --repo-id kdaterao/so101_locate_gripper --private
"""

from __future__ import annotations

from robot101.paths import REPO_ROOT

import argparse
import json
from pathlib import Path

DEFAULT_OUT = REPO_ROOT / "data" / "locate_gripper"
DEFAULT_REPO = "kdaterao/so101_locate_gripper"

# Local-only cursor; do not upload by default.
_SKIP_NAMES = frozenset(
    {"batch_state.json", "skipped_images.jsonl", ".DS_Store", "Thumbs.db"}
)


def _count_labels(labels_path: Path) -> int:
    if not labels_path.is_file():
        return 0
    n = 0
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _write_readme(data_root: Path, repo_id: str, n_labels: int, n_images: int, n_points: int) -> Path:
    readme = data_root / "README.md"
    body = f"""---
license: apache-2.0
task_categories:
  - object-detection
  - image-to-text
  - visual-question-answering
tags:
  - robotics
  - so101
  - gripper
  - grounding
pretty_name: SO-101 Gripper Points
---

# {repo_id}

SO-101 gripper point labels for visual point grounding, with legacy bounding-box labels retained.

| | |
|---|---|
| Labels | {n_labels} |
| Point labels | {n_points} |
| Images | {n_images} |

## Layout

```
images/           # JPEG frames (epXXXXXX_fXXXXXX_{{top,side}}.jpg)
point_labels.jsonl # user-clicked gripper points (pixels + normalized coordinates)
labels.jsonl       # legacy click-drag boxes (xyxy pixels), when present
```

## Point label row schema

```json
{{
  "image": "images/ep000500_f000120_top.jpg",
  "phrase": "SO-101 gripper",
  "point_xy": [x, y],
  "point_xy_norm": [x_normalized, y_normalized],
  "point_target": "center_of_gripper_jaws",
  "width": 640,
  "height": 480,
  "episode": 500,
  "frame": 120,
  "camera": "top"
}}
```

Points are collected with a single click using `src/robot101/data/collect_points.py`. Point labels are intended for the custom MolmoPoint fine-tuning workflow.

```bash
python -m robot101.data.collect_points batch
```
"""
    readme.write_text(body, encoding="utf-8")
    return readme


def push_locate_gripper(
    data_root: Path,
    repo_id: str,
    *,
    private: bool = False,
    write_readme: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload locate_gripper folder to Hub. Returns the dataset URL."""
    try:
        from robot101.data.utilities.hub_downloads import prepare_hf_hub_env

        prepare_hf_hub_env()
    except ImportError:
        pass

    from huggingface_hub import HfApi, create_repo

    data_root = Path(data_root).resolve()
    images_dir = data_root / "images"
    labels_path = data_root / "labels.jsonl"
    point_labels_path = data_root / "point_labels.jsonl"

    if not labels_path.is_file() and not point_labels_path.is_file():
        raise SystemExit(f"Missing {labels_path} and {point_labels_path}. Label some frames first.")
    if not images_dir.is_dir():
        raise SystemExit(f"Missing {images_dir}.")

    n_labels = _count_labels(labels_path)
    n_points = _count_labels(point_labels_path)
    n_images = sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if n_labels == 0 and n_points == 0:
        raise SystemExit(f"No labels in {labels_path} or {point_labels_path}")

    if write_readme:
        _write_readme(data_root, repo_id, n_labels, n_images, n_points)

    allow_patterns = ["images/**", "README.md"]
    if labels_path.is_file():
        allow_patterns.append("labels.jsonl")
    if point_labels_path.is_file():
        allow_patterns.append("point_labels.jsonl")

    ignore_patterns = sorted(_SKIP_NAMES)

    print(f"Pushing {data_root}")
    print(f"  box_labels={n_labels} point_labels={n_points} images={n_images} -> {repo_id} (private={private})")

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    msg = commit_message or f"Update locate_gripper ({n_points} point labels, {n_labels} box labels, {n_images} images)"
    api.upload_folder(
        folder_path=str(data_root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=msg,
    )

    # Record uploaded label and image counts in a sidecar.
    hub_meta = {
        "repo_id": repo_id,
        "n_labels": n_labels,
        "n_point_labels": n_points,
        "n_images": n_images,
        "files": allow_patterns,
    }
    meta_path = data_root / "hub_meta.json"
    meta_path.write_text(json.dumps(hub_meta, indent=2) + "\n", encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="hub_meta.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update hub_meta.json",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  Warning: hub_meta upload skipped: {exc}")

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Done: {url}")
    return url


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Push data/locate_gripper to Hugging Face Hub")
    p.add_argument("--data-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--repo-id", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--private", action="store_true", help="Create/keep dataset private")
    p.add_argument("--commit-message", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    push_locate_gripper(
        args.data_root,
        args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
