"""Push local ``data/locate_gripper`` labels + images to a Hugging Face dataset repo.

Examples:
  python src/locate_push_hf.py
  python src/locate_push_hf.py --repo-id kdaterao/so101_locate_gripper --private
  python src/locate_finetune.py export --push-to-hub
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "locate_gripper"
DEFAULT_REPO = "kdaterao/so101_locate_gripper"

# Local-only cursor; do not upload by default.
_SKIP_NAMES = frozenset({"batch_state.json", ".DS_Store", "Thumbs.db"})


def _count_labels(labels_path: Path) -> int:
    if not labels_path.is_file():
        return 0
    n = 0
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _write_readme(data_root: Path, repo_id: str, n_labels: int, n_images: int) -> Path:
    readme = data_root / "README.md"
    body = f"""---
license: apache-2.0
task_categories:
  - object-detection
  - visual-question-answering
tags:
  - robotics
  - so101
  - gripper
  - locateanything
  - grounding
pretty_name: SO-101 Locate Gripper
---

# {repo_id}

SO-101 gripper bounding-box labels for LocateAnything / Florence-2 grounding.

| | |
|---|---|
| Labels | {n_labels} |
| Images | {n_images} |

## Layout

```
images/           # JPEG frames (epXXXXXX_fXXXXXX_{{top,side}}.jpg)
labels.jsonl      # click-drag boxes (xyxy pixels)
locate_sft.jsonl  # optional Eagle / LocateAnything ShareGPT export
recipe.json       # optional Eagle recipe pointing at this folder
```

## Label row schema

```json
{{
  "image": "images/ep000500_f000120_top.jpg",
  "phrase": "SO-101 gripper",
  "box_xyxy": [x1, y1, x2, y2],
  "width": 640,
  "height": 480,
  "episode": 500,
  "frame": 120,
  "camera": "top"
}}
```

Collected with [`locate_collect_label.py`](https://github.com/) batch labeling. Fine-tune Florence-2 with:

```bash
python src/florence2_finetune.py train --dataset-repo {repo_id}
```
"""
    readme.write_text(body, encoding="utf-8")
    return readme


def push_locate_gripper(
    data_root: Path,
    repo_id: str,
    *,
    private: bool = False,
    include_sft: bool = True,
    write_readme: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload locate_gripper folder to Hub. Returns the dataset URL."""
    try:
        from hf_hub_windows import prepare_hf_hub_env

        prepare_hf_hub_env()
    except ImportError:
        pass

    from huggingface_hub import HfApi, create_repo

    data_root = Path(data_root).resolve()
    images_dir = data_root / "images"
    labels_path = data_root / "labels.jsonl"

    if not labels_path.is_file():
        raise SystemExit(f"Missing {labels_path}. Label some frames first.")
    if not images_dir.is_dir():
        raise SystemExit(f"Missing {images_dir}.")

    n_labels = _count_labels(labels_path)
    n_images = sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if n_labels == 0:
        raise SystemExit(f"No labels in {labels_path}")

    if write_readme:
        _write_readme(data_root, repo_id, n_labels, n_images)

    allow_patterns = ["images/**", "labels.jsonl", "README.md"]
    if include_sft:
        allow_patterns.extend(["locate_sft.jsonl", "recipe.json"])

    ignore_patterns = sorted(_SKIP_NAMES)

    print(f"Pushing {data_root}")
    print(f"  labels={n_labels} images={n_images} -> {repo_id} (private={private})")

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    msg = commit_message or f"Update locate_gripper ({n_labels} labels, {n_images} images)"
    api.upload_folder(
        folder_path=str(data_root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=msg,
    )

    # Keep recipe.json Hub-friendly: annotation/root as repo-relative paths in a sidecar.
    hub_meta = {
        "repo_id": repo_id,
        "n_labels": n_labels,
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
    p.add_argument(
        "--no-sft",
        action="store_true",
        help="Do not upload locate_sft.jsonl / recipe.json even if present",
    )
    p.add_argument("--commit-message", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    push_locate_gripper(
        args.data_root,
        args.repo_id,
        private=args.private,
        include_sft=not args.no_sft,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
