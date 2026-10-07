#!/usr/bin/env python3
"""Download SO-101 point labels and make episode-disjoint Molmo splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="kdaterao/so101_locate_gripper")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")

    local_repo = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.cache_dir,
            allow_patterns=["point_labels.jsonl", "images/**"],
        )
    )
    labels_file = local_repo / "point_labels.jsonl"
    if not labels_file.is_file():
        raise SystemExit(
            f"{args.repo_id} has no point_labels.jsonl. Upload user-clicked point labels first; "
            "legacy boxes in labels.jsonl cannot be used by this point trainer."
        )

    image_dir = local_repo / "images"
    records: list[dict] = []
    seen_images: set[str] = set()
    with labels_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {labels_file}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected a JSON object at {labels_file}:{line_number}")

            image_rel = str(row.get("image", "")).replace("\\", "/")
            image_path = (local_repo / image_rel).resolve()
            if (
                not image_rel.startswith("images/")
                or not image_path.is_relative_to(image_dir.resolve())
                or not image_path.is_file()
            ):
                raise SystemExit(f"Missing or invalid image path at line {line_number}: {image_rel!r}")
            if image_rel in seen_images:
                raise SystemExit(f"Duplicate point labels for {image_rel}; resolve duplicates first")
            seen_images.add(image_rel)

            try:
                width = int(row["width"])
                height = int(row["height"])
                point = row.get("point_xy")
                if point is None:
                    norm = row["point_xy_norm"]
                    x = float(norm[0]) * max(0, width - 1)
                    y = float(norm[1]) * max(0, height - 1)
                else:
                    x, y = float(point[0]), float(point[1])
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise SystemExit(f"Malformed point label at {labels_file}:{line_number}") from exc
            if width <= 0 or height <= 0 or not (0 <= x < width and 0 <= y < height):
                raise SystemExit(f"Out-of-image point at {labels_file}:{line_number}: {(x, y)}")

            episode = row.get("episode")
            if episode is None:
                # Sampled image names use epNNNNNN_fNNNNNN_camera.jpg.
                try:
                    episode = int(Path(image_rel).name.split("_", 1)[0][2:])
                except (ValueError, IndexError) as exc:
                    raise SystemExit(
                        f"Missing episode id at {labels_file}:{line_number}; cannot prevent split leakage"
                    ) from exc
            records.append(
                {
                    "image": str(image_path),
                    "point_xy_100": [100.0 * x / width, 100.0 * y / height],
                    "label": str(row.get("phrase") or "SO-101 gripper"),
                    "episode": int(episode),
                    "camera": str(row.get("camera") or "unknown"),
                    "source_image": image_rel,
                }
            )

    if not records:
        raise SystemExit(f"No point records found in {labels_file}")

    episode_ids = sorted({row["episode"] for row in records})
    validation_episodes: set[int] = set()
    if args.validation_fraction and len(episode_ids) > 1:
        rng = random.Random(args.seed)
        rng.shuffle(episode_ids)
        validation_count = max(1, round(len(episode_ids) * args.validation_fraction))
        validation_count = min(validation_count, len(episode_ids) - 1)
        validation_episodes = set(episode_ids[:validation_count])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split in ("train", "validation"):
        split_rows = [
            row
            for row in records
            if (row["episode"] in validation_episodes) == (split == "validation")
        ]
        output_path = args.output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_counts[split] = len(split_rows)

    if not validation_episodes:
        print("WARNING: fewer than two labeled episodes; validation split is empty.")
    print(
        f"Prepared {args.repo_id}: total={len(records)} "
        f"train={split_counts['train']} validation={split_counts['validation']} "
        f"episodes={len({row['episode'] for row in records})}"
    )
    print(f"Prepared data: {args.output_dir}")
    print(f"Image cache: {image_dir}")


if __name__ == "__main__":
    main()
