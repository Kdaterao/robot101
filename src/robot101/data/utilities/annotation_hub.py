"""Publish annotation artifacts and flat, viewer-compatible Hugging Face rows."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def add_hub_arguments(parser):
    parser.add_argument("--push-to-hub", action="store_true", help="Upload completed outputs to a Hugging Face dataset")
    parser.add_argument("--dst-repo-id", help="Destination dataset ID, e.g. kdaterao/so101_action_annotations")
    parser.add_argument("--hub-private", action="store_true", help="Create a new destination as private; existing visibility is unchanged")


def validate_hub_arguments(args):
    if args.push_to_hub and not args.dst_repo_id:
        raise SystemExit("--push-to-hub requires --dst-repo-id OWNER/DATASET")
    if args.push_to_hub and getattr(args, "dry_run", False):
        raise SystemExit("--dry-run cannot be combined with --push-to-hub")


def dataset_rows(records, stage):
    """One row per retained node; keep arbitrary diagnostics in the sidecars."""
    for record in records:
        for node in record["nodes"]:
            if stage == "aggregation" and node.get("aggregation", {}).get("status") == "not_requested":
                continue
            caption = node.get("caption", {})
            row = dict(source_repo_id=record["source_repo_id"], source_revision=record["source_revision"],
                       episode_index=record["episode_index"], node_id=node["id"], parent_id=node["parent"],
                       start_frame=node["start_frame"], end_frame=node["end_frame"],
                       fps=record["fps"], camera=record["camera"], task=record.get("task", ""),
                       selected_subtask=node["id"] in record["selected_node_ids"],
                       caption_kind=node["caption_kind"], caption=caption.get("text"),
                       caption_status=caption.get("status", "pending"),
                       caption_truncated=caption.get("truncated", False))
            if stage == "aggregation":
                state = node.get("aggregation", {})
                annotation = state.get("annotation", {})
                robotics = annotation.get("robotics", {})
                row.update(aggregation_status=state.get("status", "pending"),
                           aggregation_error=state.get("error"),
                           review_status=robotics.get("review_status", "unresolved"),
                           robotics_json=json.dumps(robotics, sort_keys=True))
                for field in ("brief_action", "detailed_action", "actor", "brief_caption", "detailed_caption"):
                    row[field] = annotation.get(field)
                for field in ("manipulated_object", "destination", "completion_condition"):
                    row[field] = robotics.get(field)
            yield row


def publish_annotations(output_dir, repo_id, stage, *, private=False):
    from huggingface_hub import HfApi
    if stage not in {"captioning", "aggregation"}:
        raise ValueError("Unknown annotation upload stage")
    root = Path(output_dir)
    records = [json.loads(path.read_text()) for path in sorted((root / "episodes").glob("ep*.json"))]
    rows = list(dataset_rows(records, stage))
    if not rows:
        print("No annotation nodes to upload; local review artifacts are preserved")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        temporary = stream.name
    os.replace(temporary, root / "dataset_rows.jsonl")
    api = HfApi()  # HF_TOKEN or saved hf auth login; never write tokens to artifacts.
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=root, path_in_repo=stage,
                      allow_patterns=["*.json", "*.jsonl", "*.npz", "*.jpg", "*.html"],
                      ignore_patterns=["request_previews/*", "**/__pycache__/*"],
                      commit_message=f"Upload {stage} episode annotations")
    available = [name for name in ("captioning", "aggregation") if name == stage or
                 api.file_exists(repo_id, f"{name}/dataset_rows.jsonl", repo_type="dataset")]
    configs = "\n".join(f"  - config_name: {name}\n    data_files:\n      - split: train\n"
                        f"        path: {name}/dataset_rows.jsonl" for name in available)
    card = f"""---
pretty_name: SO101 hierarchical episode annotations
tags:
  - robotics
  - video-captioning
  - action100m-adaptation
configs:
{configs}
---

# Hierarchical episode annotations

Action100M-inspired annotation artifacts using V-JEPA 2.1, configurable captioners
(default Qwen2.5-VL video and Qwen3-VL image), and optional Claude aggregation.
This is an adaptation, not a reproduction. Exact models are recorded in run.json.

Each `train` row represents one captioned or requested aggregation node. Context-only
nodes remain in the episode sidecars and are omitted from aggregation rows. `selected_subtask` marks
the controller cut; source frame intervals are half-open `[start_frame, end_frame)`.
The split name does not imply a held-out evaluation split or verified labels.
Source dataset IDs/revisions and episode IDs identify the original demonstrations.

`captioning/` contains caption trees, feature caches, sampled inspection images,
reports and provenance. `aggregation/` contains standardized annotations, API
rounds/usage, review records and provenance. Raw source videos and robot actions
are referenced, not duplicated. These are annotation datasets, not grounded
LeRobot training datasets. No pixel grounding is performed by these commands.

Missing, conflicting, truncated and unresolved outputs require human review.
Desired completion conditions are not observed success labels. Check source-data
and the recorded models' licensing before reuse.
"""
    api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=card.encode(),
                    path_in_repo="README.md", commit_message="Describe annotation dataset configurations")
    print(f"Uploaded {len(rows)} {stage} nodes: https://huggingface.co/datasets/{repo_id}")
