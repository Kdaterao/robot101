"""Bundle worker datasets and join captions, aggregation and grounding by identity."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from robot101.data.utilities.annotation_tree import atomic_json, fingerprint


def rate_limit_delay(exc, minimum):
    """Return a conservative wait for a 429, including wrapped download errors."""
    from robot101.data.utilities.hub_downloads import _is_rate_limit
    current, seen = exc, set()
    limited, delay = False, minimum
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        limited |= _is_rate_limit(current)
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry = headers.get("Retry-After", headers.get("retry-after", ""))
        try:
            seconds = float(retry)
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(retry)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 0
        if math.isfinite(seconds):
            delay = max(delay, seconds)
        rate = headers.get("RateLimit", headers.get("ratelimit", ""))
        match = re.search(r"(?:^|;)\s*t=(\d+)", rate)
        if match:
            delay = max(delay, float(match[1]) + 1)
        current = current.__cause__ or current.__context__
    return delay if limited else None


def with_rate_limit_pause(operation, *, minutes, retries, label):
    """Retry the same operation; successful snapshot files remain in the Hub cache."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            wait = rate_limit_delay(exc, minutes * 60)
            if wait is None or (retries and attempt >= retries):
                raise
            attempt += 1
            print(f"[hf] Rate limited during {label}. Pausing {wait / 60:.1f} minutes; "
                  f"automatic retry {attempt}. Completed downloads are retained.", flush=True)
            deadline = time.monotonic() + wait
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(60, remaining))
                remaining = deadline - time.monotonic()
                if remaining > 1:
                    print(f"[hf] Paused: {remaining / 60:.1f} minutes remaining", flush=True)
            print(f"[hf] Resuming {label}…", flush=True)


def identity(record, episode=None):
    key = (record.get("source_repo_id"), record.get("source_revision"),
           record.get("episode_index", record.get("episode", episode)))
    if not key[0] or not key[1] or type(key[2]) is not int:
        raise ValueError(f"Missing pinned source identity: {key}")
    return key


def collect(roots):
    """Index artifacts without relying on worker-local destination episode IDs."""
    episodes = {}
    for root, prefix in roots:
        for stage in ("captioning", "aggregation"):
            for path in sorted((root / stage / "episodes").glob("ep*.json")):
                record = json.loads(path.read_text())
                key = identity(record)
                item = episodes.setdefault(key, {})
                if stage in item:
                    raise ValueError(f"Duplicate {stage} episode {key}; supply disjoint worker outputs")
                item[stage] = (record, f"{prefix}/{path.relative_to(root).as_posix()}")
        for path in sorted((root / "point_tracks").glob("ep*.json")):
            record = json.loads(path.read_text())
            key = identity(record)
            item = episodes.setdefault(key, {})
            if "grounding" in item:
                raise ValueError(f"Duplicate grounding episode {key}")
            item["grounding"] = (record, f"{prefix}/{path.relative_to(root).as_posix()}")
            item["grounding_dataset"] = f"{prefix}"
    if not episodes:
        raise ValueError("No captioning, aggregation or point_tracks episode artifacts found")
    return episodes


def join(episodes):
    rows, reviews = [], []
    for key, item in sorted(episodes.items()):
        trees = [item[s][0] for s in ("captioning", "aggregation") if s in item]
        ground = item.get("grounding", (None, None))[0]
        if trees:
            base = trees[0]
            for tree in trees:
                lookup = {n["id"]: n for n in tree["nodes"]}
                ids = tree["selected_node_ids"]
                if (tree.get("bounds") != "half_open_source_frames" or not ids
                        or len(lookup) != len(tree["nodes"]) or len(ids) != len(set(ids))
                        or any(i not in lookup for i in ids)):
                    raise ValueError(f"Invalid selected annotation cut for {key}")
                cursor = 0
                for n in sorted((lookup[i] for i in ids), key=lambda n: n["start_frame"]):
                    if n["start_frame"] != cursor or n["end_frame"] <= cursor:
                        raise ValueError(f"Gaps or overlaps in annotation cut for {key}")
                    cursor = n["end_frame"]
                if cursor != tree["n_frames"]:
                    raise ValueError(f"Incomplete annotation cut for {key}")
            for tree in trees[1:]:
                if (tree["n_frames"] != base["n_frames"] or tree["fps"] != base["fps"]
                        or tree["bounds"] != base["bounds"]
                        or set(tree["selected_node_ids"]) != set(base["selected_node_ids"])):
                    raise ValueError(f"Incompatible annotation cuts for {key}")
            selected = set(base["selected_node_ids"])
            nodes = {n["id"]: n for n in base["nodes"] if n["id"] in selected}
            for tree in trees:
                for n in tree["nodes"]:
                    if n["id"] in nodes and (n["start_frame"], n["end_frame"]) != (
                            nodes[n["id"]]["start_frame"], nodes[n["id"]]["end_frame"]):
                        raise ValueError(f"Changed node bounds in {key}: {n['id']}")
        else:
            nodes = {}
        grounding = {}
        for stage in ground["subtasks"] if ground else []:
            annotation = stage.get("annotation")
            if not annotation or not annotation.get("node_id"):
                raise ValueError(f"Grounding {key} has no annotation node identity; raw gripper cuts cannot be joined")
            node_id = annotation["node_id"]
            bounds = (stage["start_frame"], stage["end_frame"] + 1)
            parent_bounds = (annotation["start_frame"], annotation["end_frame"])
            if not parent_bounds[0] <= bounds[0] < bounds[1] <= parent_bounds[1]:
                raise ValueError(f"Grounding bounds disagree with annotation for {key}/{node_id}")
            if bounds != parent_bounds and stage.get("interval_role") != "grounding_only":
                raise ValueError(f"Unmarked grounding subdivision for {key}/{node_id}")
            grounding.setdefault(node_id, []).append(stage)
            if node_id not in nodes:
                if trees:
                    raise ValueError(f"Grounding node outside selected cut: {key}/{node_id}")
                nodes[node_id] = dict(id=node_id, start_frame=parent_bounds[0], end_frame=parent_bounds[1], caption={})
            if parent_bounds != (nodes[node_id]["start_frame"], nodes[node_id]["end_frame"]):
                raise ValueError(f"Grounding parent annotation changed for {key}/{node_id}")
        aggregated = {n["id"]: n.get("aggregation", {})
                      for n in item.get("aggregation", ({"nodes": []}, None))[0]["nodes"]}
        for node_id, node in sorted(nodes.items(), key=lambda pair: pair[1]["start_frame"]):
            segments = sorted(grounding.get(node_id, []), key=lambda s: s["start_frame"])
            cursor = node["start_frame"]
            for segment in segments:
                if segment["start_frame"] != cursor:
                    raise ValueError(f"Overlapping or gapped grounding segments: {key}/{node_id}")
                cursor = segment["end_frame"] + 1
            if segments and cursor != node["end_frame"]:
                raise ValueError(f"Grounding does not cover parent interval: {key}/{node_id}")
            stage = (segments[0] if len(segments) == 1 else
                     dict(interval_role="grounding_only_group", segments=segments)) if segments else None
            state = aggregated.get(node_id, {})
            annotation = state.get("annotation", {}) if state.get("status") == "ok" else {}
            reasons = []
            if node.get("caption", {}).get("status") != "ok":
                reasons.append("missing_or_failed_caption")
            if state.get("status") != "ok":
                reasons.append("missing_or_failed_aggregation")
            elif annotation.get("robotics", {}).get("review_status") != "accepted":
                reasons.append("aggregation_requires_review")
            if not stage:
                reasons.append("missing_grounding")
            elif any(s.get("pov", {}).get("status") != "ok"
                  or s.get("pov", {}).get("gripper_filter", {}).get("manual_review_required")
                  or not s.get("third_person")
                  or any(v.get("status") != "ok" for v in s.get("third_person", {}).values()) for s in segments):
                reasons.append("grounding_requires_review")
            row = dict(source_repo_id=key[0], source_revision=key[1], episode_index=key[2],
                       node_id=node_id, start_frame=node["start_frame"], end_frame=node["end_frame"],
                       bounds="half_open_source_frames", caption=node.get("caption", {}).get("text"),
                       brief_action=annotation.get("brief_action"), detailed_action=annotation.get("detailed_action"),
                       actor=annotation.get("actor"),
                       brief_caption=annotation.get("brief_caption"), detailed_caption=annotation.get("detailed_caption"),
                       annotation_json=json.dumps(annotation),
                       grounding_json=json.dumps(stage) if stage else None,
                       grounding_dataset_path=item.get("grounding_dataset"),
                       grounding_destination_episode=ground.get("destination_episode") if ground else None,
                       captioning_path=item.get("captioning", (None, None))[1],
                       aggregation_path=item.get("aggregation", (None, None))[1],
                       grounding_path=item.get("grounding", (None, None))[1],
                       review_required=bool(reasons), review_reasons=reasons)
            rows.append(row)
            if reasons:
                reviews.append({k: row[k] for k in ("source_repo_id", "source_revision", "episode_index",
                                                    "node_id", "review_reasons")})
    return rows, reviews


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--repos", nargs="+", help="All annotation, aggregation and grounded worker Hub repos")
    inputs.add_argument("--base-manifest", type=Path,
                        help="Existing combined manifest; retain pinned inputs and selectively replace worker repos")
    p.add_argument("--replace-repo", action="append", default=[], metavar="OLD=NEW",
                   help="Replace a base-manifest repo; repeat for new grounding workers")
    p.add_argument("--revision", default="main", help="Resolved to an immutable SHA for every input")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/unified_dataset"))
    p.add_argument("--download-workers", type=int, default=4)
    p.add_argument("--rate-limit-wait-minutes", type=float, default=5,
                   help="Pause after HTTP 429; default 5 minutes, or longer if the server requires")
    p.add_argument("--rate-limit-retries", type=int, default=0,
                   help="Retries per Hub operation after rate limits; 0 means unlimited (default)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lightweight", action="store_true",
                   help="Bundle episode JSON and video metadata only; viewer fetches other assets on demand from pinned inputs")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--dst-repo-id")
    p.add_argument("--hub-private", action="store_true")
    args = p.parse_args()
    pinned = {}
    base = None
    if args.base_manifest:
        try:
            base = json.loads(args.base_manifest.read_text())
            originals = {s["repo_id"]: s["revision"] for s in base["inputs"]}
            replacements = {}
            for value in args.replace_repo:
                old, new = value.split("=", 1)
                if old not in originals or old in replacements or not new:
                    raise ValueError(f"Invalid or duplicate replacement: {value}")
                replacements[old] = new
            args.repos = [replacements.get(repo, repo) for repo in originals]
            pinned = {repo: revision for repo, revision in originals.items() if repo not in replacements}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid base manifest/replacements: {exc}")
    elif args.replace_repo:
        p.error("--replace-repo requires --base-manifest")
    if (not math.isfinite(args.rate_limit_wait_minutes) or args.rate_limit_wait_minutes <= 0
            or args.rate_limit_retries < 0):
        p.error("Rate-limit wait must be positive/finite and retries must be nonnegative")
    if args.download_workers < 1 or (args.push_to_hub and not args.dst_repo_id):
        p.error("Positive download workers and --dst-repo-id for uploads are required")
    if len(args.repos) != len(set(args.repos)) or args.dst_repo_id in args.repos:
        p.error("Input repos must be unique and destination must differ from inputs")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    def hub(operation, label):
        return with_rate_limit_pause(operation, minutes=args.rate_limit_wait_minutes,
                                     retries=args.rate_limit_retries, label=label)
    root = args.output_dir
    config = dict(repos=args.repos, revision=args.revision, schema_version=1)
    if base is not None:
        config.update(base_manifest_fingerprint=fingerprint(base), pinned_inputs=pinned)
    if args.lightweight:
        config["lightweight"] = True
    run_path = root / "manifest.json"
    if root.exists() and any(root.iterdir()) and not args.resume:
        p.error("Output exists; use --resume or a new output directory")
    if run_path.exists():
        manifest = json.loads(run_path.read_text())
        if manifest["fingerprint"] != fingerprint(config):
            p.error("Incompatible inputs; use a new output directory")
    else:
        manifest = dict(fingerprint=fingerprint(config), config=config, inputs=[
            dict(repo_id=repo, revision=pinned[repo] if repo in pinned else hub(
                lambda: api.repo_info(repo, repo_type="dataset", revision=args.revision), f"resolve {repo}").sha,
                 path=f"artifacts/input{i:02d}") for i, repo in enumerate(args.repos)])
        atomic_json(run_path, manifest)
    roots = []
    for source in manifest["inputs"]:
        print(f"Downloading {source['repo_id']} at {source['revision']}", flush=True)
        snapshot = Path(hub(lambda: snapshot_download(
            source["repo_id"], repo_type="dataset", revision=source["revision"],
            allow_patterns=(["captioning/episodes/ep*.json", "aggregation/episodes/ep*.json",
                             "point_tracks/ep*.json", "meta/info.json", "meta/episodes/**/*.parquet"]
                            if args.lightweight else None),
            max_workers=args.download_workers), f"download {source['repo_id']}"))
        roots.append((snapshot, source["path"]))
    episodes = collect(roots)
    rows, reviews = join(episodes)
    for snapshot, prefix in roots:
        # Preserve each worker's metadata/video offsets and relative mask references.
        if args.lightweight:
            import fnmatch
            patterns = ["captioning/episodes/ep*.json", "aggregation/episodes/ep*.json",
                        "point_tracks/ep*.json", "meta/info.json", "meta/episodes/**/*.parquet"]
            for path in snapshot.rglob("*"):
                relative = path.relative_to(snapshot).as_posix()
                if path.is_file() and any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                    target = root / prefix / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        else:
            shutil.copytree(snapshot, root / prefix, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".cache", ".git"))
    temporary = root / "subtasks.jsonl.tmp"
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, root / "subtasks.jsonl")
    atomic_json(root / "review.json", reviews)
    atomic_json(root / "summary.json", dict(episodes=len(episodes), subtasks=len(rows), review=len(reviews)))
    card = """---
configs:
  - config_name: subtasks
    data_files:
      - split: train
        path: subtasks.jsonl
---
# Unified robot episode artifacts

Joined caption, Claude annotation and grounding records. Frame bounds are half-open.
`artifacts/` preserves complete input snapshots, including worker videos, actions,
masks and provenance. Paths in each input's sidecars remain relative to that input.
`manifest.json` pins every input revision. Missing/failed stages remain in review.json.
The train split is not a held-out split or quality certification.

This is a unified indexed bundle, not one reindexed LeRobot dataset. Each grounded
worker remains loadable at its `grounding_dataset_path`, using that worker's local
`grounding_destination_episode`. Join training labels by source identity/node ID.
Videos and stored frame task strings are not rewritten. See each input's license.
"""
    if args.lightweight:
        card += ("\n## Lightweight test bundle\n\nVideos, masks and inspection images are referenced, not copied. "
                 "The Streamlit viewer fetches them on demand from the pinned input repos in manifest.json. "
                 "Keep access to those repos. This bundle is not a self-contained media archive.\n")
        card = card.replace("`artifacts/` preserves complete input snapshots, including worker videos, actions,\nmasks and provenance.",
                            "`artifacts/` preserves episode JSON and video metadata only.")
    (root / "README.md").write_text(card)
    if args.push_to_hub:
        hub(lambda: api.create_repo(args.dst_repo_id, repo_type="dataset", exist_ok=True,
                                     private=args.hub_private), "create destination")
        hub(lambda: api.upload_large_folder(repo_id=args.dst_repo_id, repo_type="dataset", folder_path=root,
            ignore_patterns=["*.tmp"]),
            "upload unified dataset")
    print(f"Saved {len(episodes)} episodes, {len(rows)} subtasks; {len(reviews)} require review: {root}")


if __name__ == "__main__":
    main()
