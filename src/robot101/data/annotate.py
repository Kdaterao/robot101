"""Segment LeRobot episodes and build a hierarchical caption tree (GPU pass)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from robot101.data.utilities.annotation_tree import (
    SCHEMA_VERSION, atomic_json, average_observations, build_tree, cached_json,
    fingerprint, midpoint, sampled_frames, uniform_frames, windows, write_exports, write_report,
)
from robot101.data.utilities.episode_helpers import DEFAULT_SRC, _load_lerobot, _parse_episodes
from robot101.data.utilities.episode_io import load_episode_metadata, decode_episode_cameras
from robot101.data.utilities.annotation_hub import add_hub_arguments, validate_hub_arguments, publish_annotations


class ModelInitializationError(RuntimeError):
    pass


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-repo-id", default=DEFAULT_SRC)
    p.add_argument("--src-revision", default=None, help="Default: LeRobot's compatible dataset revision")
    p.add_argument("--episodes", default="0", help="Inclusive ranges, e.g. 0-25:5")
    p.add_argument("--camera", default="top")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/annotations"))
    p.add_argument("--stage", choices=["all", "segment", "caption"], default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--download-workers", type=int, default=1)
    p.add_argument("--decode-batch-size", type=int, default=64)
    p.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    p.add_argument("--jepa-ref", default="main", help="Official repository Git ref; resolved to a commit at run creation")
    p.add_argument("--sample-step", type=int, default=4)
    p.add_argument("--window-frames", type=int, default=64)
    p.add_argument("--window-stride", type=int, default=8)
    p.add_argument("--jepa-batch-size", type=int, default=1)
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--silhouette-threshold", type=float, default=0.15)
    p.add_argument("--video-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--image-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--video-revision", default="main")
    p.add_argument("--image-revision", default="main")
    p.add_argument("--video-frames", type=int, default=32)
    p.add_argument("--video-batch-size", type=int, default=1)
    p.add_argument("--image-batch-size", type=int, default=1)
    p.add_argument("--auto-batch", action="store_true", help="Adapt caption batches to CUDA VRAM; retry oversized batches")
    p.add_argument("--auto-batch-max", type=int, default=64, help="Largest adaptive caption batch")
    p.add_argument("--auto-batch-reserve", type=float, default=0.10, help="VRAM fraction reserved, with a 2 GiB minimum")
    p.add_argument("--max-caption-tokens", type=int, default=1024)
    add_hub_arguments(p)
    return p


def load_source_episode(dataset_class, repo_id, episode, root, revision, backend):
    kwargs = dict(episodes=[episode], root=root, revision=revision, video_backend=backend)
    try:
        return dataset_class(repo_id, **kwargs)
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        print(f"Episode {episode}: syncing missing source shard", flush=True)
        return dataset_class(repo_id, force_cache_sync=True, **kwargs)


def resolve_run(args):
    """Freeze mutable source/model refs once; resume uses this same snapshot."""
    execution_options = {"episodes", "stage", "resume", "output_dir", "download_workers",
                         "push_to_hub", "dst_repo_id", "hub_private", "video_batch_size",
                         "image_batch_size", "auto_batch", "auto_batch_max", "auto_batch_reserve"}
    requested = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                 if k not in execution_options}
    path = args.output_dir / "run.json"
    key = fingerprint(requested)
    if path.exists():
        if not args.resume:
            raise ValueError(f"{path} exists; use --resume")
        existing = json.loads(path.read_text())
        previous = {k: v for k, v in existing["requested"].items() if k not in execution_options}
        if fingerprint(previous) != key:
            raise ValueError(f"Incompatible configuration in {path}; choose a different output directory")
        return existing
    from huggingface_hub import HfApi
    import requests
    _, metadata_class, _, cache_root = _load_lerobot()
    metadata = metadata_class(args.src_repo_id, root=cache_root / args.src_repo_id, revision=args.src_revision)
    api = HfApi()
    sha = api.repo_info(args.src_repo_id, repo_type="dataset", revision=metadata.revision).sha
    response = requests.get(f"https://api.github.com/repos/facebookresearch/vjepa2/commits/{args.jepa_ref}", timeout=60)
    response.raise_for_status()
    refs = dict(source=sha, jepa=response.json()["sha"],
                video=api.model_info(args.video_model, revision=args.video_revision).sha,
                image=api.model_info(args.image_model, revision=args.image_revision).sha)
    record = dict(schema_version=SCHEMA_VERSION, fingerprint=key, requested=requested, resolved=refs,
                  method="Action100M adaptation", bounds="half_open_source_frames", fps=float(metadata.fps),
                  total_episodes=int(metadata.total_episodes), prompts_version="neutral-detailed-v1")
    atomic_json(path, record)
    return record


def save_features(path, features, ids, counts, key, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, features=features, frame_ids=ids, counts=counts,
                            fingerprint=key, metadata=json.dumps(metadata))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def segment_episode(args, run, episode, source, worker):
    from robot101.perception.annotation_models import JEPA_CHECKPOINT_URL
    prefix = f"ep{episode:06d}"
    feature_config = {key: run["requested"][key] for key in (
        "src_repo_id", "camera", "jepa_ref", "sample_step", "window_frames", "window_stride", "dtype", "seed")}
    feature_key = fingerprint(dict(config=feature_config, source=run["resolved"]["source"],
                                   jepa=run["resolved"]["jepa"], checkpoint_url=JEPA_CHECKPOINT_URL,
                                   episode=episode, adapter="jepa-tubelet-amp-v3"))
    segment_key = fingerprint(dict(features=feature_key, min_duration=args.min_duration,
                                   max_segments=args.max_segments, threshold=args.silhouette_threshold, version=SCHEMA_VERSION))
    path = args.output_dir / "segments" / f"{prefix}.json"
    existing = cached_json(path, segment_key, args.resume)
    if existing:
        return existing
    feature_path = args.output_dir / "features" / f"{prefix}.npz"
    if feature_path.exists():
        if not args.resume:
            raise ValueError(f"Features exist: {feature_path}; use --resume")
        with np.load(feature_path, allow_pickle=False) as bank:
            if str(bank["fingerprint"]) != feature_key:
                raise ValueError(f"Incompatible features: {feature_path}")
            features, ids = bank["features"].copy(), bank["frame_ids"].copy()
            metadata = json.loads(str(bank["metadata"]))
    else:
        start = perf_counter()
        ds = source(episode)
        download_seconds = perf_counter() - start
        start = perf_counter()
        data = load_episode_metadata(ds, [args.camera], preserve_auxiliary=False)
        metadata_seconds = perf_counter() - start
        start = perf_counter()
        ids = sampled_frames(data["n"], args.sample_step)
        decode_episode_cameras(ds, data, [args.camera], indices=ids,
                               batch_size=args.decode_batch_size, backend=args.video_backend)
        if not all(data["masks"][args.camera][ids]):
            raise ValueError("Missing requested segmentation camera frames")
        for i in ids:
            frame = data["frames"][args.camera][i]
            if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Invalid camera frame {i}")
        decode_seconds = perf_counter() - start
        bank = windows(len(ids), args.window_frames, args.window_stride)
        model = worker()
        observations = []
        start = perf_counter()
        for offset in range(0, len(bank), args.jepa_batch_size):
            batch = bank[offset:offset + args.jepa_batch_size]
            clips = [[data["frames"][args.camera][ids[j]] for j in positions] for positions, _ in batch]
            encoded = model.encode(clips)
            for (positions, valid), features in zip(batch, encoded, strict=True):
                observations.append((positions, valid, features))
            print(f"JEPA ep{episode}: windows {min(offset + len(batch), len(bank))}/{len(bank)}", flush=True)
        features, counts = average_observations(len(ids), observations, model.tubelet)
        encoder_seconds = perf_counter() - start
        from robot101.data.utilities.stages import events_from_gripper_thresholds
        gripper_events = events_from_gripper_thresholds(data["gripper"], fps=run["fps"], min_change_frac=0.25)
        metadata = dict(n_frames=data["n"], task=Counter(data["tasks"]).most_common(1)[0][0],
                        timestamps=data["timestamps"], gripper=data["gripper"].tolist(),
                        gripper_events=gripper_events,
                        source_load_seconds=download_seconds, metadata_seconds=metadata_seconds,
                        decode_seconds=decode_seconds, encoder_seconds=encoder_seconds)
        save_features(feature_path, features, ids, counts, feature_key, metadata)
    start = perf_counter()
    tree = build_tree(features, ids, metadata["n_frames"], run["fps"], min_duration=args.min_duration,
                      max_segments=args.max_segments, threshold=args.silhouette_threshold)
    record = dict(tree, schema_version=SCHEMA_VERSION, fingerprint=segment_key,
                  jepa_checkpoint_url=JEPA_CHECKPOINT_URL,
                  feature_fingerprint=feature_key, episode_index=episode, camera=args.camera,
                  source_repo_id=args.src_repo_id, source_revision=run["resolved"]["source"],
                  fps=run["fps"], bounds="half_open_source_frames", **metadata,
                  clustering_seconds=perf_counter() - start)
    atomic_json(path, record)
    return record


def caption_records(args, run, records, source, kind, worker_factory):
    from PIL import Image
    from robot101.perception.annotation_models import release
    worker = None
    batch_size = args.video_batch_size if kind == "video" else args.image_batch_size
    from robot101.data.utilities.auto_batch import AutoBatch, is_cuda_oom
    adaptive = AutoBatch(batch_size, args.auto_batch_max, args.auto_batch_reserve) if args.auto_batch else None
    try:
        for record in records:
            jobs = []
            for node in record["nodes"]:
                if node["caption_kind"] != kind:
                    continue
                indices = uniform_frames(node, args.video_frames) if kind == "video" else [midpoint(node)]
                key = fingerprint(dict(segment=record["fingerprint"], node=node, model=run["resolved"][kind],
                                       model_id=args.video_model if kind == "video" else args.image_model,
                                       sample_frames=indices, tokens=args.max_caption_tokens, prompt="neutral-detailed-v1",
                                       dtype=args.dtype, adapter="direct-frames-metadata-v2"))
                path = args.output_dir / "node_captions" / f'ep{record["episode_index"]:06d}' / f'{node["id"]}.json'
                cached = cached_json(path, key, args.resume)
                if cached and cached.get("status") == "ok":
                    node["caption"] = cached
                else:
                    jobs.append((node, indices, path, key))
            if not jobs:
                continue
            start = perf_counter()
            try:
                ds = source(record["episode_index"])
                record.setdefault("caption_timings", {})[f"{kind}_source_load_seconds"] = perf_counter() - start
                start = perf_counter()
                data = load_episode_metadata(ds, [args.camera], preserve_auxiliary=False)
                record["caption_timings"][f"{kind}_metadata_seconds"] = perf_counter() - start
                start = perf_counter()
                indices = sorted({i for _, ids, _, _ in jobs for i in ids})
                decode_episode_cameras(ds, data, [args.camera], indices=indices,
                                       batch_size=args.decode_batch_size, backend=args.video_backend)
            except Exception as exc:
                for node, indices, path, key in jobs:
                    result = dict(fingerprint=key, status="failed", error=str(exc), sample_frames=indices, kind=kind)
                    atomic_json(path, result)
                    node["caption"] = result
                continue
            record.setdefault("caption_timings", {})[f"{kind}_decode_seconds"] = perf_counter() - start
            offset = 0
            generation_seconds = 0.0
            while offset < len(jobs):
                size = adaptive.size if adaptive else batch_size
                batch = jobs[offset:offset + size]
                requests, valid_jobs = [], []
                for node, ids, path, key in batch:
                    if not all(data["masks"][args.camera][ids]) or any(data["frames"][args.camera][i] is None for i in ids):
                        result = dict(fingerprint=key, status="failed", error="missing_caption_frame", sample_frames=ids, kind=kind)
                        atomic_json(path, result)
                        node["caption"] = result
                        continue
                    frames = [data["frames"][args.camera][i] for i in ids]
                    image_root = args.output_dir / "images" / f'ep{record["episode_index"]:06d}'
                    image_root.mkdir(parents=True, exist_ok=True)
                    for i, frame in zip(ids, frames, strict=True):
                        image_path = image_root / f"f{i:06d}.jpg"
                        if not image_path.exists():
                            Image.fromarray(frame).save(image_path, quality=85)
                    if kind == "video":
                        frames = [np.asarray(Image.fromarray(frame).resize((320, 320), Image.Resampling.BICUBIC)) for frame in frames]
                    requests.append(dict(frames=np.stack(frames), fps=record["fps"],
                                         frame_indices=[i - node["start_frame"] for i in ids],
                                         total_num_frames=node["end_frame"] - node["start_frame"]))
                    valid_jobs.append((node, ids, path, key))
                if not valid_jobs:
                    offset += len(batch)
                    continue
                if worker is None:
                    # Fatal access/load errors stop this pass rather than
                    # marking every node as if its visual input were invalid.
                    worker = worker_factory(kind)
                if adaptive:
                    adaptive.begin(worker.torch, worker.device)
                start = perf_counter()
                retry_oom = False
                saw_oom = False
                try:
                    results = worker.caption(requests, args.max_caption_tokens)
                    if len(results) != len(valid_jobs):
                        raise ValueError("Wrong number of captions")
                except Exception as exc:
                    if adaptive and is_cuda_oom(exc, worker.torch):
                        saw_oom = True
                        adaptive.oom(len(valid_jobs), perf_counter() - start)
                        retry_oom = len(valid_jobs) > 1
                    results = [dict(status="failed", error=str(exc)) for _ in valid_jobs]
                seconds = perf_counter() - start
                generation_seconds += seconds
                if adaptive:
                    record.setdefault("caption_batching", {}).setdefault(kind, []).extend(adaptive.events)
                    adaptive.events.clear()
                if saw_oom:
                    import gc
                    # Exception traceback is out of scope before clearing CUDA
                    # allocations from the failed generation call.
                    del requests
                    gc.collect()
                    worker.torch.cuda.empty_cache()
                if retry_oom:
                    del results
                    print(f'{kind} ep{record["episode_index"]}: CUDA OOM at batch {len(valid_jobs)}; '
                          f'retrying at {adaptive.size}', flush=True)
                    continue
                if adaptive and all(r.get("status", "ok") == "ok" for r in results):
                    event = adaptive.complete(worker.torch, worker.device, len(valid_jobs), seconds)
                    record["caption_batching"][kind].append(event)
                    adaptive.events.clear()
                    print(f'  auto-batch: {len(valid_jobs)} nodes, peak allocation '
                          f'{event["peak_allocated_mib"] / 1024:.2f} GiB, next {adaptive.size}', flush=True)
                for (node, ids, path, key), result in zip(valid_jobs, results, strict=True):
                    result.update(fingerprint=key, kind=kind, sample_frames=ids,
                                  sample_timestamps=[data["timestamps"][i] for i in ids],
                                  model_revision=run["resolved"][kind], seconds=seconds / len(valid_jobs))
                    result.setdefault("status", "ok")
                    atomic_json(path, result)
                    node["caption"] = result
                offset += len(batch)
                print(f'{kind} ep{record["episode_index"]}: nodes {offset}/{len(jobs)} '
                      f'(batch {len(valid_jobs)}, {seconds:.1f}s)', flush=True)
            record["caption_timings"][f"{kind}_generation_seconds"] = generation_seconds
    finally:
        if adaptive and worker is not None:
            # Peak resets used for adaptive estimates must not erase the
            # run-wide allocation peak reported by the caller.
            peak = max(adaptive.peak_allocated, worker.torch.cuda.max_memory_allocated(worker.device))
            for record in records:
                record.setdefault("caption_timings", {})[f"{kind}_peak_allocated_mib"] = peak / 2**20
        release(worker)


def main(argv=None):
    args = build_parser().parse_args(argv)
    from robot101.data.utilities.hardware_usage import HardwareUsage
    started = perf_counter()
    timings = dict(status="running", segmentation_seconds=0.0, video_captioning_seconds=0.0,
                   image_captioning_seconds=0.0, upload_seconds=0.0,
                   execution_options={k: getattr(args, k) for k in (
                       "auto_batch", "auto_batch_max", "auto_batch_reserve", "video_batch_size", "image_batch_size")})
    usage = HardwareUsage()
    try:
        _run(args, usage, timings, started)
        timings["status"] = "complete"
    except BaseException as exc:
        timings["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        usage.finish(args.output_dir)
        timings["total_wall_seconds"] = perf_counter() - started
        timings.setdefault("processing_seconds", timings["total_wall_seconds"] - timings["upload_seconds"])
        save_timings(args.output_dir, timings)
        print(f'Time spent: processing {timings["processing_seconds"]:.1f}s '
              f'(segmentation {timings["segmentation_seconds"]:.1f}s, '
              f'video captions {timings["video_captioning_seconds"]:.1f}s, '
              f'image captions {timings["image_captioning_seconds"]:.1f}s); '
              f'upload {timings["upload_seconds"]:.1f}s; total {timings["total_wall_seconds"]:.1f}s', flush=True)


def save_timings(output_dir, timings):
    atomic_json(output_dir / "timings.json", timings)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["processing_timings"] = timings
        atomic_json(summary_path, summary)


def _run(args, usage, timings, started):
    validate_hub_arguments(args)
    for name in ("download_workers", "decode_batch_size", "sample_step", "jepa_batch_size", "max_segments",
                 "video_frames", "video_batch_size", "image_batch_size", "max_caption_tokens", "auto_batch_max"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.min_duration < 0 or not np.isfinite(args.min_duration) or not -1 <= args.silhouette_threshold <= 1:
        raise SystemExit("Invalid duration or silhouette threshold")
    if not np.isfinite(args.auto_batch_reserve) or not 0 < args.auto_batch_reserve < 1:
        raise SystemExit("--auto-batch-reserve must be between 0 and 1")
    if args.auto_batch and not args.device.startswith("cuda"):
        raise SystemExit("--auto-batch requires a CUDA device")
    windows(1, args.window_frames, args.window_stride)
    os.environ["SO101_HF_DOWNLOAD_WORKERS"] = str(args.download_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = resolve_run(args)
    episodes = _parse_episodes(args.episodes)
    if not episodes or any(i < 0 or i >= run["total_episodes"] for i in episodes):
        raise SystemExit("Select existing episodes explicitly, e.g. --episodes 0-9")
    dataset_class, _, _, cache_root = _load_lerobot()
    source = lambda episode: load_source_episode(dataset_class, args.src_repo_id, episode,
                 cache_root / args.src_repo_id, run["resolved"]["source"], args.video_backend)
    import torch
    from robot101.perception.annotation_models import Captioner, JEPAEncoder, release
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the annotation environment on your GPU VM")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    worker = None
    def encoder():
        nonlocal worker
        if worker is None:
            try:
                worker = JEPAEncoder(run["resolved"]["jepa"], args.device, args.dtype)
            except Exception as exc:
                raise ModelInitializationError(f"Unable to initialize V-JEPA 2.1: {exc}") from exc
        return worker
    records, failures = [], []
    start = perf_counter()
    timings["setup_seconds"] = start - started
    try:
        for episode in episodes:
            print(f"[segment] episode {episode}", flush=True)
            if args.stage == "caption":
                path = args.output_dir / "segments" / f"ep{episode:06d}.json"
                if not path.exists():
                    raise SystemExit(f"Caption-only requires segmentation: {path}")
                # segment_episode validates fingerprints and does not load
                # JEPA when a compatible segment artifact exists.
                if not args.resume:
                    raise SystemExit("Caption-only requires --resume")
            try:
                records.append(segment_episode(args, run, episode, source, encoder))
            except ValueError as exc:
                if "exist" in str(exc).lower() or "incompatible" in str(exc).lower():
                    raise
                failures.append(dict(episode_index=episode, stage="segment", error=str(exc)))
                print(f"[skip episode {episode}] {exc}", flush=True)
            except ModelInitializationError as exc:
                failures.append(dict(episode_index=episode, stage="segment", error=str(exc)))
                atomic_json(args.output_dir / "failures.json", failures)
                raise
            except (OSError, RuntimeError) as exc:
                failures.append(dict(episode_index=episode, stage="segment", error=str(exc)))
                print(f"[skip episode {episode}] {exc}", flush=True)
                if "out of memory" in str(exc).lower():
                    atomic_json(args.output_dir / "failures.json", failures)
                    raise
            atomic_json(args.output_dir / "failures.json", failures)
    finally:
        release(worker)
        timings["segmentation_seconds"] = perf_counter() - start
    prior_peak_mib = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith("cuda") else 0
    def captioner(kind):
        return Captioner(args.video_model if kind == "video" else args.image_model,
                         run["resolved"][kind], kind, args.device, args.dtype)
    try:
        if args.stage != "segment":
            for kind in ("video", "image"):
                phase_start = perf_counter()
                try:
                    caption_records(args, run, records, source, kind, captioner)
                finally:
                    timings[f"{kind}_captioning_seconds"] = perf_counter() - phase_start
    finally:
        for record in records:
            record["review_required"] |= any(n.get("caption", {}).get("status") != "ok" or
                                             n.get("caption", {}).get("truncated", False) for n in record["nodes"])
            atomic_json(args.output_dir / "episodes" / f'ep{record["episode_index"]:06d}.json', record)
            write_report(args.output_dir, record)
        exported = [json.loads(path.read_text()) for path in sorted((args.output_dir / "episodes").glob("ep*.json"))]
        write_exports(args.output_dir, exported)
        import resource
        import sys
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_ram_mib = peak_rss / (2**20 if sys.platform == "darwin" else 1024)
        atomic_json(args.output_dir / "summary.json", dict(episodes=len(records), skipped=failures,
                    retained_nodes=sum(len(r["nodes"]) for r in records), wall_seconds=perf_counter() - start,
                    peak_cpu_ram_mib=peak_ram_mib,
                    peak_vram_mib=max([prior_peak_mib,
                        torch.cuda.max_memory_allocated(args.device) / 2**20,
                        *[r.get("caption_timings", {}).get(f"{kind}_peak_allocated_mib", 0)
                          for r in records for kind in ("video", "image")]]) if args.device.startswith("cuda") else None))
    usage.finish(args.output_dir)
    timings["processing_seconds"] = perf_counter() - started
    timings["status"] = "processing_complete"
    save_timings(args.output_dir, timings)
    print(f'Processing finished in {timings["processing_seconds"]:.1f}s '
          f'({timings["processing_seconds"] / 60:.1f} minutes), excluding upload.', flush=True)
    print(f"Saved caption trees and reports to {args.output_dir}")
    if args.push_to_hub:
        upload_start = perf_counter()
        try:
            publish_annotations(args.output_dir, args.dst_repo_id, "captioning", private=args.hub_private)
        finally:
            timings["upload_seconds"] = perf_counter() - upload_start


if __name__ == "__main__":
    main()
