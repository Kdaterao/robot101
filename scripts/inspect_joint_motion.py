#!/usr/bin/env python3
"""Report how likely robot state is to be mostly still; no videos or models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot101.data.utilities.annotation_tree import atomic_json
from robot101.data.utilities.episode_helpers import DEFAULT_SRC, _as_numpy, _load_lerobot, _parse_episodes
from robot101.data.utilities.joint_motion import estimate_stillness, select_coordinates, normalization_scales



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo-id", default=DEFAULT_SRC)
    parser.add_argument("--src-revision")
    parser.add_argument("--src-root", type=Path)
    parser.add_argument("--episodes", default="0", help="Inclusive stepped ranges, e.g. 0-25:5")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--joint-indices", help="State coordinate indices, e.g. 0-5; otherwise use named non-gripper coordinates")
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--joint-scales", help="Fixed positive raw-unit range per selected coordinate, or one shared range")
    parser.add_argument("--angle-period", type=float, help="Unwrap periodic selected angles, e.g. 360 for degrees; omit for non-angle states")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--speed-threshold", type=float, default=0.01, help="Low-motion speed in fractions of the normalization range per second")
    parser.add_argument("--excursion-threshold", type=float, default=0.01, help="Low-motion displacement in fractions of normalization range per window")
    parser.add_argument("--max-gap-seconds", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/joint_motion"))
    args = parser.parse_args()
    if args.angle_period is not None and (not np.isfinite(args.angle_period) or args.angle_period <= 0):
        parser.error("--angle-period must be finite and positive")
    ids = _parse_episodes(args.episodes)
    if not ids or min(ids) < 0:
        parser.error("Select nonnegative episode IDs")
    dataset_class, _, _, cache_root = _load_lerobot()
    root = args.src_root or cache_root / args.src_repo_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, failures = [], []
    for episode in ids:
        try:
            kwargs = dict(episodes=[episode], root=root, revision=args.src_revision, download_videos=False)
            try:
                source = dataset_class(args.src_repo_id, **kwargs)
            except ValueError as exc:
                if "corresponds to no data" not in str(exc):
                    raise
                source = dataset_class(args.src_repo_id, force_cache_sync=True, **kwargs)
            feature = source.meta.features.get(args.state_key)
            if feature is None:
                raise ValueError(f"Source feature missing: {args.state_key}")
            rows = source.hf_dataset
            states = np.stack([_as_numpy(row[args.state_key]).reshape(-1) for row in rows])
            times = np.array([float(_as_numpy(row["timestamp"]).item()) for row in rows])
            selected, names, excluded, kind = select_coordinates(feature, states.shape[1], args.joint_indices, args.include_gripper)
            scales, scale_source = normalization_scales(source.meta, args.state_key, selected, args.joint_scales)
            measured = states[:, selected].astype(float)
            if args.angle_period:
                if not np.isfinite(measured).all():
                    raise ValueError("Cannot unwrap angles containing invalid samples")
                measured = np.unwrap(measured, period=args.angle_period, axis=0)
            result = estimate_stillness(measured, times, scales, window_seconds=args.window_seconds,
                speed_threshold=args.speed_threshold, excursion_threshold=args.excursion_threshold,
                max_gap_seconds=args.max_gap_seconds)
            result.update(episode_index=episode, source_repo_id=args.src_repo_id,
                source_revision=getattr(source.meta, "revision", args.src_revision), state_key=args.state_key,
                signal_kind=kind, coordinate_indices=selected, coordinate_names=names,
                excluded_gripper_coordinates=excluded, scales=scales.tolist(), scale_source=scale_source,
                angle_period=args.angle_period)
            if kind != "joint_state":
                print(f"WARNING ep{episode}: {kind}; this is not confirmed raw joint-angle motion. Coordinates: {names}", flush=True)
            prefix = args.output_dir / f"ep{episode:06d}"
            atomic_json(prefix.with_suffix(".json"), result)
            with prefix.with_suffix(".csv").open("w", newline="") as stream:
                keys = ["frame", "timestamp", "stillness_score", "likely_still", "normalized_speed_per_second",
                        "normalized_excursion", "window_start_frame", "window_end_frame", "reason"]
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(result["frames"])
            summary = dict(episode_index=episode, signal_kind=kind, **result["summary"])
            summaries.append(summary)
            score, fraction = summary["mean_stillness_score"], summary["likely_still_fraction"]
            print(f"Episode {episode}: mean stillness score={'N/A' if score is None else f'{score:.3f}'}, "
                  f"likely-still samples={'N/A' if fraction is None else f'{100 * fraction:.1f}%'}, "
                  f"unknown={summary['unknown_frames']}/{summary['frames']}", flush=True)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            failures.append(dict(episode_index=episode, error=str(exc)))
            print(f"[FAILED episode {episode}] {exc}", file=sys.stderr, flush=True)
    atomic_json(args.output_dir / "summary.json", dict(episodes=summaries, failures=failures))
    print(f"Reports saved to {args.output_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
