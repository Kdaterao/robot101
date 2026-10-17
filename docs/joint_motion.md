# Joint/state stillness inspection

Run from the repository root in an environment with LeRobot, NumPy and the Hub
dependencies installed, such as `.venv-annotation` or the existing viewer
environment. This is a standalone CPU utility; it does not run vision models,
download/decode videos or change segmentation. Separately, `annotate.py` now
uses the same heuristic to provide per-node motion hints to video captioning;
see [episode_annotation.md](episode_annotation.md#measured-motion-hints).

```bash
python scripts/inspect_joint_motion.py \
  --src-repo-id felsager/community_dataset_v3_ee_smolVLA \
  --episodes 0 --output-dir outputs/joint_motion
```

It reads measured `observation.state`, uses coordinate names to exclude gripper,
finger and jaw channels by default, and prints a mean **stillness score** plus
the fraction of samples classified as likely still. It writes per-frame JSON
and CSV, plus `summary.json`. Larger scores mean less motion; this is **not a
calibrated probability**, object-motion estimate, grasp detector or success label.
Stillness of the selected coordinates does not guarantee every robot link is still.

## Select the actual joints

Inspect `coordinate_names` and `signal_kind` in the episode report. Some datasets
provide end-effector pose rather than joint angles; the utility warns and labels
pose/unverified state coordinates rather than calling them confirmed joints.
For genuine joint-motion estimates, select a feature/source with measured joint
positions. Do not interpret pose/quaternion changes as joint-angle changes.

```bash
python scripts/inspect_joint_motion.py \
  --src-repo-id OWNER/JOINT_DATASET --episodes 0-25:5 \
  --state-key observation.state --joint-indices 0-5 \
  --output-dir outputs/joint_motion
```

`--joint-indices` uses zero-based inclusive ranges. Explicit indices are required
if the feature has no coordinate names. Named gripper/finger channels are still
excluded unless `--include-gripper` is supplied. If names are missing, ensure your
explicit selection excludes the gripper yourself.

## Scoring and thresholds

Each frame uses a centered **one-second** window, clipped at episode boundaries.
Positions are divided by fixed dataset-wide min/max ranges per coordinate, not
the current episode's range. This avoids magnifying tiny jitter in static episodes.
The metric combines the largest per-coordinate 90th-percentile absolute speed
and the largest position excursion across the window. It detects slow drift as
well as faster motion. Defaults:

- `--speed-threshold 0.01`: 1% of a coordinate's normalization range per second.
- `--excursion-threshold 0.01`: 1% of that range within a window.
- `--window-seconds 1`: centered offline analysis, not a causal live detector.
- `--max-gap-seconds 0.1`: larger timestamp gaps make affected windows unknown.

With `r = max(speed/speed_threshold, excursion/excursion_threshold)`, the score is
`2 ** (-r*r)`. A score at least **0.5** means likely still under those thresholds.
Larger thresholds are more tolerant of movement. Tune against known idle and
moving intervals; these defaults are not calibrated robot-specific limits.
Missing/nonfinite state windows or insufficient observations produce null scores,
not presumed stillness. Non-increasing timestamps reject the episode.

Use `--joint-scales 180` for a shared fixed range in raw units, or comma-separated
scales in the order of the selected coordinates. Explicit scales are required
when dataset ranges are missing, constant or invalid. These are normalization
ranges, not velocity thresholds. For wrapped joint angles only, set
`--angle-period 360` for degrees or `--angle-period 6.283185307179586` for radians;
omit it for normalized actuator values, poses and mixed coordinate types.

The importable array API is
`robot101.data.utilities.joint_motion.estimate_stillness(states, timestamps, scales)`.
Its output contains per-frame scores/measurements and a sample-weighted summary
that excludes unknown windows. Source loading failures are recorded in the summary
and make the command exit with a nonzero status.
