"""Command arguments and defaults for the grounded preprocessing pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
from robot101.paths import DEFAULT_CHECKPOINT
from robot101.data.utilities.episode_helpers import DEFAULT_SRC

DEFAULT_DST = "kdaterao/community_v3_ee_smolvla_molmo_grounded"
DEFAULT_MOLMO = "allenai/Molmo2-4B"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-repo-id", default=DEFAULT_SRC)
    p.add_argument("--dst-repo-id", default=DEFAULT_DST)
    p.add_argument("--episodes", default="0", help="Episode selection (default: 0), e.g. 0-9, 0-25:5, or 0,1,5")
    p.add_argument("--molmo-model", default=DEFAULT_MOLMO, help="Base HF model ID or local Transformers checkpoint")
    p.add_argument("--molmo-backend", choices=["molmo2", "molmopoint"], default="molmo2")
    p.add_argument("--molmo-connector", type=Path, help="Local trusted so101_connector.pt; overrides Hub download")
    p.add_argument("--molmo-connector-repo", default="kdaterao/so101-molmo2-4b-gripper")
    p.add_argument("--molmo-connector-revision", default="main")
    p.add_argument("--molmo-batch-size", type=int, default=4,
                   help="Independent Molmo2 image/prompt requests per generation batch")
    p.add_argument("--molmo-dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--max-task-objects", type=int, default=8)
    p.add_argument("--object-proximity-threshold", type=float, default=0.08)
    p.add_argument("--ambiguity-margin", type=float, default=0.01)
    p.add_argument("--third-person-tracking-fps", type=float, default=1.0,
                   help="TAPIR rate for top/side cameras; endpoints included, coordinates interpolated to source FPS")
    p.add_argument("--tapnet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dry-run", action="store_true", help="Only detect gripper stages; do not load models or write")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--first-primitive", default="grasp", choices=["grasp", "release"])
    p.add_argument("--max-stages", type=int, default=32)
    p.add_argument("--min-stage-frames", type=int, default=1, help="Minimum gap between event boundaries; preserves quick reversals")
    p.add_argument("--gripper-source", default="state", choices=["state", "action"])
    p.add_argument("--gripper-event-mode", choices=["movement", "bands"], default="movement")
    p.add_argument("--gripper-min-change-frac", type=float, default=0.25,
                   help="Movement required as a fraction of the smoothed episode gripper range")
    p.add_argument("--gripper-min-change-abs", type=float, default=0.0,
                   help="Additional minimum displacement in dataset gripper units")
    p.add_argument("--gripper-closed-frac", type=float, default=0.15)
    p.add_argument("--gripper-open-frac", type=float, default=0.85)
    p.add_argument("--gripper-min-dwell-frames", type=int, default=3)
    p.add_argument("--gripper-vel-stall", type=float, default=0.02)
    p.add_argument("--gripper-vel-min", type=float, default=0.05)
    p.add_argument("--gripper-smooth-window", type=int, default=5)
    p.add_argument("--cluster-tail-frames", type=int, default=30)
    p.add_argument("--num-sample-points", type=int, default=128, help="Shared query budget across episodes/stages/source frames; at least 8 per source frame")
    p.add_argument("--query-frames-per-stage", type=int, default=5, help="Spread feature seeds across each wrist tail, as in tapnetCreate.py")
    p.add_argument("--num-poi-points", type=int, default=16)
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--static-thresh", type=float, default=0.03)
    p.add_argument("--funnel-quantile", type=float, default=0.4)
    p.add_argument("--goal-tail-frames", type=int, default=5)
    p.add_argument("--pov-track-previous-stage", action=argparse.BooleanOptionalAction, default=True,
                   help="Continue the preceding stage's POV points forward alongside current-stage points")
    p.add_argument("--transition-persist-seconds", type=float, default=1.5,
                   help="Track preceding-subtask points into the next subtask for this long; 0 disables carryover")
    p.add_argument("--pov-visibility-gap-seconds", type=float, default=0.5, help="Bridge short bounded TAPIR confidence gaps; 0 disables")
    p.add_argument("--third-person-visibility-gap-seconds", type=float, default=2.0, help="Bridge bounded confidence gaps after sparse tracking; 0 disables")
    p.add_argument("--heatmap-sigma", type=float, default=40.0)
    p.add_argument("--heatmap-alpha", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tapir-frame-batch-size", type=int, default=256,
                   help="Ordered frames per causal TAPIR call; 1 restores per-frame tracking")
    p.add_argument("--tapir-tf32", action="store_true", help="Allow faster TF32 FP32 operations on Ampere GPUs; may slightly change tracks")
    p.add_argument("--decode-batch-size", type=int, default=64, help="Frames per video seek; increase with available CPU RAM")
    p.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    p.add_argument("--viz-dir", type=Path, default=None)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    return p
