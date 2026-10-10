"""Repository and asset paths independent of entrypoint location."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
SO101_DIR = PACKAGE_ROOT / "legacy" / "simulation" / "assets" / "so101"
DEFAULT_CHECKPOINT = REPO_ROOT / "tapnet" / "checkpoints" / "causal_bootstapir_checkpoint.pt"
