"""SO-101 dataset generation, perception, robot control and policy tools."""
import sys
from .paths import REPO_ROOT

# Use the repository's pinned LeRobot checkout when available.
_lerobot_src = REPO_ROOT / "lerobot" / "src"
if _lerobot_src.is_dir() and str(_lerobot_src) not in sys.path:
    sys.path.insert(0, str(_lerobot_src))
