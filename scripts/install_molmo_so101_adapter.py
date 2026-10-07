#!/usr/bin/env python3
"""Install the SO-101 point dataset adapter into an AllenAI Molmo checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def insert_once(path: Path, marker: str, addition: str, *, before: bool = True) -> None:
    text = path.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if marker not in text:
        raise SystemExit(f"Could not find expected Molmo integration point in {path}")
    replacement = f"{addition}{marker}" if before else f"{marker}{addition}"
    path.write_text(text.replace(marker, replacement, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molmo-repo", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    repo = args.molmo_repo.resolve()
    data_init = repo / "olmo/data/__init__.py"
    launcher = repo / "launch_scripts/train_multitask_model.py"
    if not data_init.is_file() or not launcher.is_file():
        raise SystemExit(f"Not an AllenAI Molmo training checkout: {repo}")

    adapter_dest = repo / "olmo/data/so101_point_dataset.py"
    shutil.copy2(args.adapter, adapter_dest)

    import_marker = "from olmo.torch_util import get_global_rank, get_world_size\n"
    insert_once(
        data_init,
        import_marker,
        "from olmo.data.so101_point_dataset import So101GripperPoints\n",
    )
    registry_marker = (
        '    elif dataset_name in ["point_count", "pixmo_points_counting"]:\n'
    )
    insert_once(
        data_init,
        registry_marker,
        '    elif dataset_name == "so101_gripper_points":\n'
        '        return So101GripperPoints(split)\n',
    )

    mixture_marker = '    elif args.mixture in ["small1", "debug"]:\n'
    mixture_branch = (
        '    elif args.mixture == "so101-point":\n'
        '        # Replay general point-grounding data while adapting to SO-101 clicks.\n'
        '        eval_tasks = []\n'
        '        tasks = [\n'
        '            ["general_pointing", ["pixmo_points"], 0.5],\n'
        '            ["so101_pointing", ["so101_gripper_points"], 0.5],\n'
        '        ]\n'
    )
    insert_once(launcher, mixture_marker, mixture_branch)
    print(f"Installed SO-101 point adapter in {repo}")


if __name__ == "__main__":
    main()
