#!/usr/bin/env python3
"""Install the SO-101 point dataset and single-GPU save hook into Molmo2."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASET_SOURCE = '''"""SO-101 user-clicked gripper points for Molmo2 pointing SFT."""

import json
import os
from pathlib import Path

import numpy as np

from olmo.data.dataset import Dataset


class So101GripperPoints(Dataset):
    def __init__(self, split):
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported SO-101 split: {split}")
        root = os.environ.get("SO101_POINT_DATA_ROOT")
        if not root:
            raise RuntimeError("Set SO101_POINT_DATA_ROOT to the prepared SO-101 point data")
        path = Path(root) / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if split == "train" and not self.rows:
            raise ValueError(f"No SO-101 training annotations found in {path}")

    def __len__(self):
        return len(self.rows)

    def get(self, item, rng):
        row = self.rows[item]
        return {
            "image": row["image"],
            "message_list": [{
                "label": row["label"],
                "points": np.asarray(row["point_xy_100"], dtype=np.float32).reshape(1, 2),
                "point_scale": 100,
                "clip_points": True,
                "style": "pointing",
            }],
            "metadata": {
                "episode": row["episode"],
                "camera": row["camera"],
                "source_image": row["source_image"],
            },
        }
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"Expected Molmo2 integration marker not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()

    dataset_module = repo / "olmo/data/so101_point_dataset.py"
    get_dataset = repo / "olmo/data/get_dataset.py"
    sft = repo / "launch_scripts/sft.py"
    run_trainer = repo / "olmo/train/run_trainer.py"
    preprocessor = repo / "olmo/models/molmo2/molmo2_preprocessor.py"
    for path in (get_dataset, sft, run_trainer, preprocessor):
        if not path.is_file():
            raise SystemExit(f"Not an AllenAI Molmo2 training checkout: {repo}; missing {path.name}")

    # The upstream SFT defaults reserve tokens for 128 video frames and five
    # images. Allow video=None so our single-image launcher can disable video
    # preprocessing instead of increasing the sequence length to fit it.
    replace_once(
        preprocessor,
        "import dataclasses\n",
        "import dataclasses\nfrom typing import Optional\n",
    )
    replace_once(
        preprocessor,
        "    video: VideoPreprocessorConfig = dataclasses.field(default_factory=VideoPreprocessorConfig)",
        "    video: Optional[VideoPreprocessorConfig] = dataclasses.field(default_factory=VideoPreprocessorConfig)",
    )
    replace_once(
        preprocessor,
        "            video_preprocessor=self.video.build_video_preprocessor(tokenizer, image_preprocessor),",
        "            video_preprocessor=(\n"
        "                self.video.build_video_preprocessor(tokenizer, image_preprocessor)\n"
        "                if self.video is not None else None\n"
        "            ),",
    )

    dataset_module.write_text(DATASET_SOURCE, encoding="utf-8")
    replace_once(
        get_dataset,
        "from olmo.data.dataset import Dataset\n",
        "from olmo.data.dataset import Dataset\n"
        "from olmo.data.so101_point_dataset import So101GripperPoints\n",
    )
    replace_once(
        get_dataset,
        "def get_dataset_by_name(dataset_name, split) -> Dataset:\n",
        "def get_dataset_by_name(dataset_name, split) -> Dataset:\n"
        "    if dataset_name == \"so101_gripper_points\":\n"
        "        return So101GripperPoints(split)\n",
    )

    sft_text = sft.read_text(encoding="utf-8")
    custom_mixture = (
        '    if name == "so101_point":\n'
        '        training_mixture = [["so101_points", '
        '[WeightedDataset("so101_gripper_points", sampling_rate=1.0)], 1.0]]\n'
    )
    broken_mixture = (
        "def get_training_mixture(name):\n"
        + custom_mixture
        + '    elif name == "debug":\n'
        + '    if name == "debug":\n'
    )
    fixed_mixture = (
        "def get_training_mixture(name):\n"
        + custom_mixture
        + '    elif name == "debug":\n'
    )
    if broken_mixture in sft_text:
        sft_text = sft_text.replace(broken_mixture, fixed_mixture, 1)
        sft.write_text(sft_text, encoding="utf-8")
    elif fixed_mixture not in sft_text:
        original_mixture = (
            'def get_training_mixture(name):\n'
            '    if name == "debug":\n'
        )
        if original_mixture not in sft_text:
            raise SystemExit(f"Expected Molmo2 mixture marker not found in {sft}")
        sft_text = sft_text.replace(original_mixture, fixed_mixture, 1)
        sft.write_text(sft_text, encoding="utf-8")
    replace_once(
        sft,
        "    if args.mixture == \"debug\":\n",
        "    if args.mixture == \"so101_point\":\n"
        "        loss_eval_tasks = []\n"
        "        eval_tasks = []\n"
        "    elif args.mixture == \"debug\":\n",
    )

    replace_once(
        run_trainer,
        '            trainer.fit()\n            log.info("Training complete")',
        '            trainer.fit()\n'
        '            if os.environ.get("SO101_SAVE_TRAINABLE_ONLY") == "1":\n'
        '                model = trainer.fsdp_model\n'
        '                state = {\n'
        '                    name: param.detach().to(device="cpu", copy=True)\n'
        '                    for name, param in model.named_parameters() if param.requires_grad\n'
        '                }\n'
        '                if not state:\n'
        '                    raise RuntimeError("No trainable SO-101 adapter parameters found")\n'
        '                if get_global_rank() == 0:\n'
        '                    output = Path(cfg.save_folder) / "so101_connector.pt"\n'
        '                    torch.save(state, output)\n'
        '                    log.info(f"Saved {len(state)} trainable tensors to {output}")\n'
        '                barrier()\n'
        '            log.info("Training complete")',
    )
    print(f"Installed SO-101 point adapter into {repo}")


if __name__ == "__main__":
    main()
