"""SO-101 point annotations in the Molmo training example format."""

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
            raise RuntimeError("Set SO101_POINT_DATA_ROOT to prepared SO-101 point data")
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
        message = {
            "label": row["label"],
            "points": np.asarray(row["point_xy_100"], dtype=np.float32).reshape(1, 2),
            "point_scale": 100,
            "style": "pointing",
        }
        return {
            "image": row["image"],
            "message_list": [message],
            "metadata": {
                "episode": row["episode"],
                "camera": row["camera"],
                "source_image": row["source_image"],
            },
        }
