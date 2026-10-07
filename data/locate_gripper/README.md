---
license: apache-2.0
task_categories:
  - object-detection
  - visual-question-answering
tags:
  - robotics
  - so101
  - gripper
  - locateanything
  - grounding
pretty_name: SO-101 Locate Gripper
---

# kdaterao/so101_locate_gripper

SO-101 gripper point labels for visual point grounding. Legacy bounding-box labels are retained for the Florence-2 experiments.

| | |
|---|---|
| Labels | 290 |
| Point labels | 0 (collect with `locate_collect_label.py`) |
| Images | 288 |

## Layout

```
images/           # JPEG frames (epXXXXXX_fXXXXXX_{top,side}.jpg)
point_labels.jsonl # user-clicked gripper points (pixels + normalized coordinates)
labels.jsonl       # legacy click-drag boxes (xyxy pixels)
locate_sft.jsonl  # optional Eagle / LocateAnything ShareGPT export
recipe.json       # optional Eagle recipe pointing at this folder
```

## Point label row schema

```json
{
  "image": "images/ep000500_f000120_top.jpg",
  "phrase": "SO-101 gripper",
  "point_xy": [x, y],
  "point_xy_norm": [x_normalized, y_normalized],
  "point_target": "center_of_gripper_jaws",
  "width": 640,
  "height": 480,
  "episode": 500,
  "frame": 120,
  "camera": "top"
}
```

Collected by clicking the center of the gripper jaws in `src/locate_collect_label.py`; intended for custom MolmoPoint fine-tuning. Existing box annotations remain available for the legacy Florence-2 workflow.

Start point collection with:

```bash
python src/locate_collect_label.py batch
```
