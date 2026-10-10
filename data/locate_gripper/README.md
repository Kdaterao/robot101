---
license: apache-2.0
task_categories:
  - object-detection
  - visual-question-answering
tags:
  - robotics
  - so101
  - gripper
  - grounding
pretty_name: SO-101 Gripper Points
---

# kdaterao/so101_locate_gripper

SO-101 gripper point labels for visual point grounding. Existing bounding-box labels are retained as historical annotations.

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

Collected by clicking the center of the gripper jaws in `src/robot101/data/collect_points.py`; intended for custom MolmoPoint fine-tuning. Existing box annotations remain available as historical annotations.

Start point collection with:

```bash
python -m robot101.data.collect_points batch
```
