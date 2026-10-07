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

SO-101 gripper bounding-box labels for LocateAnything / Florence-2 grounding.

| | |
|---|---|
| Labels | 290 |
| Images | 288 |

## Layout

```
images/           # JPEG frames (epXXXXXX_fXXXXXX_{top,side}.jpg)
labels.jsonl      # click-drag boxes (xyxy pixels)
locate_sft.jsonl  # optional Eagle / LocateAnything ShareGPT export
recipe.json       # optional Eagle recipe pointing at this folder
```

## Label row schema

```json
{
  "image": "images/ep000500_f000120_top.jpg",
  "phrase": "SO-101 gripper",
  "box_xyxy": [x1, y1, x2, y2],
  "width": 640,
  "height": 480,
  "episode": 500,
  "frame": 120,
  "camera": "top"
}
```

Collected with [`locate_collect_label.py`](https://github.com/) batch labeling; export with:

```bash
python src/locate_finetune.py export --data-root .
```
