# Setup / convenience scripts

One-off tools for camera capture, iPhone lens calibration, and Blender LilyTag authoring. Not part of the runtime simulation stack.

| Script | Purpose |
|--------|---------|
| [`capture_photos.py`](capture_photos.py) | Interval capture from OpenCV into repo `captured_images/` (or `--out`) |
| [`calibrate_iphone.py`](calibrate_iphone.py) | ChArUco board + offline calibration from `calibration_images/` |
| [`blender_lily.py`](blender_lily.py) | Run inside Blender: place LilyTags on meshes and export GLB |

```bash
uv run python -m robot101.calibration.capture_photos --camera 1
uv run python -m robot101.calibration.calibrate_iphone
# Blender: Scripting workspace → Open blender_lily.py → Run
```

Calibration outputs (`calibration.json`, `camera_matrix.npy`, etc.) are written under the repo root unless the script says otherwise. See [source layout](../../README.md) for entrypoints and section boundaries.
