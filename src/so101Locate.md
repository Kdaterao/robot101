# SO-101 LocateAnything + SmolVLA TapNet preprocess

## Setup (Linux CUDA VM) — Florence-2 fine-tune

```bash
bash packages-florence.sh
source .venv/bin/activate
source .venv/florence.env          # PYTHONPATH=src, CUDA, HF, alloc conf
hf auth login
```

Installs CUDA torch + `requirements-florence.txt` only (no Eagle / tapnet / deepspeed).
Writes `.venv/florence.env` for every new shell.

```bash
python src/florence2_finetune.py train --dataset-repo kdaterao/so101_locate_gripper --push-to-hub
# local: python src/florence2_finetune.py train --data-root data/locate_gripper
```

### Optional: full Locate / TapNet / Eagle stack

```bash
bash packages-locate.sh
source .venv/bin/activate
source .venv/locate-eagle.env   # LAUNCHER=pytorch + Eagle PYTHONPATH
hf auth login
```

`packages-locate.sh` also pulls tapnet, lerobot, Eagle, and `requirements-locate-finetune.txt`.
Not required for Florence-2 training.

## A. Collect SO-101 gripper point labels

The current label UI collects one user-clicked point at the center of the gripper jaws and writes `point_labels.jsonl` (pixel and normalized coordinates). It does not change the existing `labels.jsonl` box annotations. These point labels are for the planned custom MolmoPoint training data.

```bash
# 0. Log in (avoids Hub 429 rate limits)
hf auth login

# 1. Recommended: pull a small batch, click-label, repeat
# Clear old identical dumps if needed: Remove-Item -Recurse data/locate_gripper/images
python src/locate_collect_label.py batch
# Default: 1 episode, 1 timestamp, top+side -> 2 JPEGs per round (not every frame in the clip)

#    Jump to a later episode (needs Hub download if not cached yet):
# python src/locate_collect_label.py batch --start-episode 500 --reset-cursor

#    One image only:
# python src/locate_collect_label.py batch --batch-size 1 --cameras top

#    Two timestamps (e.g. start + late) on the same episode:
# python src/locate_collect_label.py batch --timestamps-per-episode 2 --batch-size 4

#    Each round:
#      - downloads only the next few episodes' videos (not the whole dataset)
#      - opens the single-click point-label UI (center of gripper jaws)
#      - asks: Pull next batch? [Y/n/q]
#    Progress is saved in data/locate_gripper/batch_state.json

# Or do it manually:
python src/locate_collect_label.py sample --num-frames 4 --timestamps-per-episode 1 --episodes 0 --episode-by-episode
python src/locate_collect_label.py label
```

Controls while labeling: click to place/move point, `s` save, `n` skip, `u` clear, `q` quit (resume later). Point records go to `data/locate_gripper/point_labels.jsonl`.

```bash
# 1b. Push point labels + images to Hugging Face (default: kdaterao/so101_locate_gripper)
python src/locate_push_hf.py
# or: python src/locate_collect_label.py push
# or quit batch with: python src/locate_collect_label.py batch --push-to-hub

# Legacy: fine-tune Florence-2 using the existing box labels.jsonl (the point UI does not create boxes)
python src/florence2_finetune.py preview --data-root data/locate_gripper
python src/florence2_finetune.py train --data-root data/locate_gripper
# From Hub only (no local images needed):
# python src/florence2_finetune.py train --dataset-repo kdaterao/so101_locate_gripper
# 8GB smoke:
# python src/florence2_finetune.py train --data-root data/locate_gripper --max-steps 1 --batch-size 1 --overwrite-output-dir

# 3. Push fine-tuned Florence checkpoint
python src/florence2_finetune.py push --output-dir work_dirs/florence2_so101_gripper
# or: ... train ... --push-to-hub

# 4. Smoke live grounding with Florence
python src/testing2.py --locator florence2 --prompt "SO-101 gripper"
# Resolves: local work_dirs/florence2_so101_gripper → Hub kdaterao/florence2_so101_gripper
# → microsoft/Florence-2-large
```

LocateAnything / Eagle fine-tune (`locate_finetune.py`) is optional and not required for the Florence path.

### Windows Hub download notes

- Scripts force Hugging Face to **copy** files instead of creating symlinks (fixes `WinError 1314`).
- Downloads use **`max_workers=1`** and auto-retry on **429** (this community dataset has ~1k meta files; parallel pulls burn the free API quota).
- Do **not** wipe a nearly finished download — just re-run the same command; Hub/LeRobot resume from cache. After a 429, wait ~5 minutes or let the script backoff.

```powershell
# Prefer batch mode (small pulls). Resume keeps completed Hub files.
python src/locate_collect_label.py batch

# Only if the tree is badly corrupted:
# Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\lerobot\felsager\community_dataset_v3_ee_smolVLA"
```

Optional: enable Windows **Developer Mode** if you prefer Hub symlink caching. Upgrade to HF PRO if you need higher rate limits: https://huggingface.co/pricing


## B. Preprocess HF teleop → POV + third-person grounded tracks

The grounded entrypoint keeps gripper open/close segmentation, clusters shared POV candidates only on each stage tail, and backward-tracks selected POV points through full stages. It extracts task noun phrases with spaCy, grounds those nouns plus the gripper with MolmoPoint on each available third-person view, chooses the endpoint object by gripper proximity (or gripper fallback), and forward-tracks the selected start-frame points. Per-subtask diagnostics are saved as JSON sidecars.

```bash
# Install point-grounding dependencies in the training environment
pip install -r requirements-molmo-grounding.txt
python -m spacy download en_core_web_sm

# Dry-run stage boundaries (no MolmoPoint or TapIR)
python src/hf_preprocess_smolvla_grounded.py --episodes 0-2 --dry-run

# Full preprocess (small slice with debug images)
python src/hf_preprocess_smolvla_grounded.py \
  --src-repo-id felsager/community_dataset_v3_ee_smolVLA \
  --dst-repo-id kdaterao/community_v3_ee_smolvla_molmo_grounded \
  --episodes 0-2 --cluster-tail-frames 30 \
  --object-proximity-threshold 0.08 \
  --viz-dir outputs/molmo_grounded_viz

# Continue/resume a larger run; use a custom checkpoint after MolmoPoint fine-tuning
python src/hf_preprocess_smolvla_grounded.py --episodes 0-99 --resume \
  --molmo-model work_dirs/molmo_so101_gripper_hf
```

The script writes per-subtask diagnostics under the destination's `point_tracks/` directory and renders selected tracks as heatmaps in the camera streams.

### Notes

- **Wrist:** shared tail-window tracking + cross-demo funneling/clustering → full-stage backward TapIR tracks
- **Top/side:** MolmoPoint noun/gripper grounding → endpoint proximity decision → selected forward TapIR tracks
- Gripper open/closed bands = episode-local `g_min + frac*(g_max-g_min)` plus velocity stall/min
- Ambiguous candidates, missing frames, invalid points, grounding errors, and tracker loss are retained in sidecar failure fields for inspection.
