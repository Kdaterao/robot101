# SO-101 LocateAnything + SmolVLA TapNet preprocess

## Setup (Linux CUDA VM)

```bash
bash packages-locate.sh
source .venv/bin/activate
hf auth login
```

Installs CUDA torch, tapnet checkpoint, vendored lerobot/tapnet editables,
`requirements-locate.txt`, **Eagle Embodied** (`third_party/Eagle`), and
`requirements-locate-finetune.txt` (peft / deepspeed / sentencepiece / … for
`locate_finetune.py train`). Does not replace `packages.sh` or `packages-tapnet-gpu.sh`.

```bash
# Fine-tune after setup (Eagle path printed by packages-locate.sh)
source .venv/locate-eagle.env   # sets LAUNCHER=pytorch + PYTHONPATH (or rely on locate_finetune.py)
python src/locate_finetune.py export
python src/locate_finetune.py train --eagle-root third_party/Eagle/Embodied --push-model-to-hub
# if DeepSpeed fails: add --deepspeed none
# Eagle defaults LAUNCHER=slurm — locate_finetune.py forces LAUNCHER=pytorch for torchrun
```

## A. Fine-tune LocateAnything on the SO-101 gripper

```bash
# 0. Log in (avoids Hub 429 rate limits)
hf auth login

# 1. Recommended: pull a small batch, label, repeat
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
#      - opens the click-drag label UI
#      - asks: Pull next batch? [Y/n/q]
#    Progress is saved in data/locate_gripper/batch_state.json

# Or do it manually:
python src/locate_collect_label.py sample --num-frames 4 --timestamps-per-episode 1 --episodes 0 --episode-by-episode
python src/locate_collect_label.py label
```

Controls while labeling: drag box, `s` save, `n` skip, `u` clear, `q` quit (resume later).

```bash
# 1b. Push labels + images to Hugging Face (default: kdaterao/so101_locate_gripper)
python src/locate_push_hf.py
# or: python src/locate_collect_label.py push
# or quit batch with: python src/locate_collect_label.py batch --push-to-hub

# 2. Export for Eagle (after you have enough labels)
python src/locate_finetune.py export
# Export + push in one shot:
python src/locate_finetune.py export --push-to-hub

# 3. Fine-tune (Eagle cloned by packages-locate.sh into third_party/Eagle)
python src/locate_finetune.py train --eagle-root third_party/Eagle/Embodied --dry-run
python src/locate_finetune.py train --eagle-root third_party/Eagle/Embodied --push-model-to-hub
# python src/locate_finetune.py train --eagle-root third_party/Eagle/Embodied --deepspeed none --push-model-to-hub

# 4. Push fine-tuned ckpt to Hub (only the dataset was pushed before — not the model)
python src/locate_finetune.py push-model --output-dir work_dirs/locate_so101_gripper
# or train with: ... train --eagle-root ... --push-model-to-hub

# 5. Smoke live grounding
python src/testing2.py
# Resolves: local work_dirs/locate_so101_gripper → Hub kdaterao/locate_so101_gripper
# → nvidia/LocateAnything-3B (if neither fine-tune exists yet)

# Optional A/B: Florence-2 instead of LocateAnything (LocateAnything kept intact)
python src/testing2.py --locator florence2 --prompt "SO-101 gripper"
# or microsoft/Florence-2-base / Florence-2-large-ft:
# python src/testing2.py --locator florence2 --locate-model microsoft/Florence-2-base --prompt "robot gripper"
```

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


## B. Preprocess HF teleop → heatmap SmolVLA dataset

Reads `felsager/community_dataset_v3_ee_smolVLA`, splits substages from **per-episode gripper min/max** + gripper velocity, seeds wrist POIs (TapNet), marks static 3rd-person gripper goals (LocateAnything, no tracking), writes a new LeRobot dataset with heatmap cameras.

```bash
# Dry-run stages only (no TapNet / no write)
python src/hf_preprocess_smolvla.py --episodes 0-2 --dry-run

# Full preprocess (small slice)
python src/hf_preprocess_smolvla.py `
  --src-repo-id felsager/community_dataset_v3_ee_smolVLA `
  --dst-repo-id kdaterao/community_v3_ee_smolvla_tapnet `
  --episodes 0-99 `
  --locate-model kdaterao/locate_so101_gripper `
  --gripper-closed-frac 0.15 `
  --gripper-open-frac 0.85 `
  --resume

# Same preprocess with Florence-2 grounding (no LocateAnything load)
python src/hf_preprocess_smolvla.py --episodes 0-2 --locator florence2 --dry-run
python src/hf_preprocess_smolvla.py --episodes 0-99 --locator florence2 --resume

# Optional Hub publish
python src/hf_preprocess_smolvla.py --episodes 0-99 --resume --push-to-hub
```

Then point [`src/train.py`](train.py) at `kdaterao/community_v3_ee_smolvla_tapnet` (same `observation.images.{top,wrist,side}` keys).

### Notes

- **Wrist:** Stage-2 backward TapNet on motion-clustered POIs → time-varying heatmap
- **Top/side:** LocateAnything or Florence-2 box at `stage.end` → **static** goal heatmap (camera is fixed)
- Gripper open/closed bands = episode-local `g_min + frac*(g_max-g_min)` plus velocity stall/min
- Locators: `--locator locateanything` (default) or `--locator florence2` (`src/florence2_worker.py`)
