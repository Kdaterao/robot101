# Build the Molmo gripper point dataset

This workflow samples third-person images from the SO-101 dataset and lets a
person click the gripper jaw center. The resulting point labels can supervise
Molmo gripper grounding.

From the repository root, prepare a local labeling environment and sign in to
Hugging Face:

```bash
python3 -m venv .venv-labeling
source .venv-labeling/bin/activate
python scripts/install_requirements.py --profile labeling
hf auth login
```

Collect and label small batches (default source: `felsager/community_dataset_v3_ee_smolVLA`):

```bash
python -m robot101.data.collect_points batch
```

## Useful options

The batch command defaults to one episode and one timestamp, with top and side
camera images. Common adjustments:

```bash
# Label just the top camera, or collect two moments from each episode
python -m robot101.data.collect_points batch --cameras top
python -m robot101.data.collect_points batch --timestamps-per-episode 2 --batch-size 4

# Start at episode 500. Use --reset-cursor when an earlier batch run saved a cursor.
python -m robot101.data.collect_points batch --start-episode 500 --reset-cursor

# Sample a fixed set of episodes without the repeating batch workflow
python -m robot101.data.collect_points sample --episodes 0-10 --num-frames 20

# Continue pulling batches without asking between them
python -m robot101.data.collect_points batch --auto-continue
```

`--batch-size` caps new images per round; `--max-episodes-per-batch` controls
how many episodes it fetches at a time. `--stride` changes which frame is chosen
as batches advance. Add `--shuffle --seed 42` to randomize episode selection
reproducibly. `--repo-id` chooses the source dataset, `--out` changes the local
label folder, and `--video-backend pyav|torchcodec` selects video decoding.

Click the center of the gripper jaws. Press `s` to save, `n` to skip, and `q` to
quit. The batch cursor and labels are saved under `data/locate_gripper/`, so the
next run can continue. To sample images without the batch loop, use the
`sample` command, then run `label`. Use `label --phrase "SO-101 gripper"` to set
the text associated with each point, or `label --relabel-all` to revisit saved
images.

Upload the labeled images and points to the default dataset repo
`kdaterao/so101_locate_gripper`:

```bash
python -m robot101.data.collect_points push
# Choose a different repo or make it private:
python -m robot101.data.collect_points push --hf-repo-id YOUR_USER/REPO --private
```

The output is point supervision; it does not fine-tune or upload Molmo model
weights.
