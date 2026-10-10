# View datasets and gripper stages locally

## Packages

From the repository root:

```bash
python3 -m venv .venv-viewer
source .venv-viewer/bin/activate
python scripts/install_requirements.py --profile viewer
```

## View random episodes and subtasks

```bash
python scripts/view_grounded_subtasks.py \
  --repo-id kdaterao/so101_pov_clustering_test_v2 \
  --random-episodes 3 --seed 100 --open-browser
```

Change `--repo-id` to select another dataset. By default, the viewer loads at
most three episodes. Use `--random-episodes 5` for five random episodes.
Add `--recompute-stages` to preview current gripper segmentation without
rewriting saved data. On grounded datasets, point heatmaps are drawn over the
clean videos from the `point_tracks/` sidecars. Source datasets also support
this viewer and show no point overlay unless tracking sidecars are present.

## View prepared gripper labels

```bash
python scripts/view_molmo_so101_dataset.py \
  --repo-id kdaterao/so101_molmo2_gripper_preprocessed --open-browser
```

Pass `--help` for all viewer flags. The standalone
`scripts/inspect_gripper_stages.py` needs the full preprocessing environment;
its command is in [Preprocessing](preprocessing.md).
