# Molmo2-4B gripper point training

## Install packages

```bash
cd ~/robot101
bash scripts/setup/linux_setup.sh --profile molmo2
source .venv-molmo2/bin/activate
hf auth login
```

This prepares the pinned Molmo2 checkout at `molmo2_so101_run/molmo2` and its
training dependencies.

## Prepare point labels

```bash
python scripts/prepare_molmo_so101.py \
  --repo-id kdaterao/so101_locate_gripper \
  --cache-dir molmo2_so101_run/hub_dataset \
  --output-dir molmo2_so101_run/prepared_dataset \
  --push-to-hub --hub-repo-id kdaterao/so101_molmo2_gripper_preprocessed
```

This reads user-clicked points and makes episode-disjoint train/validation
splits. Omit `--push-to-hub` for local preparation; add `--public` when creating
a public dataset (private by default). Robot video processing has its own
[guide](preprocessing.md).

## Install the adapter and download the training checkpoint

```bash
REPO_ROOT="$PWD"
WORK_ROOT="$REPO_ROOT/molmo2_so101_run"
MOLMO2_REPO_ROOT="$WORK_ROOT/molmo2"
python scripts/install_molmo2_so101_adapter.py --repo "$MOLMO2_REPO_ROOT"
mkdir -p "$WORK_ROOT/Molmo2-4B-SFT"
curl --fail --location --retry 5 \
  https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B-SFT.tar \
  --output "$WORK_ROOT/Molmo2-4B-SFT.tar.part"
mv "$WORK_ROOT/Molmo2-4B-SFT.tar.part" "$WORK_ROOT/Molmo2-4B-SFT.tar"
tar -xf "$WORK_ROOT/Molmo2-4B-SFT.tar" -C "$WORK_ROOT/Molmo2-4B-SFT"
CONFIG_FILE="$(find "$WORK_ROOT/Molmo2-4B-SFT" -name config.yaml -type f -print -quit)"
test -n "$CONFIG_FILE"
CHECKPOINT_DIR="$(dirname "$CONFIG_FILE")"
```

Reuse an existing archive or extracted checkpoint to skip downloading or
extracting it. This native training checkpoint differs from the HF inference base.

## Train the connector on one GPU

```bash
export MOLMO_DATA_DIR="$WORK_ROOT/molmo_data"
export SO101_POINT_DATA_ROOT="$WORK_ROOT/prepared_dataset"
export SO101_SAVE_TRAINABLE_ONLY=1
export WANDB_MODE=disabled
export PYTHONPATH="$MOLMO2_REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SAVE_FOLDER="$WORK_ROOT/checkpoints/so101_molmo2_4b"
mkdir -p "$MOLMO_DATA_DIR"
cd "$MOLMO2_REPO_ROOT"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  launch_scripts/sft.py "$CHECKPOINT_DIR" so101_point \
  --save_folder="$SAVE_FOLDER" --max_duration=500 \
  --global_train_batch_size=1 --device_batch_size=1 --seq_len=1024 \
  --num_workers=2 --prefetch_factor=2 \
  --model.mm_preprocessor.video=null \
  --model.mm_preprocessor.image.max_images=null \
  --model.mm_preprocessor.image.max_crops=2 \
  --model.mm_preprocessor.image.high_res_max_crops=4 \
  --model.mm_preprocessor.image.p_high_res=0 \
  --ft_llm=false --ft_vit=false --ft_connector=true \
  --save_num_checkpoints_to_keep=0 \
  --save_final_unsharded_checkpoint=false --save_final_optim=false \
  --eval_interval=-1 --inf_eval_interval=-1 --wandb=null \
  --compile=null --compile_loss=false --save_overwrite=false
cd "$REPO_ROOT"
```

Choose a fresh `SAVE_FOLDER` for another run, including when a failed run left
only `config.yaml`. Change `--max_duration` to set the number of steps. This
recipe freezes the vision encoder and language model and saves tuned tensors to
`$SAVE_FOLDER/so101_connector.pt` at completion. It saves no intermediate optimizer
checkpoint.

## Upload the connector

```bash
hf upload kdaterao/so101-molmo2-4b-gripper \
  "$SAVE_FOLDER/so101_connector.pt" so101_connector.pt
```

The [preprocessor](preprocessing.md) loads this connector into `allenai/Molmo2-4B`.
