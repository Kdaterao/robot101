"""Fine-tune Florence-2 on SO-101 gripper boxes from Hub / local locate_gripper.

Dataset layout (matches ``kdaterao/so101_locate_gripper`` / ``data/locate_gripper``)::

    labels.jsonl   # image, phrase, box_xyxy, width, height, ...
    images/*.jpg

Examples::

  # Train from Hub dataset (downloads labels + images)
  python src/florence2_finetune.py train --dataset-repo kdaterao/so101_locate_gripper

  # Train from local folder
  python src/florence2_finetune.py train --data-root data/locate_gripper

  # Smoke (1 step) on 8GB GPU
  python src/florence2_finetune.py train --data-root data/locate_gripper --max-steps 1 \\
      --model-id microsoft/Florence-2-base-ft --batch-size 1

  # Push checkpoint
  python src/florence2_finetune.py push --output-dir work_dirs/florence2_so101_gripper
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from florence2_worker import (
    TASK_OPEN_VOCAB,
    _load_florence_model,
    _patch_florence_attn_flags,
)

DEFAULT_DATASET_REPO = "kdaterao/so101_locate_gripper"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "locate_gripper"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "work_dirs" / "florence2_so101_gripper"
DEFAULT_MODEL_REPO = "kdaterao/florence2_so101_gripper"
DEFAULT_TRAIN_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_PHRASE = "SO-101 gripper"


def _prepare_hub() -> None:
    try:
        from hf_hub_windows import prepare_hf_hub_env

        prepare_hf_hub_env()
    except ImportError:
        pass


def _norm_loc(v: float, size: int) -> int:
    """Quantize pixel coord to Florence ``<loc_N>`` range (0–999)."""
    if size <= 0:
        raise ValueError("image size must be positive")
    return int(max(0, min(999, round(v / size * 1000))))


def box_to_loc_tokens(box_xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    nx1 = _norm_loc(x1, width)
    ny1 = _norm_loc(y1, height)
    nx2 = _norm_loc(x2, width)
    ny2 = _norm_loc(y2, height)
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1
    return f"<loc_{nx1}><loc_{ny1}><loc_{nx2}><loc_{ny2}>"


def florence_prefix(phrase: str, task: str = TASK_OPEN_VOCAB) -> str:
    phrase = (phrase or DEFAULT_PHRASE).strip()
    return f"{task}{phrase}"


def florence_suffix(phrase: str, box_xyxy: list[float], width: int, height: int) -> str:
    phrase = (phrase or DEFAULT_PHRASE).strip()
    return f"{phrase}{box_to_loc_tokens(box_xyxy, width, height)}"


def resolve_data_root(
    *,
    data_root: Path | None,
    dataset_repo: str | None,
    cache_dir: Path | None = None,
) -> Path:
    """Prefer local ``--data-root``; otherwise snapshot the Hub dataset repo."""
    if data_root is not None:
        root = Path(data_root).resolve()
        labels = root / "labels.jsonl"
        if not labels.is_file():
            raise SystemExit(f"Missing {labels}")
        return root

    repo = dataset_repo or DEFAULT_DATASET_REPO
    _prepare_hub()
    from huggingface_hub import snapshot_download

    dest = cache_dir or (Path.home() / ".cache" / "robot101" / "datasets" / repo.replace("/", "__"))
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {repo} -> {dest}")
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["labels.jsonl", "images/**", "README.md", "hub_meta.json"],
    )
    root = Path(path)
    if not (root / "labels.jsonl").is_file():
        raise SystemExit(f"Hub dataset {repo} has no labels.jsonl after download")
    return root


def load_label_rows(data_root: Path, *, skip_missing: bool = True) -> list[dict[str, Any]]:
    labels_path = data_root / "labels.jsonl"
    rows: list[dict[str, Any]] = []
    skipped = 0
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_rel = str(row["image"]).replace("\\", "/")
            image_path = data_root / image_rel
            if not image_path.is_file():
                skipped += 1
                if skip_missing:
                    continue
                raise SystemExit(f"Missing image: {image_path}")
            box = row.get("box_xyxy")
            if not box or len(box) != 4:
                raise SystemExit(f"Bad box_xyxy in {labels_path}: {box}")
            w = int(row.get("width") or 0)
            h = int(row.get("height") or 0)
            if w <= 0 or h <= 0:
                with Image.open(image_path) as im:
                    w, h = im.size
            rows.append(
                {
                    "image": image_rel,
                    "phrase": row.get("phrase") or DEFAULT_PHRASE,
                    "box_xyxy": [float(v) for v in box],
                    "width": w,
                    "height": h,
                    "path": image_path,
                }
            )
    if skipped:
        print(f"Skipped {skipped} labels with missing images")
    if not rows:
        raise SystemExit(f"No usable labels in {labels_path}")
    return rows


@dataclass
class FlorenceSample:
    prefix: str
    suffix: str
    image_path: Path


class FlorenceGripperDataset(Dataset):
    def __init__(self, samples: list[FlorenceSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {"prefix": s.prefix, "suffix": s.suffix, "image": image}


def rows_to_samples(rows: list[dict[str, Any]], task: str) -> list[FlorenceSample]:
    out: list[FlorenceSample] = []
    for row in rows:
        out.append(
            FlorenceSample(
                prefix=florence_prefix(row["phrase"], task=task),
                suffix=florence_suffix(row["phrase"], row["box_xyxy"], row["width"], row["height"]),
                image_path=row["path"],
            )
        )
    return out


class FlorenceCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prefixes = [b["prefix"] for b in batch]
        suffixes = [b["suffix"] for b in batch]
        images = [b["image"] for b in batch]

        inputs = self.processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = self.processor.tokenizer(
            text=suffixes,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.masked_fill(labels == pad_id, -100)
        inputs["labels"] = labels
        return inputs


def _freeze_vision(model) -> None:
    tower = getattr(model, "vision_tower", None)
    if tower is None:
        return
    for p in tower.parameters():
        p.requires_grad = False


def cmd_train(args: argparse.Namespace) -> None:
    data_root = resolve_data_root(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo if args.data_root is None else None,
    )
    rows = load_label_rows(data_root, skip_missing=True)
    samples = rows_to_samples(rows, task=args.task)

    rng = random.Random(args.seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = 0
    if args.val_ratio > 0 and len(samples) >= 5:
        n_val = max(1, int(round(len(samples) * args.val_ratio)))
        n_val = min(n_val, len(samples) // 5)  # keep most for train
    val_idx = set(idx[:n_val]) if n_val else set()
    train_samples = [samples[i] for i in idx if i not in val_idx]
    val_samples = [samples[i] for i in idx if i in val_idx]

    print(f"Data root: {data_root}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} task={args.task}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    # Keep master weights in fp32; Trainer fp16/bf16 autocast handles the forward.
    # Loading the whole model in fp16 breaks GradScaler ("unscale FP16 gradients").
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Florence-2 {args.model_id} (fp32 masters, amp={args.precision}) on {device}...")
    _patch_florence_attn_flags(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _load_florence_model(args.model_id, torch.float32, device)
    model.train()
    if args.freeze_vision:
        _freeze_vision(model)
        print("Frozen vision_tower")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    train_ds = FlorenceGripperDataset(train_samples)
    eval_ds = FlorenceGripperDataset(val_samples) if val_samples else None
    collator = FlorenceCollator(processor)

    use_fp16 = device == "cuda" and args.precision == "fp16"
    use_bf16 = device == "cuda" and args.precision == "bf16"

    targs = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None and args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        overwrite_output_dir=args.overwrite_output_dir,
        gradient_checkpointing=args.grad_checkpoint,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if args.dry_run:
        batch = collator([train_ds[0]])
        print("Dry-run batch keys:", {k: tuple(v.shape) for k, v in batch.items()})
        print(f"Example prefix: {train_samples[0].prefix}")
        print(f"Example suffix: {train_samples[0].suffix}")
        return

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    meta = {
        "base_model": args.model_id,
        "dataset_root": str(data_root),
        "dataset_repo": args.dataset_repo,
        "task": args.task,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "phrase_default": DEFAULT_PHRASE,
    }
    (output_dir / "florence2_finetune_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved checkpoint -> {output_dir}")

    if args.push_to_hub:
        cmd_push(
            argparse.Namespace(
                output_dir=output_dir,
                repo_id=args.hub_model_repo,
                private=args.private,
                commit_message=args.commit_message,
            )
        )


def cmd_push(args: argparse.Namespace) -> None:
    _prepare_hub()
    from huggingface_hub import HfApi, create_repo

    output_dir = Path(args.output_dir).resolve()
    if not (output_dir / "config.json").is_file():
        raise SystemExit(f"No config.json in {output_dir}; train first")

    repo_id = args.repo_id or DEFAULT_MODEL_REPO
    create_repo(repo_id, repo_type="model", private=bool(args.private), exist_ok=True)
    api = HfApi()
    msg = args.commit_message or "Florence-2 SO-101 gripper fine-tune"
    print(f"Pushing {output_dir} -> {repo_id}")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
        ignore_patterns=["checkpoint-*/**", "runs/**", "*.pt", "trainer_state.json"],
    )
    print(f"Done: https://huggingface.co/{repo_id}")


def cmd_preview(args: argparse.Namespace) -> None:
    data_root = resolve_data_root(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo if args.data_root is None else None,
    )
    rows = load_label_rows(data_root)[: args.n]
    for row in rows:
        print(
            json.dumps(
                {
                    "image": row["image"],
                    "prefix": florence_prefix(row["phrase"], task=args.task),
                    "suffix": florence_suffix(
                        row["phrase"], row["box_xyxy"], row["width"], row["height"]
                    ),
                },
                ensure_ascii=False,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Florence-2 on so101_locate_gripper")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--data-root",
            type=Path,
            default=None,
            help=f"Local locate_gripper folder (default if present: {DEFAULT_DATA_ROOT})",
        )
        sp.add_argument(
            "--dataset-repo",
            default=DEFAULT_DATASET_REPO,
            help="HF dataset repo when --data-root is omitted",
        )
        sp.add_argument(
            "--task",
            default=TASK_OPEN_VOCAB,
            choices=[TASK_OPEN_VOCAB, "<CAPTION_TO_PHRASE_GROUNDING>"],
            help="Florence task token (must match inference)",
        )

    prev = sub.add_parser("preview", help="Print prefix/suffix for a few labels")
    add_data_args(prev)
    prev.add_argument("-n", type=int, default=3)
    prev.set_defaults(func=cmd_preview)

    tr = sub.add_parser("train", help="Fine-tune Florence-2")
    add_data_args(tr)
    tr.add_argument("--model-id", default=DEFAULT_TRAIN_MODEL, help="Base Florence checkpoint")
    tr.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    tr.add_argument("--epochs", type=float, default=3.0)
    tr.add_argument("--max-steps", type=int, default=-1, help="Override epochs when > 0")
    tr.add_argument("--batch-size", type=int, default=1)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-6)
    tr.add_argument("--warmup-ratio", type=float, default=0.05)
    tr.add_argument("--val-ratio", type=float, default=0.05)
    tr.add_argument("--logging-steps", type=int, default=10)
    tr.add_argument("--save-steps", type=int, default=100)
    tr.add_argument("--eval-steps", type=int, default=100)
    tr.add_argument("--save-total-limit", type=int, default=2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument(
        "--precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Train dtype (fp16 recommended on GTX 10-series)",
    )
    tr.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--grad-checkpoint", action="store_true")
    tr.add_argument("--dataloader-num-workers", type=int, default=0)
    tr.add_argument("--report-to", default="none")
    tr.add_argument("--overwrite-output-dir", action="store_true")
    tr.add_argument("--resume-from-checkpoint", default=None)
    tr.add_argument("--cpu", action="store_true")
    tr.add_argument("--dry-run", action="store_true", help="Build one batch and exit")
    tr.add_argument("--push-to-hub", action="store_true")
    tr.add_argument("--hub-model-repo", default=DEFAULT_MODEL_REPO)
    tr.add_argument("--private", action="store_true")
    tr.add_argument("--commit-message", default=None)
    tr.set_defaults(func=cmd_train)

    pu = sub.add_parser("push", help="Upload a trained checkpoint to the Hub")
    pu.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    pu.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    pu.add_argument("--private", action="store_true")
    pu.add_argument("--commit-message", default=None)
    pu.set_defaults(func=cmd_push)

    return p


def main(argv: list[str] | None = None) -> None:
    # Default local data when present and user omitted both roots.
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "data_root", None) is None and getattr(args, "cmd", None) in ("train", "preview"):
        # Prefer local mirror when it exists so offline train works.
        if "--dataset-repo" not in argv and DEFAULT_DATA_ROOT.joinpath("labels.jsonl").is_file():
            args.data_root = DEFAULT_DATA_ROOT

    args.func(args)


if __name__ == "__main__":
    main()
