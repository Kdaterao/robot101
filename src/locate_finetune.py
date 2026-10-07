"""Export LocateAnything SFT JSONL and launch LoRA fine-tuning via Eagle Embodied."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "locate_gripper"
DEFAULT_MODEL = "nvidia/LocateAnything-3B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "work_dirs" / "locate_so101_gripper"
DEFAULT_PHRASE = "SO-101 gripper"
DEFAULT_HF_REPO = "kdaterao/so101_locate_gripper"


def _norm_coord(v: float, size: int) -> int:
    if size <= 0:
        raise ValueError("image size must be positive")
    return int(round(max(0.0, min(1.0, v / size)) * 1000))


def _box_tokens(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    nx1 = _norm_coord(x1, width)
    ny1 = _norm_coord(y1, height)
    nx2 = _norm_coord(x2, width)
    ny2 = _norm_coord(y2, height)
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1
    return f"<box><{nx1}><{ny1}><{nx2}><{ny2}></box>"


def _resolve_image_size(row: dict, root: Path) -> tuple[int, int]:
    if "width" in row and "height" in row:
        return int(row["width"]), int(row["height"])
    img_path = root / row["image"]
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def _human_prompt(phrase: str) -> str:
    return f"Locate a single instance that matches the following description: {phrase}."


def _gpt_answer(phrase: str, box_token: str) -> str:
    return f"<ref>{phrase}</ref>{box_token}"


def cmd_export(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    labels_path = data_root / "labels.jsonl"
    if not labels_path.is_file():
        raise SystemExit(f"Missing labels at {labels_path}. Run locate_collect_label.py label first.")

    sft_path = data_root / "locate_sft.jsonl"
    recipe_path = data_root / "recipe.json"

    n = 0
    with labels_path.open("r", encoding="utf-8") as src, sft_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            phrase = row.get("phrase") or args.phrase
            box = row["box_xyxy"]
            if len(box) != 4:
                raise SystemExit(f"Bad box_xyxy in labels: {box}")
            width, height = _resolve_image_size(row, data_root)
            x1, y1, x2, y2 = [int(v) for v in box]
            box_tok = _box_tokens(x1, y1, x2, y2, width, height)
            sample = {
                "conversations": [
                    {"from": "human", "value": _human_prompt(phrase)},
                    {"from": "gpt", "value": _gpt_answer(phrase, box_tok)},
                ],
                "image": row["image"].replace("\\", "/"),
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    if n == 0:
        raise SystemExit(f"No labels found in {labels_path}")

    recipe = {
        "so101_gripper": {
            "annotation": str(sft_path).replace("\\", "/"),
            "root": str(data_root).replace("\\", "/"),
            "repeat_time": float(args.repeat_time),
            "data_augment": bool(args.data_augment),
        }
    }
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {n} samples -> {sft_path}")
    print(f"Recipe -> {recipe_path}")
    print(
        "Next: python src/locate_finetune.py train "
        f"--meta-path {recipe_path} --eagle-root <path/to/Eagle/Embodied>"
    )

    if getattr(args, "push_to_hub", False):
        from locate_push_hf import push_locate_gripper

        push_locate_gripper(
            data_root,
            args.repo_id,
            private=bool(getattr(args, "private", False)),
            include_sft=True,
            commit_message=f"Export locate_sft ({n} samples)",
        )


def _find_train_script(eagle_root: Path) -> Path:
    candidates = [
        eagle_root / "eaglevl" / "train" / "locany_finetune_magi_stream.py",
        eagle_root / "Embodied" / "eaglevl" / "train" / "locany_finetune_magi_stream.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "Could not find eaglevl/train/locany_finetune_magi_stream.py under "
        f"{eagle_root}. Clone https://github.com/NVlabs/Eagle.git and pass "
        "--eagle-root path/to/Eagle/Embodied (or path/to/Eagle)."
    )


def _cwd_for_eagle(train_script: Path) -> Path:
    # .../Embodied/eaglevl/train/script.py -> Embodied
    return train_script.resolve().parents[2]


def build_train_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    eagle_root = Path(args.eagle_root).resolve()
    train_script = _find_train_script(eagle_root)
    cwd = _cwd_for_eagle(train_script)

    meta_path = Path(args.meta_path).resolve()
    if not meta_path.is_file():
        raise SystemExit(f"Missing recipe at {meta_path}. Run `export` first.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    deepspeed = args.deepspeed
    if deepspeed:
        ds_path = Path(deepspeed)
        if not ds_path.is_file():
            # resolve relative to Eagle Embodied cwd
            alt = cwd / deepspeed
            if alt.is_file():
                deepspeed = str(alt)
            else:
                print(f"Warning: deepspeed config not found at {deepspeed}; passing through anyway")

    nproc = max(1, int(args.nproc_per_node))
    # Invoke via path relative to Embodied cwd so Eagle imports resolve.
    train_entry = "eaglevl/train/locany_finetune_magi_stream.py"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port",
        str(args.master_port),
        train_entry,
        "--model_name_or_path",
        args.model_name_or_path,
        "--max_steps",
        str(args.max_steps),
        "--output_dir",
        str(output_dir),
        "--meta_path",
        str(meta_path),
        "--overwrite_output_dir",
        "True" if args.overwrite_output_dir else "False",
        "--block_size",
        str(args.block_size),
        "--attn_implementation",
        args.attn_implementation,
        "--causal_attn",
        "False",
        "--freeze_llm",
        "True",
        "--freeze_mlp",
        "False",
        "--freeze_backbone",
        "True",
        "--use_llm_lora",
        str(args.use_llm_lora),
        "--use_backbone_lora",
        str(args.use_backbone_lora),
        "--vision_select_layer",
        "-1",
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--bf16",
        "True",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        "3",
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        "0.01",
        "--warmup_steps",
        str(args.warmup_steps),
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--max_seq_length",
        str(args.max_seq_length),
        "--do_train",
        "True",
        "--grad_checkpoint",
        "True",
        "--group_by_length",
        "False",
        "--report_to",
        args.report_to,
        "--mlp_connector_layers",
        "2",
    ]
    if deepspeed:
        cmd.extend(["--deepspeed", deepspeed])

    return cmd, cwd


def cmd_train(args: argparse.Namespace) -> None:
    cmd, cwd = build_train_command(args)
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"cwd: {cwd}")
    print(f"cmd: {printable}")
    print()
    print("After training, load the checkpoint with:")
    print(
        f'  python src/testing2.py --locate-model "{Path(args.output_dir).resolve()}" '
        f'--prompt "{DEFAULT_PHRASE}"'
    )
    print()

    if args.dry_run:
        print("Dry run only — not launching.")
        return

    env = os.environ.copy()
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
        print("Warning: HF_TOKEN not set; model download may fail if gated/private.")

    log_path = Path(args.output_dir).resolve() / "training_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_path}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n# cmd: {printable}\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Training failed with exit code {proc.returncode}. See {log_path}")
    print(f"Training finished. Checkpoint dir: {Path(args.output_dir).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export SO-101 gripper labels to LocateAnything JSONL and LoRA fine-tune."
    )
    sub = p.add_subparsers(dest="command", required=True)

    ep = sub.add_parser("export", help="Convert labels.jsonl -> locate_sft.jsonl + recipe.json")
    ep.add_argument("--data-root", type=Path, default=DEFAULT_OUT)
    ep.add_argument("--phrase", default=DEFAULT_PHRASE, help="Fallback phrase if missing in a label row")
    ep.add_argument("--repeat-time", type=float, default=1.0)
    ep.add_argument("--data-augment", action="store_true", help="Enable Eagle resize augmentation")
    ep.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After export, upload data/locate_gripper to Hugging Face",
    )
    ep.add_argument("--repo-id", default=DEFAULT_HF_REPO, help="HF dataset repo for --push-to-hub")
    ep.add_argument("--private", action="store_true", help="Private HF dataset")
    ep.set_defaults(func=cmd_export)

    tp = sub.add_parser("train", help="Launch Eagle LoRA continual SFT (requires NVlabs/Eagle Embodied)")
    tp.add_argument(
        "--eagle-root",
        type=Path,
        required=True,
        help="Path to Eagle/Embodied (or Eagle repo root containing Embodied/)",
    )
    tp.add_argument("--meta-path", type=Path, default=DEFAULT_OUT / "recipe.json")
    tp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    tp.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    tp.add_argument("--nproc-per-node", type=int, default=1)
    tp.add_argument("--master-port", type=int, default=29500)
    tp.add_argument("--max-steps", type=int, default=1000)
    tp.add_argument("--learning-rate", type=float, default=2e-5)
    tp.add_argument("--warmup-steps", type=int, default=50)
    tp.add_argument("--save-steps", type=int, default=100)
    tp.add_argument("--per-device-train-batch-size", type=int, default=1)
    tp.add_argument("--gradient-accumulation-steps", type=int, default=1)
    tp.add_argument("--dataloader-num-workers", type=int, default=2)
    tp.add_argument("--block-size", type=int, default=6)
    tp.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "magi"],
        help="sdpa for consumer GPUs; magi for Hopper/Blackwell",
    )
    tp.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Use ~4096 with sdpa; magi can go higher",
    )
    tp.add_argument("--use-llm-lora", type=int, default=64)
    tp.add_argument("--use-backbone-lora", type=int, default=0)
    tp.add_argument(
        "--deepspeed",
        default="deepspeed_configs/zero_stage1_config.json",
        help="DeepSpeed config path (relative to Embodied cwd ok). Pass 'none' to disable.",
    )
    tp.add_argument("--report-to", default="tensorboard")
    tp.add_argument("--overwrite-output-dir", action="store_true")
    tp.add_argument("--hf-token", default=None)
    tp.add_argument("--dry-run", action="store_true", help="Print command only")
    tp.set_defaults(func=cmd_train)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "deepspeed", None) in ("", "none", "None", "null"):
        args.deepspeed = None
    args.func(args)


if __name__ == "__main__":
    main()
