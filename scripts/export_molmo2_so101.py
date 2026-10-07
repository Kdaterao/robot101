#!/usr/bin/env python3
"""Merge SO-101 connector weights into a complete Transformers checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file


def hf_key(name: str) -> str:
    # Match AllenAI's olmo/hf_model/convert_molmo2_to_hf.py.
    if name == "transformer.ff_out.weight":
        return "lm_head.weight"
    parts = name.split(".")
    if len(parts) >= 5 and parts[:2] == ["transformer", "blocks"]:
        if parts[3] in {"att_proj", "attn_out", "q_norm", "k_norm"}:
            parts.insert(3, "self_attn")
        elif parts[3] in {"ff_proj", "ff_out"}:
            parts.insert(3, "mlp")
    return "model." + ".".join(parts)


def load_connector(path: Path) -> dict:
    # The training hook may have saved one-rank FSDP2 DTensors. A process group
    # is needed to deserialize their mesh; no model or GPU tensors are created.
    with tempfile.TemporaryDirectory(prefix="so101-export-") as temporary:
        owns_group = not dist.is_initialized()
        if owns_group:
            dist.init_process_group(
                "gloo", init_method=Path(temporary, "rendezvous").as_uri(),
                rank=0, world_size=1,
            )
        try:
            # Only load the trusted connector produced by our training script.
            state = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or not state:
                raise ValueError("Connector must be a nonempty tensor dictionary")
            result = {}
            for name, tensor in state.items():
                if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid connector entry: {name!r}")
                if hasattr(tensor, "to_local"):
                    if tensor.to_local().shape != tensor.shape:
                        raise ValueError("Export requires the one-GPU connector, not a partial shard")
                    tensor = tensor.to_local()
                result[name] = tensor.detach().cpu()
            return result
        finally:
            if owns_group:
                dist.destroy_process_group()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / "molmo2_so101_run"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=work / "Molmo2-4B-SFT")
    parser.add_argument("--connector", type=Path,
                        default=work / "checkpoints/so101_molmo2_4b/so101_connector.pt")
    parser.add_argument("--output-dir", type=Path, default=work / "hf_model")
    parser.add_argument("--repo-id", help="Upload the finished model to this HF model repository")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--base-hf-revision", default="main")
    args = parser.parse_args()

    if not args.connector.is_file():
        raise FileNotFoundError(args.connector)
    candidates = list(args.base_checkpoint.rglob("model.pt"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one base model.pt under {args.base_checkpoint}; found {len(candidates)}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Export directory is not empty: {output}; choose a new --output-dir")

    connector = load_connector(args.connector)
    print(f"Loading base checkpoint {candidates[0]} with memory mapping", flush=True)
    base = torch.load(candidates[0], map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(base, dict):
        raise ValueError("Expected a base model state dictionary")
    for name, tensor in connector.items():
        if name not in base:
            raise ValueError(f"Connector parameter absent from base model: {name}")
        if tensor.shape != base[name].shape:
            raise ValueError(f"Connector shape mismatch for {name}")
        base[name] = tensor

    # Download only HF configs, tokenizer, processor and custom model code.
    # All actual weights come from the exact checkpoint used for training.
    metadata = Path(snapshot_download(
        "allenai/Molmo2-4B", revision=args.base_hf_revision,
        allow_patterns=["*.json", "*.py", "*.txt", "*.jinja", "LICENSE*"],
    ))
    reference_index = json.loads((metadata / "model.safetensors.index.json").read_text())
    converted = {hf_key(name): tensor for name, tensor in base.items()}
    if len(converted) != len(base):
        raise ValueError("Duplicate names after converting model parameters")
    expected = set(reference_index["weight_map"])
    actual = set(converted)
    if expected != actual:
        raise ValueError(f"HF parameter mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    output.mkdir(parents=True, exist_ok=True)
    for file in metadata.iterdir():
        if file.is_file() and file.name not in {"model.safetensors.index.json", "README.md"}:
            shutil.copyfile(file, output / file.name)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    # Match the reduced crop budget used during SO-101 training.
    image_config_path = output / "preprocessor_config.json"
    image_config = json.loads(image_config_path.read_text())
    image_config["max_crops"] = 2
    image_config_path.write_text(json.dumps(image_config, indent=2) + "\n")

    # Convert at most 2 GB at a time; do not instantiate a second 4B model.
    shard_limit = 2_000_000_000
    shard, shard_bytes, total_bytes = {}, 0, 0
    shard_files, weight_map = [], {}

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        filename = f"weights-{len(shard_files)+1:05d}.safetensors"
        print(f"Writing {filename} ({shard_bytes / 1e9:.2f} GB)", flush=True)
        save_file(shard, str(output / filename), metadata={"format": "pt"})
        shard_files.append(filename)
        weight_map.update({name: filename for name in shard})
        shard, shard_bytes = {}, 0
        gc.collect()

    for name, tensor in converted.items():
        dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
        size = tensor.numel() * torch.empty((), dtype=dtype).element_size()
        if shard_bytes + size > shard_limit:
            flush()
        shard[name] = tensor.to(dtype=dtype).contiguous()
        shard_bytes += size
        total_bytes += size
    flush()
    for number, old in enumerate(shard_files, 1):
        new = f"model-{number:05d}-of-{len(shard_files):05d}.safetensors"
        (output / old).rename(output / new)
        weight_map = {key: new if value == old else value for key, value in weight_map.items()}
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes}, "weight_map": weight_map,
    }, indent=2) + "\n")
    (output / "export_info.json").write_text(json.dumps({
        "base_checkpoint": str(candidates[0]), "connector": str(args.connector.resolve()),
        "hf_metadata_revision": metadata.name, "merged_parameters": sorted(connector),
        "dtype": "bfloat16",
    }, indent=2) + "\n")
    (output / "README.md").write_text(
        "---\nbase_model: allenai/Molmo2-4B\nlicense: apache-2.0\n"
        "library_name: transformers\npipeline_tag: image-text-to-text\n"
        "datasets:\n- kdaterao/so101_locate_gripper\n---\n\n"
        "# Molmo2-4B SO-101 gripper grounding\n\n"
        "Complete BF16 model with the SO-101 tuned connector merged into the original "
        "Molmo2-4B SFT weights. Includes tokenizer, processor, and custom model code.\n\n"
        "Fine-tuned on user-clicked gripper points, updating the connector "
        "and additional token embeddings. Held-out localization has not been evaluated.\n\n"
        "Load with `AutoModelForImageTextToText.from_pretrained(repo_id, "
        "trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')` "
        "and `AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)`.\n"
    )
    print(f"Exported full model: {output} ({total_bytes / 1e9:.2f} GB)", flush=True)
    if args.repo_id:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=str(output),
                          commit_message="Upload complete SO-101 fine-tuned Molmo2-4B model")
        print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
