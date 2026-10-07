"""Shared loading and name conversion for trusted SO-101 connector checkpoints."""
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist

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


def unwrap_parameter_name(name: str) -> str:
    """Remove training wrappers present in named_parameters() save hooks."""
    wrappers = {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
    return ".".join(part for part in name.split(".") if part not in wrappers)


def load_connector(path: Path) -> dict:
    # The training hook may have saved one-rank FSDP2 DTensors. A process group
    # is needed to deserialize their mesh; no model or GPU tensors are created.
    with tempfile.TemporaryDirectory(prefix="so101-connector-") as temporary:
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
                clean_name = unwrap_parameter_name(name)
                if clean_name in result:
                    raise ValueError(f"Duplicate connector parameter after removing wrappers: {clean_name}")
                if hasattr(tensor, "to_local"):
                    if tensor.to_local().shape != tensor.shape:
                        raise ValueError("Loading requires the one-GPU connector, not a partial shard")
                    tensor = tensor.to_local()
                result[clean_name] = tensor.detach().cpu()
            return result
        finally:
            if owns_group:
                dist.destroy_process_group()


