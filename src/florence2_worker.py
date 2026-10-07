"""Florence-2 grounding worker with a LocateAnything-compatible API.

Drop-in alternative for open-vocab / phrase grounding. LocateAnything stays
untouched — switch via ``--locator florence2`` in testing2 / preprocess.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-large"
DEFAULT_FLORENCE_HF = "kdaterao/florence2_so101_gripper"
_DEFAULT_FLORENCE_LOCAL = (
    Path(__file__).resolve().parent.parent / "work_dirs" / "florence2_so101_gripper"
)
# Preferred Hub id after you push a LocateAnything fine-tune (may not exist yet).
DEFAULT_LOCATE_HF = "kdaterao/locate_so101_gripper"
DEFAULT_LOCATE_BASE = "nvidia/LocateAnything-3B"
_DEFAULT_LOCATE_LOCAL = (
    Path(__file__).resolve().parent.parent / "work_dirs" / "locate_so101_gripper"
)


def resolve_locate_model(explicit: str | None = None) -> str:
    """Pick fine-tuned Hub/local ckpt if present, else NVIDIA base.

    Order: explicit arg → local ``work_dirs/locate_so101_gripper`` →
    Hub ``kdaterao/locate_so101_gripper`` (if reachable) → ``nvidia/LocateAnything-3B``.
    """
    if explicit:
        return str(explicit)

    local = _DEFAULT_LOCATE_LOCAL
    if (local / "config.json").is_file() or (local / "adapter_config.json").is_file():
        return str(local.resolve())

    try:
        from huggingface_hub import model_info

        model_info(DEFAULT_LOCATE_HF)
        return DEFAULT_LOCATE_HF
    except Exception:
        pass

    return DEFAULT_LOCATE_BASE


def resolve_florence_model(explicit: str | None = None) -> str:
    """Pick fine-tuned Florence ckpt if present, else Microsoft base.

    Order: explicit → local ``work_dirs/florence2_so101_gripper`` →
    Hub ``kdaterao/florence2_so101_gripper`` → ``microsoft/Florence-2-large``.
    """
    if explicit:
        return str(explicit)

    local = _DEFAULT_FLORENCE_LOCAL
    if (local / "config.json").is_file():
        return str(local.resolve())

    try:
        from huggingface_hub import model_info

        model_info(DEFAULT_FLORENCE_HF)
        return DEFAULT_FLORENCE_HF
    except Exception:
        pass

    return DEFAULT_FLORENCE_MODEL


# Back-compat alias used by other scripts.
DEFAULT_LOCATE_MODEL = DEFAULT_LOCATE_HF

# Open-vocab phrase grounding (closest to LocateAnything ground_single).
TASK_OPEN_VOCAB = "<OPEN_VOCABULARY_DETECTION>"
TASK_PHRASE_GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"


def _default_dtype(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _patch_florence_attn_flags(model_path: str) -> None:
    """Florence-2 remote code lacks ``_supports_sdpa``; transformers>=4.50 crashes.

    Load the dynamic module first, then set the class flags newer transformers expect.
    """
    from transformers import AutoConfig

    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    for mod in list(sys.modules.values()):
        cls = getattr(mod, "Florence2ForConditionalGeneration", None)
        if cls is None:
            continue
        # Class attrs (instance lookup via Module.__getattr__ fails otherwise).
        cls._supports_sdpa = False
        cls._supports_flash_attn_2 = False
        cls._supports_flex_attn = False
        cls._supports_attention_backend = False


def _load_florence_model(model_path: str, dtype, device: str):
    """Load Florence-2 with eager attention (compatible with transformers 4.57+)."""
    _patch_florence_attn_flags(model_path)
    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "dtype": dtype,  # transformers>=4.56; falls back below
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, attn_implementation="eager", torch_dtype=dtype
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype
            )
    return model.to(device).eval()


def _boxes_from_parsed(parsed: dict, task: str) -> list[dict]:
    """Normalize Florence post_process dict -> [{x1,y1,x2,y2}, ...]."""
    payload = parsed.get(task) or {}
    if not isinstance(payload, dict):
        return []
    bboxes = payload.get("bboxes") or payload.get("bboxes_labels") or []
    out: list[dict] = []
    for box in bboxes:
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


class Florence2Worker:
    """Stateful Florence-2 worker; mirrors LocateAnythingWorker grounding calls."""

    def __init__(
        self,
        model_path: str = DEFAULT_FLORENCE_MODEL,
        device: str = "cuda",
        dtype=None,
        task: str = TASK_OPEN_VOCAB,
    ):
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.dtype = dtype if dtype is not None else _default_dtype(self.device)
        self.task = task
        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = _load_florence_model(model_path, self.dtype, self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        *,
        task: str | None = None,
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        **_kwargs: Any,
    ) -> dict:
        """Run a Florence task. ``question`` is the text_input (phrase / caption)."""
        task = task or self.task
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Official Florence-2 pattern: task token + phrase concatenated.
        prompt = f"{task}{question}" if question else task
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        # Greedy by default: beam search hits past_key_values=None bugs with
        # Florence remote code + transformers 4.57.
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": False,
        }
        if num_beams and num_beams > 1:
            gen_kwargs["num_beams"] = int(num_beams)
        generated = self.model.generate(**gen_kwargs)
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)[0]

        w, h = image.size
        try:
            parsed = self.processor.post_process_generation(
                raw, task=task, image_size=(w, h)
            )
        except Exception:
            parsed = {task: {"bboxes": [], "labels": []}}

        boxes = _boxes_from_parsed(parsed, task)
        return {
            "answer": raw,
            "parsed": parsed,
            "boxes": boxes,
            "task": task,
        }

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        phrase = ", ".join(categories)
        return self.predict(image, phrase, task=TASK_OPEN_VOCAB, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        return self.predict(image, phrase, task=self.task, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        # Florence open-vocab returns all matches for the phrase.
        return self.predict(image, phrase, task=TASK_OPEN_VOCAB, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        # No native point task — return box centers as points.
        result = self.ground_single(image, phrase, **kwargs)
        points = []
        for b in result.get("boxes") or []:
            points.append(
                {
                    "x": 0.5 * (float(b["x1"]) + float(b["x2"])),
                    "y": 0.5 * (float(b["y1"]) + float(b["y2"])),
                }
            )
        result["points"] = points
        return result

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse Florence ``<loc_N>`` tokens (quantized 0–1000) into pixel boxes."""
        locs = [int(x) for x in re.findall(r"<loc_(\d+)>", answer or "")]
        boxes: list[dict] = []
        for i in range(0, len(locs) - 3, 4):
            x1, y1, x2, y2 = locs[i : i + 4]
            boxes.append(
                {
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                }
            )
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse pairs of loc tokens as points (fallback)."""
        locs = [int(x) for x in re.findall(r"<loc_(\d+)>", answer or "")]
        points: list[dict] = []
        # Prefer box centers when we have full boxes.
        boxes = Florence2Worker.parse_boxes(answer, image_width, image_height)
        if boxes:
            for b in boxes:
                points.append(
                    {
                        "x": 0.5 * (b["x1"] + b["x2"]),
                        "y": 0.5 * (b["y1"] + b["y2"]),
                    }
                )
            return points
        for i in range(0, len(locs) - 1, 2):
            points.append(
                {
                    "x": locs[i] / 1000.0 * image_width,
                    "y": locs[i + 1] / 1000.0 * image_height,
                }
            )
        return points


def boxes_from_ground_result(worker, result: dict, image_width: int, image_height: int) -> list[dict]:
    """Prefer structured boxes; fall back to worker.parse_boxes on the answer string."""
    boxes = result.get("boxes")
    if boxes:
        return list(boxes)
    answer = result.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)
    return type(worker).parse_boxes(answer, image_width, image_height)


def load_locator(name: str, model_path: str | None = None, device: str = "cuda"):
    """Factory: ``locateanything`` | ``florence2``."""
    key = (name or "locateanything").strip().lower()
    if key in ("florence2", "florence", "florence-2"):
        path = resolve_florence_model(model_path)
        print(f"Loading Florence-2 from {path}...")
        return Florence2Worker(path, device=device)
    if key in ("locateanything", "locate", "la"):
        from locateanything_worker import LocateAnythingWorker

        path = resolve_locate_model(model_path)
        print(f"Loading LocateAnything from {path}...")
        return LocateAnythingWorker(path, device=device)
    raise ValueError(f"Unknown locator {name!r}; use locateanything or florence2")
