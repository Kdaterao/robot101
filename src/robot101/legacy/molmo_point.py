"""Small inference adapter for Hugging Face MolmoPoint checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


class MolmoPointWorker:
    """Load MolmoPoint once and return pixel points plus the raw response."""

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "auto"):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MolmoPoint inference needs transformers with MolmoPoint support; "
                "see the official allenai/molmo2 MOLMO_POINT_README.md setup."
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype: Any = dtype
        if dtype == "auto":
            model_dtype = "auto"
        elif dtype in {"bf16", "bfloat16"}:
            model_dtype = torch.bfloat16
        elif dtype in {"fp16", "float16"}:
            model_dtype = torch.float16
        elif dtype in {"fp32", "float32"}:
            model_dtype = torch.float32

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=model_dtype,
            device_map="auto" if self.device == "auto" else {"": self.device},
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, padding_side="left"
        )

    @torch.inference_mode()
    def ground(self, image_rgb: np.ndarray, prompt: str) -> dict[str, Any]:
        """Return ``points_xy`` in source-image pixels and a raw text response."""
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        autocast_dtype = None
        if model_device.type == "cuda":
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(
            device_type=model_device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = self.model.generate(
                **inputs,
                logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=200,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.post_process_image_text_to_text(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        decoded = self.model.extract_image_points(
            raw,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        points = []
        for item in np.asarray(decoded).reshape(-1, 4):
            x, y = float(item[-2]), float(item[-1])
            points.append([x, y])
        return {"points_xy": np.asarray(points, dtype=np.float32).reshape(-1, 2), "raw_response": str(raw)}
