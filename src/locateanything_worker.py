"""Minimal LocateAnything worker (from NVIDIA HF release docs)."""

from __future__ import annotations

import re
from typing import Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        do_sample = temperature > 0
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=max(temperature, 1e-5),
            do_sample=do_sample,
            top_p=0.9 if do_sample else None,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        # Support both tagged <box>...</box> and bare coordinate tokens.
        patterns = (
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
            r"<(\d+)><(\d+)><(\d+)><(\d+)>",
        )
        for pattern in patterns:
            for m in re.finditer(pattern, answer):
                x1, y1, x2, y2 = [int(g) for g in m.groups()]
                boxes.append(
                    {
                        "x1": x1 / 1000 * image_width,
                        "y1": y1 / 1000 * image_height,
                        "x2": x2 / 1000 * image_width,
                        "y2": y2 / 1000 * image_height,
                    }
                )
            if boxes:
                break
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points.

        Points are 2-coord tokens: <box><x><y></box>. Coordinates are
        normalized integers in [0, 1000].
        """
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append(
                {
                    "x": x / 1000 * image_width,
                    "y": y / 1000 * image_height,
                }
            )
        return points
