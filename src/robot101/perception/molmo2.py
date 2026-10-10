"""Molmo2 inference with the separately uploaded SO-101 connector."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from robot101.perception.connector import hf_key, load_connector


def parse_image_points(text: str, width: int, height: int) -> np.ndarray:
    """Parse Molmo2 html-v2: image ID, then object ID / x / y at scale 1000."""
    points = []
    for coords in re.findall(r'<points\b[^>]*\bcoords="([^"]*)"[^>]*>', text):
        for group in re.split(r'[\t:;,]', coords):
            tokens = group.split()
            if not tokens or tokens[0] != '1' or (len(tokens) - 1) % 3:
                continue
            for offset in range(1, len(tokens), 3):
                obj, xs, ys = tokens[offset:offset + 3]
                if not obj.isdigit() or not re.fullmatch(r'\d{3,4}', xs) or not re.fullmatch(r'\d{3,4}', ys):
                    continue
                x, y = int(xs), int(ys)
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    points.append([min(x * width / 1000, width - 1),
                                   min(y * height / 1000, height - 1)])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def apply_connector(model, state: dict) -> int:
    """Validate every connector parameter before changing the base model."""
    parameters = dict(model.named_parameters())
    expected = {key for key in parameters
                if (key.startswith('model.vision_backbone.')
                    and not key.startswith('model.vision_backbone.image_vit.'))
                or key == 'model.transformer.wte.new_embedding'}
    mapped = {hf_key(key): value for key, value in state.items()}
    if not expected or set(mapped) != expected:
        raise ValueError(f'Connector mismatch: missing={sorted(expected-set(mapped))}, '
                         f'extra={sorted(set(mapped)-expected)}')
    for key, value in mapped.items():
        if parameters[key].shape != value.shape:
            raise ValueError(f'Connector shape mismatch for {key}: {value.shape} != {parameters[key].shape}')
    with torch.no_grad():
        for key, value in mapped.items():
            parameters[key].copy_(value)
    return len(mapped)


class Molmo2Worker:
    def __init__(self, model_id='allenai/Molmo2-4B', device=None, dtype='bf16',
                 connector_path: Path | None = None,
                 connector_repo='kdaterao/so101-molmo2-4b-gripper', connector_revision='main',
                 batch_size: int = 4):
        if batch_size < 1:
            raise ValueError("Molmo batch_size must be positive")
        self.batch_size = int(batch_size)
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype == 'auto':
            dtype = 'bf16' if device != 'cpu' else 'fp32'
        model_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
        if connector_path is None:
            connector_path = Path(hf_hub_download(
                connector_repo, 'so101_connector.pt', revision=connector_revision))
        # This is the user's own trusted PyTorch checkpoint (may contain DTensors).
        state = load_connector(connector_path)
        print(f'Loading Molmo2 base {model_id} for inference ({dtype})', flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=model_dtype,
            device_map='auto' if device == 'auto' else {'': device},
        )
        count = apply_connector(self.model, state)
        del state
        self.model.requires_grad_(False).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        print(f'Molmo grounding batch size: {self.batch_size}', flush=True)
        print(f'Applied {count} fine-tuned tensors from {connector_path}; inference only', flush=True)

    def ground(self, image_rgb: np.ndarray, prompt: str) -> dict:
        result = self.ground_batch([(image_rgb, prompt)])[0]
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    @torch.inference_mode()
    def _generate_batch(self, requests) -> list[dict]:
        images, texts = [], []
        for image_rgb, prompt in requests:
            image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
            if prompt == 'Point to the robot gripper.':
                prompt = 'Point to the SO-101 gripper.'
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            images.append(image)
        # The official processor accepts a list of texts and a flat image list
        # in matching order. Left padding is required for decoder generation.
        inputs = self.processor(
            text=texts, images=images, padding=True, add_special_tokens=False,
            return_tensors='pt',
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, 'to') else value
                  for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        prompt_length = inputs['input_ids'].shape[1]
        raw_outputs = self.processor.tokenizer.batch_decode(
            output[:, prompt_length:], skip_special_tokens=True)
        return [
            {'points_xy': parse_image_points(raw, image.width, image.height), 'raw_response': raw}
            for raw, image in zip(raw_outputs, images)
        ]

    def ground_batch(self, requests) -> list[dict]:
        """Ground independent (RGB image, prompt) requests, preserving order.

        On batch failure, retry smaller groups. Unresolved singleton requests
        carry errors so one bad input cannot discard the other annotations.
        """
        requests = list(requests)

        def generate(group):
            try:
                results = self._generate_batch(group)
                if len(results) != len(group):
                    raise RuntimeError("Molmo returned an unexpected batch length")
                return results
            except Exception as exc:
                message = str(exc)
            # Leave the exception scope first so temporary generation tensors
            # can be released before retrying a smaller batch.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(group) > 1:
                middle = len(group) // 2
                print(f"Molmo batch of {len(group)} failed; retrying smaller groups: {message}", flush=True)
                return generate(group[:middle]) + generate(group[middle:])
            return [{'points_xy': np.zeros((0, 2), np.float32),
                     'raw_response': '', 'error': message}]

        results = []
        for start in range(0, len(requests), self.batch_size):
            results.extend(generate(requests[start:start + self.batch_size]))
        return results
