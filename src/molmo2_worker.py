"""Molmo2 inference with the separately uploaded SO-101 connector."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from molmo2_connector import hf_key, load_connector


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
                 connector_repo='kdaterao/so101-molmo2-4b-gripper', connector_revision='main'):
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
        print(f'Applied {count} fine-tuned tensors from {connector_path}; inference only', flush=True)

    @torch.inference_mode()
    def ground(self, image_rgb: np.ndarray, prompt: str) -> dict:
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
        # Match the gripper label used by the point annotation training dataset.
        if prompt == 'Point to the robot gripper.':
            prompt = 'Point to the SO-101 gripper.'
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, 'to') else value
                  for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        raw = self.processor.tokenizer.decode(
            output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return {'points_xy': parse_image_points(raw, image.width, image.height), 'raw_response': raw}
