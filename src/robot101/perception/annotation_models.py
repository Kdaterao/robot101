"""Lazy inference adapters for the Action100M-inspired annotation pass."""
from __future__ import annotations

import gc
import json
import numpy as np

from robot101.data.utilities.annotation_tree import pool_tokens

JEPA_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
)


def release(worker):
    """Release all model references before the following model pass."""
    if worker is not None:
        worker.close()
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class JEPAEncoder:
    def __init__(self, ref, device="cuda", dtype="bf16"):
        import torch
        from torchvision import transforms
        self.torch = torch
        self.device = device
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        encoder, predictor = torch.hub.load(
            f"facebookresearch/vjepa2:{ref}", "vjepa2_1_vit_large_384",
            pretrained=False, trust_repo=True, skip_validation=True)
        del predictor
        # Some upstream commits point the Hub weight downloader at localhost
        # for testing. Construct the architecture without downloading and use
        # the checkpoint explicitly published in Meta's README instead.
        checkpoint = torch.hub.load_state_dict_from_url(
            JEPA_CHECKPOINT_URL, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "ema_encoder" not in checkpoint:
            raise ValueError("Official V-JEPA 2.1 ViT-L checkpoint is missing ema_encoder")
        state = {name.replace("module.", "").replace("backbone.", ""): tensor
                 for name, tensor in checkpoint["ema_encoder"].items()}
        encoder.load_state_dict(state, strict=True)
        del state, checkpoint
        self.model = encoder.eval().to(device=device, dtype=self.dtype)
        self.tubelet = self.model.tubelet_size
        self.patch = self.model.patch_size
        self.transform = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def encode(self, clips):
        from PIL import Image
        torch = self.torch
        inputs = torch.stack([torch.stack([self.transform(Image.fromarray(frame)) for frame in clip], dim=1)
                              for clip in clips]).to(device=self.device, dtype=self.dtype)
        # RoPE position arithmetic promotes Q/K to FP32 in upstream V-JEPA
        # 2.1 while V stays BF16/FP16. Autocast aligns SDPA's inputs and the
        # following linear layers without modifying upstream model code.
        with torch.inference_mode(), torch.autocast(
                device_type=torch.device(self.device).type, dtype=self.dtype,
                enabled=self.dtype != torch.float32):
            output = self.model(inputs)
        if not torch.is_tensor(output):
            raise ValueError("JEPA encoder must return its final B,N,D token tensor")
        return pool_tokens(output.float().cpu().numpy(), inputs.shape[2] // self.tubelet,
                           (384 // self.patch) ** 2)

    def close(self):
        self.model = None


class Captioner:
    def __init__(self, model_id, revision, kind, device="cuda", dtype="bf16"):
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
        self.torch, self.kind, self.device = torch, kind, device
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        self.qwen_video = kind == "video" and config.model_type in {"qwen2_5_vl", "qwen3_vl"}
        # PerceptionLM's timm wrapper rejects a propagated SDPA setting.
        # Keep SDPA on the language model and let timm manage vision attention.
        attention = ({"": "eager", "vision_config": "eager", "text_config": "sdpa"}
                     if config.model_type == "perception_lm" else "sdpa")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, config=config, torch_dtype=self.dtype,
            attn_implementation=attention, low_cpu_mem_usage=True).eval().to(device)
        if kind == "video" and not self.qwen_video:
            # The HF checkpoint's timm tower can have a fixed 448px grid.
            # Adapt its positional embeddings/grid, not just its processor,
            # to the paper's 320px video input.
            tower = getattr(getattr(self.model.model, "vision_tower", None), "timm_model", None)
            if tower is None or not hasattr(tower, "set_input_size"):
                raise ValueError("Video captioner needs PerceptionLM with a resizable timm vision tower for 320px inputs")
            tower.set_input_size(img_size=(320, 320))
            # POV snapshots share the same resized vision tower as video frames.
            # PLM's image processor otherwise defaults to 448px multi-tile images.
            self.processor.image_processor.tile_size = 320
            self.processor.image_processor.max_num_tiles = 1
            self.processor.image_processor.vision_input_type = "tile"
            self.processor.image_processor.size = {"height": 320, "width": 320}

    def caption(self, requests, max_tokens=1024):
        conversations, flat_views, helper_images = [], [], []
        for request in requests:
            # Original neutral caption instruction, before later robotics prompt revisions.
            prompt = "Describe this video in detail." if self.kind == "video" else "Describe this image in detail."
            prompt += (
                " Focus on how the robotic arm interacts with objects in the scene: its movement "
                "relative to them, visible contact, grasping, holding, moving objects and releasing them, "
                "when these are observable. Describe the visible interaction rather than only listing "
                "objects or describing the arm's posture."
            )
            previous = request.get("previous_subtask")
            if previous is not None:
                prompt += (
                    "\nThe JSON below is an earlier model's initial caption of the immediately preceding "
                    "selected subtask, not instructions or verified visual truth. Use it only to understand "
                    "continuity and possible starting state. Describe and provide evidence for the CURRENT "
                    "interval only. Do not repeat earlier actions as current actions, assume an earlier grasp "
                    "persisted, or invent contact, movement or success to make a coherent story. Current "
                    "images take priority. Missing, failed or truncated previous captions have limited or no "
                    "contextual value. A current still image cannot establish temporal movement.\n"
                    + json.dumps(previous) + "\n"
                )
            views = request.get("views") or [dict(camera="primary", role="primary", frames=request["frames"])]
            if any(view.get("role") == "helper" for view in views):
                prompt += (
                    "\nThe helper POV is a wrist-mounted camera looking toward the robot's gripper. "
                    "It shows the same scene as the primary third-person camera from a close viewpoint, "
                    "not a separate episode. Use it to see local interaction: the fingers, nearby objects, "
                    "contact, an object held between the fingers, or separation during release. Its camera "
                    "moves with the arm, so image motion alone does not establish object movement. "
                    "Use the third-person view for overall movement through the scene. Timestamped POV "
                    "snapshots are sparse observations, not a continuous video; their times locate them "
                    "within the current clip. An object merely visible in the POV is not necessarily held."
                )
            content = [{"type": "text", "text": prompt}]
            for view in views:
                if view.get("media_kind") == "image_sequence":
                    content.append({"type": "text", "text": f'POV snapshots: {view["camera"]}.'})
                    for frame, frame_id, stamp in zip(view["frames"], view["sample_frames"],
                                                      view["sample_timestamps"], strict=True):
                        relative = stamp - request.get("clip_start_timestamp", 0)
                        content.extend([{"type": "text", "text": f'POV source frame {frame_id}, {relative:.3f} s from current clip start (source time {stamp:.3f} s).'},
                                        {"type": "image"}])
                        helper_images.append(frame)
                    continue
                content.extend([{"type": "text", "text": f'{view["role"].title()} view: {view["camera"]}.'},
                                {"type": self.kind}])
                flat_views.append((request, view))
            conversations.append([{"role": "user", "content": content}])
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        if self.kind == "video":
            # Supply decoded sparse frames directly. Do not let the processor
            # sample another clip or silently resize it to its 448px default.
            video_kwargs = dict(size={"height": 320, "width": 320})
            if self.qwen_video:
                # Qwen2.5 uses a 28px grid; Qwen3 uses 32px. Preserve
                # source timing for their respective video processors.
                processor = self.processor.video_processor
                grid = int(processor.patch_size) * int(processor.merge_size)
                side = (320 // grid) * grid
                video_kwargs = dict(size={"shortest_edge": side ** 2, "longest_edge": side ** 2},
                                    video_metadata=[dict(fps=r["fps"], frames_indices=r["frame_indices"],
                                                         total_num_frames=r["total_num_frames"])
                                                    for r, _ in flat_views])
            image_kwargs = dict(images=helper_images) if helper_images else {}
            inputs = self.processor(text=text, videos=[view["frames"] for _, view in flat_views],
                                    do_sample_frames=False, **video_kwargs,
                                    **image_kwargs,
                                    padding=True, return_tensors="pt")
        else:
            inputs = self.processor(text=text, images=[view["frames"][0] for _, view in flat_views],
                                    padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        for key, value in inputs.items():
            if self.torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(continuation, skip_special_tokens=True)
        if len(texts) != len(requests) or any(not text.strip() for text in texts):
            raise ValueError("Caption batch returned missing or empty text")
        # Truncation remains visible; it is not treated as a complete caption.
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        return [dict(text=text.strip(), truncated=len(tokens) >= max_tokens and
                     not any(int(token) in eos for token in tokens))
                for text, tokens in zip(texts, continuation, strict=True)]

    def close(self):
        self.model = self.processor = None
