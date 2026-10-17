"""Lazy inference adapters for the Action100M-inspired annotation pass."""
from __future__ import annotations

import gc
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
        self.qwen_video = kind == "video" and config.model_type == "qwen2_5_vl"
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

    def caption(self, requests, max_tokens=1024):
        conversations, flat_views = [], []
        for request in requests:
            prompt = (
                "Describe the visible actions and object movements in detail, in temporal order. "
                if self.kind == "video" else
                "Describe the visible objects and their spatial relationships in detail at this instant. "
            )
            prompt += (
                "The labeled camera views show the same scene at matching source times, not separate actions. "
                "Use the primary third-person view to describe the overall action, object location and world motion. "
                "Use the helper POV mainly for local interaction evidence: visible contact, whether an object is "
                "between the fingers, or whether it visibly separates from them. The wrist camera moves with the arm. "
                "An object moving upward, growing larger or shifting sideways in the POV can be caused entirely "
                "by camera motion; this does not establish that the object was lifted or moved. "
                "Do not infer object translation, lifting, transport or direction from POV image motion alone. "
                "Describe these movements only when supported by the primary view; otherwise describe only the "
                "local interaction that is actually visible. Do not invent an extra object just because it appears in both views. "
                "Retain concrete action steps and relevant changes in object relationships; avoid generic scene filler. "
                "Describe what happens rather than guessing intentions from posture. Do not add task-success "
                "or failure verdicts, sensor commentary, or an evidence audit. Mention uncertainty briefly only "
                "when a relevant action cannot be seen. Do not infer that the whole interval is stationary "
                "from one unchanged frame."
            )
            views = request.get("views") or [dict(camera="primary", role="primary", frames=request["frames"])]
            content = [{"type": "text", "text": prompt}]
            for view in views:
                content.extend([{"type": "text", "text": f'{view["role"].title()} view: {view["camera"]}.'},
                                {"type": self.kind}])
                flat_views.append((request, view))
            if len(views) == 1:
                content.append({"type": "text", "text": "Only the primary view is available; do not assume helper-view evidence."})
            conversations.append([{"role": "user", "content": content}])
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        if self.kind == "video":
            # Supply decoded sparse frames directly. Do not let the processor
            # sample another clip or silently resize it to its 448px default.
            video_kwargs = dict(size={"height": 320, "width": 320})
            if self.qwen_video:
                # Qwen's 14px patches and 2x merge require a 28px grid.
                # Use 308px (nearest grid below 320), and preserve source
                # timing so the processor derives the sparse clip's FPS.
                video_kwargs = dict(size={"shortest_edge": 308 ** 2, "longest_edge": 308 ** 2},
                                    video_metadata=[dict(fps=r["fps"], frames_indices=r["frame_indices"],
                                                         total_num_frames=r["total_num_frames"])
                                                    for r, _ in flat_views])
            inputs = self.processor(text=text, videos=[view["frames"] for _, view in flat_views],
                                    do_sample_frames=False, **video_kwargs,
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
