"""TF-C model with DINOv3 encoders for time and frequency views.

This keeps the four-output interface of the authors' January 2023 TF-C code, so
``trainer.py`` and the original contrastive losses continue to work. Two DINOv3
vision backbones replace the two TransformerEncoder blocks. A differentiable
adapter converts each multichannel 1-D sensor view to a 3-channel image.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel


DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_local_dinov3(directory) -> DINOv3ViTModel:
    """Load a Transformers-format DINOv3 directory without network access.

    A raw Meta .pth is a different format and must be converted first; do not
    silently replace missing or incompatible pretrained tensors with random ones.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Local DINOv3 folder not found: {directory}. "
            "Expected config.json and model.safetensors (or pytorch_model.bin). "
            "A raw .pth checkpoint must be converted first."
        )
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DINOv3 architecture file: {config_path}")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if config_data.get("model_type") != "dinov3_vit":
        raise ValueError("Local weights must be a DINOv3 ViT model (model_type=dinov3_vit)")
    weight_names = (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )
    if not any((directory / name).is_file() for name in weight_names):
        raise FileNotFoundError(f"Missing pretrained weight files in {directory}")
    backbone, info = DINOv3ViTModel.from_pretrained(
        str(directory), local_files_only=True, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Local DINOv3 checkpoint does not fully match its architecture: {info}")
    return backbone


class SignalToImageAdapter(nn.Module):
    """Convert ``[batch, channels, length]`` into a DINOv3 RGB tensor."""

    def __init__(self, input_channels: int, image_size: int = 224):
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of DINOv3 patch size 16")
        self.input_channels = input_channels
        self.image_size = image_size
        self.channel_adapter = nn.Conv1d(input_channels, 3, kernel_size=1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Start from an interpretable channel mapping, then allow learning."""
        nn.init.zeros_(self.channel_adapter.weight)
        nn.init.zeros_(self.channel_adapter.bias)
        with torch.no_grad():
            for output_channel in range(3):
                source_channel = output_channel % self.input_channels
                self.channel_adapter.weight[output_channel, source_channel, 0] = 1.0

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"Expected [batch, channels, length], got {tuple(signal.shape)}")
        if signal.shape[1] != self.input_channels:
            raise ValueError(
                f"Adapter expects {self.input_channels} channels, got {signal.shape[1]}. "
                "Set configs.input_channels to match the dataset."
            )
        signal = signal.float()
        mean = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        signal = (signal - mean) / std
        rgb_traces = self.channel_adapter(signal)
        rgb_traces = F.interpolate(rgb_traces, size=self.image_size, mode="linear", align_corners=False)
        rgb_traces = (torch.tanh(rgb_traces / 3.0) + 1.0) / 2.0
        image = rgb_traces.unsqueeze(2).expand(-1, -1, self.image_size, -1).contiguous()
        return (image - self.image_mean) / self.image_std


def _tiny_backbone_config(image_size: int) -> DINOv3ViTConfig:
    """Small random DINOv3 used only by tests and the smoke test."""
    return DINOv3ViTConfig(
        image_size=image_size,
        patch_size=16,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_register_tokens=2,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=None,
    )


class DINOv3SignalEncoder(nn.Module):
    """One TF-C view adapter followed by one DINOv3 backbone."""

    def __init__(self, configs, view_name: str, backbone: nn.Module | None = None):
        super().__init__()
        self.view_name = view_name
        self.freeze_backbone = bool(getattr(configs, "dinov3_freeze_backbone", True))
        self.adapter = SignalToImageAdapter(
            input_channels=int(configs.input_channels),
            image_size=int(getattr(configs, "dinov3_image_size", 224)),
        )
        if backbone is None:
            backbone = self._load_backbone(configs)
        self.backbone = backbone
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.set_backbone_trainable(not self.freeze_backbone)

    def _load_backbone(self, configs) -> nn.Module:
        backbone_config = getattr(configs, "dinov3_backbone_config", None)
        if backbone_config is not None:
            # Deployment bundles carry the architecture config and full state
            # dict, so inference does not need to contact Hugging Face.
            return DINOv3ViTModel(DINOv3ViTConfig.from_dict(backbone_config))
        if bool(getattr(configs, "dinov3_random_init_for_smoke_test", False)):
            return DINOv3ViTModel(_tiny_backbone_config(self.adapter.image_size))
        if bool(getattr(configs, "dinov3_train_from_scratch", False)):
            # Full ViT-S/16 architecture, not the reduced smoke-test model.
            return DINOv3ViTModel(DINOv3ViTConfig(
                image_size=self.adapter.image_size, num_register_tokens=4,
                pos_embed_shift=None, pos_embed_jitter=None, pos_embed_rescale=None,
            ))
        local_directory = getattr(configs, "dinov3_local_model_dir", None)
        if local_directory:
            return load_local_dinov3(local_directory)
        return AutoModel.from_pretrained(
            getattr(configs, "dinov3_model_name", DEFAULT_DINOV3_MODEL),
            local_files_only=bool(getattr(configs, "dinov3_local_files_only", False)),
        )

    def set_backbone_trainable(self, trainable: bool):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if trainable:
            self.backbone.train(self.training)
        else:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        image = self.adapter(signal)
        # Do not wrap a frozen backbone in no_grad: gradients must still flow
        # through it to train the signal-to-image adapter.
        output = self.backbone(pixel_values=image)
        return output.last_hidden_state[:, 0, :]


class TFC(nn.Module):
    """TF-C with separate DINOv3 time and frequency encoders."""

    def __init__(self, configs, time_backbone: nn.Module | None = None, frequency_backbone: nn.Module | None = None):
        super().__init__()
        self.time_encoder = DINOv3SignalEncoder(configs, "time", time_backbone)
        self.frequency_encoder = DINOv3SignalEncoder(configs, "frequency", frequency_backbone)
        if self.time_encoder.hidden_size != self.frequency_encoder.hidden_size:
            raise ValueError("Time and frequency DINOv3 backbones must have the same hidden size")
        hidden_size = self.time_encoder.hidden_size
        self.projector_t = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.projector_f = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x_in_t: torch.Tensor, x_in_f: torch.Tensor):
        h_time = self.time_encoder(x_in_t)
        z_time = self.projector_t(h_time)
        h_freq = self.frequency_encoder(torch.log1p(x_in_f.clamp_min(0)))
        z_freq = self.projector_f(h_freq)
        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):
    """Original TF-C classifier; input remains two 128-D projectors."""

    def __init__(self, configs):
        super().__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        return self.logits_simple(emb)
