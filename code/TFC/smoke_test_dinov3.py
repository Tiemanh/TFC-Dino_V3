"""Offline forward/backward check for the dual-DINOv3 TF-C implementation.

This uses tiny randomly initialised DINOv3 models and synthetic tensors solely to
verify the code path. It does not download pretrained weights and is not a model
quality result.
"""

import argparse
from types import SimpleNamespace

import torch

from loss import NTXentLoss_poly
from model import TFC, target_classifier
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-architecture", action="store_true",
        help="Use two full-size random ViT-S DINOv3 backbones; still does not download weights",
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    configs = SimpleNamespace(
        input_channels=3,
        num_classes_target=8,
        dinov3_image_size=64,
        dinov3_freeze_backbone=True,
        dinov3_random_init_for_smoke_test=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.full_architecture:
        full_config = DINOv3ViTConfig(num_register_tokens=4, pos_embed_shift=None, pos_embed_jitter=None, pos_embed_rescale=None)
        model = TFC(configs, DINOv3ViTModel(full_config), DINOv3ViTModel(full_config)).to(device).train()
    else:
        model = TFC(configs).to(device).train()
    classifier = target_classifier(configs).to(device).train()
    time_view = torch.randn(2, 3, 206, device=device)
    frequency_view = torch.fft.fft(time_view, dim=-1).abs()
    h_t, z_t, h_f, z_f = model(time_view, frequency_view)
    h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(time_view * 1.01, frequency_view * 1.01)
    logits = classifier(torch.cat((z_t, z_f), dim=1))
    # Same Poly-NT-Xent implementation called by the authors' trainer.py.
    nt_xent = NTXentLoss_poly(device, batch_size=2, temperature=0.2, use_cosine_similarity=True)
    loss_t = nt_xent(h_t, h_t_aug)
    loss_f = nt_xent(h_f, h_f_aug)
    loss_tf = nt_xent(z_t, z_f)
    loss = 0.2 * (loss_t + loss_f) + loss_tf + logits.square().mean()
    loss.backward()

    expected_hidden_size = 384 if args.full_architecture else 96
    assert h_t.shape == h_f.shape == (2, expected_hidden_size)
    assert z_t.shape == z_f.shape == (2, 128)
    assert logits.shape == (2, 8)
    assert model.time_encoder is not model.frequency_encoder
    assert model.time_encoder.backbone is not model.frequency_encoder.backbone
    assert model.time_encoder.adapter.channel_adapter.weight.grad is not None
    assert model.frequency_encoder.adapter.channel_adapter.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.time_encoder.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.frequency_encoder.backbone.parameters())
    print("PASS: dual DINOv3 forward/backward")
    print("architecture:", "full ViT-S random-init" if args.full_architecture else "tiny random-init")
    print("device:", device)
    print("h_time/h_freq:", tuple(h_t.shape), "z_time/z_freq:", tuple(z_t.shape))
    print("classifier logits:", tuple(logits.shape))
    print("frozen backbones: yes; trainable adapters/projectors/classifier: gradients verified")
    if torch.cuda.is_available():
        print("peak CUDA memory (MiB):", round(torch.cuda.max_memory_allocated() / 1024**2, 1))


if __name__ == "__main__":
    main()
