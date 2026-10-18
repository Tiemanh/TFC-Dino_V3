"""Measure full dual ViT-S/16 training memory before the real run.

Random tensors are only for hardware measurement, never for reported accuracy.
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "TFC"))
sys.path.insert(0, str(ROOT / "code"))
from config_files.InGesture_Configs import Config
from model import TFC, target_classifier
from train_gestures import contrastive_loss


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this hardware benchmark")
    config = Config()
    config.dinov3_train_from_scratch = True
    config.dinov3_freeze_backbone = False
    config.dinov3_image_size = 64
    model = TFC(config).cuda().train()
    classifier = target_classifier(config).cuda().train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
    batch_size = 8
    signal = torch.randn(batch_size, 3, 206, device="cuda")
    freq = torch.fft.fft(signal, dim=-1).abs()
    start = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(model, signal, signal + .05 * torch.randn_like(signal),
                                freq, freq * 1.01, batch_size, config, torch.device("cuda"))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    report = {"gpu": torch.cuda.get_device_name(), "image_size": 64,
              "batch_size": batch_size, "architecture": "two full DINOv3 ViT-S/16",
              "seconds_per_pretrain_batch": (time.monotonic() - start) / 3,
              "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "purpose": "hardware benchmark only, not model quality"}
    print(json.dumps(report, indent=2), flush=True)
    out = ROOT / "artifacts" / "hardware_benchmark.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
