"""Reusable inference API for the three-column web application in part 3."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from model import TFC, target_classifier


class GesturePredictor:
    def __init__(self, bundle_path, device="cpu"):
        self.device = torch.device(device)
        bundle = torch.load(bundle_path, map_location=self.device, weights_only=False)
        self.training_info = bundle.get("training_info", {})
        if self.training_info.get("initialization") == "smoke_test":
            raise ValueError("Smoke-test bundle is not a trained gesture model")
        self.metrics = bundle.get("metrics", {})
        options = bundle["model_options"]
        self.metadata = bundle["metadata"]
        self.class_names = bundle["class_names"]
        config = SimpleNamespace(
            input_channels=options["input_channels"],
            num_classes_target=len(self.class_names),
            dinov3_image_size=options["image_size"],
            dinov3_freeze_backbone=True,
            dinov3_backbone_config=bundle["dinov3_backbone_config"],
        )
        self.model = TFC(config).to(self.device)
        self.classifier = target_classifier(config).to(self.device)
        self.model.load_state_dict(bundle["model_state_dict"])
        self.classifier.load_state_dict(bundle["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()

    def preprocess(self, xyz):
        values = np.asarray(xyz, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Input must have shape [number_of_samples, 3] for X/Y/Z")
        window_size = int(self.metadata["window_size"])
        if len(values) < window_size:
            raise ValueError(f"Need at least {window_size} sensor samples")
        values = values[-window_size:]
        normalization = self.metadata["normalization"]
        low = np.asarray(normalization["clip_low"], dtype=np.float32)
        high = np.asarray(normalization["clip_high"], dtype=np.float32)
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        values = (np.clip(values, low, high) - mean) / std
        return torch.from_numpy(values.T).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, xyz):
        values = np.asarray(xyz, dtype=np.float32)
        window_size = int(self.metadata["window_size"])
        # Validate before creating windows; preprocess enforces channel/length checks.
        self.preprocess(values)
        if not np.isfinite(values).all():
            raise ValueError("Input contains non-finite sensor values")
        stride = int(self.metadata.get("stride", max(1, window_size // 2)))
        if stride <= 0:
            raise ValueError("Model stride must be positive")
        starts = list(range(0, len(values) - window_size + 1, stride))
        if starts[-1] != len(values) - window_size:
            starts.append(len(values) - window_size)
        # Average real predictions over the recording, rather than using only its tail.
        total_probabilities = torch.zeros(len(self.class_names), device=self.device)
        for offset in range(0, len(starts), 16):
            batch_starts = starts[offset:offset + 16]
            time_view = torch.cat([self.preprocess(values[start:start + window_size])
                                   for start in batch_starts])
            frequency_view = torch.fft.fft(time_view, dim=-1).abs()
            _, z_t, _, z_f = self.model(time_view, frequency_view)
            logits = self.classifier(torch.cat((z_t, z_f), dim=1))
            total_probabilities += logits.softmax(dim=1).sum(dim=0)
        probabilities = (total_probabilities / len(starts)).cpu()
        best = int(probabilities.argmax())
        return {
            "label": self.class_names[best],
            "confidence": float(probabilities[best]),
            "probabilities": {name: float(score) for name, score in zip(self.class_names, probabilities)},
            "training_info": self.training_info,
            "evaluation": self.metrics,
            "window_size": int(self.metadata["window_size"]),
            "window_selection": "mean_probabilities_all_windows",
            "windows_used": len(starts),
        }
