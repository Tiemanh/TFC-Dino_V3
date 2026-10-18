import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


TFC_DIR = Path(__file__).resolve().parents[1] / "code" / "TFC"
sys.path.insert(0, str(TFC_DIR))
from model import SignalToImageAdapter, TFC, target_classifier, load_local_dinov3  # noqa: E402
from inference import GesturePredictor  # noqa: E402


class DualDINOv3Tests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            input_channels=3,
            num_classes_target=8,
            dinov3_image_size=64,
            dinov3_freeze_backbone=True,
            dinov3_random_init_for_smoke_test=True,
        )

    def test_adapter_shape_and_finite_values(self):
        adapter = SignalToImageAdapter(3, 64)
        image = adapter(torch.randn(2, 3, 206))
        self.assertEqual(image.shape, (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(image).all())

    def test_bad_channel_count_is_rejected(self):
        adapter = SignalToImageAdapter(3, 64)
        with self.assertRaisesRegex(ValueError, "expects 3 channels"):
            adapter(torch.randn(2, 1, 206))

    def test_dual_encoders_shapes_and_identity(self):
        model = TFC(self.config).eval()
        time_view = torch.randn(2, 3, 206)
        frequency_view = torch.fft.fft(time_view, dim=-1).abs()
        outputs = model(time_view, frequency_view)
        self.assertEqual([tuple(value.shape) for value in outputs], [(2, 96), (2, 128), (2, 96), (2, 128)])
        self.assertIsNot(model.time_encoder.backbone, model.frequency_encoder.backbone)

    def test_gradients_reach_adapters_but_not_frozen_backbones(self):
        model = TFC(self.config).train()
        time_view = torch.randn(2, 3, 206)
        frequency_view = torch.fft.fft(time_view, dim=-1).abs()
        _, z_t, _, z_f = model(time_view, frequency_view)
        (z_t.square().mean() + z_f.square().mean()).backward()
        self.assertIsNotNone(model.time_encoder.adapter.channel_adapter.weight.grad)
        self.assertIsNotNone(model.frequency_encoder.adapter.channel_adapter.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.time_encoder.backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.frequency_encoder.backbone.parameters()))

    def test_classifier_has_eight_outputs(self):
        classifier = target_classifier(self.config)
        self.assertEqual(classifier(torch.randn(2, 256)).shape, (2, 8))

    def test_from_scratch_uses_full_vits16_and_trains_backbones(self):
        self.config.dinov3_random_init_for_smoke_test = False
        self.config.dinov3_train_from_scratch = True
        self.config.dinov3_freeze_backbone = False
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            model = TFC(self.config)
        self.assertEqual(model.time_encoder.hidden_size, 384)
        self.assertEqual(model.time_encoder.backbone.config.num_hidden_layers, 12)
        self.assertEqual(model.frequency_encoder.backbone.config.num_register_tokens, 4)
        self.assertTrue(all(p.requires_grad for p in model.time_encoder.backbone.parameters()))

    def test_local_weights_load_into_both_encoders_without_network(self):
        # Tiny random weights only test serialization, not gesture quality.
        source = TFC(self.config).time_encoder.backbone
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source.save_pretrained(directory)
            self.config.dinov3_random_init_for_smoke_test = False
            self.config.dinov3_local_model_dir = directory
            with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                model = TFC(self.config).eval()
            self.assertIsNot(model.time_encoder.backbone, model.frequency_encoder.backbone)
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, model.time_encoder.backbone.state_dict()[name]))
                self.assertTrue(torch.equal(tensor, model.frequency_encoder.backbone.state_dict()[name]))
            values = model(torch.randn(2, 3, 206), torch.rand(2, 3, 206))
            self.assertTrue(all(torch.isfinite(value).all() for value in values))

    def test_local_weights_missing_does_not_fall_back_to_download(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch('model.AutoModel.from_pretrained', side_effect=AssertionError('Download forbidden')):
                with self.assertRaisesRegex(FileNotFoundError, 'architecture file'):
                    load_local_dinov3(directory)

    def test_local_incomplete_checkpoint_is_rejected(self):
        from safetensors.torch import save_file
        source = TFC(self.config).time_encoder.backbone
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source.config.save_pretrained(directory)
            weights = dict(source.state_dict())
            weights.pop(next(iter(weights)))
            save_file(weights, str(Path(directory) / 'model.safetensors'), metadata={'format': 'pt'})
            with self.assertRaisesRegex(ValueError, 'does not fully match'):
                load_local_dinov3(directory)

    def test_export_bundle_can_be_loaded_for_web_inference(self):
        model = TFC(self.config).eval()
        classifier = target_classifier(self.config).eval()
        metadata = {
            "window_size": 64,
            "normalization": {
                "clip_low": [-5.0, -5.0, -5.0],
                "clip_high": [5.0, 5.0, 5.0],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
        bundle = {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "dinov3_backbone_config": model.time_encoder.backbone.config.to_dict(),
            "class_names": [f"gesture_{index}" for index in range(8)],
            "metadata": metadata,
            "model_options": {"input_channels": 3, "image_size": 64, "freeze_backbone": True},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            bundle_path = Path(temporary) / "model_bundle.pt"
            torch.save(bundle, bundle_path)
            predictor = GesturePredictor(bundle_path)
            result = predictor.predict(torch.randn(64, 3).numpy())
            long_result = predictor.predict(torch.randn(160, 3).numpy())
        self.assertIn(result["label"], bundle["class_names"])
        self.assertEqual(len(result["probabilities"]), 8)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertGreater(long_result["windows_used"], 1)
        self.assertAlmostEqual(sum(long_result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
