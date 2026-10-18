import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_gesture_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_gesture_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TFC_DIR = Path(__file__).resolve().parents[1] / "code" / "TFC"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(TFC_DIR))
sys.path.insert(0, str(CONFIG_DIR))
from config_files.HAR_Configs import Config  # noqa: E402
from dataloader import Load_Dataset  # noqa: E402
from model import TFC, target_classifier  # noqa: E402
from train_gestures import run_finetune, run_pretrain  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class GestureDataPipelineTests(unittest.TestCase):
    def test_pretrain_augmentations_work_on_cpu(self):
        config = Config()
        config.TSlength_aligned = 32
        payload = {
            "samples": torch.randn(16, 3, 32),
            "labels": torch.arange(16) % 8,
        }
        dataset = Load_Dataset(payload, config, "pre_train", subset=False)
        sample = dataset[0]
        self.assertEqual(len(sample), 5)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in (sample[0], sample[2], sample[3], sample[4])))
        self.assertTrue(all(value.device.type == "cpu" for value in (sample[0], sample[2], sample[3], sample[4])))

    def test_tiny_end_to_end_training_loops(self):
        config = Config()
        config.TSlength_aligned = 32
        config.dinov3_image_size = 64
        config.dinov3_freeze_backbone = True
        config.dinov3_random_init_for_smoke_test = True
        payload = {"samples": torch.randn(4, 3, 32), "labels": torch.arange(4)}
        pretrain_loader = DataLoader(Load_Dataset(payload, config, "pre_train"), batch_size=2, drop_last=True)
        train_loader = DataLoader(Load_Dataset(payload, config, "fine_tune_test"), batch_size=2)
        model = TFC(config)
        classifier = target_classifier(config)
        model_optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
        run_pretrain(model, pretrain_loader, model_optimizer, config, torch.device("cpu"), 1)
        run_finetune(model, classifier, train_loader, model_optimizer, classifier_optimizer, torch.device("cpu"), 1)

    def test_clean_split_segment_and_normalize(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            raw = Path(temporary) / "raw"
            raw.mkdir()
            for label_index in range(8):
                for take in range(2):
                    values = np.linspace(0, 1, 24, dtype=np.float32)
                    frame = pd.DataFrame({
                        "x": values + label_index,
                        "y": values * 2 + take,
                        "z": values * 3,
                        "label": f"gesture_{label_index}",
                        "recording_id": f"g{label_index}_take{take}",
                    })
                    if label_index == 0 and take == 0:
                        frame.loc[4, "x"] = np.nan
                    frame.to_csv(raw / f"g{label_index}_take{take}.csv", index=False)
            data = MODULE.read_recordings(raw)
            train_ids, test_ids = MODULE.split_recordings(data, 0.5, 42)
            classes = sorted(data["label"].unique())
            mapping = {name: index for index, name in enumerate(classes)}
            stats = MODULE.fit_normalization(data, train_ids)
            train = MODULE.make_windows(data, train_ids, mapping, 16, 8, stats)
            test = MODULE.make_windows(data, test_ids, mapping, 16, 8, stats)
            self.assertEqual(train["samples"].shape, (16, 3, 16))
            self.assertEqual(test["samples"].shape, (16, 3, 16))
            self.assertEqual(set(train["labels"].tolist()), set(range(8)))
            self.assertTrue(torch.isfinite(train["samples"]).all())


if __name__ == "__main__":
    unittest.main()
