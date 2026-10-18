import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "TFC"))
from train_gestures import cap_per_class, classification_metrics


class OfflineTrainingTests(unittest.TestCase):
    def test_metrics_expose_majority_class_collapse(self):
        confusion = torch.zeros(8, 8, dtype=torch.long)
        confusion[:, 0] = 10
        confusion[0, 0] = 930
        report = classification_metrics(confusion, [str(i) for i in range(8)])
        self.assertAlmostEqual(report["test_accuracy"], 0.93)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.125)
        self.assertLess(report["macro_f1"], 0.13)

    def test_training_subset_keeps_original_label_ids(self):
        payload = {"samples": torch.randn(80, 3, 206), "labels": torch.arange(80) % 8}
        subset = cap_per_class(payload, 3, 42)
        self.assertEqual(len(subset["labels"]), 24)
        self.assertEqual(torch.bincount(subset["labels"]).tolist(), [3] * 8)
        self.assertTrue(torch.equal(subset["samples"], cap_per_class(payload, 3, 42)["samples"]))


if __name__ == "__main__":
    unittest.main()
