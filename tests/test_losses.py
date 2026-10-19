import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "TFC"))
from loss import ClassBalancedFocalLoss


class FocalLossTests(unittest.TestCase):
    def test_recovers_cross_entropy(self):
        logits = torch.randn(6, 3, requires_grad=True)
        labels = torch.arange(6) % 3
        loss = ClassBalancedFocalLoss([100, 10, 2], beta=0, gamma=0)(logits, labels)
        torch.testing.assert_close(loss, F.cross_entropy(logits, labels))

    def test_rare_class_gets_larger_weight_and_gradients_are_finite(self):
        criterion = ClassBalancedFocalLoss([1000, 10, 1])
        self.assertGreater(criterion.class_weights[2], criterion.class_weights[0])
        logits = torch.tensor([[100., -100., 0.], [-100., 100., 0.]], requires_grad=True)
        loss = criterion(logits, torch.tensor([1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_missing_training_class_is_rejected(self):
        with self.assertRaises(ValueError):
            ClassBalancedFocalLoss([10, 0, 5])
