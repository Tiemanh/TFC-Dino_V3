import io
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.main import parse_xyz  # noqa: E402
from app.data_io import parse_xyz_bytes


class WebInputTests(unittest.TestCase):
    def test_oversized_upload_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "5 MB"):
            parse_xyz_bytes(b"x" * (5 * 1024 * 1024 + 1))
    def test_csv_aliases_and_missing_value_interpolation(self):
        csv = b"timestamp,accX,accY,accZ\n0,1,2,3\n1,,4,5\n2,3,6,7\n"
        values = parse_xyz(csv)
        self.assertEqual(values.shape, (3, 3))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[1, 0], 2.0)

    def test_missing_axis_is_rejected(self):
        with self.assertRaises(Exception):
            parse_xyz(b"x,y\n1,2\n")


if __name__ == "__main__":
    unittest.main()
