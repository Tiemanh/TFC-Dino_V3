import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'TFC'))
from train_gestures import cap_per_class, classification_metrics

spec = importlib.util.spec_from_file_location('import_private_recording', ROOT / 'tools' / 'import_private_recording.py')
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
package_spec = importlib.util.spec_from_file_location('package_private_dataset', ROOT / 'tools' / 'package_private_dataset.py')
packager = importlib.util.module_from_spec(package_spec)
package_spec.loader.exec_module(packager)


class OfflineTrainingTests(unittest.TestCase):
    def test_kaggle_packager_rejects_public_data_without_private_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            raw, prepared = root / 'raw', root / 'prepared'
            raw.mkdir(); prepared.mkdir()
            (raw / 'ingesture.csv').write_text('x,y,z\n1,2,3\n', encoding='utf-8')
            (raw / 'ingesture.json').write_text(json.dumps({'source_type': 'public_ingesture'}))
            (prepared / 'metadata.json').write_text(json.dumps({'class_names': list('abcdefgh')}))
            with self.assertRaisesRegex(ValueError, 'self-collection provenance'):
                packager.package(raw, prepared, root / 'upload', 'owner/dataset', 'title', 'CC-BY-4.0')
            self.assertFalse((root / 'upload').exists())

    def test_kaggle_packager_copies_real_input_without_publishing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            raw, prepared = root / 'raw', root / 'prepared'
            raw.mkdir(); prepared.mkdir()
            (raw / 'take.csv').write_text('x,y,z\n1,2,3\n', encoding='utf-8')
            (raw / 'take.json').write_text(json.dumps({'source_type': 'user_recorded'}))
            (prepared / 'metadata.json').write_text(json.dumps({'class_names': list('abcdefgh')}))
            for name in ('train.pt', 'test.pt'):
                torch.save({'samples': torch.randn(8, 3, 206), 'labels': torch.arange(8)}, prepared / name)
            output = packager.package(raw, prepared, root / 'upload', 'owner/dataset', 'title', 'CC-BY-4.0')
            self.assertEqual((output / 'raw' / 'take.csv').read_bytes(), (raw / 'take.csv').read_bytes())
            self.assertEqual(json.loads((output / 'dataset-metadata.json').read_text())['id'], 'owner/dataset')

    def test_metrics_expose_majority_class_collapse(self):
        confusion = torch.zeros(8, 8, dtype=torch.long)
        confusion[:, 0] = 10
        confusion[0, 0] = 930
        report = classification_metrics(confusion, [str(i) for i in range(8)])
        self.assertAlmostEqual(report['test_accuracy'], .93)
        self.assertAlmostEqual(report['balanced_accuracy'], .125)
        self.assertLess(report['macro_f1'], .13)

    def test_training_subset_keeps_original_label_ids(self):
        payload = {'samples': torch.randn(80, 3, 206), 'labels': torch.arange(80) % 8}
        subset = cap_per_class(payload, 3, 42)
        self.assertEqual(len(subset['labels']), 24)
        self.assertEqual(torch.bincount(subset['labels']).tolist(), [3] * 8)
        self.assertTrue(torch.equal(subset['samples'], cap_per_class(payload, 3, 42)['samples']))

    def test_phone_import_converts_units_and_preserves_label(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            raw = folder / 'phone.csv'
            times = np.arange(0, 2, .01)
            pd.DataFrame({'timestamp': times, 'x': np.ones(len(times)) * 9.80665,
                          'y': np.zeros(len(times)), 'z': np.zeros(len(times))}).to_csv(raw, index=False)
            result = importer.import_recording(raw, folder / 'private', 'waving_left', 'p01', '001')
            frame = pd.read_csv(result)
            self.assertEqual(frame['label'].unique().tolist(), ['waving_left'])
            self.assertAlmostEqual(frame['x'].mean(), 1.)
            self.assertTrue(result.with_suffix('.json').exists())
            self.assertGreaterEqual(len(frame), 206)
            with self.assertRaises(FileExistsError):
                importer.import_recording(raw, folder / 'private', 'waving_left', 'p01', '001')


if __name__ == '__main__':
    unittest.main()
