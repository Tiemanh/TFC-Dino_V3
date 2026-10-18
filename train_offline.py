"""Reproducible offline baseline training with no pretrained downloads."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets' / 'InGestureOffline')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'gesture_model')
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--finetune-epochs', type=int, default=12)
    parser.add_argument('--train-per-class', type=int, default=256)
    args = parser.parse_args()
    if (args.output_dir / 'model_bundle.pt').exists():
        print('Model already exists. Use --output-dir with a new folder to preserve this model.')
        return 1
    if not (args.data_dir / 'metadata.json').exists():
        if args.data_dir.resolve() != (ROOT / 'datasets' / 'InGestureOffline').resolve():
            parser.error('Prepare this private dataset before starting training')
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'prepare_ingesture_data.py'),
                        '--input-dir', str(ROOT / 'data' / 'external' / 'InGesture'),
                        '--output-dir', str(args.data_dir), '--val-ratio', '0.16'], check=True, cwd=ROOT)
    command = [sys.executable, '-u', str(ROOT / 'code' / 'TFC' / 'train_gestures.py'),
               '--from-scratch', '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir),
               '--image-size', '64', '--batch-size', '16', '--train-per-class', str(args.train_per_class),
               '--validation-per-class', '64', '--pretrain-epochs', str(args.pretrain_epochs),
               '--finetune-epochs', str(args.finetune_epochs), '--learning-rate', '0.0003']
    print('Offline full DINOv3 ViT-S/16 baseline: no pretrained image weights. Keep this window open.', flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
