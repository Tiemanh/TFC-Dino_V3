"""Matched loss ablation on validation. Test data is never accessed by this tool."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/InGestureOffline')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/loss_ablation')
    parser.add_argument('--local-model-dir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--finetune-epochs', type=int, default=12)
    parser.add_argument('--train-per-class', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Tiny random model: protocol check only, no scientific ranking')
    parser.add_argument('--contrastive-objective',choices=['legacy','projected_multiview'],default='legacy')
    args = parser.parse_args()
    if args.smoke_test and args.local_model_dir:
        parser.error('Choose either smoke-test or local pretrained weights')
    # Change only the objective. All runs use the same subject split, full
    # validation, sampler, initialization seed, batch size and learning rate.
    candidates = [
        ('ce', 'cross_entropy', 0., 0.),
        ('cb_ce', 'cb_ce', 0., 0.),
        ('cb_focal_g1', 'cb_focal', 1., 0.),
        ('cb_focal_g2', 'cb_focal', 2., 0.),
        ('cb_focal_joint', 'cb_focal', 1., .1),
    ]
    results = []
    for name, loss, gamma, joint in candidates:
        scores = []
        for seed in args.seeds:
            output = args.output_dir / f'{name}_seed{seed}'
            command = [sys.executable, '-u', str(ROOT/'code/TFC/train_gestures.py'),
                       '--data-dir', str(args.data_dir), '--output-dir', str(output),
                       '--classification-loss', loss, '--focal-gamma', str(gamma), '--cb-beta', '.999',
                       '--finetune-tfc-weight', str(joint), '--sampler', 'shuffle', '--validation-only',
                       '--contrastive-objective',args.contrastive_objective,
                       '--seed', str(seed), '--train-per-class', str(args.train_per_class),
                       '--pretrain-epochs', str(args.pretrain_epochs),
                       '--finetune-epochs', str(args.finetune_epochs), '--batch-size', '16',
                       '--image-size', '64', '--learning-rate', '.0003']
            command += (['--smoke-test'] if args.smoke_test else
                        ['--local-model-dir', str(args.local_model_dir)] if args.local_model_dir else ['--from-scratch'])
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            metric_path = output / 'validation_metrics.json'
            if not metric_path.exists():
                subprocess.run(command, cwd=ROOT, check=True)
            report = json.loads(metric_path.read_text())
            expected = {'seed':seed,'classification_loss':loss,'focal_gamma':gamma,
                        'finetune_tfc_weight':joint,'sampler':'shuffle',
                        'pretrain_epochs':args.pretrain_epochs,'finetune_epochs':args.finetune_epochs,
                        'train_per_class':args.train_per_class,'data_dir':str(args.data_dir),
                        'local_model_dir':str(args.local_model_dir) if args.local_model_dir else None}
            expected['smoke_test'] = args.smoke_test
            expected['contrastive_objective'] = args.contrastive_objective
            if any(report['arguments'].get(k) != v for k,v in expected.items()):
                raise ValueError(f'Existing result config mismatch: {output}; use a new output-dir')
            scores.append(report['metrics']['macro_f1'])
        if scores:
            results.append({'candidate':name,'mean_validation_macro_f1':float(np.mean(scores)),
                            'std_validation_macro_f1':float(np.std(scores)), 'scores':scores})
    if results:
        results.sort(key=lambda x:x['mean_validation_macro_f1'],reverse=True)
        (args.output_dir/'comparison.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))
        print('SMOKE TEST ONLY: ranking has no scientific meaning.' if args.smoke_test else
              'Winner is validation-selected; do one final test evaluation before deployment.')


if __name__ == '__main__':
    main()
