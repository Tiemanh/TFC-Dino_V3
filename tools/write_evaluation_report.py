"""Write a report only from a completed run's measured metrics and metadata."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/gesture_model'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/offline_baseline'))
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / 'metrics.json').read_text(encoding='utf-8'))
    info = json.loads((args.run_dir / 'training_info.json').read_text(encoding='utf-8'))
    classes = list(metrics['per_class'])
    matrix = metrics['confusion_matrix']
    total = sum(map(sum, matrix))
    majority = max(map(sum, matrix)) / total
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true_label / predicted_label', *classes])
        for name, row in zip(classes, matrix):
            writer.writerow([name, *row])
    lines = [
        '# Kết quả baseline TF-C + DINOv3 học từ đầu', '',
        f"Nguồn khởi tạo: `{info['initialization']}`. Dataset: {info['dataset']}.", '',
        'Đây là kết quả đo thật trên InGesture công khai, không phải dữ liệu smartphone private.',
        'Hai backbone đầy đủ ViT-S/16 được học từ đầu, không có pretrained ảnh.', '',
        f"- Train dùng: {info['train_windows_used']} cửa sổ.",
        f"- Validation dùng: {info['validation_windows_used']} cửa sổ, chọn epoch {info['best_validation_epoch']}.",
        f"- Test dùng: {total} cửa sổ, giữ nguyên toàn bộ phân bố lớp.",
        f"- TF-C pretrain: {info['pretrain_epochs']} epoch; phân loại: {info['finetune_epochs']} epoch.",
        f"- Ảnh: {info['image_size']}×{info['image_size']}; batch: {info['batch_size']}; seed: {info['seed']}.",
        f"- Accuracy: {metrics['test_accuracy']:.2%}.",
        f"- Macro F1: {metrics['macro_f1']:.2%}.",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.2%}.",
        f'- Accuracy nếu luôn đoán lớp đa số: {majority:.2%}.', '',
        'Accuracy của baseline hiện thấp hơn cách luôn đoán lớp đa số; macro F1 và',
        'balanced accuracy cho thấy model còn yếu. Không dùng để tuyên bố hệ thống',
        'đã đạt chất lượng nhận dạng smartphone. Confidence softmax chưa được hiệu chỉnh.', '',
        '| Nhãn | Precision | Recall | F1 | Số cửa sổ test |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, values in metrics['per_class'].items():
        lines.append(f"| {name} | {values['precision']:.2%} | {values['recall']:.2%} | {values['f1']:.2%} | {values['support']} |")
    lines += ['', 'Tách 32 người train, 8 người validation và 10 người test trước khi',
              'fit normalization. Epoch được chọn bằng validation, không dựa vào test.',
              'Đánh giá trên từng cửa sổ; web tổng hợp xác suất qua các cửa sổ của file.',
              'Không so sánh accuracy cửa sổ này với một dự đoán demo trên file train.', '',
              'Các việc còn lại: thu và xác nhận nhãn private, huấn luyện/đánh giá lại,',
              'cải thiện chất lượng, rồi xác nhận công khai GitHub/Kaggle/deploy.']
    (args.output_dir / 'EVALUATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.output_dir / 'training_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(args.output_dir / 'EVALUATION.md')


if __name__ == '__main__':
    main()
