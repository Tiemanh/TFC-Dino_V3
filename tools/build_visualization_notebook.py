"""Build and execute a self-contained Kaggle notebook from real InGesture CSVs."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT/'datasets/InGestureOffline/metadata.json')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    source = (ROOT/'tools/visualize_ingesture.py').read_text(encoding='utf-8').split('\ndef main():')[0]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# InGesture: Dataset Visualization for TF-C + DINOv3

Notebook trực quan hóa dữ liệu gia tốc X/Y/Z cho tám cử chỉ. Đây là dữ liệu
**InGesture công khai**, không phải bộ dữ liệu tự thu thập.

Dataset dùng trên Kaggle: [Gesture Dataset của dự án](https://www.kaggle.com/datasets/nguyntinmnh41/gesture-dataset).

Nguồn: [InGesture v5, DOI 10.17632/fdxst56tcj.5](https://doi.org/10.17632/fdxst56tcj.5),
giấy phép CC BY 4.0. Cảm biến IMU đeo cổ tay; tần số danh định khoảng 200 Hz.

**Chạy trên Kaggle:** tạo Notebook, chọn Add Input và gắn dataset InGesture có
thư mục `subject_*`. Nếu input là ZIP chưa được Kaggle giải nén, cần giải nén
trước vào `/kaggle/working` rồi chỉnh `DATA_ROOT`. Không cần GPU hoặc trọng số AI.
Các biểu đồ dưới đây chỉ được tạo từ file thật, không có dữ liệu mô phỏng.

### Câu hỏi khám phá
1. Dữ liệu có mất cân bằng lớp hay thiếu người/cử chỉ không?
2. Thời lượng, dữ liệu thiếu và số cửa sổ sau xử lý ra sao?
3. Tín hiệu time/frequency và ảnh đầu vào DINOv3 có đặc điểm gì?
4. Cách chia subject và chuẩn hóa có ngăn rò rỉ dữ liệu không?'''),
             md('''### Đặc điểm dataset

| Thuộc tính | Giá trị / cách kiểm chứng |
|---|---|
| Đơn vị dữ liệu gốc | Một segmented CSV, một nhãn cử chỉ ở mức recording |
| Kênh dùng trong dự án | Gia tốc X/Y/Z; các kênh gyro trong CSV không dùng cho mô hình hiện tại |
| Số người | 50; kiểm tra lại từ toàn bộ inventory |
| Nhãn | 8 lớp, giữ nguyên định nghĩa của InGesture |
| Tần số | Khoảng 200 Hz danh định; không đo lại từ timestamp |
| Đầu vào TF-C | `[3, 206]`, stride 103 |
| Tái lập hình minh họa | Seed 42, tối đa 12 người train khác nhau mỗi lớp |

### Cách tổ chức học từ notebook tham khảo

Tham khảo [Sentinel-2 RPG - Dataset Visualization](https://www.kaggle.com/code/loanhviet/sentinel-2-rpg-dataset-visualization)
về trình tự: giới thiệu → inventory/format → mẫu thật và nhãn → biến thiên
thời gian/nhóm → thống kê toàn bộ dataset → kết luận. Các mục ảnh vệ tinh,
mask và vùng địa lý được chuyển thành tín hiệu, nhãn recording và subject.
Toàn bộ code, hình và nhận xét InGesture trong notebook này được xây dựng
từ dữ liệu của dự án; không sao chép kết quả của dataset tham khảo.'''),
             md('''Tham khảo bổ sung [Sensor Logger: Five-Gesture Dataset Visualization](https://www.kaggle.com/code/vuvandat000099/sensor-logger-five-gesture-dataset-visualization)
về PCA đặc trưng thống kê, RMS kênh, STFT và kiểm tra trùng. Phiên bản InGesture
vẫn dùng tám lớp và ba kênh gia tốc; số liệu và đồ thị đều được tính lại.'''),
             md('## 0. Setup\nNotebook tự chứa các hàm đọc và vẽ; không cần clone GitHub hoặc tải trọng số AI.'), code(source)]
    config = ('METADATA = ' + repr(metadata) + '\n'
              "DATA_ROOT = find_root(Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('data/external/InGesture'))\n"
              "OUTPUT = Path('/kaggle/working/ingesture_visualization') if Path('/kaggle/working').exists() else Path('reports/data_visualization_notebook')\n"
              "OUTPUT.mkdir(parents=True, exist_ok=True)\n"
              "frame, previews, points = inventory(DATA_ROOT, METADATA)\n"
              "validate_counts(frame, METADATA)\n"
              "frame.to_csv(OUTPUT/'recording_inventory.csv', index=False)\n"
              "print('Source:', DATA_ROOT)\n"
              "print('Recordings:', len(frame), '| Subjects:', frame.subject.nunique(), '| Windows:', int(frame.windows.sum()))\n"
              "print(frame.groupby('split').agg(recordings=('file','size'), subjects=('subject','nunique'), windows=('windows','sum')).to_string())\n"
              "groups = [set(METADATA[k]) for k in ['train_subjects','validation_subjects','test_subjects']]\n"
              "assert all(not groups[i] & groups[j] for i in range(3) for j in range(i+1,3))\n"
              "print('PASS: no subject overlap; all class window counts match metadata')\n"
              "from IPython.display import Image, display\n")
    config += ("pool, pool_manifest = sample_pool(DATA_ROOT, frame, METADATA, per_class=12, seed=42)\n"
               "pool_manifest.to_csv(OUTPUT/'sample_pool.csv', index=False)\n"
               "previews = {name: items[0]['values'][:1200] for name,items in pool.items()}\n"
               "print('Seeded sample pool:', len(pool_manifest), 'recordings; selected file paths in sample_pool.csv')\n")
    cells += [md('''## 1. Dataset Structure and Inspection — kiểm kê và chia tập

Chỉ dùng segmented CSV có tên `gesture__*.csv` với tám nhãn hợp lệ, tránh đếm
trùng file recording chưa tách. Những người train/validation/test được cố định
theo metadata của pipeline dự án (32/8/10 người); không chia ngẫu nhiên các
cửa sổ chồng lấn. Tất cả ngưỡng clipping, mean/std được fit từ train.
Notebook dùng metadata nhúng sẵn để chạy độc lập và kiểm tra lại số cửa sổ.
Inventory quét tất cả CSV một lần để đếm chính xác; các biểu đồ mẫu đọc từ
pool nhỏ. Đây là điểm khác với bài tham khảo có manifest được cấp sẵn.'''), code(config),
              md('''### 1.1 Sample Format Check — kiểm tra định dạng một file

In tên file, shape/dtype, ba dòng dữ liệu thật và shape cửa sổ. Chỉ rút nhãn
từ tên segmented CSV theo quy ước dataset; không suy diễn thời điểm bắt đầu
hoặc kết thúc cử chỉ từ dạng sóng.'''),
              code("sample_format(DATA_ROOT, pool, METADATA)\nprint(pd.DataFrame({'label_id':range(8),'gesture':CLASSES,'display_name':NAMES}).to_string(index=False))"),
              md('''### 1.2 Sample Pool — chọn mẫu tái lập được

Mỗi lớp chọn tối đa 12 subject train khác nhau bằng seed 42, rồi chọn một
recording hợp lệ mỗi subject. Các hình tín hiệu/nhãn/ảnh adapter dùng cùng
recording đầu tiên trong pool. File `sample_pool.csv` ghi chính xác tên file
đã dùng; bảng thống kê toàn dataset luôn dùng inventory đầy đủ.'''),
              code("print(pool_manifest.groupby('gesture').agg(recordings=('file','size'), subjects=('subject','nunique')).to_string())\nprint(pool_manifest.head(8).to_string(index=False))"),
              md('''### 1.3 Exact Duplicate and Constant Recording Audit

Băm SHA-256 trên mảng X/Y/Z float32 sau cleaning để kiểm tra recording trùng
và hash xuất hiện ở nhiều split. Đây là kiểm tra mức recording, không kiểm tra
mọi cửa sổ chồng lấn hay mẫu gần giống. Tín hiệu đứng yên trùng nhau không tự
động chứng minh rò rỉ; cần xem provenance trước khi kết luận.'''),
              code("print(json.dumps(duplicate_audit(frame,METADATA,OUTPUT),indent=2))")]
    sections = [
        ('2. Signal Visualization — tín hiệu X/Y/Z', "signal_examples(previews, METADATA, OUTPUT)",
         'Một recording mỗi cử chỉ từ sample pool cố định, tối đa 1.200 mẫu, đơn vị g. X/Y/Z giữ cùng màu ở mọi hình. Không khẳng định một recording đại diện cho cả lớp.'),
        ('3. Signal–Label Alignment — đối chiếu tín hiệu và nhãn', "label_alignment(pool, METADATA, OUTPUT)",
         'Đối chiếu cùng recording: tín hiệu gốc → nhãn ở mức file → cửa sổ 206 mẫu đã chuẩn hóa nhận nhãn kế thừa. Khác segmentation trong bài tham khảo, InGesture ở đây không cung cấp onset/offset cho từng điểm; dải màu là nhãn recording, không phải ranh giới cử chỉ được đo.'),
        ('4. Temporal / Frequency Visualization — phổ theo cử chỉ', "signal_examples(previews, METADATA, OUTPUT, frequency=True)",
         'Dùng đúng các recording ở mục 2. Trừ mean và dùng Hann window để nhìn dao động, giới hạn 0–30 Hz. FFT minh họa này khác frequency view huấn luyện abs(FFT)+log1p.'),
        ('5. Multi-Subject Inspection — cùng cử chỉ ở nhiều người', "multisubject_examples(pool, METADATA, OUTPUT)",
         'Bốn cử chỉ, mỗi cử chỉ lấy ba người khác nhau từ pool. Vẽ độ lớn gia tốc sqrt(x²+y²+z²), giữ cùng trục y trong mỗi hàng để thấy biến thiên giữa người. Tên subject hiện ngay trên hình.'),
        ('6. Subject Coverage — độ phủ người và nhãn', "subject_coverage(frame, METADATA, OUTPUT)",
         'Tương ứng phần coverage trong bài tham khảo, dùng toàn bộ inventory thay vì chỉ sample pool. Mỗi ô là số segmented recording của một người/cử chỉ.'),
        ('7. Class Distribution — phân bố lớp toàn dataset', "class_distribution(frame, METADATA, OUTPUT)",
         'So sánh số recording và số cửa sổ: một recording dài có thể sinh nhiều cửa sổ. Trục đếm dùng log; không được hiểu số cửa sổ là số người độc lập.'),
        ('8. Split Distribution — phân bố train/validation/test', "split_distribution(frame, METADATA, OUTPUT)",
         'Giữ phân bố thực của validation/test. Số liệu trên ô là số cửa sổ; màu dùng log(1 + count).'),
        ('9. Temporal Coverage and Quality — thời lượng và chất lượng', "quality(frame, METADATA, OUTPUT)",
         'Thời lượng được ước lượng từ số mẫu / 200 Hz, không suy ra từ timestamp. NaN/Inf được nội suy trong từng recording. Không loại bỏ outlier khỏi biểu đồ thống kê chất lượng; boxplot thời lượng ẩn các điểm outlier để dễ đọc.'),
        ('10. Gesture Profiles — đường trung bình và độ biến thiên', "gesture_profiles(pool, METADATA, OUTPUT)",
         'Tương ứng đường mean ± SD theo lớp trong bài tham khảo. Mỗi recording được nội suy về 200 điểm theo tiến độ tương đối 0–100%, chỉ để vẽ. Dải màu là ±1 độ lệch chuẩn giữa recording, không phải confidence interval. Không có căn chỉnh onset/offset và không áp dụng phép nội suy này vào đầu vào huấn luyện.'),
        ('11. Normalization and Axis Correlation', "train_statistics(points, METADATA, OUTPUT)",
         'Mẫu train gồm tối đa 50 điểm/recording, chọn seed 42. Đây là lấy mẫu theo recording, không phải phân bố theo toàn bộ thời gian. Không fit lại thống kê trên validation/test.'),
        ('12. Model Input Inspection — ảnh đầu vào DINOv3', "adapter_examples(previews, METADATA, OUTPUT)",
         'Minh họa adapter lúc khởi tạo với ánh xạ kênh identity, ảnh 64×64 trước chuẩn hóa ảnh. Không phải ảnh từ adapter đã học, GAF hay spectrogram. Các hàng lặp lại do cách trải trace thành ảnh; loss tốt hơn không bảo đảm khắc phục hạn chế của cách biểu diễn này.'),
        ('13. PCA — chiếu đặc trưng thống kê lên hai chiều', "pca_features(pool,METADATA,OUTPUT)",
         'Mỗi recording trong pool lấy cửa sổ đầu tiên dài 206 mẫu, chuẩn hóa bằng thống kê train. Đặc trưng gồm mean, std, RMS và khoảng percentile 5–95 cho ba trục (12 chiều). StandardScaler/PCA chỉ fit trên pool train. Mức tách cụm và phương sai giải thích không phải accuracy hoặc embedding DINOv3.'),
        ('14. Channel RMS — độ lớn theo kênh và cử chỉ', "channel_rms(pool,METADATA,OUTPUT)",
         'RMS được tính trên đúng các cửa sổ chuẩn hóa dùng ở mục PCA, rồi lấy trung bình theo lớp. Giá trị là đơn vị chuẩn hóa, không phải năng lượng vật lý.'),
        ('15. Time–Frequency View — STFT', "time_frequency(pool,METADATA,OUTPUT)",
         'STFT trục X của cùng mẫu đã chọn, NFFT=64, overlap=48. Tần số cycles/sample không đòi hỏi xác minh tần số vật lý. Hình này chỉ là EDA, chưa thay phép adapter/FFT của model đang deploy.'),
    ]
    for title, expression, explanation in sections:
        cells += [md('## ' + title + '\n\n' + explanation), code('path = ' + expression + '\ndisplay(Image(filename=str(path)))')]
    cells += [md('''## 16. Summary — kết luận từ dữ liệu

Mục dưới tính trực tiếp số liệu toàn dataset và sample pool thay vì ghi tay
thống kê. Các nhận xét mô hình phải dựa trên tập chia theo người, không dựa
trên một recording đẹp.'''), code("counts = frame.groupby('gesture').windows.sum().sort_values(ascending=False)\nprint(f'{len(frame):,} recordings | {frame.subject.nunique()} subjects | {int(frame.windows.sum()):,} windows')\nprint(f'Majority class: {counts.index[0]} = {counts.iloc[0]/counts.sum():.2%} of windows')\nprint(f'Short recordings excluded from windowing: {int((frame.windows==0).sum())}')\nprint(f'Nonfinite X/Y/Z cells: {int(frame.nonfinite_cells.sum())}')\nprint(f'Figures use {len(pool_manifest)} seeded training recordings from distinct subjects per gesture.')"), md('''### Ý nghĩa đối với TF-C + DINOv3

- Tỷ lệ lớp phải được tính trên dữ liệu thật; không đổi nhãn để làm cân bằng.
- Với phân bố lệch lớp, so sánh Cross Entropy, CB Cross Entropy, CB Focal; dùng
  cùng sampler, tập chia và seed. Chọn cấu hình bằng macro F1 validation.
- CB weighting bằng nhau nếu chỉ lấy cùng số cửa sổ mỗi lớp. Vì vậy nên chạy
  đánh giá loss trên train đầy đủ hoặc tập con giữ phân bố tự nhiên.
- Báo cáo cả macro F1, balanced accuracy, recall từng lớp; accuracy có thể tăng
  trong khi một số cử chỉ hiếm kém đi.
- Không tuyên bố đã thu thập dữ liệu private hoặc đã dùng pretrained DINOv3.

**Bàn giao theo mẫu:** gửi link Kaggle Dataset **và** link Notebook đã Save
Version/Run All, có outputs, đặt Public. Các PNG nằm trong output notebook để tải xuống.

Nguồn: [TF-C](https://github.com/mims-harvard/TFC-pretraining),
[DINOv3](https://github.com/facebookresearch/dinov3),
[Class-Balanced Loss](https://arxiv.org/abs/1901.05555).''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    # Execute every code cell in order. Capture actual stdout and display MIME
    # payloads; no fabricated outputs or synthetic data are inserted.
    from IPython.utils.capture import capture_output
    namespace = {'__name__':'ingesture_visualization_notebook'}
    number = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code': continue
        number += 1
        with capture_output() as captured:
            exec(compile(cell.source, f'<notebook cell {number}>', 'exec'), namespace)
        outputs = []
        if captured.stdout: outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr: outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
        for display in captured.outputs:
            outputs.append(nbf.v4.new_output('display_data', data=display.data, metadata=display.metadata))
        # In a non-interactive Python process IPython display() can print an
        # Image repr. The image was actually generated in this cell; attach its
        # bytes explicitly to keep the executed notebook useful on Kaggle.
        if cell.source.startswith('path = ') and not any('image/png' in o.get('data',{}) for o in outputs):
            outputs = [nbf.v4.new_output('display_data', data={'image/png':base64.b64encode(namespace['path'].read_bytes()).decode('ascii')}, metadata={})]
        cell.execution_count, cell.outputs = number, outputs
        print(f'Executed notebook cell {number}', flush=True)
    destination = ROOT/'kaggle'
    destination.mkdir(exist_ok=True)
    nbf.validate(notebook)
    nbf.write(notebook, destination/'ingesture_visualization.ipynb')
    shutil.make_archive(str(destination/'ingesture_visualization_assets'), 'zip', namespace['OUTPUT'])
    print(destination/'ingesture_visualization.ipynb')


if __name__ == '__main__':
    main()
