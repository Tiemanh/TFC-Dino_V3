# Chạy TF-C + DINOv3 khi có dữ liệu 8 cử chỉ

## 1. Đặt dữ liệu vào dự án

Chép CSV vào `data/raw/`. Đọc `data/raw/README.md` để biết tên cột. Bộ dữ liệu
phải có đúng 8 nhãn; mỗi nhãn cần ít nhất hai lần ghi (`recording_id`) và mỗi
lần ghi nên dài ít nhất 206 mẫu.

## 2. Làm sạch và tạo tập train/test

Chạy tại thư mục gốc `TFC-DivV3`:

```powershell
.\.venv\Scripts\python.exe tools\prepare_gesture_data.py `
  --raw-dir data\raw `
  --output-dir datasets\Gesture
```

Pipeline thực hiện: chuyển giá trị lỗi thành rỗng, nội suy theo từng lần ghi,
cắt ngoại lệ theo percentile của tập train, chia train/test theo `recording_id`,
chuẩn hóa bằng thống kê tập train và segmentation cửa sổ 206 mẫu/stride 103.

Kết quả là `datasets/Gesture/train.pt`, `test.pt` và `metadata.json`. Mở
`metadata.json` và kiểm tra đủ 8 tên cử chỉ, số recording và số window.

## 3. Đăng nhập để lấy DINOv3 pretrained

**Cách dùng trọng số trên máy (không cần đăng nhập):** xem
`weights/README.md`. Đặt config và trọng số trong `weights/dinov3-vits16`,
rồi chạy từ thư mục gốc dự án:

```powershell
python code/TFC/train_gestures.py --data-dir datasets/InGesture --local-model-dir weights/dinov3-vits16 --output-dir artifacts/gesture_model --pretrain-epochs 1 --finetune-epochs 1 --batch-size 2
```

Đổi data-dir thành datasets/Gesture khi dùng dữ liệu private đã xử lý.
Chế độ này chỉ đọc file trên máy và không cần bước đăng nhập bên dưới.
Trọng số `.pth` gốc phải được chuyển đổi sang định dạng được mô tả trong
weights/README.md; không thể dùng bằng cách đổi tên file.

Sau khi tạo tài khoản Hugging Face và được cấp quyền vào model DINOv3:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Token chỉ cần quyền đọc và không được đưa vào GitHub.

## 4. Pretrain, fine-tune, đánh giá và xuất model

```powershell
cd code\TFC
..\..\.venv\Scripts\python.exe train_gestures.py `
  --data-dir ..\..\datasets\Gesture `
  --output-dir ..\..\artifacts\gesture_model `
  --pretrain-epochs 10 `
  --finetune-epochs 20 `
  --batch-size 16
cd ..\..
```

Với GPU 4 GB, giữ DINOv3 ở trạng thái freeze và giảm `--batch-size` nếu thiếu
bộ nhớ. Chỉ dùng `--unfreeze-dinov3` trên GPU có nhiều VRAM.

Kết quả cần giữ là `artifacts/gesture_model/model_bundle.pt` và `metrics.json`.
Bundle gồm hai DINOv3, adapter, projector, classifier, danh sách nhãn, thống kê
chuẩn hóa và cấu hình kiến trúc. Đây là đầu vào trực tiếp cho phần 3.

## 5. Tích hợp phần 3

Backend web có thể dùng trực tiếp:

```python
from inference import GesturePredictor

predictor = GesturePredictor("artifacts/gesture_model/model_bundle.pt")
result = predictor.predict(xyz_samples)  # mảng [số_mẫu, 3]
```

`result` chứa `label`, `confidence` và xác suất của cả 8 cử chỉ. Giao diện ba
cột dùng CSV ở **Input Data**, ba đồ thị X/Y/Z ở **Signal Preview**, và kết quả
trên ở **Prediction Result**.

Không dùng metric sinh từ dữ liệu mẫu tổng hợp để báo cáo hoặc nộp bài.

## Dùng bộ InGesture công khai đã tải

Dữ liệu gốc nằm tại `data/external/InGesture`. Chuyển sang định dạng TF-C bằng:

```powershell
.\.venv\Scripts\python.exe tools\prepare_ingesture_data.py
```

Kết quả nằm trong `datasets/InGesture`. Công cụ tách train/test theo subject,
không để dữ liệu của cùng một người xuất hiện ở cả hai tập. Khi train, truyền
`--data-dir ..\..\datasets\InGesture` cho `train_gestures.py`.

Có thể thêm `--smoke-test --pretrain-epochs 1 --finetune-epochs 1` để kiểm tra
luồng chạy bằng backbone nhỏ ngẫu nhiên, không cần tài khoản. Bundle và metric
từ chế độ này chỉ có giá trị kiểm tra kỹ thuật, không dùng cho phần 3 chính thức.
