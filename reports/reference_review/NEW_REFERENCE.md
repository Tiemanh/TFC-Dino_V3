# Đối chiếu bài TF-C + DINOv3 và Sensor Logger

## Những nguồn đã đọc

- PDF bài nộp người dùng gửi: chỉ mục GitHub / Dataset / Visualization / Deploy
  và file dự đoán thử, không phải bài báo khoa học mô tả một thuật toán mới.
- https://github.com/vuvandat109/TF-C-DINOv3 : README ở commit 268df45.
- https://www.kaggle.com/code/vuvandat000099/sensor-logger-five-gesture-dataset-visualization : Version 4.

README repo tham khảo mô tả thử nghiệm pretrained DINOv3 trên EEG hai lớp.
Notebook Sensor Logger dùng năm nhãn và sáu kênh. Đây là các điều kiện khác với
InGesture tám lớp X/Y/Z; không đưa số liệu của chúng vào báo cáo kết quả dự án.
Nguồn này được dùng để học cấu trúc bàn giao, lựa chọn kiểm tra và ứng viên
loss, không được xem là bằng chứng ứng viên đó tối ưu cho InGesture.

## Những phần đã áp dụng

| Phần tham khảo | Áp dụng vào dự án |
|---|---|
| Loss trên bốn projected views | Thêm ProjectedMultiViewTFCLoss có thể chọn trong train_gestures.py; giữ legacy mặc định. |
| Kiểm tra loss và gradient | Test positive pairing, tính bất biến theo scale sau L2 normalization và gradient ở bốn view. |
| PCA đặc trưng thống kê | Mean, std, RMS, percentile range cho X/Y/Z; StandardScaler/PCA fit trên pool train. |
| RMS theo cử chỉ/kênh | Heatmap standardized RMS; không gọi là năng lượng vật lý. |
| Time-frequency view | STFT minh họa trục X, tần số cycles/sample, giữ FFT/adapter deploy hiện tại. |
| Duplicate/constant checks | Hash exact cleaned float32 recording arrays; xuất bảng để xem provenance; không tuyên bố đã audit mọi window hoặc near-duplicate. |
| File demo cho người chấm | Tám CSV thật, provenance manifest và ô chọn mẫu trong Streamlit. |

## Loss ứng viên mới

Time và frequency losses dùng z từ projector thay vì h backbone. Cross loss
lấy trung bình bốn cặp gốc/augmentation. Mỗi cặp là symmetric NT-Xent với
positive là cùng mẫu ở hai view, self-similarity được mask. Các mẫu khác trong
batch đóng vai trò negatives. Hệ số time/frequency 0.2, cross 1.0; temperature
lấy từ config. Đây là ứng viên, chưa thay thế kết quả thực nghiệm đang báo cáo.

```powershell
python code/TFC/train_gestures.py --from-scratch `
  --data-dir datasets/InGestureOffline --output-dir artifacts/multiview_candidate `
  --contrastive-objective projected_multiview --classification-loss cb_focal `
  --focal-gamma 1 --sampler shuffle --finetune-tfc-weight 0.1 `
  --validation-only --image-size 64 --batch-size 16
```

Khi so sánh với legacy phải giữ các tham số và tập dữ liệu như nhau, dùng
validation để lựa chọn, không chép metric EEG hay Sensor Logger sang InGesture.
Model đang deploy vẫn là ViT-S/16 học từ đầu với CB Focal, không phải pretrained.

## Những giới hạn cần nói rõ

PCA chỉ là hình chiếu EDA, không phải DINOv3 embedding hoặc kết quả classifier.
RMS chuẩn hóa không có đơn vị năng lượng vật lý. Mean ± SD có thể che mất cấu
trúc khi các recording không căn chỉnh onset/offset. Nhãn InGesture ở mức file,
không tự tạo ground truth theo từng thời điểm. Recording hash trùng có thể
do tín hiệu không đổi, cần điều tra trước khi kết luận leakage.

Tám CSV demo dùng dữ liệu train, timestamp sinh theo tần số danh định. Mỗi file
có nhãn nguồn trong manifest, nhưng model có thể dự đoán khác. Không đổi nhãn
hoặc tạo dự đoán giả để làm cho demo trông chính xác.

PDF chứa thông tin truy cập website tham khảo; các thông tin đó không được
chép vào source hoặc tài liệu công khai của dự án. Phần đánh giá ở đây dùng
README và notebook công khai, không khẳng định đã kiểm thử website sau đăng nhập.
