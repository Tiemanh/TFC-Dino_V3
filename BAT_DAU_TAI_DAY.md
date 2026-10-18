# Bắt đầu với dự án TF-C + DINOv3

## Chạy trong VS Code trên máy hiện tại

Mở `TFC-DivV3.code-workspace` hoặc File → Open Folder → TFC-DivV3.
Môi trường `.venv` đã được tạo trong thư mục này. Môi trường tái sử dụng
PyTorch CUDA và một số thư viện đã cài ở Python 3.12 trên máy; không cần
môi trường ở thư mục TFC-pretraining-dinov3 nữa. Nếu chuyển sang máy khác,
phải tạo lại venv và cài PyTorch cùng requirements-web.txt; không chép venv.

Vào Run and Debug → **Web app: TF-C Prediction** → F5. Mở
http://127.0.0.1:8000, chọn Dùng CSV mẫu → Run Prediction. Không cần train lại
để chạy thử bundle hiện có. Nếu cổng 8000 đang chạy, dừng cửa sổ server cũ
bằng Ctrl+C trước khi F5, hoặc dùng luôn trang web server đó đang phục vụ.

Muốn train lại từ VS Code, chọn **Training: offline InGesture (new output folder)**.
Lượt mới lưu artifacts/offline_vscode_run, không ghi đè model web. Nếu thư mục
này đã có bundle, đổi --output-dir trong launch.json sang tên mới.

## Hai cách huấn luyện

- `--local-model-dir weights/dinov3-vits16`: dùng trọng số pretrained hợp lệ đã tải về.
- `--from-scratch`: không cần tải trọng số hay tài khoản. Hai DINOv3 ViT-S/16
  đầy đủ được khởi tạo ngẫu nhiên và được cập nhật khi học trên cảm biến.
  Đây là baseline thử nghiệm, không tương đương pretrained trên ảnh.

`--smoke-test` là chế độ riêng, dùng kiến trúc nhỏ để kiểm tra code. Bundle của
chế độ này bị chặn khỏi web chính thức. Không dùng metric của nó để báo cáo.

## Mở web

Nhấp đúp `start_web.cmd`, giữ cửa sổ mở, vào http://127.0.0.1:8000.
Bấm **Dùng CSV mẫu** rồi **Run Prediction**. Web đọc bundle tại
`artifacts/gesture_model/model_bundle.pt`, không đoán theo tên file hay nhãn CSV.
File dài được chia thành các cửa sổ rồi lấy trung bình xác suất thật từ model.
Một file nên chứa một loại cử chỉ; file có nhiều cử chỉ sẽ bị gộp xác suất.

## Huấn luyện lại không cần pretrained

Nhấp đúp `train_offline.cmd` hoặc chạy trong Terminal VS Code:

```powershell
python train_offline.py --output-dir artifacts/offline_run_2
```

Mặc định dùng InGesture, 64×64 pixel, batch 16, tối đa 256 cửa sổ mỗi lớp,
2 epoch TF-C pretraining và 12 epoch phân loại. Toàn bộ backbone, adapter,
projector và classifier đều được học. Giới hạn train là để có một baseline
chạy được trên laptop; không tuyên bố đã tận dụng hết dữ liệu.

Train/validation/test được tách theo người tham gia trước khi chuẩn hóa.
Thống kê chuẩn hóa chỉ tính từ train. Model tốt nhất được chọn bằng macro F1
trên validation; test được đánh giá sau huấn luyện, không dùng chọn epoch.
Giới hạn validation mặc định 64 cửa sổ/lớp để giảm thời gian; tập test dùng đầy đủ.

Các tệp đầu ra:

- `model_bundle.pt`: toàn bộ model và metadata cần cho web.
- `metrics.json`: accuracy, macro F1, balanced accuracy và từng lớp trên test.
- `history.json`: loss/accuracy/validation theo epoch.
- `training_info.json`: nguồn khởi tạo, số cửa sổ, seed, epoch, thời gian.
- `best_validation.pt`: checkpoint được chọn theo validation.

Không ghi đè bundle đang có. Chọn thư mục đầu ra mới cho lượt tiếp theo.
Muốn dùng model mới, thay bundle trong artifacts/gesture_model sau khi đánh giá
và khởi động lại web. Các nhãn luôn lấy từ metadata của model.

## Dữ liệu private và nộp bài

Xem `THU_THAP_PRIVATE.md` cho cách thu, nhập CSV và chuẩn bị dữ liệu.
Xem `DEPLOY_VA_BAN_GIAO.md` cho các phần cần kiểm tra trước khi công khai.
InGesture là bộ cảm biến đeo cổ tay công khai, không phải dữ liệu smartphone
private. Các nhãn InGesture khác tám cử chỉ UWave trong thí nghiệm TF-C gốc.
Model trên InGesture không tự biết các nhãn private mới.
