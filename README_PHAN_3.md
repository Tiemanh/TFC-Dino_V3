# Phần 3 — Web TF-C Prediction

Xem BAT_DAU_TAI_DAY.md cho bản baseline offline học từ đầu và
DEPLOY_VA_BAN_GIAO.md cho trạng thái nộp bài. Nguồn khởi tạo và metric thật
được hiển thị trong cột kết quả; không coi baseline là DINOv3 pretrained.

Giao diện bám theo thiết kế ba cột đã cung cấp:

1. **Input Data:** tải hoặc kéo thả CSV/TXT có X/Y/Z.
2. **Signal Preview:** vẽ time series ba trục và FFT magnitude trục X.
3. **Prediction Result:** nhãn, confidence và xác suất đủ 8 lớp.

Backend chỉ dự đoán khi có model thật tại
`artifacts/gesture_model/model_bundle.pt`. Nếu chưa có bundle, giao diện vẫn cho
xem tín hiệu nhưng vô hiệu hóa nút dự đoán; không sinh kết quả giả.

## Chạy local

Trên Windows: nhấp đúp `start_web.cmd`, giữ cửa sổ này mở và truy cập
`http://127.0.0.1:8000`. Không mở `index.html` bằng Live Server hoặc trực tiếp.
Nếu đóng cửa sổ chạy web, tải file sẽ báo mất kết nối máy chủ.
Khi đã đọc CSV nhưng nút dự đoán vẫn khóa, kiểm tra thông báo model ở cột phải:
cần huấn luyện và xuất `artifacts/gesture_model/model_bundle.pt` trước.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Có thể đặt model ở vị trí khác bằng biến môi trường
`MODEL_BUNDLE_PATH`.

## Định dạng file người dùng tải lên

```csv
timestamp,x,y,z
0.000,0.12,-0.04,0.18
0.005,0.15,-0.03,0.20
```

Chấp nhận các tên trục `x/y/z`, `ax/ay/az`, `accX/accY/accZ` hoặc
`accel_x/accel_y/accel_z`. File cần ít nhất số mẫu ghi trong metadata của model,
mặc định là 206.

## Deploy

`Dockerfile` đã chuẩn bị cổng 7860 cho Hugging Face Spaces hoặc một nền tảng chạy
Docker. Model bundle phải được thêm sau khi train dữ liệu thật. Không deploy bản
chính thức khi chưa có model bundle thật.
