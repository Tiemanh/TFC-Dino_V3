# Kế hoạch kiểm thử thực hành TF-C + DINOv3

## 1. Phạm vi và nguyên tắc

Bộ kiểm thử gồm đúng 20 test case: 10 trường hợp kiểm thử hệ thống AI và 10
trường hợp mô phỏng tấn công. Các test API dùng FastAPI qua ASGI transport;
model được mock ở những trường hợp cần cô lập lỗi. Những thử nghiệm có khả năng
gây tải chỉ chạy trong CI/local, không tấn công dịch vụ công khai.

Quy ước đánh giá:

- `PASS`: tất cả assertion của test đạt.
- `FAIL/ERROR`: ít nhất một assertion sai hoặc chương trình phát sinh lỗi.
- `BLOCKED`: không thể thực hiện do thiếu điều kiện môi trường.
- `RISK`: test chạy ổn định nhưng cho thấy model chưa phát hiện dữ liệu bất thường.

## 2. Mười test case hệ thống, API, AI và mock

| Mã | Loại | Mục tiêu | Dữ liệu chính | Kết quả mong đợi |
|---|---|---|---|---|
| TC01 | Automation/System | Mở giao diện chính | `GET /` | HTTP 200, có tiêu đề TF-C Prediction |
| TC02 | Automation | Tải CSS và JavaScript | `GET /style.css`, `/app.js` | HTTP 200, JavaScript gọi `/api/predict` |
| TC03 | API/Mock | Kiểm tra model sẵn sàng | FakePredictor | `/api/live` OK, `/api/health.ready=true` |
| TC04 | API/Mock | Kiểm tra khi thiếu model | `get_predictor=None` | Health vẫn 200 và `ready=false` |
| TC05 | System | Tải CSV mẫu | `GET /sample-data/...` | File không rỗng |
| TC06 | API/AI data | Tạo preview tín hiệu | CSV 206 mẫu X/Y/Z | Có time view, FFT và đúng số mẫu |
| TC07 | API negative | Từ chối định dạng sai | `gesture.exe` | HTTP 400 |
| TC08 | AI/Mock | Kiểm tra hợp đồng dự đoán | CSV hợp lệ + FakePredictor | Đủ 8 lớp, tổng xác suất bằng 1 |
| TC09 | AI boundary | Kiểm tra cửa sổ ngắn | CSV 205 mẫu | HTTP 400, yêu cầu 206 mẫu |
| TC10 | Mock/Security | Giả lập lỗi suy luận | ExplodingPredictor | HTTP 500 đã làm sạch, không lộ lỗi nội bộ |

## 3. Mười loại tấn công

| Mã | Loại tấn công | Payload | Kết quả mong đợi/đánh giá |
|---|---|---|---|
| AT01 | Malformed request | POST không có file | HTTP 400, API không sập |
| AT02 | Extension spoofing | CSV đặt tên `.exe` | HTTP 400 |
| AT03 | Bounded upload DoS | File lớn hơn 5 MB | HTTP 400, thông báo giới hạn |
| AT04 | Content spoofing | Byte nhị phân đặt tên CSV | HTTP 400, không lộ stack trace |
| AT05 | Schema manipulation | Thiếu trục Z | HTTP 400 |
| AT06 | CSV formula injection | Giá trị `=1+1` | HTTP 400, không đưa vào model |
| AT07 | Numeric overflow | Giá trị `1e309` | HTTP 400 |
| AT08 | Truncated input | Chỉ có 20 mẫu | HTTP 400 |
| AT09 | Adversarial noise | Nhiễu biên độ rất lớn | API không sập; đánh dấu RISK do chưa có OOD detector |
| AT10 | Sensor manipulation | Hoán đổi X/Y/Z | API không sập; đánh dấu RISK do chưa xác minh hướng cảm biến |

## 4. Lệnh chạy

```powershell
python tools/run_practice_tests.py
```

Kết quả được ghi vào:

- `reports/testing/test-results.json`: dữ liệu chi tiết có thể dùng tạo báo cáo.
- `reports/testing/SUMMARY.md`: bảng tổng hợp PASS/FAIL/RISK và thời gian.

GitHub Actions chạy lại bộ test sau mỗi lần push lên nhánh triển khai và tải ba
file minh chứng thành artifact `ai-testing-evidence`.
