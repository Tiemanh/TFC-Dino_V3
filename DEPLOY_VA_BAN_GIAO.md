# Trạng thái bàn giao và triển khai

## GitHub

Code dựa trên mims-harvard/TFC-pretraining, commit nền 9667582. Xem
README_DINOV3_VI.md cho thay đổi hai encoder. Giữ README và LICENSE tác giả.
Không đưa data/external, dữ liệu private, cache, venv và token vào repo code.
Chưa có URL nộp GitHub mới nếu chưa push lên tài khoản của người dùng.

## Kaggle

Chỉ gói dữ liệu private sau khi tự thu và kiểm tra quyền công khai.
InGesture phải được ghi rõ là dữ liệu công khai của tác giả khác.
Xem THU_THAP_PRIVATE.md. Không đánh dấu đã nộp nếu chưa có public dataset URL.

## Web model

Mở start_web.cmd và http://127.0.0.1:8000. Model cần nằm tại
artifacts/gesture_model/model_bundle.pt. Trình duyệt hiển thị nguồn huấn luyện
và kết quả đánh giá khi dùng baseline học từ đầu. Có thể dùng MODEL_BUNDLE_PATH
để chọn bundle khác. Dự đoán phải đến từ model thật, không từ tên file hay label.

## Docker (chuẩn bị triển khai)

```powershell
docker build -t tfc-dinov3 .
docker run --rm -p 7860:7860 tfc-dinov3
```

Truy cập http://127.0.0.1:7860. Dockerfile dùng CPU cho inference và cổng 7860.
.dockerignore chỉ đưa bundle đang dùng vào image; bỏ tập dữ liệu và các run cũ.
Đây là cấu hình chuẩn bị, không khẳng định Docker đã được kiểm thử nếu máy
chưa cài/chạy Docker. Model và repo public cần tuân thủ giấy phép tương ứng.

Hugging Face Spaces loại Docker hoặc nền tảng hỗ trợ container có thể dùng
Dockerfile này. Việc tạo tài nguyên, đưa bundle lên và publish chỉ làm sau
khi người dùng chọn tài khoản/nền tảng và xác nhận công khai. Không có deploy
URL khi mới chạy localhost. Baseline InGesture chưa đạt kiểm chứng smartphone
thì phải ghi rõ là demo trong trang nộp bài.
