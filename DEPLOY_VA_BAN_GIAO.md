# Trạng thái bàn giao và triển khai

## GitHub

Code dựa trên mims-harvard/TFC-pretraining, commit nền 9667582. Xem
README_DINOV3_VI.md cho thay đổi hai encoder. Giữ README và LICENSE tác giả.
Không đưa data/external, dữ liệu private, cache, venv và token vào repo code.
Repository hiện tại: https://github.com/Tiemanh/TFC-Dino_V3. Mọi cập nhật
Streamlit phải được commit và push trước khi tạo app trên Community Cloud.

## Kaggle

Chỉ gói dữ liệu private sau khi tự thu và kiểm tra quyền công khai.
InGesture phải được ghi rõ là dữ liệu công khai của tác giả khác.
Xem THU_THAP_PRIVATE.md. Không đánh dấu đã nộp nếu chưa có public dataset URL.

## Web model

Ưu tiên `start_streamlit.cmd` và http://localhost:8501 cho bản sẽ deploy miễn
phí. FastAPI gốc vẫn chạy bằng `start_web.cmd` tại http://127.0.0.1:8000.
Model cần nằm tại
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

Streamlit Community Cloud dùng `streamlit_app.py` và `requirements.txt` để tạo
URL `.streamlit.app`. Việc push cập nhật và tạo app công khai chỉ thực hiện sau
khi người dùng xác nhận. Không có deploy URL khi mới chạy localhost. Baseline
InGesture chưa đạt kiểm chứng smartphone thì phải ghi rõ là demo trong trang
nộp bài.
