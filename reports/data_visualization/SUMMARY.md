# InGesture visualization - kết quả kiểm kê thực tế

Notebook: `kaggle/ingesture_visualization.ipynb` (tính từ thư mục gốc).
Dataset Kaggle: https://www.kaggle.com/datasets/nguyntinmnh41/gesture-dataset
Nguồn gốc: InGesture v5, DOI 10.17632/fdxst56tcj.5, CC BY 4.0.

Đã đọc 5.345 segmented recording thuộc 50 người và tám nhãn, tổng cộng
7.958.400 dòng mẫu X/Y/Z. Có 9 recording ngắn hơn 206 mẫu nên không sinh
cửa sổ; không phát hiện ô X/Y/Z NaN/Inf trong bản dữ liệu này.

| Tập | Số người | Recording | Cửa sổ |
|---|---:|---:|---:|
| Train | 32 | 3.484 | 44.862 |
| Validation | 8 | 647 | 8.334 |
| Test | 10 | 1.214 | 15.658 |
| Tổng | 50 | 5.345 | 68.854 |

Cửa sổ dài 206 mẫu, stride 103; số cửa sổ mỗi lớp ở cả ba tập khớp metadata.
Lớp other có 51.987 cửa sổ (75,50%); các lớp hiếm chỉ có khoảng 660-859 cửa
sổ toàn bộ dataset. Accuracy đơn lẻ dễ che khuất các cử chỉ hiếm.

Mười bốn figure: phân bố lớp, phân bố split, độ phủ subject/gesture, chất lượng
và thời lượng, time series, FFT, thống kê train, ảnh adapter, đối chiếu tín hiệu
và nhãn, so sánh nhiều người, profile mean ± SD, PCA, RMS theo kênh và STFT.
Notebook có 20 code cell đã chạy, 14 ảnh output, không có dữ liệu mô phỏng.

Audit bổ sung theo notebook Sensor Logger tham khảo: 0 recording hằng số,
85 recording trùng sau lần xuất hiện đầu tiên theo exact cleaned float32 X/Y/Z.
Không thấy hash recording được dùng ở nhiều split subject. Đây là mức
recording, chưa kiểm tra mọi cửa sổ hoặc near-duplicate. Số file và window
trên vẫn bao gồm bản trùng vì pipeline hiện có chưa deduplicate. Cần kiểm tra
`duplicate_recordings.csv` và huấn luyện/đánh giá lại nếu thay đổi dataset.

Bố cục học từ notebook Kaggle của bài tham khảo: giới thiệu/đặc điểm → setup
→ inventory và format → sample pool → tín hiệu/nhãn → biến thiên thời gian
và nhóm → thống kê toàn dataset → summary. Pool minh họa chọn 12 subject train
khác nhau mỗi lớp bằng seed 42, tổng 96 recording; xem `sample_pool.csv`.
Các thống kê toàn dataset không được tính từ pool minh họa này.

Giới hạn diễn giải:
- Cửa sổ chồng lấn không phải các quan sát độc lập.
- Thời lượng dùng tần số danh định 200 Hz, không ước lượng lại timestamp.
- FFT minh họa dùng mean removal/Hann; FFT huấn luyện dùng abs(FFT)+log1p.
- Histogram/correlation dùng tối đa 50 điểm/recording train; không phải phân
  bố theo toàn bộ thời gian. Chuẩn hóa dùng thống kê train đã lưu.
- Ảnh adapter là phép ánh xạ khởi tạo identity, chưa phải output đã học.
- Đường mean ± SD dùng tiến độ tương đối của recording sau nội suy chỉ để
  vẽ; không phải căn chỉnh onset/offset, không phải confidence interval và
  không thay đổi preprocessing huấn luyện.
- Tập dữ liệu là InGesture công khai, không phải dữ liệu private tự thu thập.
