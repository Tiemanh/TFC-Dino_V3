# Thu dữ liệu smartphone private thật

## Chốt nhãn trước khi thu

Bài TF-C gốc dùng tám cử chỉ UWave: trái, phải, lên, xuống, tròn ngược chiều
kim đồng hồ, tròn cùng chiều, hình vuông, mũi tên phải. InGesture hiện dùng
tám nhãn khác được giữ nguyên trong datasets/InGestureOffline/metadata.json.
Hãy chốt đúng tám nhãn cần nộp với giảng viên; không tự đổi tên InGesture
thành UWave. Công cụ private giữ nguyên nhãn bạn nhập và huấn luyện lại theo
chính nhãn đó; không ép dùng danh sách InGesture.

## Cách thu

1. Dùng phyphox trên Android/iOS: https://phyphox.org/.
   Ứng dụng hỗ trợ xuất dữ liệu cảm biến và điều khiển từ xa. Tài liệu:
   https://phyphox.org/docs/remote-interface/.
2. Chọn loại gia tốc thống nhất cho mọi lần thu, ví dụ **Acceleration with g**.
   Giữ hướng điện thoại, tay cầm và cách thực hiện nhất quán. Ghi loại cảm biến
   và đơn vị vào ghi chú thu thập. Không trộn dữ liệu có/không có trọng lực.
3. Một lần ghi là một cử chỉ, khoảng 2–4 giây, tránh phần chờ dài ở đầu/cuối.
   Dừng, lưu, rồi bắt đầu bản ghi mới. Dùng mã p01, p02 cho người tham gia.
4. Mục tiêu ban đầu: khoảng 20 lần/nhãn/người, từ 5 người nếu có thể.
   Đây là đề xuất thiết kế, chưa phải số lượng đã thu. Ghi qua nhiều phiên,
   tránh mọi dữ liệu đều đến từ một người hoặc một lần cầm điện thoại.
5. Xuất **CSV với dấu phẩy phân cột và dấu chấm thập phân**, đưa về máy.
   Nếu ứng dụng xuất ZIP, giải nén và lấy bảng có cùng timestamp và cả X/Y/Z.
   Không dùng bảng chỉ có một trục hoặc chỉ độ lớn gia tốc.
6. Giữ bản gốc bên ngoài data/raw, ví dụ data/phone_exports. Đặt tệp có nhãn,
   mã người và mã lần ghi; không dùng họ tên/số điện thoại làm mã.

Mỗi người phải đồng ý việc thu và công khai dữ liệu trước khi đăng Kaggle.

## Nhập và chuẩn hóa định dạng

Với CSV đã có timestamp (giây), x, y, z:

```powershell
python tools/import_private_recording.py --input data/phone_exports/take001.csv --label waving_left --participant p01 --take 001 --units ms2
```

`waving_left` chỉ là tên minh họa, phải thay bằng nhãn thật đã chốt.
Nếu file phyphox dùng tiêu đề khác, chỉ rõ tên cột đúng như trong CSV:

```powershell
python tools/import_private_recording.py --input data/phone_exports/take001.csv --label waving_left --participant p01 --take 001 --units ms2 --time-column "Time (s)" --x-column "Acceleration x (m/s^2)" --y-column "Acceleration y (m/s^2)" --z-column "Acceleration z (m/s^2)"
```

Tên cột thực tế có thể khác theo phiên bản/ngôn ngữ; kiểm tra hàng đầu của CSV.
Nếu gia tốc xuất ở đơn vị g, dùng `--units g`. Timestamp đầu vào bắt buộc là
giây; công cụ không tự đoán mili giây. Công cụ sắp xếp timestamp, bỏ trùng,
nội suy giá trị thiếu, resample 200 Hz và chuyển gia tốc m/s² sang g nếu cần.
Resample không tạo thêm thông tin đo; giữ ghi chú tần số gốc của điện thoại.

Đầu ra: `data/raw/<nhãn>/p01_<nhãn>_001.csv` và tệp JSON ghi nguồn, đơn vị,
tần số và SHA256 bản gốc. Không dùng CSV mẫu/InGesture làm dữ liệu tự thu.

## Cleaning → segmentation → normalization → train

Sau khi đủ tám nhãn và ít nhất hai bản ghi riêng mỗi nhãn:

```powershell
python tools/prepare_gesture_data.py --raw-dir data/raw --output-dir datasets/Gesture
python train_offline.py --data-dir datasets/Gesture --output-dir artifacts/private_gesture_model
```

Pipeline tách theo recording_id trước khi tạo cửa sổ; thống kê cắt ngoại lệ và
chuẩn hóa chỉ fit trên train. Cách tách mặc định này chưa đảm bảo người độc lập:
nếu cần báo cáo khả năng nhận người mới, phải chia theo người trước khi train.
Private chưa có val.pt thì lưu epoch cuối, không chọn model dựa vào test.

Trong giai đoạn private, dữ liệu tải lên web cũng phải cùng cảm biến, đơn vị g
và tần số 200 Hz như dữ liệu train đã nhập. Không đưa thẳng CSV m/s² vào model
đã học g. Web dự đoán thử nghiệm; điểm confidence là softmax, chưa được hiệu chỉnh.

## Chuẩn bị gói Kaggle (chưa công khai)

```powershell
python tools/package_private_dataset.py --dataset-id TEN_TAI_KHOAN/ten-dataset --title "Smartphone gesture recordings" --license CC-BY-4.0
```

Chỉ chọn CC-BY-4.0 nếu bạn có quyền cấp giấy phép này và người tham gia đồng ý.
Thư mục artifacts/kaggle_private_upload chứa dữ liệu, metadata và manifest.
Công cụ không đăng nhập, upload hay publish. Kiểm tra nội dung và xác nhận
trước thao tác công khai; Kaggle public URL chỉ có sau khi đăng thật.
