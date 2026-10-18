# Phần 2 — Kế hoạch dữ liệu private 8 cử chỉ

## Kết luận lựa chọn dữ liệu

Bộ dữ liệu chính để nộp Kaggle phải được tự thu bằng smartphone. Không đổi tên
một bộ công khai thành private. Bộ InGesture công khai chỉ dùng để tham khảo
protocol, kiểm tra code hoặc thí nghiệm transfer learning có ghi nguồn rõ ràng.

Nguồn tham khảo:

- InGesture (8 lớp, accelerometer + gyroscope, 50 người, 200 Hz):
  https://data.mendeley.com/datasets/fdxst56tcj/5
- Bài mô tả và quy trình InGesture:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13136770/
- phyphox — ứng dụng smartphone đọc cảm biến và xuất CSV:
  https://phyphox.com/
- Tài liệu định dạng/export phyphox:
  https://phyphox.org/docs/file-format/
- UCI HAR — dữ liệu smartphone CC BY 4.0 dùng làm nguồn pretrain tham khảo:
  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Tám nhãn đề xuất

Giữ bộ nhãn có chuyển động tay gần nhau để bài toán có ý nghĩa:

| ID | Nhãn trong dữ liệu | Mô tả |
|---:|---|---|
| 0 | `neutral` | Cầm điện thoại tự nhiên, không thực hiện cử chỉ |
| 1 | `drink` | Đưa tay/điện thoại mô phỏng động tác uống |
| 2 | `answer_phone` | Đưa tay lên tai như nghe điện thoại |
| 3 | `scratch_head` | Đưa tay lên gãi đầu |
| 4 | `hand_over_face` | Đưa tay qua mặt |
| 5 | `adjust_glasses` | Đưa tay chỉnh kính |
| 6 | `hold_chin` | Đưa tay giữ cằm |
| 7 | `hands_behind_neck` | Đưa tay ra sau cổ |

Đây là bộ 8 lớp theo InGesture. Nếu đề bài/bài báo đã quy định tám tên khác,
phải thay danh sách trước khi thu; không đổi nhãn sau khi đã thu xong.

## Công cụ thu

1. Cài **phyphox** trên Android hoặc iPhone.
2. Mở thí nghiệm **Acceleration without g** nếu có; nếu không, dùng
   **Acceleration with g** và ghi rõ lựa chọn trong mô tả dataset.
3. Đặt điện thoại cố định trong tay thuận, cùng một chiều màn hình cho mọi lần.
4. Mỗi lần bấm Start, chờ 1 giây, thực hiện đúng một cử chỉ trong 3–4 giây,
   chờ 1 giây rồi Stop.
5. Xuất CSV và đặt tên theo quy tắc bên dưới.

Không trộn hai cử chỉ trong một file. Không thay đổi tư thế cầm điện thoại giữa
các lần nếu chưa ghi biến thể đó vào metadata.

## Quy mô thu đề xuất

- Tối thiểu để chạy kỹ thuật: 3 người × 15 lần × 8 lớp = 360 recording.
- Nên dùng để nộp: 5 người × 20 lần × 8 lớp = 800 recording.
- Tốt hơn: 10 người × 20 lần × 8 lớp = 1.600 recording.

Mỗi người nên thu ở ít nhất hai phiên khác nhau. Tập test phải tách theo người
hoặc theo phiên, không chia ngẫu nhiên các cửa sổ của cùng một recording.

## Cấu trúc thư mục và tên file

```text
data/raw/
  subject_01/
    session_01/
      neutral/neutral_take_001.csv
      drink/drink_take_001.csv
      answer_phone/answer_phone_take_001.csv
      ...
  subject_02/
    session_01/...
```

Sau khi xuất từ phyphox, chuẩn hóa CSV về:

```csv
timestamp,ax,ay,az,label,recording_id,subject_id,session_id
0.000,0.10,-0.02,0.15,drink,s01_session01_drink_001,subject_01,session_01
0.020,0.12,-0.03,0.18,drink,s01_session01_drink_001,subject_01,session_01
```

Không lưu tên thật, số điện thoại, email, GPS, hình ảnh hoặc âm thanh của người
tham gia. Mã `subject_01` là đủ.

## Kiểm tra trước khi xử lý

- Đủ đúng 8 nhãn và không sai chính tả.
- Mỗi file chỉ có một `recording_id` và một `label`.
- Có đủ X/Y/Z, timestamp tăng dần và không có khoảng mất mẫu lớn.
- Mỗi recording dài hơn 206 mẫu sau khi resample.
- Đơn vị và loại accelerometer giống nhau giữa các máy hoặc được ghi trong metadata.
- Có bảng số recording theo nhãn/người/phiên.

## Nội dung Kaggle public

Dataset Kaggle nên có:

```text
raw/                       # CSV nguyên bản đã ẩn danh
processed/train.pt
processed/test.pt
processed/metadata.json
dataset_card.md            # mục tiêu, thiết bị, sampling, protocol, license
participants.csv           # chỉ mã ẩn danh và thông tin được phép công khai
```

Dataset card phải nói rõ ai thu, ngày thu, loại điện thoại, vị trí/cách cầm máy,
tần số lấy mẫu, định nghĩa 8 cử chỉ, cleaning, segmentation, normalization và
cách chia train/test. Chỉ đăng khi mọi người tham gia đồng ý công khai dữ liệu
cảm biến đã ẩn danh.
