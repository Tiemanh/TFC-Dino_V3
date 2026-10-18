# Kết quả baseline TF-C + DINOv3 học từ đầu

Nguồn khởi tạo: `random_full_vits16`. Dataset: InGesture v5.

Đây là kết quả đo thật trên InGesture v5 công khai.
Hai backbone đầy đủ ViT-S/16 được học từ đầu, không có pretrained ảnh.

- Train dùng: 2048 cửa sổ.
- Validation dùng: 512 cửa sổ, chọn epoch 12.
- Test dùng: 15658 cửa sổ, giữ nguyên toàn bộ phân bố lớp.
- TF-C pretrain: 2 epoch; phân loại: 12 epoch.
- Ảnh: 64×64; batch: 16; seed: 42.
- Accuracy: 39.79%.
- Macro F1: 18.57%.
- Balanced accuracy: 31.27%.
- Accuracy nếu luôn đoán lớp đa số: 76.15%.

Accuracy của baseline hiện thấp hơn cách luôn đoán lớp đa số; macro F1 và
balanced accuracy cho thấy model còn yếu. Không dùng để tuyên bố hệ thống
đã đạt chất lượng nhận dạng cao. Confidence softmax chưa được hiệu chỉnh.

| Nhãn | Precision | Recall | F1 | Số cửa sổ test |
|---|---:|---:|---:|---:|
| other | 96.54% | 41.39% | 57.94% | 11924 |
| fluid_intake | 45.47% | 47.76% | 46.59% | 1922 |
| answering_cell_phone | 19.69% | 13.04% | 15.69% | 966 |
| scratching_head | 2.92% | 46.98% | 5.49% | 149 |
| passing_hand_over_face | 4.57% | 26.63% | 7.80% | 169 |
| adjusting_glasses_touching_temples | 1.48% | 13.29% | 2.66% | 158 |
| holding_chin | 2.69% | 15.73% | 4.60% | 178 |
| stretching_with_hands_behind_neck | 4.26% | 45.31% | 7.78% | 192 |

Tách 32 người train, 8 người validation và 10 người test trước khi
fit normalization. Epoch được chọn bằng validation, không dựa vào test.
Đánh giá trên từng cửa sổ; web tổng hợp xác suất qua các cửa sổ của file.
Không so sánh accuracy cửa sổ này với một dự đoán demo trên file train.

Hướng cải thiện: dùng DINOv3 pretrained hợp lệ, huấn luyện trên toàn bộ
tập train, cân bằng lớp và chạy thêm nhiều seed trước khi báo cáo.
