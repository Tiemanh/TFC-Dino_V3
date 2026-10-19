# Kết quả baseline TF-C + DINOv3 học từ đầu

Lượt thử này dùng CB Focal Loss và shuffle, thay cho Cross Entropy với weighted
sampling của baseline cũ. Chỉ dùng 256 cửa sổ train mỗi lớp nên các trọng số CB
bằng nhau. Một seed chưa đủ để kết luận loss nào tốt hơn; balanced accuracy
giảm dù accuracy và macro F1 tăng. Chi tiết đối chiếu nằm trong README gốc.

Nguồn khởi tạo: `random_full_vits16`. Dataset: InGesture v5.

Loss phân loại: `cb_focal`.

Đây là kết quả đo thật trên InGesture v5 công khai.
Hai backbone đầy đủ ViT-S/16 được học từ đầu, không có pretrained ảnh.

- Train dùng: 2048 cửa sổ.
- Validation dùng: 512 cửa sổ, chọn epoch 11.
- Test dùng: 15658 cửa sổ, giữ nguyên toàn bộ phân bố lớp.
- TF-C pretrain: 2 epoch; phân loại: 12 epoch.
- Ảnh: 64×64; batch: 16; seed: 42.
- Accuracy: 50.72%.
- Macro F1: 20.12%.
- Balanced accuracy: 29.61%.
- Accuracy nếu luôn đoán lớp đa số: 76.15%.

Accuracy của baseline hiện thấp hơn cách luôn đoán lớp đa số; macro F1 và
balanced accuracy cho thấy model còn yếu. Không dùng để tuyên bố hệ thống
đã đạt chất lượng nhận dạng cao. Confidence softmax chưa được hiệu chỉnh.

| Nhãn | Precision | Recall | F1 | Số cửa sổ test |
|---|---:|---:|---:|---:|
| other | 93.35% | 57.56% | 71.21% | 11924 |
| fluid_intake | 54.75% | 37.46% | 44.49% | 1922 |
| answering_cell_phone | 14.02% | 14.91% | 14.45% | 966 |
| scratching_head | 3.53% | 48.32% | 6.58% | 149 |
| passing_hand_over_face | 4.86% | 18.34% | 7.68% | 169 |
| adjusting_glasses_touching_temples | 1.21% | 8.23% | 2.11% | 158 |
| holding_chin | 3.99% | 14.04% | 6.21% | 178 |
| stretching_with_hands_behind_neck | 4.61% | 38.02% | 8.22% | 192 |

Tách 32 người train, 8 người validation và 10 người test trước khi
fit normalization. Epoch được chọn bằng validation, không dựa vào test.
Đánh giá trên từng cửa sổ; web tổng hợp xác suất qua các cửa sổ của file.
Không so sánh accuracy cửa sổ này với một dự đoán demo trên file train.

Hướng cải thiện: dùng DINOv3 pretrained hợp lệ, huấn luyện trên toàn bộ
tập train, cân bằng lớp và chạy thêm nhiều seed trước khi báo cáo.
