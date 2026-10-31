# Lựa chọn loss cho TF-C + DINOv3 trên InGesture

## Kết luận áp dụng

Không có loss tối ưu chỉ vì backbone mang tên DINOv3. Với dự án này, lựa chọn
thực dụng là giữ TF-C ở pretraining và so sánh loss phân loại có cân bằng lớp
ở fine-tuning. CB Focal đang deploy là baseline; cấu hình mới cần thắng trên
validation trước khi thay model. Code mới bổ sung joint objective tùy chọn:

`L_pretrain = 0.2 * (L_time + L_frequency) + L_cross`

`L_finetune = L_classification + lambda * L_pretrain`

Đây là dạng đối chiếu với objective thực sự dùng trong code TF-C công khai,
không khẳng định đã hiện thực mọi công thức trong bài báo. `L_cross` dùng
NTXentLoss_poly giữa hai projector. Mặc định lambda=0 giữ cách fine-tune của
model đang phục vụ; lambda=0.1 là ứng viên nghiên cứu, chưa chứng minh tốt hơn.

## Vì sao không chuyển sang loss pretraining DINOv3 của Meta?

Model card chính thức liệt kê DINO self-distillation, iBOT masked-image loss,
KoLeo và Gram anchoring. Đây là recipe học biểu diễn thị giác với teacher,
student, masking và nhiều crop. Gram anchoring phục vụ chất lượng patch features
trong training dài; không phải loss phân loại tám cử chỉ có thể thay trực tiếp
vào CrossEntropyLoss.

Chúng ta dùng DINOv3 làm encoder trong TF-C, lấy CLS token để phân loại và học
adapter chuỗi-thành-ảnh. Không có pipeline teacher–student của Meta. Thêm riêng
một số hạng Gram vào mạng hiện tại sẽ không tái tạo recipe DINOv3 gốc.
Khi có pretrained hợp lệ, nên thử đóng băng backbone và học adapter/projector/
classifier trước. Với backbone khởi tạo ngẫu nhiên như hiện tại, phải học
backbone; không được đóng băng một mạng ngẫu nhiên rồi gọi là pretrained.

## Các ứng viên loss

| Ứng viên | Công thức phân loại | Vai trò |
|---|---|---|
| CE + shuffle | `-log(p_y)` | Baseline có cùng sampler để so sánh công bằng. |
| CB CE + shuffle | `-w_y log(p_y)` | Kiểm tra tác dụng của trọng số lớp, gamma=0. |
| CB Focal gamma=1 + shuffle | `-w_y (1-p_y) log(p_y)` | Mức tập trung mẫu khó vừa phải, ứng viên khởi đầu. |
| CB Focal gamma=2 + shuffle | `-w_y (1-p_y)^2 log(p_y)` | Đối chiếu với model hiện có; có thể nhạy với nhiễu nhãn. |
| CB Focal + TF-C | Loss phân loại + lambda × TF-C | Kiểm tra giữ consistency trong fine-tuning. |

`w_c = a_c / mean(a)`, `a_c = (1-beta)/(1-beta^n_c)`. Số mẫu n_c chỉ tính từ
tập train thực sự được dùng. Khởi đầu beta=0.999, so sánh thêm 0.9999 trên
validation nếu dùng train đầy đủ. Không vừa dùng class weights vừa oversample
theo nghịch đảo tần suất trong cùng ứng viên.

## Những gì đã đo và chưa đo

Model CE trước đây: accuracy 39.79%, macro F1 18.57%, balanced accuracy 31.27%.
Model CB Focal hiện tại: 50.72%, 20.12%, 29.61%, trên cùng 15,658 cửa sổ test.
Tuy nhiên đây không phải ablation thuần loss: sampler cũng thay đổi. Train của
cả hai lần chỉ lấy 256 cửa sổ mỗi lớp nên trọng số CB bằng nhau. Vì vậy không
thể dùng kết quả này để chứng minh class balancing tốt hơn, hoặc gamma=2 là
tối ưu. Test này đã được xem trong quá trình phát triển; không gọi nó là một
benchmark hoàn toàn chưa từng quan sát. Để công bố chắc chắn hơn cần thêm
subject holdout hoặc group cross-validation.

Lần sửa này đã kiểm tra forward/backward của joint loss và đường chạy
validation-only. Chưa hoàn tất ma trận thực nghiệm backbone ViT-S/16 đầy đủ
trên nhiều seed. Model deploy không bị thay bởi kết quả smoke test.

## Quy trình tối ưu hợp lệ

1. Giữ split subject 32/8/10 và normalization fit từ train. Không chia ngẫu
   nhiên các cửa sổ chồng lấn sang nhiều tập.
2. Dùng train đầy đủ (44,862 cửa sổ) hoặc một mẫu phân tầng giữ tỷ lệ lớp.
   Full validation là 8,334 cửa sổ; không cap validation thành số đều mỗi lớp.
3. Giữ sampler=shuffle, seed, số epoch, image size và learning rate như nhau.
   Reset RNG trước khởi tạo model để các cấu hình cùng seed có cùng trọng số
   ban đầu, kể cả khi joint loss tạo thêm augmentation.
4. So sánh năm ứng viên, ít nhất ba seed 42/43/44. Chọn mean macro F1 validation,
   đồng thời xem độ lệch chuẩn, balanced accuracy và recall từng cử chỉ.
5. Chỉ đánh giá test cuối sau khi chốt cấu hình. Không chọn loss bằng accuracy
   test, một file CSV mẫu hoặc độ thấp của training loss.
6. Nếu vẫn dùng random init/ảnh 64×64, giới hạn kết luận trong điều kiện này;
   kết quả không chứng minh hiệu quả transfer của pretrained DINOv3.

Chạy từ thư mục gốc:

```powershell
python tools/compare_loss_configs.py --dry-run
python tools/compare_loss_configs.py --output-dir artifacts/loss_ablation_v1
```

Chế độ mặc định dùng train đầy đủ và ba seed, tốn thời gian GPU đáng kể.
Nếu có checkpoint local, thêm `--local-model-dir weights/dinov3-vits16`.
Công cụ gọi `--validation-only`, không đọc test.pt, không xuất model deploy,
và lưu comparison.json theo macro F1 validation. `--smoke-test` chỉ kiểm tra
protocol với backbone nhỏ, không được dùng kết quả làm bằng chứng chất lượng.

Chạy riêng một ứng viên (chưa đánh giá test):

```powershell
python code/TFC/train_gestures.py --from-scratch `
  --data-dir datasets/InGestureOffline --output-dir artifacts/joint_candidate `
  --classification-loss cb_focal --cb-beta 0.999 --focal-gamma 1 `
  --finetune-tfc-weight 0.1 --sampler shuffle --validation-only `
  --image-size 64 --batch-size 16 --learning-rate 0.0003 `
  --pretrain-epochs 2 --finetune-epochs 12
```

## Hạn chế ngoài loss

Adapter hiện trải trace thành ảnh với các hàng lặp lại. Đây không phải ảnh tự
nhiên hay spectrogram đã được kiểm chứng. EDA kèm dự án hiển thị ảnh time và
frequency lúc adapter khởi tạo để thấy rõ giới hạn này. Nên đối chiếu GAF/STFT
hoặc adapter tốt hơn bằng cùng protocol trước khi quy toàn bộ vấn đề cho loss.
Các cửa sổ overlap có tương quan; n_c không phải số quan sát hoàn toàn độc lập.

## Đối chiếu bài nộp tham khảo

PDF người dùng gửi gồm link GitHub, Kaggle Dataset, Kaggle Visualization
Notebook và website. Repo DeepSatModels-DINOv3 dùng frozen pretrained DINOv3
cho segmentation ảnh vệ tinh và Masked Cross Entropy bỏ void pixels. Cách
tách Dataset/Notebook phù hợp để tham khảo cách bàn giao. Loss segmentation
không thể sao chép nguyên sang tám nhãn gesture. Notebook Kaggle mẫu đã được
đọc trực tiếp bằng trình duyệt: có inventory, sample format/pool, ảnh/nhãn và
overlay, temporal/multi-season, coverage, class distribution, temporal coverage
và đường NDVI mean ± SD. Notebook InGesture học cách bố trí này, chuyển thành
tín hiệu/nhãn recording/cửa sổ, multi-subject và profile gia tốc mean ± SD.

## Nguồn gốc nghiên cứu

- DINOv3 model card: https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md
- DINOv3 paper: https://arxiv.org/abs/2508.10104
- TF-C paper: https://zitniklab.hms.harvard.edu/publications/papers/TF-C-self_supervised_contrastive_main.pdf
- Class-Balanced Loss, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Focal Loss, ICCV 2017: https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_Dense_Object_Detection_ICCV_2017_paper.html
- Repo tham khảo: https://github.com/loanhviet/DeepSatModels-DINOv3
