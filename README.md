# TF-C + DINOv3 cho nhận dạng 8 cử chỉ InGesture

Dữ liệu chính của dự án là **InGesture v5** với ba trục gia tốc X/Y/Z và tám
nhãn cử chỉ. Model đã xuất nằm tại
`artifacts/gesture_model/model_bundle.pt`.

## 1. Nguồn mã và phạm vi thay đổi

- Mã TF-C của tác giả: <https://github.com/mims-harvard/TFC-pretraining>
- Mốc đối chiếu: commit [`9667582`](https://github.com/mims-harvard/TFC-pretraining/tree/9667582)
- Bài báo: *Self-Supervised Contrastive Pre-Training For Time Series via
  Time-Frequency Consistency*, NeurIPS 2022.
- Repository này giữ giấy phép MIT trong `LICENSE`.

Các thành phần TF-C được giữ lại gồm time/frequency views, augmentation,
`NTXentLoss_poly`, time-frequency consistency loss, hai projector 128 chiều và
classifier tám lớp. Phần encoder của hai miền được thay bằng hai backbone
DINOv3 ViT-S/16 độc lập.

## 2. Kiến trúc sau khi thay encoder

```text
Gia tốc X/Y/Z [B, 3, L]
        |
        +---------------- Time view ----------------+
        |                                            |
        |   SignalToImageAdapter -> ảnh RGB          |
        |   -> DINOv3 Time Encoder -> CLS -> z_time  |
        |                                            |
        +-- abs(FFT) -> log1p -- Frequency view -----+
                                                     |
            SignalToImageAdapter -> ảnh RGB          |
            -> DINOv3 Frequency Encoder -> CLS       |
            -> z_frequency                           |

TF-C loss = loss_time + loss_frequency + loss_time-frequency
Phân loại = concat(z_time, z_frequency) -> MLP -> 8 logits
```

`SignalToImageAdapter` chuẩn hóa từng chuỗi, dùng `Conv1d(3, 3, kernel_size=1)`
có thể học, nội suy chiều dài và trải ba trace thành ảnh RGB vuông. Cách chuyển
tín hiệu thành ảnh là phần mở rộng của repository này vì DINOv3 nhận ảnh, trong
khi đầu vào TF-C là chuỗi thời gian.

Hai backbone là hai object riêng, không dùng chung trọng số:

```python
self.time_encoder = DINOv3SignalEncoder(configs, "time", time_backbone)
self.frequency_encoder = DINOv3SignalEncoder(configs, "frequency", frequency_backbone)
```

## 3. Vị trí đã cập nhật cho yêu cầu 1

| Vị trí | Nội dung cập nhật |
|---|---|
| `code/TFC/model.py` — `SignalToImageAdapter` | Chuyển chuỗi ba trục thành ảnh RGB cho DINOv3. |
| `code/TFC/model.py` — `DINOv3SignalEncoder` | Nạp DINOv3 pretrained từ máy, DINOv3 ViT-S/16 học từ đầu hoặc backbone nhỏ cho smoke test. |
| `code/TFC/model.py` — `TFC` | Thay encoder time và frequency cũ bằng hai DINOv3 riêng; giữ đầu ra `h_time, z_time, h_freq, z_freq`. |
| `code/TFC/dataloader.py` — `Load_Dataset` | Giữ đủ ba kênh X/Y/Z và tạo frequency view bằng FFT magnitude. |
| `code/config_files/InGesture_Configs.py` | Cấu hình ba kênh, chiều dài 206 và tám lớp InGesture. |
| `code/TFC/train_gestures.py` | Pretrain TF-C, fine-tune classifier, đánh giá và xuất `model_bundle.pt`. |
| `tools/prepare_ingesture_data.py` | Làm sạch, tách subject, chuẩn hóa và chia cửa sổ InGesture. |
| `code/TFC/inference.py` | Nạp bundle đã xuất để dự đoán mà không cần tải model từ mạng. |
| `code/TFC/smoke_test_dinov3.py` | Kiểm tra forward/backward, hai backbone và gradient adapter. |
| `tests/` | Kiểm tra shape, tám logits, serialization, preprocessing và web inference. |

Muốn đối chiếu với code tác giả, mở liên kết commit nền ở mục 1 rồi so sánh
trực tiếp các file trong bảng trên.

## 4. Dữ liệu chính: InGesture

- Cảm biến: WT901BLECL5 đeo cổ tay, tần số khoảng 200 Hz.
- Giấy phép dữ liệu: CC BY 4.0.
- Quy mô bản đã dùng: 50 subject, 5.410 CSV và 5.345 đoạn cử chỉ.

Tám nhãn được giữ đúng theo InGesture:

| ID | Nhãn |
|---:|---|
| 0 | `other` |
| 1 | `fluid_intake` |
| 2 | `answering_cell_phone` |
| 3 | `scratching_head` |
| 4 | `passing_hand_over_face` |
| 5 | `adjusting_glasses_touching_temples` |
| 6 | `holding_chin` |
| 7 | `stretching_with_hands_behind_neck` |

Giải nén dataset vào:

```text
data/external/InGesture/
  subject_1/
  subject_2/
  ...
  subject_50/
```

Thư mục dữ liệu lớn được `.gitignore`; repository chỉ lưu mã xử lý và thông tin
nguồn tại `data/external/SOURCE_INGESTURE.md`.

## 5. Pipeline xử lý InGesture

Chạy từ thư mục gốc:

```powershell
python tools/prepare_ingesture_data.py `
  --input-dir data/external/InGesture `
  --output-dir datasets/InGestureOffline `
  --window-size 206 `
  --stride 103 `
  --test-ratio 0.20 `
  --val-ratio 0.16 `
  --seed 42
```

Pipeline thực hiện:

1. Đọc các cột `x, y, z`, chuyển về số và nội suy giá trị thiếu.
2. Tách subject trước khi chuẩn hóa để tránh rò rỉ dữ liệu.
3. Fit ngưỡng 0,5%–99,5%, mean và standard deviation chỉ trên train.
4. Chuẩn hóa rồi cắt cửa sổ 206 mẫu, stride 103.
5. Xuất `train.pt`, `val.pt`, `test.pt` và `metadata.json`.

Tensor đầu ra có dạng `[số_cửa_sổ, 3, 206]`. Mỗi file `.pt` chứa hai khóa
`samples` và `labels`.

## 6. Cài đặt trên Windows/VS Code

Khuyến nghị Python 3.11:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch phù hợp với CPU/CUDA của máy theo
<https://pytorch.org/get-started/locally/>, sau đó:

```powershell
python -m pip install -r requirements-dinov3.txt
```

Trong VS Code chọn interpreter `.venv\Scripts\python.exe`.

## 7. Huấn luyện và xuất model

### Phương án hiện tại: ViT-S/16 học từ đầu

Không cần Hugging Face hoặc checkpoint tải ngoài. Lệnh dưới đây tự chuẩn bị
InGesture nếu `datasets/InGestureOffline/metadata.json` chưa tồn tại:

```powershell
python train_offline.py --output-dir artifacts/gesture_model_v2
```

Có thể truyền cùng tham số qua `train_offline.cmd` khi chạy trong Terminal.
Không ghi đè `artifacts/gesture_model` hiện có; dùng một thư mục output mới cho
mỗi lần thử.

### Dùng trọng số DINOv3 có sẵn trên máy

Đặt checkpoint Transformers hợp lệ vào `weights/dinov3-vits16/`. Thư mục cần có
`config.json` và model weights (`model.safetensors`, `pytorch_model.bin` hoặc
các shard kèm index), rồi chạy:

```powershell
python code/TFC/train_gestures.py `
  --data-dir datasets/InGestureOffline `
  --local-model-dir weights/dinov3-vits16 `
  --output-dir artifacts/gesture_model_pretrained `
  --image-size 224 `
  --batch-size 4 `
  --pretrain-epochs 10 `
  --finetune-epochs 20
```

Mặc định pretrained backbone được đóng băng để giảm VRAM; adapter, projector và
classifier vẫn học. Thêm `--unfreeze-dinov3` chỉ khi GPU đủ bộ nhớ.

Mỗi lần huấn luyện xuất:

```text
artifacts/<ten_run>/
  model_bundle.pt
  metrics.json
  training_info.json
  history.json
```

## 8. Kiểm tra code

Smoke test không cần dataset hoặc pretrained weights:

```powershell
python code/TFC/smoke_test_dinov3.py
python -m unittest discover -s tests -v
```

Kiểm tra kiến trúc ViT-S/16 đầy đủ, vẫn dùng khởi tạo ngẫu nhiên:

```powershell
python code/TFC/smoke_test_dinov3.py --full-architecture
```

Smoke test chỉ xác nhận luồng kỹ thuật, không phải kết quả nhận dạng.

## 9. Model hiện có và giới hạn

Bundle hiện tại được huấn luyện bằng hai DINOv3 ViT-S/16 đầy đủ, khởi tạo ngẫu
nhiên, ảnh 64×64, 2 epoch pretrain và 12 epoch fine-tune trên một tập train giới
hạn để tạo baseline chạy được.

| Chỉ số test InGesture | Kết quả |
|---|---:|
| Accuracy | 39,79% |
| Macro F1 | 18,57% |
| Balanced accuracy | 31,27% |
| Số cửa sổ test | 15.658 |

Đây là baseline kỹ thuật, chưa phải model có độ chính xác cao. Kết quả pretrained
chỉ được công bố sau khi có checkpoint DINOv3 hợp lệ và huấn luyện/đánh giá lại.
Chi tiết từng lớp và confusion matrix nằm trong `reports/offline_baseline/`.

## 10. Cấu trúc chính của repository

```text
TFC-DivV3/
├── code/TFC/                         # TF-C và hai DINOv3 encoder
├── code/config_files/                # cấu hình, gồm InGesture_Configs.py
├── tools/prepare_ingesture_data.py   # preprocessing InGesture
├── datasets/                         # tensor đã xử lý, không đưa lên Git
├── data/external/InGesture/          # dữ liệu gốc, không đưa lên Git
├── artifacts/gesture_model/          # model bundle và metric hiện có
├── reports/offline_baseline/         # báo cáo đánh giá
├── tests/                             # kiểm thử
├── environments/                      # Conda specs của các baseline gốc
├── train_offline.py                  # entrypoint huấn luyện dễ dùng
├── streamlit_app.py                  # kiểm thử model đã xuất
├── requirements-dinov3.txt
└── README.md                         # tài liệu duy nhất của dự án
```

## Trích dẫn

Khi sử dụng repository, cần trích dẫn bài báo TF-C, mã nguồn của tác giả và bộ
dữ liệu InGesture theo DOI ở mục 4. Trọng số DINOv3 pretrained có giấy phép và
điều khoản riêng của nhà phát hành; không đưa checkpoint hoặc token truy cập lên
GitHub.
