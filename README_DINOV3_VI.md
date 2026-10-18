# TF-C + DINOv3 cho nhận dạng 8 cử chỉ smartphone

Đây là bản mở rộng trực tiếp từ repo công khai
[`mims-harvard/TFC-pretraining`](https://github.com/mims-harvard/TFC-pretraining),
commit nền `9667582`. Bản mở rộng giữ dataloader, augmentation, Poly-NT-Xent,
TF-C consistency loss, trainer, projector và classifier flow của tác giả. Hai
`TransformerEncoder` trong code công khai được thay bằng hai DINOv3 riêng biệt.

Code hiện thực yêu cầu kiến trúc. Có hai lựa chọn: pretrained từ file cục bộ,
hoặc `--from-scratch` học DINOv3 ViT-S/16 đầy đủ từ khởi tạo ngẫu nhiên.
Lựa chọn thứ hai không cần quyền tải checkpoint, nhưng kết quả không tương đương
pretrained. Xem BAT_DAU_TAI_DAY.md. Smoke test vẫn là chế độ kỹ thuật riêng.

Khi đã có CSV của 8 cử chỉ, xem quy trình tự động từ làm sạch dữ liệu đến xuất
model dùng cho web tại [`HUONG_DAN_CHAY_DU_LIEU_MOI.md`](HUONG_DAN_CHAY_DU_LIEU_MOI.md).

## Kiến trúc đã thay

```text
X/Y/Z time view [B,3,L]
  -> learnable 1x1 channel adapter -> RGB signal image [B,3,224,224]
  -> DINOv3 Time Encoder -> CLS h_time -> projector_t -> z_time [B,128]

abs(FFT) frequency view [B,3,L]
  -> log1p -> learnable 1x1 channel adapter -> RGB spectrum image [B,3,224,224]
  -> DINOv3 Frequency Encoder -> CLS h_freq -> projector_f -> z_freq [B,128]

Original TF-C losses: L_time + L_frequency + L_TF
Fine-tuning: concat(z_time,z_freq) -> original MLP -> 8 gesture logits
```

Hai backbone là hai object riêng và có tham số riêng. Mặc định chúng được đóng băng
để phù hợp GPU RTX 3050 Ti 4 GB; adapter, hai projector và classifier vẫn học.
Không dùng `torch.no_grad()` quanh backbone vì gradient cần đi xuyên qua backbone
đóng băng để cập nhật adapter. Có thể mở toàn bộ backbone bằng cờ
`--unfreeze_dinov3`, nhưng cần GPU nhiều VRAM hơn.

## Vị trí đã cập nhật

| File | Thay đổi |
|---|---|
| `code/TFC/model.py` | Thêm `SignalToImageAdapter`, `DINOv3SignalEncoder`; thay cả Time Encoder và Frequency Encoder; giữ API bốn tensor của TF-C và projector 128 chiều |
| `code/TFC/dataloader.py` | Giữ `configs.input_channels` thay vì luôn cắt còn một kênh; FFT magnitude của tác giả vẫn được dùng |
| `code/config_files/HAR_Configs.py` | Chọn 3 kênh gia tốc, 8 lớp Gesture và cấu hình checkpoint DINOv3 |
| `code/TFC/main.py` | Thêm tham số DINOv3, đường dẫn checkpoint mới, chỉ đưa tham số trainable vào optimizer, và cờ `--subset` rõ ràng |
| `code/TFC/smoke_test_dinov3.py` | Kiểm tra forward/backward, loss gốc, hai backbone riêng và gradient adapter |
| `tests/test_dinov3_model.py` | Unit test shape, channel validation, 8 logits và chính sách freeze |
| `requirements-dinov3.txt` | Thư viện bổ sung cho DINOv3 |

Xem diff chính xác so với tác giả:

```powershell
git diff origin/main -- code/TFC/model.py code/TFC/dataloader.py code/TFC/main.py code/config_files/HAR_Configs.py
```

## Cài trên VS Code / Windows

Python 3.12 và PyTorch CUDA đã được kiểm tra trên máy phát triển. Từ thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Cài PyTorch CUDA phù hợp theo hướng dẫn chính thức:
https://pytorch.org/get-started/locally/. Sau đó:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dinov3.txt
```

Chọn `.venv\Scripts\python.exe` bằng **Python: Select Interpreter** trong VS Code.

## Kiểm tra code không cần tài khoản hoặc dataset

```powershell
cd code\TFC
..\..\.venv\Scripts\python.exe smoke_test_dinov3.py
cd ..\..
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Smoke test dùng DINOv3 thu nhỏ khởi tạo ngẫu nhiên và tensor tổng hợp chỉ để kiểm tra
luồng code. Kiểm tra đúng kiến trúc ViT-S đầy đủ mà không tải trọng số:

```powershell
cd code\TFC
..\..\.venv\Scripts\python.exe smoke_test_dinov3.py --full-architecture
```

Không dùng đầu ra smoke test làm kết quả nghiên cứu hoặc dự đoán cử chỉ.

## Lấy trọng số pretrained DINOv3

Model mặc định là `facebook/dinov3-vits16-pretrain-lvd1689m`. Checkpoint yêu cầu
tài khoản Hugging Face và chấp nhận điều kiện trên trang model:
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m.

Đăng nhập trên máy bằng token chỉ có quyền đọc. Không commit token, `.env` hoặc
cache model lên GitHub:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Sau lần tải đầu có thể dùng `--dinov3_local_files_only`. Nếu cung cấp thư mục
checkpoint local qua `--dinov3_model_name`, thư mục đó phải có cấu trúc
`from_pretrained` hợp lệ.

## Dataset theo định dạng tác giả

Đặt các file PyTorch của tác giả tại:

```text
datasets/HAR/train.pt
datasets/Gesture/train.pt
datasets/Gesture/test.pt
```

Mỗi `.pt` là dictionary có `samples` và `labels`. `samples` cần dạng
`[N, channels, length]` hoặc `[N, length, channels]`; dataloader chuẩn hóa thành
`[N, 3, TSlength_aligned]`. Kịch bản code của tác giả dùng HAR để pretrain và
Gesture để fine-tune 8 lớp.

Đây là ba file mà `data_generator` hiện tại thực sự đọc. Các file `val.pt` hoặc
`HAR/test.pt` có thể tồn tại trong một gói dữ liệu đầy đủ, nhưng không được hai
chế độ chạy bên dưới sử dụng.

## Chạy pretrain và fine-tune

Chạy từ `code/TFC` để các đường dẫn tương đối của repo tác giả được giữ đúng:

```powershell
cd code\TFC
..\..\.venv\Scripts\python.exe main.py `
  --training_mode pre_train `
  --pretrain_dataset HAR `
  --target_dataset Gesture `
  --run_description dinov3_v1

..\..\.venv\Scripts\python.exe main.py `
  --training_mode fine_tune_test `
  --pretrain_dataset HAR `
  --target_dataset Gesture `
  --run_description dinov3_v1
```

Thêm `--subset` cho vòng debug ngắn. Không dùng `--subset` cho kết quả bàn giao.
Hai lệnh phải dùng cùng `run_description`, seed và model để fine-tune tìm đúng
checkpoint pretrain. Checkpoint DINOv3 được lưu trong thư mục có hậu tố `_dinov3`,
không nạp nhầm checkpoint Transformer cũ đi kèm repo tác giả.

## Giới hạn đã biết

- Chưa có dữ liệu private thật trong repo này, nên chưa có checkpoint 8 cử chỉ hay metric.
- Trọng số pretrained chưa thể được kiểm tra cho đến khi chủ tài khoản chấp nhận giấy phép DINOv3.
- Adapter tạo ảnh tín hiệu là thiết kế của bản mở rộng này, không phải đóng góp được
  mô tả trong bài TF-C gốc. README phải giữ thông tin này khi nộp.
- Với 4 GB VRAM, cấu hình mặc định freeze backbone và cần batch nhỏ khi chạy dữ liệu thật.

## Nguồn và giấy phép

- TF-C code/paper: https://github.com/mims-harvard/TFC-pretraining
- DINOv3 model card: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- DINOv3 Transformers docs: https://huggingface.co/docs/transformers/v4.56.2/model_doc/dinov3

Repo gốc có giấy phép MIT. Checkpoint DINOv3 có giấy phép riêng; người dùng phải
đọc và chấp nhận trước khi tải hoặc phân phối model.
