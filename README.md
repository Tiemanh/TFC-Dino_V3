# TF-C: Thay thế hai Encoder bằng DINOv3

**Phát triển từ:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining)<br>
**Bài báo:** [NeurIPS 2022](https://openreview.net/forum?id=OJ4mMfGKLN) · [arXiv](https://arxiv.org/abs/2206.08496)<br>
**Backbone thay thế:** [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)<br>
**Ứng dụng:** [tfc-dinov3-gesture.streamlit.app](https://tfc-dinov3-gesture.streamlit.app/)

> **Yêu cầu chính:** thay **cả Time Encoder và Frequency Encoder** của TF-C bằng
> hai DINOv3 ViT-S/16 độc lập, giữ lại luồng học tương phản time–frequency,
> projector và classifier của TF-C. Repository có code huấn luyện, kiểm tra,
> xuất model và chạy dự đoán từ file CSV/TXT.

## 1. Tổng quan

TF-C (*Time-Frequency Consistency*) học biểu diễn chuỗi thời gian từ hai góc
nhìn: tín hiệu trong **miền thời gian** và độ lớn FFT trong **miền tần số**.
Mã nguồn công khai của tác giả tại mốc
[`9667582`](https://github.com/mims-harvard/TFC-pretraining/tree/9667582)
dùng hai `TransformerEncoder` riêng cho hai miền.

Trong dự án này, hai encoder đó được thay bằng:

| Nhánh TF-C | Encoder mới | Đầu ra |
|---|---|---|
| Time domain | `DINOv3SignalEncoder` số 1 | CLS token → `projector_t` → 128 chiều |
| Frequency domain | `DINOv3SignalEncoder` số 2 | CLS token → `projector_f` → 128 chiều |

Hai encoder có cùng kiến trúc nhưng là **hai instance độc lập**, không dùng
chung tham số. Giao diện bốn đầu ra của TF-C vẫn được giữ:
`h_time, z_time, h_freq, z_freq`.

## 2. Kiến trúc sau khi thay encoder

```text
Gia tốc X/Y/Z [B, 3, L]
        |
        +---------------- Time view ----------------+
        |                                            |
        |   SignalToImageAdapter -> ảnh RGB          |
        |   -> DINOv3 Time Encoder -> CLS -> z_time  |
        |                                            |
        +-- |FFT| -------- Frequency view -----------+
                                                     |
            log1p -> SignalToImageAdapter -> ảnh RGB |
            -> DINOv3 Frequency Encoder -> CLS       |
            -> z_frequency                           |

TF-C loss = loss_time + loss_frequency + loss_time-frequency
Phân loại = concat(z_time, z_frequency) -> classifier -> 8 logits
```

DINOv3 nhận ảnh RGB, còn TF-C nhận chuỗi 1D. Vì vậy
`SignalToImageAdapter` thực hiện bốn bước:

1. Chuẩn hóa từng chuỗi cảm biến.
2. Dùng `Conv1d(input_channels, 3, kernel_size=1)` để học phép ánh xạ X/Y/Z.
3. Nội suy chiều dài và trải ba trace thành tensor ảnh vuông.
4. Chuẩn hóa theo thống kê ảnh trước khi đưa vào DINOv3.

Đây là adapter có thể học; gradient đi từ loss qua DINOv3 về adapter kể cả khi
backbone pretrained đang được đóng băng.

## 3. Những vị trí đã cập nhật

### 3.1. Thay hai encoder trong `model.py`

**File chính:** [`code/TFC/model.py`](code/TFC/model.py)

| Vị trí | Nội dung cập nhật |
|---|---|
| [`load_local_dinov3`](code/TFC/model.py#L23) | Nạp checkpoint DINOv3 từ máy và kiểm tra cấu hình/trọng số. |
| [`SignalToImageAdapter`](code/TFC/model.py#L56) | Chuyển chuỗi cảm biến `[B, C, L]` thành ảnh RGB cho DINOv3. |
| [`DINOv3SignalEncoder`](code/TFC/model.py#L116) | Ghép adapter, backbone DINOv3 và lấy CLS token. |
| [`TFC`](code/TFC/model.py#L177) | Khởi tạo hai DINOv3 riêng cho time và frequency. |
| [`TFC.forward`](code/TFC/model.py#L194) | Giữ đúng giao diện bốn tensor của TF-C gốc. |

Phần thay encoder nằm trực tiếp trong class `TFC`:

```python
self.time_encoder = DINOv3SignalEncoder(configs, "time", time_backbone)
self.frequency_encoder = DINOv3SignalEncoder(
    configs, "frequency", frequency_backbone
)

def forward(self, x_in_t, x_in_f):
    h_time = self.time_encoder(x_in_t)
    z_time = self.projector_t(h_time)
    h_freq = self.frequency_encoder(torch.log1p(x_in_f.clamp_min(0)))
    z_freq = self.projector_f(h_freq)
    return h_time, z_time, h_freq, z_freq
```

### 3.2. Các file hỗ trợ để chạy đầu cuối

| File | Vai trò |
|---|---|
| [`code/TFC/dataloader.py`](code/TFC/dataloader.py#L39) | Giữ ba kênh X/Y/Z và tạo frequency view bằng FFT magnitude. |
| [`code/TFC/train_gestures.py`](code/TFC/train_gestures.py#L128) | Pretrain TF-C, fine-tune classifier, đánh giá và xuất model bundle. |
| [`code/TFC/inference.py`](code/TFC/inference.py#L13) | Nạp model đã xuất và dự đoán tám lớp mà không cần Internet. |
| [`code/TFC/smoke_test_dinov3.py`](code/TFC/smoke_test_dinov3.py#L18) | Kiểm tra forward/backward, hai backbone độc lập và gradient adapter. |
| [`code/config_files/InGesture_Configs.py`](code/config_files/InGesture_Configs.py) | Cấu hình đầu vào ba kênh, chiều dài 206 và tám lớp. |
| [`streamlit_app.py`](streamlit_app.py) | Giao diện tải CSV/TXT, xem tín hiệu và nhận kết quả dự đoán. |

Các thành phần TF-C được giữ lại gồm augmentation, `NTXentLoss_poly`,
time-frequency consistency loss, hai projector 128 chiều và downstream
classifier.

## 4. Cách sử dụng DINOv3

Code hỗ trợ hai chế độ:

### DINOv3 pretrained từ máy

Đặt checkpoint theo định dạng Transformers vào `weights/dinov3-vits16/`.
Thư mục phải có `config.json` và file trọng số tương ứng. Khi đó hai encoder
được khởi tạo từ pretrained DINOv3 ViT-S/16; checkpoint được đọc hoàn toàn từ
máy, không bắt buộc đăng nhập Hugging Face lúc chạy.

### DINOv3 ViT-S/16 học từ đầu

Thêm `--from-scratch` để dùng đúng kiến trúc ViT-S/16 nhưng khởi tạo trọng số
ngẫu nhiên. Chế độ này chạy được khi chưa có checkpoint pretrained, tuy nhiên
không được xem là DINOv3 pretrained.

Model bundle đang có trong `artifacts/gesture_model/model_bundle.pt` được huấn
luyện theo chế độ **ViT-S/16 học từ đầu**. Repository không giả định hoặc ghi
nhãn pretrained cho model này.

## 5. Cài đặt và kiểm tra

Khuyến nghị Python 3.11. Trong VS Code, mở terminal tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch phù hợp với CPU/CUDA theo
[hướng dẫn chính thức](https://pytorch.org/get-started/locally/), sau đó chạy:

```powershell
python -m pip install -r requirements-dinov3.txt
python code/TFC/smoke_test_dinov3.py
python -m unittest discover -s tests -v
```

Smoke test không cần dataset hoặc checkpoint pretrained. Nó xác nhận:

- Cả hai nhánh đều chạy forward và backward.
- Time Encoder và Frequency Encoder là hai object khác nhau.
- Adapter, projector và classifier nhận được gradient.
- Classifier trả đúng tám logits.

## 6. Huấn luyện và xuất model

### Chạy baseline không cần pretrained

```powershell
python train_offline.py --output-dir artifacts/gesture_model_v2
```

### Chạy với checkpoint DINOv3 pretrained trên máy

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

Mỗi lần huấn luyện xuất `model_bundle.pt`, `metrics.json`,
`training_info.json` và `history.json`. Nên dùng thư mục output mới để không
ghi đè model cũ.

## 7. Dữ liệu

Dự án hiện dùng **InGesture v5** làm dữ liệu thực nghiệm chính. Mỗi mẫu gồm
chuỗi gia tốc X/Y/Z và một trong tám nhãn cử chỉ. Dữ liệu được làm sạch, chuẩn
hóa và cắt thành cửa sổ 206 mẫu trước khi huấn luyện.

- Nguồn: [InGesture v5](https://doi.org/10.17632/fdxst56tcj.5)
- Dữ liệu gốc: `data/external/InGesture/`
- Tensor sau xử lý: `datasets/InGestureOffline/`
- Script chuẩn bị: [`tools/prepare_ingesture_data.py`](tools/prepare_ingesture_data.py)

Dữ liệu dung lượng lớn không được đưa vào GitHub; repository lưu code xử lý,
model đã xuất và file CSV mẫu.

## 8. Chạy ứng dụng dự đoán

```powershell
python -m pip install -r requirements-web.txt
streamlit run streamlit_app.py
```

Mở `http://localhost:8501`, tải CSV/TXT có các cột `timestamp, x, y, z`, sau đó
chọn **Run Prediction**. Ứng dụng hiển thị tín hiệu miền thời gian, phổ tần số,
nhãn dự đoán, độ tin cậy và xác suất của tám lớp.

Bản triển khai công khai:
[https://tfc-dinov3-gesture.streamlit.app/](https://tfc-dinov3-gesture.streamlit.app/)

## 9. Trạng thái model hiện tại

Model hiện có là baseline kỹ thuật để xác nhận toàn bộ pipeline chạy thật từ
file đầu vào đến kết quả dự đoán. Model dùng hai DINOv3 ViT-S/16 khởi tạo ngẫu
nhiên, đạt test accuracy **39,79%** và macro F1 **18,57%** trên InGesture.

Các chỉ số này không chứng minh pretrained DINOv3 cải thiện TF-C. Muốn đánh giá
lợi ích của pretrained cần nạp checkpoint hợp lệ, huấn luyện lại với cùng cách
chia dữ liệu và so sánh trên cùng test set.

## 10. Tài liệu tham khảo

- Xiang Zhang và cộng sự, [*Self-Supervised Contrastive Pre-Training for Time
  Series via Time-Frequency Consistency*](https://openreview.net/forum?id=OJ4mMfGKLN),
  NeurIPS 2022.
- [Mã nguồn TF-C của nhóm tác giả](https://github.com/mims-harvard/TFC-pretraining).
- [Repository DINOv3 chính thức của Meta](https://github.com/facebookresearch/dinov3).
- [DINOv3 ViT-S/16 checkpoint](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

*Đây là bản phát triển phục vụ yêu cầu thay encoder/backbone trong TF-C, không
phải bản phát hành chính thức của nhóm tác giả TF-C hoặc Meta AI.*
