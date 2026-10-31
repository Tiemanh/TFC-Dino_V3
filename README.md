# TF-C: Thay thế hai Encoder bằng DINOv3 và cập nhật Loss Function

**Phát triển từ:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining)<br>
**Bài báo:** [NeurIPS 2022](https://openreview.net/forum?id=OJ4mMfGKLN) · [arXiv](https://arxiv.org/abs/2206.08496)<br>
**Backbone thay thế:** [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)<br>
**Ứng dụng:** [tfc-dinov3-gesture.streamlit.app](https://tfc-dinov3-gesture.streamlit.app/)

> **Yêu cầu chính:** thay **cả Time Encoder và Frequency Encoder** của TF-C bằng
> hai DINOv3 ViT-S/16 độc lập và cập nhật loss phân loại thành
> **Class-Balanced Focal Loss**. Giữ lại luồng học tương phản time–frequency,
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
| [`code/TFC/loss.py`](code/TFC/loss.py) — `ClassBalancedFocalLoss` | Thêm loss mới với trọng số effective number và focal modulation. |
| [`code/TFC/train_gestures.py`](code/TFC/train_gestures.py) — `run_finetune`, `main` | Chọn loss, tính trọng số từ train, cập nhật sampler, huấn luyện và xuất model. |
| [`tests/test_losses.py`](tests/test_losses.py) | Kiểm tra tính ổn định gradient, trọng số lớp và trường hợp tương đương Cross Entropy. |
| [`code/TFC/inference.py`](code/TFC/inference.py#L13) | Nạp model đã xuất và dự đoán tám lớp mà không cần Internet. |
| [`code/TFC/smoke_test_dinov3.py`](code/TFC/smoke_test_dinov3.py#L18) | Kiểm tra forward/backward, hai backbone độc lập và gradient adapter. |
| [`code/config_files/InGesture_Configs.py`](code/config_files/InGesture_Configs.py) | Cấu hình đầu vào ba kênh, chiều dài 206 và tám lớp. |
| [`streamlit_app.py`](streamlit_app.py) | Giao diện tải CSV/TXT, xem tín hiệu và nhận kết quả dự đoán. |

Các thành phần TF-C được giữ lại gồm augmentation, `NTXentLoss_poly`,
time-frequency consistency loss, hai projector 128 chiều và downstream
classifier.

## 4. Cập nhật Loss Function

### Loss cũ và loss mới

| Giai đoạn | Trước cập nhật | Sau cập nhật |
|---|---|---|
| Pretraining | TF-C contrastive loss dùng `NTXentLoss_poly` | Tiếp tục dùng loss này cho time, frequency và liên kết hai miền. |
| Fine-tuning phân loại | `CrossEntropyLoss` và weighted sampling | Mặc định `ClassBalancedFocalLoss` và shuffle; vẫn có tùy chọn chạy baseline cũ. |

Thay đổi loss nằm ở giai đoạn fine-tuning tám lớp. Focal loss giảm đóng góp
của mẫu đã dự đoán dễ, giúp mô hình tập trung vào các mẫu khó. Class balancing
điều chỉnh đóng góp giữa các lớp dựa trên số mẫu train.

### Công thức và cách triển khai

`code/TFC/loss.py` bổ sung `ClassBalancedFocalLoss`. Trọng số lớp được tính
từ số mẫu train theo effective number: `w_c ∝ (1-beta)/(1-beta^n_c)`;
loss mỗi mẫu là `-w_y * (1-p_y)^gamma * log(p_y)`. Mặc định `beta=0.999`,
`gamma=2`. Cách này giảm đóng góp của mẫu dễ và tăng trọng số lớp ít mẫu.
Loss tương phản TF-C vẫn được dùng trong pretraining.

Trong đó `n_c` là số mẫu train của lớp `c`, `p_y` là xác suất softmax của
nhãn đúng. Các trọng số `w_c` được chuẩn hóa để có trung bình bằng 1;
loss cuối cùng là trung bình trên batch. Code dùng `log_softmax` để tính
`log(p_y)` ổn định và không dùng số mẫu validation/test để tính trọng số.

```python
criterion = ClassBalancedFocalLoss(class_counts, beta=0.999, gamma=2.0)
loss = criterion(logits, labels)
```

`train_gestures.py` mặc định dùng `--classification-loss cb_focal`.
Có thể chọn `--classification-loss cross_entropy` để chạy baseline cũ.
Khi dùng CB Focal, sampler không oversample để tránh bù mất cân bằng hai lần.
Tên loss, tham số và số mẫu từng lớp được lưu trong `training_info.json`.
Model bundle hiện tại đã được huấn luyện lại với CB Focal; tên loss và tham số
được lưu ngay trong bundle. Báo cáo: `reports/cb_focal/`.

Tham khảo: [Class-Balanced Loss](https://arxiv.org/abs/1901.05555),
[Focal Loss](https://arxiv.org/abs/1708.02002).

## 5. Cách sử dụng DINOv3

### Nghiên cứu loss và joint fine-tuning

DINOv3 là backbone; loss DINO/iBOT/Gram anchoring của Meta không tự động là
loss phù hợp cho phân loại cử chỉ. Phân tích nguồn và giới hạn thực nghiệm ở
[`reports/loss_research/RESEARCH.md`](reports/loss_research/RESEARCH.md).

Code bổ sung `--classification-loss cb_ce`, `--sampler shuffle` và
`--finetune-tfc-weight 0.1` để so sánh joint objective
`L_classification + lambda * [0.2*(L_time+L_frequency)+L_cross]`.
Lambda mặc định 0 giữ cấu hình của model đã xuất. Chọn lambda/gamma/beta bằng
validation; chưa có bằng chứng cho một cấu hình tối ưu tuyệt đối.

`tools/compare_loss_configs.py` chạy ablation cùng sampler/seed/tập chia;
`--validation-only` không đọc test, không xuất bundle deploy. Xem lệnh và
quy trình ba seed trong báo cáo nghiên cứu.

### Loss bổ sung sau khi đối chiếu bài TF-C + DINOv3 tham khảo

`code/TFC/loss.py` bổ sung `ProjectedMultiViewTFCLoss`. Chọn bằng
`--contrastive-objective projected_multiview` trong `train_gestures.py`:

```text
Lt = NT-Xent(zt, zt_aug)
Lf = NT-Xent(zf, zf_aug)
Lcross = mean[NT-Xent(zt,zf), NT-Xent(zt,zf_aug),
              NT-Xent(zt_aug,zf), NT-Xent(zt_aug,zf_aug)]
Lpretrain = 0.2*(Lt + Lf) + Lcross
Lfinetune = Lclassification + lambda*Lpretrain
```

Cả bốn tensor projector đều nhận gradient. Đây là loss TF-C có mở rộng
multi-view, không phải loss teacher–student dùng để Meta pretrain DINOv3.
Mặc định `legacy` vẫn khớp model hiện có. Tùy chọn mới đã kiểm tra loss,
positive pairing và backward; chưa có kết quả chất lượng để thay model deploy.
Tham khảo [repo TF-C-DINOv3](https://github.com/vuvandat109/TF-C-DINOv3);
đối chiếu chi tiết ở [`reports/reference_review/NEW_REFERENCE.md`](reports/reference_review/NEW_REFERENCE.md).

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

## 6. Cài đặt và kiểm tra

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

## 7. Huấn luyện và xuất model

### Chạy baseline không cần pretrained

```powershell
python train_offline.py --classification-loss cb_focal --output-dir artifacts/gesture_model_v2
```

Để chạy lại baseline Cross Entropy với cùng cấu hình entrypoint:

```powershell
python train_offline.py --classification-loss cross_entropy --output-dir artifacts/gesture_model_ce_v2
```

Với `train_gestures.py`, có thể điều chỉnh `--cb-beta 0.999` và
`--focal-gamma 2.0`. Không thay các tham số này dựa trên kết quả test;
dùng validation để chọn cấu hình.

### Chạy với checkpoint DINOv3 pretrained trên máy

```powershell
python code/TFC/train_gestures.py `
  --data-dir datasets/InGestureOffline `
  --local-model-dir weights/dinov3-vits16 `
  --output-dir artifacts/gesture_model_pretrained `
  --image-size 224 `
  --batch-size 4 `
  --classification-loss cb_focal `
  --pretrain-epochs 10 `
  --finetune-epochs 20
```

Mỗi lần huấn luyện xuất `model_bundle.pt`, `metrics.json`,
`training_info.json` và `history.json`. Nên dùng thư mục output mới để không
ghi đè model cũ.

## 8. Dữ liệu

### Kaggle Dataset và Visualization Notebook

- Dataset: [Gesture Dataset](https://www.kaggle.com/datasets/nguyntinmnh41/gesture-dataset).
- Notebook đã chạy trên dữ liệu thật: [`kaggle/ingesture_visualization.ipynb`](kaggle/ingesture_visualization.ipynb).
- Mười bốn biểu đồ và kiểm kê: [`reports/data_visualization/`](reports/data_visualization/).

Trên Kaggle: mở dataset → **New Notebook** → **File / Import Notebook** và chọn
file `.ipynb` trên. Đảm bảo mục **Input** có `nguyntinmnh41/gesture-dataset`;
chọn **Run All**, sau đó **Save Version / Save & Run All**. Chỉ đặt Public khi
đã kiểm tra toàn bộ outputs. Nộp hai link: Dataset và Notebook Visualization.
Notebook trong repo đã có outputs chạy local; link public Notebook chỉ có sau
khi tạo và lưu thành công trên Kaggle, không phải link Dataset hiện tại.

Notebook học bố cục của [bài Kaggle tham khảo](https://www.kaggle.com/code/loanhviet/sentinel-2-rpg-dataset-visualization):
giới thiệu → inventory/format/pool → mẫu tín hiệu và nhãn → temporal/multi-subject
→ coverage/phân bố lớp/thời lượng → mean ± SD theo cử chỉ → kết luận.
Có thêm chuẩn hóa/tương quan và ảnh adapter DINOv3. Mẫu minh họa dùng seed 42,
tối đa 12 subject train mỗi lớp, tên file lưu trong `sample_pool.csv`.
Các thống kê toàn dataset dùng đủ inventory, không dùng số lượng sample pool.
Đối chiếu thêm [Sensor Logger Visualization](https://www.kaggle.com/code/vuvandat000099/sensor-logger-five-gesture-dataset-visualization),
notebook bổ sung PCA của 12 đặc trưng thống kê, RMS theo kênh, STFT và audit
recording trùng/đứng yên. PCA dùng pool train, không thể diễn giải thành accuracy;
audit trùng ở mức recording, chưa phải cửa sổ hay mẫu gần giống.
Kết quả audit bản local: 85 recording trùng sau lần đầu (84 train, 1 test),
không có nhãn mâu thuẫn và không có hash recording chung giữa các split.
Các số đếm và metric hiện tại vẫn dùng bản dữ liệu chưa deduplicate;
thay đổi tập dữ liệu cần huấn luyện/đánh giá lại.
Không cần GPU, model weights hoặc Hugging Face. Chạy EDA local:

```powershell
python -m pip install -r requirements-analysis.txt
python tools/visualize_ingesture.py
python tools/build_visualization_notebook.py
```

Notebook tự chứa code và metadata về split/normalization. Nếu cấu trúc dataset
khác bản InGesture v5 đã dùng, chương trình dừng khi số cửa sổ không khớp để
tránh xuất biểu đồ sai. `kernel-metadata.json` gắn đúng dataset, mặc định private.

Dự án hiện dùng **InGesture v5** làm dữ liệu thực nghiệm chính. Mỗi mẫu gồm
chuỗi gia tốc X/Y/Z và một trong tám nhãn cử chỉ. Dữ liệu được làm sạch, chuẩn
hóa và cắt thành cửa sổ 206 mẫu trước khi huấn luyện.

- Nguồn: [InGesture v5](https://doi.org/10.17632/fdxst56tcj.5)
- Dữ liệu gốc: `data/external/InGesture/`
- Tensor sau xử lý: `datasets/InGestureOffline/`
- Script chuẩn bị: [`tools/prepare_ingesture_data.py`](tools/prepare_ingesture_data.py)

Dữ liệu dung lượng lớn không được đưa vào GitHub; repository lưu code xử lý,
model đã xuất và file CSV mẫu.

## 9. Chạy ứng dụng dự đoán

```powershell
python -m pip install -r requirements-web.txt
streamlit run streamlit_app.py
```

Mở `http://localhost:8501`, tải CSV/TXT có các cột `timestamp, x, y, z`, sau đó
chọn **Run Prediction**. Ứng dụng hiển thị tín hiệu miền thời gian, phổ tần số,
nhãn dự đoán, độ tin cậy và xác suất của tám lớp.

Ô **Chọn dữ liệu mẫu** có thêm tám CSV cử chỉ thật trong `sample_data/examples/`.
Nguồn, split và nhãn gốc nằm trong `sample_data/example_manifest.csv`. Timestamp
được sinh theo tần số danh định; X/Y/Z lấy từ recording gốc. Các file này là mẫu
train để kiểm tra luồng ứng dụng; không lấy tỷ lệ đoán đúng trên chúng làm
test accuracy. Có thể tạo lại bằng `python tools/export_demo_samples.py`.

Bản triển khai công khai:
[https://tfc-dinov3-gesture.streamlit.app/](https://tfc-dinov3-gesture.streamlit.app/)

Link cloud hoạt động độc lập với máy chạy VS Code. Để mọi người truy cập,
đặt quyền ứng dụng là Public trong Streamlit Community Cloud. Bản miễn phí
có thể ngủ sau 12 giờ không có traffic; khách có thể bấm đánh thức ứng dụng.
Không có cam kết uptime tuyệt đối trên gói miễn phí.

Ứng dụng giới hạn upload 5 MB/100.000 mẫu, dùng một model cache chung,
xử lý tuần tự và tối đa 64 cửa sổ trải đều mỗi file, batch inference 4.
Các giới hạn này giúp kiểm soát RAM khi nhiều người dùng; xác suất trả về
với file dài là trung bình trên các cửa sổ được chọn.

## 10. Docker và CI/CD

Image production chạy giao diện Streamlit cùng model đã xuất ở cổng `8501`:

```powershell
docker build -t tfc-dinov3-gesture:local .
docker run --rm -p 8501:8501 tfc-dinov3-gesture:local
```

Mở `http://localhost:8501`. Container có health check tại
`/_stcore/health`, chạy bằng user không có quyền root và chỉ chứa code inference,
file mẫu cùng model bundle. Dataset huấn luyện không được đóng gói vào image.

Pipeline [`.github/workflows/ci.yml`](.github/workflows/ci.yml) tự động chạy khi
push hoặc tạo pull request vào `main`: cài dependencies, compile source, chạy
unit test và xác nhận Docker image build thành công.

Pipeline [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml)
build và public image `linux/amd64` lên Docker Hub khi chạy thủ công hoặc push
tag `v*`. Repository GitHub cần hai Actions secrets:

- `DOCKERHUB_USERNAME`: tên tài khoản Docker Hub.
- `DOCKERHUB_TOKEN`: access token Docker Hub có quyền Read & Write.

Image được gắn ba dạng tag: `latest`, tag phiên bản như `v1.0.0` và
`sha-<commit>`. Ví dụ sau khi public:

```powershell
docker pull tiemanh05/tfc-dinov3-gesture:latest
docker run --rm -p 8501:8501 tiemanh05/tfc-dinov3-gesture:latest
```

Image public: [Docker Hub — tiemanh05/tfc-dinov3-gesture](https://hub.docker.com/r/tiemanh05/tfc-dinov3-gesture).

Model 166 MB được quản lý bằng Git LFS; cả hai workflow đều bật tải LFS trước
khi kiểm thử hoặc build để tránh đóng gói nhầm file LFS pointer.

## 11. Trạng thái model hiện tại

Model hiện có là baseline kỹ thuật để xác nhận toàn bộ pipeline chạy thật từ
file đầu vào đến kết quả dự đoán. Model dùng hai DINOv3 ViT-S/16 khởi tạo ngẫu
nhiên, huấn luyện lại với CB Focal Loss. Kết quả trên 15.658 cửa sổ test:

| Chỉ số | Cross Entropy cũ | CB Focal hiện tại |
|---|---:|---:|
| Accuracy | 39,79% | 50,72% |
| Macro F1 | 18,57% | 20,12% |
| Balanced accuracy | 31,27% | 29,61% |

Đây là một lần thử seed 42 với 256 cửa sổ train mỗi lớp. Phân bố train thử
nghiệm này đã cân bằng, nên trọng số CB bằng nhau; thành phần focal vẫn hoạt
động. Đồng thời sampler được đổi từ weighted sampling sang shuffle, vì vậy
chênh lệch trên không cô lập tác động của riêng loss. Balanced accuracy giảm,
cho thấy hiệu quả chưa đồng đều giữa các lớp. Checkpoint được chọn bằng macro
F1 validation tại epoch 11; test được đánh giá sau khi chọn checkpoint.

Các chỉ số này không chứng minh pretrained DINOv3 cải thiện TF-C. Muốn đánh giá
lợi ích của pretrained cần nạp checkpoint hợp lệ, huấn luyện lại với cùng cách
chia dữ liệu và so sánh trên cùng test set.

## 12. Tài liệu tham khảo

- Xiang Zhang và cộng sự, [*Self-Supervised Contrastive Pre-Training for Time
  Series via Time-Frequency Consistency*](https://openreview.net/forum?id=OJ4mMfGKLN),
  NeurIPS 2022.
- [Mã nguồn TF-C của nhóm tác giả](https://github.com/mims-harvard/TFC-pretraining).
- [Repository DINOv3 chính thức của Meta](https://github.com/facebookresearch/dinov3).
- [DINOv3 ViT-S/16 checkpoint](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

*Đây là bản phát triển phục vụ yêu cầu thay encoder/backbone trong TF-C, không
phải bản phát hành chính thức của nhóm tác giả TF-C hoặc Meta AI.*
