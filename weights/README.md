# Trọng số DINOv3 trên máy

Đặt model ở `weights/dinov3-vits16/`, gồm:

```text
weights/dinov3-vits16/
  config.json
  model.safetensors
```

Cũng hỗ trợ `pytorch_model.bin` hoặc các shard cùng tệp index tương ứng.
Đây phải là trọng số DINOv3 ViT hợp lệ ở định dạng Transformers, do nguồn
cung cấp xuất hoặc được chuyển đổi đầy đủ. `config.json` phải có
`"model_type": "dinov3_vit"` và khớp trọng số.

File `.pth` gốc từ Meta dùng tên tensor khác; không đổi đuôi hoặc đổi tên
thành `model.safetensors`. Cần chuyển đổi đúng kiến trúc trước khi dùng.
Chưa có file trọng số pretrained nào được cung cấp trong thư mục này.

Chạy từ thư mục gốc TFC-DivV3 trong Terminal VS Code, với môi trường đã cài
requirements-dinov3.txt và PyTorch phù hợp:

```powershell
python code/TFC/train_gestures.py --data-dir datasets/InGesture --local-model-dir weights/dinov3-vits16 --output-dir artifacts/gesture_model --pretrain-epochs 1 --finetune-epochs 1 --batch-size 2
```

Một epoch mỗi pha là lượt thử ban đầu, chưa đảm bảo chất lượng nhận dạng.
Mặc định chương trình chọn CUDA nếu PyTorch nhận GPU; nếu không sẽ chạy CPU.
Hai encoder riêng biệt cùng nạp trọng số từ thư mục này. Không đăng nhập
Hugging Face, không tải trọng số qua mạng khi dùng `--local-model-dir`.
Chương trình dừng nếu thiếu file hoặc tensor không khớp, không tự dùng
trọng số ngẫu nhiên. Tên file không chứng minh nguồn gốc pretrained;
cần dùng nguồn và giấy phép hợp lệ.

Sau khi train xong: model_bundle.pt và metrics.json nằm trong
artifacts/gesture_model/. Khởi động lại web để nạp bundle mới.
Bundle chứa đầy đủ hai backbone nên web không cần thư mục weights khi chạy.
