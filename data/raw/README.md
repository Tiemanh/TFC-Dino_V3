# Dữ liệu cảm biến thô

Đặt các file CSV thu từ smartphone vào thư mục này. Mỗi dòng là một mẫu gia
tốc và nên có các cột sau:

```csv
timestamp,ax,ay,az,label,recording_id
0.00,0.12,-0.04,9.71,gesture_1,g1_take_001
0.02,0.15,-0.03,9.69,gesture_1,g1_take_001
```

- `ax`, `ay`, `az`: gia tốc ba trục. Có thể dùng `x/y/z` hoặc
  `accel_x/accel_y/accel_z`.
- `label`: tên một trong đúng 8 cử chỉ.
- `recording_id`: mã duy nhất của một lần thực hiện cử chỉ.
- `timestamp`: nên có để kiểm tra dữ liệu, nhưng công cụ không bắt buộc.

Nếu mỗi file chỉ chứa một lần thực hiện, có thể bỏ `recording_id`; đường dẫn file
sẽ được dùng làm mã. Nếu file nằm trong thư mục mang tên cử chỉ, có thể bỏ
`label`; tên thư mục cha sẽ được dùng làm nhãn.

Không đưa dữ liệu mẫu tổng hợp vào bộ dữ liệu nộp chính thức.
