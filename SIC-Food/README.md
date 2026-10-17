# SIC-Food — Bản bàn giao chạy dự đoán

Bản bàn giao ngày 17/10/2026. Giải nén toàn bộ ZIP trước khi chạy. Không cần dataset để mở website hoặc dự đoán; không cần huấn luyện lại.

## Cài đặt trên máy mới

Cài Python **3.11 64-bit**. Mở terminal tại thư mục có `main.py` sau khi giải nén. Trên Windows (CMD hoặc PowerShell):

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-handover.txt
.venv\Scripts\python.exe -B check_handover.py
.venv\Scripts\python.exe -m streamlit run main.py
```

Mở http://127.0.0.1:8501. Dừng bằng Ctrl+C. Không cần kích hoạt venv. Nếu dùng Conda, tạo môi trường Python 3.11 rồi dùng `python` thay `.venv\Scripts\python.exe` trong các lệnh trên. Trên Linux/macOS dùng `python3.11 -m venv .venv`, sau đó `.venv/bin/python`.

`requirements-handover.txt` cố định các gói trực tiếp theo môi trường đã kiểm thử trên máy bàn giao. `requirements.txt` là danh sách linh hoạt của project gốc. Cần Internet khi tải/cài thư viện lần đầu; hai checkpoint đã nằm trong ZIP, ứng dụng dự đoán không cần tải weights hoặc dataset. Môi trường mới vẫn cần kiểm tra bằng `check_handover.py`; các phụ thuộc gián tiếp không được khóa toàn bộ.

## Các file bắt buộc phải giữ

- `best.pt`: YOLO Detection, 214 nhãn.
- `models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth`: EfficientNet-B0, 37 lớp và metadata/class_names.
- Toàn bộ mã `.py` ở gốc, `config/*.json`, `.streamlit/config.toml` và các thư viện đã cài.
- `images/`: giữ ảnh/video mẫu để người nhận tự kiểm tra.

Giữ đúng cấu trúc thư mục. Nếu đã đặt biến môi trường `SIC_MODEL_PATH` hoặc `SIC_CONDITION_MODEL_PATH` trên máy nhận, bỏ giá trị cũ hoặc trỏ đến checkpoint trong bản bàn giao. Script kiểm tra dùng đường dẫn tuyệt đối đến checkpoint đóng gói và chặn kết nối mạng trong bước dự đoán.

## Phạm vi đã tinh gọn

Đã loại khỏi bản ZIP toàn bộ `datasets/` (khoảng 70,5 GiB), `feature_cache/`, `training_state.pt`, checkpoint cũ/thử nghiệm, backup, bytecode và notebook checkpoint. Các notebook nghiên cứu vẫn giữ mã lệnh, đã xóa output, execution count và metadata của cell để giảm dung lượng. Các checkpoint đang phục vụ dự đoán được giữ nguyên từng byte.

Mã nghiên cứu trong notebook, `scripts/` và `efficientnet_37_fasttrack/` được giữ để tham khảo/phát triển. Chạy lại training, EDA hay đánh giá toàn dataset cần cung cấp dataset/manifest và phụ thuộc nghiên cứu tương ứng; các phần này không phải thao tác khởi động website. Tài liệu lịch sử trong `docs/` có thể nhắc đường dẫn máy cũ hoặc dữ liệu đã loại; xem hướng dẫn này để cài bản bàn giao.

Không xóa dataset hoặc thay đổi mã/mô hình gốc trên máy bàn giao. Đây là bản sao riêng để gửi người nhận.

## Kiểm tra bàn giao

```bat
.venv\Scripts\python.exe -B check_handover.py
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py"
```

Script đầu kiểm tra SHA256 các file bàn giao, nạp cả hai checkpoint thật, chạy YOLO với ảnh/video mẫu và EfficientNet với ba ảnh mẫu; không đọc dataset, không đánh giá độ chính xác. `tests/check_website_models.py` trong bản này cũng gọi kiểm tra không dataset đó. Unit/AppTest kiểm tra logic và giao diện với fixture tổng hợp. Kết quả thực thi trên máy bàn giao ở `HANDOVER_VALIDATION.txt`.

Chức năng website và giới hạn mô hình được mô tả trong `docs/README_website_original.md`. EfficientNet phân loại một nguyên liệu/crop; không định vị nhiều vật thể. Số 97,01% là Validation V3 trên 26/37 lớp, không phải độ chính xác đủ 37 lớp. Trạng thái hình ảnh không chứng nhận an toàn thực phẩm; dữ liệu dinh dưỡng hiện có chưa được kiểm chứng.

## Gửi cho người nhận

Gửi `SIC-Food-Handover(1).zip` kèm mã SHA256 bên cạnh ZIP để đối chiếu sau khi tải. Người nhận giải nén, làm theo phần cài đặt, thử ảnh/video của họ và kiểm tra cả hai trang. Không cần gửi dataset, môi trường Conda/venv hay cache. Khi cần huấn luyện lại, bàn giao dataset riêng theo phạm vi và quyền sử dụng đã thống nhất.
