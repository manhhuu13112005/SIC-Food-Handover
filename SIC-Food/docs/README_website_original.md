# Food AI · SIC-Food

Website Streamlit gồm hai module **độc lập**:

- **Món ăn & dinh dưỡng:** YOLO Detection từ `best.pt`, 214 nhãn thực tế; nhận diện ảnh/nhiều box hoặc một khung hình video. Người dùng xác nhận món, nhập gram, chọn hồ sơ dinh dưỡng.
- **Trạng thái nguyên liệu:** EfficientNet-B0 từ `models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth`, 37 lớp đọc theo thứ tự trong checkpoint. Phân loại một ảnh nguyên liệu/crop, không định vị nhiều vật thể.

Không chạy training, không tự tải weights và không đọc dataset khi phục vụ dự đoán.

## Chạy trên Windows / Anaconda

Trong Anaconda Prompt:

```bat
conda activate sic-food
cd /d C:\Users\ASUS\SIC-Food
python -m streamlit run main.py --server.address 127.0.0.1 --server.port 8517
```

Mở http://127.0.0.1:8517. Nếu cổng đã có server của phiên kiểm thử, dùng server đó hoặc dừng bằng Ctrl+C trước khi chạy lại; có thể đổi sang cổng 8501.

PowerShell khi chưa kích hoạt Conda:

```powershell
Set-Location -LiteralPath 'C:\Users\ASUS\SIC-Food'
& 'C:\MY_ MANH\APP DOWLOAD\Anaconda\envs\sic-food\python.exe' -m streamlit run main.py --server.address 127.0.0.1 --server.port 8517
```

Chạy từ Jupyter trong đúng kernel sic-food:

```python
import subprocess, sys
from pathlib import Path
project = Path(r"C:\Users\ASUS\SIC-Food")
log = open(project / "streamlit.log", "a", encoding="utf-8")
process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(project / "main.py"),
     "--server.address", "127.0.0.1", "--server.port", "8517"],
    cwd=project, stdout=log, stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NO_WINDOW)
log.close()
print("http://127.0.0.1:8517")
# Khi cần dừng server do cell này tạo:
# process.terminate()
# process.wait(timeout=15)
```

Các notebook nghiên cứu được giữ nguyên. Không chạy notebook training để mở website.

## Sử dụng

### Món ăn & dinh dưỡng

1. Chọn ảnh JPG/JPEG/PNG/WEBP (tối đa 20 MB, 25 triệu pixel), hoặc video MP4/MOV/AVI (64 MB).
2. Bấm nhận diện. Với video chọn chỉ số khung hình từ 0. Video không tracking và không cộng dồn qua các frame.
3. Mỗi bounding box là một mục riêng, kể cả các box cùng nhãn. Kiểm tra/sửa tên món nếu cần.
4. Nhập gram, xác nhận thông tin, chọn hồ sơ dinh dưỡng và chấp nhận dùng dữ liệu chưa kiểm chứng nếu phù hợp.
5. Xem kcal, protein, carbohydrate, fat, biểu đồ và tổng khẩu phần; tải JSON nếu cần.

Giữ công thức `per_100g × grams / 100` cùng kiểm tra hồ sơ khớp nguyên liệu/trạng thái chế biến. Cả 47 hồ sơ hiện tại có trạng thái chế biến unknown và nguồn gốc chưa kiểm chứng; chọn sống/chín có thể không còn hồ sơ phù hợp. Dữ liệu thiếu không điền 0. Gram bằng 0, âm, NaN/Inf hoặc không phải số không được tính. Có thể nhập thủ công khi thiếu YOLO. Có thể quan sát trạng thái thủ công trong phần mở rộng của từng món; YOLO không tự gán độ tươi.

Sửa gram không chạy lại YOLO. Ảnh, kết quả và phần chỉnh sửa giữ trong phiên khi chuyển trang; tải ảnh mới hoặc xóa ảnh sẽ xóa kết quả cũ. Tải lại toàn bộ tab/khởi động lại server có thể tạo phiên mới.

### Trạng thái nguyên liệu

Tải ảnh của **một nguyên liệu** hoặc crop đã cắt; bấm Phân tích trạng thái. Hiển thị tên Việt, trạng thái, softmax và Top-3. Dưới ngưỡng 0,6 thì thông báo chưa đủ căn cứ. Ngưỡng này chưa được hiệu chỉnh cho 37 lớp.

Preprocessing xác minh từ script train: EXIF transpose → RGB → Resize cạnh ngắn 256, Bicubic → CenterCrop 224 → tensor → mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]. Không đổi thành resize vuông. Checkpoint nạp `weights_only=True`, strict state dict, CPU, eval và inference mode. Model được cache theo đường dẫn/mtime/kích thước.

Metadata checkpoint: `model_state_dict`, `class_names`, `num_classes=37`, `epoch=15`, `pretrained_weights=IMAGENET1K_V1`, `training_type=prototype_frozen_backbone`. Classifier shape [37,1280].

**Giới hạn:** 97,01% chỉ là Validation V3 trên 26/37 lớp, không phải độ chính xác 37 lớp. Các lớp mới chưa có đánh giá độc lập đầy đủ. Không có lớp tôm, không có broccoli visible_defect. Softmax cao không đảm bảo đúng hoặc phát hiện được ảnh ngoài tập nhãn. Trạng thái hình ảnh không chứng nhận an toàn để ăn.

## Cấu trúc

```text
main.py                     Trang chủ, điều hướng
page1.py                    YOLO ảnh + biểu mẫu dinh dưỡng
page2.py                    YOLO một khung hình video
condition_page.py           EfficientNet độc lập
model_loader.py             Đường dẫn, cache, tải hai model CPU
condition_service.py        Preprocessing, inference, Top-3
detection_service.py        Detection, bounding box và số thứ tự
nutrition_service.py        Công thức và kiểm tra hồ sơ dinh dưỡng
image_service.py            Đọc ảnh an toàn, RGB/EXIF, giới hạn dung lượng
app_state.py                Lưu input, kết quả, sửa khẩu phần theo phiên
ui_components.py            Giao diện chung, tên tiếng Việt, biểu đồ
pipeline_config.py          Parse nhãn mới __ và tương thích nhãn cũ
.streamlit/config.toml      Theme, giới hạn upload, ẩn lỗi nội bộ
config/                     Dữ liệu dinh dưỡng và cấu hình nghiên cứu hiện có
models/                     Checkpoint hiện có, giữ nguyên
tests/                      Unit, AppTest và kiểm thử model thật
docs/website_validation.md  Báo cáo thực thi và danh sách file
```

Có thể đổi checkpoint qua biến môi trường `SIC_MODEL_PATH` và `SIC_CONDITION_MODEL_PATH`. Đường dẫn tương đối được tính từ project root; không đổi sang checkpoint giả khi model thiếu. EfficientNet trên website yêu cầu 37 lớp, định dạng đã xác minh. Module còn lại vẫn hoạt động nếu một model thiếu/hỏng.

## Môi trường và kiểm thử

Đã dùng Python 3.11.15, Streamlit 1.65.0, Torch 2.14.1, Torchvision 0.29.1, Ultralytics 8.4.172, Pillow 12.3.0, NumPy 2.4.6, OpenCV 5.0.0.93. Không cài/nâng CUDA/PyTorch. Dùng môi trường sic-food hiện có. Khi tạo môi trường mới, dùng requirements.txt và các phiên bản tương thích trên; giao diện hiện sử dụng API Streamlit 1.65.

```bat
python -B -m unittest discover -s tests -p "test_*.py"
python -B tests/check_website_models.py
```

- Unit/AppTest dùng fixture/mock để kiểm tra logic, giữ trạng thái, ảnh lỗi và điều hướng.
- `check_website_models.py` dùng checkpoint thật, ảnh/video mẫu và 52 ảnh Test V3 (2 ảnh/lớp có sẵn); xuất báo cáo vào docs. Không dùng kết quả này để tuyên bố Accuracy toàn bộ 37 lớp.
- Các lệnh cũ `check_streamlit.py`, `check_image_ui.py`, `check_video_ui.py` vẫn gọi bộ kiểm thử giao diện hiện tại.
- Kết quả và ảnh chụp trình duyệt: [website_validation.md](docs/website_validation.md).

## Khôi phục

Project hiện không có Git. Bản sao mã trước chỉnh sửa nằm tại `backups/website_20261017_before/`; giữ cả README cũ. Dừng website trước khi phục hồi các file cùng tên. Các file mới được liệt kê riêng trong báo cáo; không chép đè checkpoint/dataset. `protected_hashes.json` ghi SHA256 các checkpoint, notebook và database được kiểm tra bảo toàn.

Các tài liệu nghiên cứu/dataset cũ trong docs và datasets là lịch sử riêng; không dùng mô tả pipeline YOLO → EfficientNet cũ làm tài liệu website hiện tại.

