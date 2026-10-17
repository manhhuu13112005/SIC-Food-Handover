# Xử lý hàng loạt dataset trạng thái EfficientNet V4

Công cụ: `scripts/bulk_state_dataset.py`. Cấu hình: `config/state_sources_v4.yaml`.
Xử lý một đợt cho lettuce, broccoli, onion, eggplant, pork, shrimp. Không tải dữ liệu,
không huấn luyện, không gộp vào V3, không sửa ZIP/dataset gốc hay checkpoint.

## 1. Chuẩn bị ZIP trên Windows

Đặt các ZIP tải thủ công vào đúng thư mục (không cần tự giải nén):

```text
C:\Users\ASUS\SIC-Food\datasets\raw\state_sources\extension_v4\archives\
├── lettuce_clean.zip
├── smartveggies.zip
├── onion.zip
├── pork.zip
└── shrimp.zip
```

`smartveggies.zip` là nguồn ứng viên cho cả eggplant và broccoli. ZIP khác tên
sẽ được ghi `unconfigured_archive`, không tự nhận nhãn. ZIP còn thiếu được ghi
`missing_archive`; chương trình không tải bù và không tạo dữ liệu giả.

**Cập nhật nguồn thực tế:** lettuce dùng `lettuce_clean.zip`; ZIP `lettuce.zip`
gốc giữ nguyên và chỉ tạo thông báo `unconfigured_archive` mức info, không làm
prepare thất bại. SmartVeggies hiện chỉ chứa 1.674 ảnh trong train; YAML khai báo
val/test không có trong ZIP. Cấu hình `optional_missing_splits: [val, test]`
ghi `split_absent_in_archive` mức warning, không tạo holdout thay thế và không
cho phép đường dẫn vượt phạm vi ZIP.

Shrimp thực tế là **Classification CSV**: `valid/_classes.csv` cùng 217 ảnh
trực tiếp trong `valid/`. Header gốc là `filename, fresh, not`; khoảng trắng
định dạng CSV được bỏ khi đọc nhưng tên nhãn giữ nguyên chữ thường. 91 ảnh chỉ
có fresh, 126 ảnh bật cả fresh và not. Đây chưa phải bằng chứng đủ để tự xác
nhận trạng thái: `fresh` giữ `needs_review`, `not` bị chặn. Cấu hình Shrimp không
có approved_mappings. Không chọn argmax hoặc suy trạng thái từ filename. Mọi
dòng đa nhãn, không có nhãn, cờ sai hoặc ảnh thiếu annotation đều được báo cáo;
ảnh đa nhãn không được prepare kể cả khi một nhãn riêng được phê duyệt sau này.

Pork giữ tám nhãn Fresh/Spoiled ở needs_approval. `inspect` xuất
`datasets/reports/state_v4/pork_review/README.md` và tám contact sheet, mỗi lớp
tối đa 6 ảnh lấy từ các source split. Mỗi mẫu có ảnh toàn cảnh với box đỏ và
crop bên cạnh; CSV đi kèm giữ tên file, SHA, box và thứ tự mẫu. Contact sheet
không tự phê duyệt nhãn hoặc chứng minh tình trạng hư hỏng thực tế.

Thư viện: Python 3.10+, Pillow, NumPy, PyYAML. Không cần Torch/GPU/Kaggle API.
Nên dùng môi trường sic-food đã có các thư viện. Nếu thiếu, cài trong đúng kernel:

```python
%pip install Pillow numpy PyYAML
```

## 2. Chạy trực tiếp từ Jupyter Notebook

Tạo cell thiết lập sau. Notebook không cần nằm trong thư mục project. Dùng
`sys.executable` để tránh gọi nhầm Python của terminal và không dùng `shell=True`.

```python
import sys
import subprocess
from pathlib import Path

PROJECT = Path(r"C:\Users\ASUS\SIC-Food")
SCRIPT = PROJECT / "scripts" / "bulk_state_dataset.py"
PYTHON = sys.executable
print("Python:", PYTHON)
assert SCRIPT.is_file()
subprocess.run([PYTHON, "-c", "import PIL, numpy, yaml; print('Dependencies OK')"],
               cwd=PROJECT, check=True)

def run_state(command):
    subprocess.run([PYTHON, str(SCRIPT), command], cwd=PROJECT, check=True)
```

**Cell 1 — xem nhãn gốc, chưa xuất ảnh ứng viên:**

```python
run_state("inspect")
```

Mở `datasets/reports/state_v4/inspect_labels.csv` và `inspect_issues.csv`.
`image_count` đếm ảnh chứa nhãn; `bounding_box_count` đếm box hợp lệ đối với YOLO.
Ảnh lỗi chỉ được xác minh bằng giải mã ở prepare; số đếm inspect chưa khẳng định
ảnh hợp lệ. `data.yaml` có nhãn nhưng chưa có box cũng được liệt kê với số 0.
Các lỗi annotation có số dòng; segmentation TXT (không phải 5 trường) bị từ chối.

**Duyệt mapping trước khi chuẩn bị:** sửa `approved_mappings` trong YAML bằng
đúng tên nhãn từ CSV, phân biệt hoa/thường, dấu cách, dấu gạch. Các mapping người
dùng đã nêu nằm sẵn trong cấu hình và chỉ có hiệu lực khi tên nhãn khớp chính xác.
Không fuzzy-match và không suy trạng thái từ tên nguyên liệu đơn thuần.

Pork chưa có tên nhãn gốc được xác minh, nên mặc định `approved_mappings: {}`.
Ví dụ **chỉ sau khi inspect thực sự thấy** `Fresh Kasim` và `Spoiled Kasim`:

```yaml
pork:
  archive: pork.zip
  format: auto
  classification_label_mode: relative
  approved_mappings:
    Fresh Kasim: pork__normal_appearance
    Spoiled Kasim: pork__visible_defect
```

Tương tự Fresh/Spoiled cho Liempo, Pork chop, Tenderloin. Nếu nguồn dùng cây thư
mục `train/Kasim/Fresh/image.png`, nhãn inspect là `Kasim/Fresh`; dùng nguyên
chuỗi đó làm khóa mapping. Gợi ý pork không đồng nghĩa đã phê duyệt.

`Larvae`, `Healthyd`, `Broccoli`, `generic Broccoli`, `Not Pork Meat`, `Not`
luôn bị giữ ngoài kể cả vô tình thêm vào approved_mappings. `Rotten broccoli`
chưa được duyệt, nên không có mapping mặc định. Không đổi shrimp `Not` thành hỏng.
Không sửa config để tạo đủ số lớp; lớp thiếu luôn được báo cáo.

**Cell 2 — xuất ứng viên và tự động audit:**

```python
run_state("prepare")
```

Chỉ nhãn đã duyệt được xuất. YOLO crop từng box hợp lệ vào lớp đích; box nhỏ
hơn 16 pixel hoặc annotation không hợp lệ bị loại và ghi lý do. Box chạm biên
được cắt về phạm vi ảnh. Tọa độ YOLO được áp vào lưới pixel gốc, không xoay EXIF
trước crop. Classification không crop, giữ nguyên độ phân giải, chuyển RGB và
hiệu chỉnh EXIF khi có; ảnh đầu ra chuẩn hóa PNG.

`prepare` tự chạy audit, có thể tốn thời gian ở lần đầu do so sánh với toàn bộ
V3 và cabbage candidates. Cache pHash tham chiếu được dùng lại theo size/mtime;
nguồn tham chiếu chỉ đọc. `prepare` không tự chuyển ảnh vào train/val/test.

**Cell 3 — chạy lại audit sau khi cần kiểm tra:**

```python
run_state("audit")
```

Nếu ZIP hoặc YAML thay đổi, phải chạy lại prepare. Nếu sửa ảnh ứng viên thì audit
ghi ảnh thay đổi/lỗi để giữ ngoài split. Không tự sửa SHA hoặc tin tên file.

**Cell 4 — tạo báo cáo tổng hợp:**

```python
run_state("summary")
```

Xem `datasets/reports/state_v4/summary.md`, `summary.json`, `class_summary.csv`.
`audit_current` kiểm tra config/manifest, size/mtime ZIP và cây ảnh ứng viên/tham
chiếu so với lần audit. Đây là kiểm tra độ mới, không thay thế audit lại nội dung.

## 3. Lệnh tương đương trong terminal

Chạy từ `C:\Users\ASUS\SIC-Food`, với Python đúng môi trường:

```powershell
python scripts/bulk_state_dataset.py inspect
python scripts/bulk_state_dataset.py prepare
python scripts/bulk_state_dataset.py audit
python scripts/bulk_state_dataset.py summary
```

Có thể dùng `--config C:\path\custom.yaml` sau tên lệnh; đường dẫn dữ liệu trong
YAML luôn tương đối với project, không phải thư mục notebook. Work/output/reports
phải riêng biệt và không chồng lên raw, V3, cabbage candidates hoặc models.

## 4. Ảnh ứng viên, manifest và kiểm duyệt

```text
datasets/
├── state_v4_work/                 # Giải nén/cache/lock riêng của công cụ
│   └── extracted/<archive_sha256>/
├── state_v4_candidates/
│   ├── lettuce__normal_appearance/<sha256>.png
│   ├── lettuce__visible_defect/<sha256>.png
│   └── ...                       # Chỉ lớp thực sự có ảnh hợp lệ
└── reports/state_v4/
```

Các cột manifest bắt buộc đều có: source_dataset, source_image, source_split,
original_label, target_class, bounding_box, image_group, crop_path, sha256, phash.
Có thêm source_sha256, group_method, record_id để truy nguồn và nối các crop.
`source_image` là đường dẫn tương đối bên trong ZIP; SHA ZIP nằm trong source_status.
`bounding_box` lưu cả YOLO xywh chuẩn hóa và pixel xyxy đã crop (exclusive cạnh phải/dưới).
`crop_path` tương đối project; classification có bounding_box rỗng.

| Báo cáo | Nội dung cần kiểm tra |
| --- | --- |
| inspect_labels.csv | Toàn bộ nhãn khai báo/phát hiện, số ảnh/box, mapping đề xuất và đã duyệt |
| source_folders.csv / source_metadata.json | Cấu trúc thư mục, số ảnh, README/data.yaml và header CSV nguyên gốc |
| classification_annotations.csv | Từng nhãn đang bật trên ảnh CSV, vị trí annotation và lý do giữ lại để duyệt |
| inspect_class_summary.csv / inspect_summary.json | Số nguồn đọc được, nhóm nguồn tạm thời, số mẫu được duyệt và lớp còn thiếu |
| pork_review/README.md | Contact sheet từng lớp Pork và CSV truy nguồn ảnh mẫu, luôn needs_approval |
| source_status.json | ZIP thiếu/lỗi, SHA ZIP, trạng thái xử lý và số mẫu được giữ/loại |
| inspect_issues.csv | YAML, annotation, đường dẫn hoặc nguồn lỗi |
| manifest.csv | Tất cả bản ghi ứng viên của lần prepare hiện tại |
| prepare_review.csv | Nhãn chặn/chưa duyệt và ảnh/crop lỗi |
| audit_pairs.csv | Liên kết trùng SHA, pHash, gần trùng; mâu thuẫn nhãn và giao với nguồn tham chiếu |
| duplicate_groups.csv | Thành viên đầy đủ của nhóm SHA/pHash trùng |
| audit_issues.csv | Ảnh hỏng, bị sửa hoặc lỗi ảnh tham chiếu |
| split_plan.csv | Ảnh duy nhất, audit_group, split đề xuất, lý do review |
| unreferenced_candidates.csv | Ảnh cũ không còn thuộc manifest hiện tại; không được dùng huấn luyện |
| audit_state.json | Trạng thái audit, độ đầy đủ tham chiếu, cờ giới hạn báo cáo cặp |
| class_summary.csv / summary.md / summary.json | Số ảnh theo 6 nguyên liệu × 2 trạng thái và lớp còn thiếu |

Ảnh trùng SHA trong cùng lớp chỉ có một file đầu ra; manifest vẫn giữ mọi bản
ghi nguồn gốc. Split plan chỉ giữ một ảnh duy nhất cho mỗi lớp+SHA. Nhãn mâu
thuẫn hoặc giao/gần trùng tham chiếu được giữ ở review. pHash giống/gần giống là
tín hiệu cần kiểm duyệt, có thể false positive; không tự suy nhãn hoặc xóa ảnh.
Báo cáo cặp có giới hạn cấu hình; cờ `pair_report_truncated` báo rõ khi bị giới
hạn, nhưng việc liên kết nhóm vẫn xử lý mọi hash match.

## 5. Nhóm nguồn và split: không tạo Test giả

Giữ cùng nhóm: mọi crop cùng ảnh; SHA nguồn giống nhau; file video có tên
`clip_mp4-001...` / `clip_frame_001...`; ảnh augmentation có prefix rotate/flip/
noise và hậu tố Roboflow; ảnh screenshot cùng ngày; các ảnh trùng/gần trùng.
Khi không biết provenance, mặc định coi toàn bộ acquisition/export của một nguồn
là một nhóm (thận trọng hơn chia ngẫu nhiên ảnh). Không chứng minh độc lập chỉ
bằng filename hoặc source split do dataset bên ngoài cung cấp.

`group_patterns` cho phép chỉ định capture session/video ID đã được xác minh.
Ví dụ nếu thư mục ghi rõ mã mẫu và tất cả ảnh/frame của một mẫu nằm cùng đó:

```yaml
group_patterns:
  - 'samples/(sample_[0-9]+)/'
```

Capture group 1 phải giữ đúng **nguồn gốc độc lập**, không dùng số frame hay
augmentation ID. Khi một ảnh có nhiều nhãn, mọi crop vẫn được nối chung group.
Nhóm có mẫu review khiến toàn nhóm review; không để phần còn lại lọt vào holdout.

Tỷ lệ đề xuất 70/15/15. Mỗi lớp cần tối thiểu 10 nhóm (cấu hình), tối thiểu 2
nhóm ở mỗi holdout. Nếu thiếu nhóm hoặc không tìm được phân bổ hợp lệ, giữ toàn
bộ family lớp liên kết ở `train_candidate`. Không tạo ảnh Test hay lớp âm giả.

`train`, `val`, `test` trong split_plan chỉ là **đề xuất**; công cụ này không có
lệnh merge. Người dùng cần duyệt báo cáo, nguồn/license/nhãn/provenance và overlap
với V3 trước khi xuất dataset hợp nhất bằng một bước riêng. Không huấn luyện trực
tiếp bằng ImageFolder trên thư mục candidates: nó có thể chứa ảnh review và asset
cũ không còn trong manifest. Giữ nguyên mọi split V3 hiện có.

## 6. Tái chạy, an toàn và kiểm thử

ZIP được kiểm tra toàn bộ trước ghi: chặn traversal/đường dẫn Windows nguy hiểm,
symlink, trùng tên khác hoa thường, file/directory collision, ZIP mã hóa và giới
hạn số file/kích thước/tỷ lệ nén. Cùng SHA ZIP dùng chung cache giải nén, không
giải nén lại. Không có chức năng download. Cache lỗi/dở dang bị từ chối thay vì
tự ghi đè; chỉ kiểm tra/xử lý thư mục work do công cụ tạo nếu cần phục hồi.

Manifest/báo cáo là snapshot được cập nhật nguyên tử khi chạy lại; file ảnh có
tên SHA, không ghi đè file khác nội dung. Ảnh cũ không tự xóa, được liệt kê riêng.
Lock ngăn hai lệnh chạy đè nhau. Nếu notebook bị ngắt và còn `.bulk_state.lock`,
chỉ bỏ file lock sau khi xác nhận không còn tiến trình xử lý. Không xóa nguồn.

Chạy test nhỏ và syntax từ notebook:

```python
subprocess.run([PYTHON, "-B", "-m", "unittest", "discover", "-s", "tests",
                "-p", "test_bulk_state_dataset.py"], cwd=PROJECT, check=True)
subprocess.run([PYTHON, "-c",
    "import ast,pathlib; ast.parse(pathlib.Path('scripts/bulk_state_dataset.py').read_text(encoding='utf-8')); print('Syntax OK')"],
    cwd=PROJECT, check=True)
```

Các test dùng ảnh/ZIP tổng hợp trong thư mục tạm riêng thuộc tests; không chạm
ảnh sản phẩm. Không cần chạy notebook huấn luyện để dùng công cụ.
