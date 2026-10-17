# Kiểm tra bulk state dataset V4 — 16/10/2026

## Mã nguồn và kiểm thử

- Syntax Python: kiểm tra AST cho `scripts/bulk_state_dataset.py` và `tests/test_bulk_state_dataset.py`; YAML được nạp bằng bộ kiểm tra cấu hình của công cụ.
- Unit tests của bulk dataset: **24/24 PASS**.
- Toàn bộ project: `python -B -m unittest discover -s tests -p 'test_*.py'`: **41/41 PASS** (8,098 giây ở lần chạy xác nhận).
- Lần chạy toàn bộ đầu tiên có hai lỗi quyền truy cập file tạm của test EfficientNet/OpenCV trong sandbox. Chạy lại cùng bộ test ngoài sandbox thành công; không thay đổi test để bỏ qua lỗi.
- Test mới kiểm tra split YOLO vắng mặt và đường dẫn nguy hiểm, Classification CSV đa nhãn, binary flag/ảnh thiếu/annotation trùng, contact sheet không phê duyệt nhãn, và ZIP không được cấu hình chỉ mang tính thông tin.
- Các test hiện có tiếp tục kiểm tra ZIP traversal/collision/symlink, cache SHA, crop, mapping, SHA/pHash/nhãn mâu thuẫn, overlap reference, nhóm video/augmentation và split không rò rỉ nhóm.

## Inspect trên ZIP thực tế

**5/5 nguồn đọc được; 5/5 không có lỗi cấu trúc chặn xử lý.** Điều này không có nghĩa mọi nhãn đã được duyệt.

| Nguồn | Tổng ảnh | Box / nhãn CSV đang bật |
| --- | ---: | ---: |
| Lettuce clean | 2.170 | 1.456 box |
| SmartVeggies | 1.674 | 2.112 box |
| Onion | 1.616 | 1.616 box |
| Pork | 6.898 | 8.496 box |
| Shrimp | 217 | 343 nhãn CSV, không có box |

- SmartVeggies: ZIP chỉ có train; hai tham chiếu val/test không tồn tại được ghi `split_absent_in_archive` mức warning. Không tắt kiểm tra đường dẫn và không tạo holdout thay thế.
- Lettuce gốc: `lettuce.zip` giữ nguyên, `unconfigured_archive` mức info; nguồn hoạt động là `lettuce_clean.zip`.
- Shrimp: `valid/_classes.csv` có nhãn chính xác `fresh`, `not`; 91 ảnh fresh-only, 126 ảnh có cả hai. `fresh` giữ needs_review; `not` bị chặn. Không suy nhãn từ tên file.
- Pork: tám nhãn Fresh/Spoiled vẫn needs_approval. Đã xuất và xem tám contact sheet, 48 mẫu. Not Pork Meat bị chặn. Cần người dùng xác minh nhãn và provenance, đặc biệt tên file yellow_realistic trong khi README ghi không augmentation.
- Không có nhãn broccoli hỏng được xác minh. Broccoli chung chung, Healthyd, Larvae, Not Pork Meat, Not và Bad tiếp tục bị chặn.

Danh sách đầy đủ từng nhãn và kết quả xem mẫu: [state_v4_source_review.md](state_v4_source_review.md).

## Prepare và audit trên dữ liệu thực tế

- `prepare` hoàn tất với audit bắt buộc; `summary` xác nhận `audit_current=true`.
- 2.701 bản ghi crop từ 2.650 ảnh nguồn; 2.689 file crop duy nhất, không có ảnh Classification được đưa vào ứng viên. Tất cả bản ghi khớp `approved_exact`; Pork/Shrimp có 0 bản ghi.
- Lettuce: 520 normal + 487 defect. Eggplant: 51 normal + 12 defect. Broccoli: 33 normal. Onion: 1.471 normal + 127 defect; 12 bản ghi normal trùng SHA nên còn 1.459 file normal duy nhất.
- Đã audit 23.092 ảnh V3 và 801 ảnh Bắp cải; không có lỗi giải mã ứng viên/tham chiếu được ghi nhận. Pillow có cảnh báo palette transparency khi đọc một số ảnh tham chiếu; giữ nguyên ảnh nguồn.
- Báo cáo 211 liên kết: 12 SHA256, 34 exact pHash, 165 near pHash. Các nhóm liên kết có thể chồng nhau, không cộng thành số ảnh lỗi. Không phát hiện liên kết khác nhãn hoặc overlap tham chiếu ở ngưỡng pHash <= 4; đây không phải chứng minh không còn trùng ngoài ngưỡng.
- Ba nhóm audit thận trọng (Lettuce 1.007 ảnh; SmartVeggies 96; Onion 1.586), **không phải ba mẫu vật đã được xác minh độc lập**. Chưa có metadata đủ để chia thành nhóm thu thập độc lập nhỏ hơn.
- Mỗi nhóm chứa tín hiệu pHash cần duyệt, nên cả 2.689 ảnh ở `review` theo quy tắc giữ chung nhóm. Không suy rằng mọi ảnh đều hỏng/trùng; không tạo Train/Val/Test giả. Không có file ứng viên ngoài manifest hiện tại.
- Còn năm lớp chưa có ứng viên được duyệt: broccoli visible_defect, hai trạng thái Pork, hai trạng thái Shrimp.
- Kiểm tra size/mtime trước và sau trên **23.903 file** thuộc ZIP gốc, V3, Bắp cải và models: không đổi. Snapshot lưu ở `validation_protected_before.json` và `validation_protected_after.json`. Đây là đối chiếu metadata, không tuyên bố băm lại toàn bộ nội dung các file này.

Chi tiết: `datasets/reports/state_v4/audit_state.json`, `audit_pairs.csv`, `class_summary.csv`, `split_plan.csv` và `summary.json`.

Không huấn luyện, không gộp dataset, không sửa nhãn gốc hoặc checkpoint. Hướng dẫn CLI/Jupyter: [bulk_state_dataset_README.md](bulk_state_dataset_README.md).

