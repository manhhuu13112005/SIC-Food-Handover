# Kế hoạch triển khai: YOLO + EfficientNet + Nutrition Engine

Kế hoạch này thay thế phương án một YOLO với nhãn ghép nguyên liệu/trạng thái.

## 1. Hai model AI

1. YOLO: 20 lớp nguyên liệu, bounding box + class_id + confidence trên ảnh/video.
2. EfficientNet-B0: nhận crop từng box, phân loại trạng thái theo nguyên liệu. Checkpoint chứa architecture, image_size, class_names và state_dict. Chuẩn hóa ImageNet, resize 224×224 giống nhau khi train/inference.

Nutrition Engine là tra cứu dữ liệu và nhân theo khối lượng; không phải model thứ ba. Chuẩn bị sống/chín vẫn là thuộc tính người dùng xác nhận. Model trạng thái chỉ mô tả ngoại hình.

Danh sách 20 nguyên liệu trong config/ai_pipeline.json là đề xuất ban đầu vì người dùng chưa cung cấp danh sách cụ thể. Chốt tên, phạm vi mỗi lớp (đặc biệt rice, egg, fish) và chính sách nhãn trước thu thập. Không coi danh sách catalog cũ là bộ lớp YOLO mới.

## 2. Dataset YOLO

Mục tiêu 6.000–10.000 ảnh thật, 20 loại nguyên liệu, nhiều thiết bị/nền/ánh sáng/góc nhìn, vật thể che khuất và ảnh nhiều nguyên liệu. Mỗi vật thể một box; cùng nguyên liệu ở trạng thái khác nhau vẫn cùng lớp YOLO. Có ảnh nền âm tính kèm TXT rỗng, nhưng không dùng chúng thay thế ảnh dương tính. Định dạng YOLO: class_id x_center y_center width height (tọa độ chuẩn hóa 0–1). Bộ train/val/test đều có images và labels, cấu hình data.yaml đã tạo.

## 3. Dataset EfficientNet

Mục tiêu khoảng 8.000 ảnh/crop, mỗi ảnh một nguyên liệu nhìn đủ rõ. Baseline hai lớp fresh/defect mỗi nguyên liệu, tổng 40 lớp, hướng tới khoảng 200 ảnh/lớp trước augmentation nếu cân bằng. Chỉ là mục tiêu thu thập, không bảo đảm chất lượng hay độ chính xác.

Fresh = có vẻ tươi/chưa thấy khuyết tật theo hướng dẫn riêng từng nguyên liệu. Defect = có biểu hiện bất thường. Không gán thịt đã chín là defect chỉ vì khác màu; ghi chế biến trong metadata và thu mẫu phù hợp phạm vi. Với bánh mì/cơm, định nghĩa fresh là ngoại hình bình thường, không kết luận thời gian sản xuất hay độ an toàn.

Khi đủ dữ liệu, thay hoặc bổ sung các lớp cụ thể theo nguyên liệu như tomato_bruised, lettuce_wilted, bread_mold. Cần hướng dẫn xử lý nhiều dấu hiệu đồng thời và có người duyệt. Kiến trúc hiện là single-label: chọn một nhãn theo quy tắc ưu tiên đã duyệt; ảnh mơ hồ để ngoài tập chính. Hai lớp fresh/defect không đủ để phân biệt riêng héo/dập/mốc. Code ánh xạ các nhãn cụ thể khi checkpoint tương ứng được huấn luyện.

## 4. Chống rò rỉ và chất lượng nhãn

- Lưu sample_id, batch_id, source_id, thời điểm, quyền sử dụng, reviewer, preparation_state và parent_image cho crop. Manifest ví dụ chỉ là schema, không phải dữ liệu thật.
- Chia nhóm train/val/test trước crop và augmentation, có thể bắt đầu 70/15/15 theo nhóm. Mọi ảnh cùng mẫu/lô/phiên chụp, mọi frame cùng video và mọi crop từ cùng ảnh gốc thuộc cùng split, thống nhất giữa hai dataset.
- Loại ảnh trùng/gần trùng qua split; không dùng ảnh augmentation để đạt mục tiêu số ảnh gốc. Có người thứ hai duyệt trạng thái và box.
- Không tạo mẫu hỏng trong điều kiện không an toàn. Ảnh không xác định được vi sinh hay chứng nhận thực phẩm an toàn.

## 5. Huấn luyện và đánh giá

scripts/prepare_datasets.py tạo skeleton, scripts/train_yolo.py và scripts/train_efficientnet.py là entrypoint huấn luyện, chưa chạy training thật. Dùng weights pretrained cục bộ, seed 42; lưu phiên bản thư viện, tham số, checksum dataset/model và nguồn pretrained trong hồ sơ đợt chạy. EfficientNet dùng loss cân bằng theo tần suất lớp và chọn best checkpoint bằng validation macro-F1.

- YOLO: precision/recall, mAP50-95, confusion matrix theo 20 lớp.
- EfficientNet: precision/recall/F1 theo từng nguyên liệu-trạng thái, macro-F1, confusion matrix; chú ý bỏ sót mốc/nghi hỏng.
- Pipeline: đánh giá crop YOLO thật, sai nguyên liệu, box thiếu, ánh sáng lạ, nhiều vật thể, ngoài danh mục và thời gian xử lý. Tập classifier chỉ gồm crop sạch chưa đủ đo hiệu quả toàn pipeline.
- Hiệu chỉnh ngưỡng confidence trên validation; báo tỷ lệ unknown và lỗi trạng thái trong các mẫu được chấp nhận. Test độc lập chỉ dùng đánh giá cuối; chưa có số đo mới.

## 6. Tích hợp và dinh dưỡng

SIC_MODEL_PATH trỏ checkpoint YOLO mới; SIC_CONDITION_MODEL_PATH trỏ EfficientNet mới. App giữ khả năng thử model YOLO 214 lớp cũ với cảnh báo; thiếu EfficientNet thì giữ nguyên liệu và để trạng thái unknown. Hai model không suy gram. Người dùng xác nhận trạng thái, nhập gram phần ăn được và chọn hồ sơ dinh dưỡng đúng chế biến.

Hồ sơ dinh dưỡng cần nguồn chính thức, publisher, record_id/reference, phần ăn được, trạng thái chế biến, đơn vị và verification_status. Giữ 47 hồ sơ cũ ở unverified; chưa có vitamin thực tế được bổ sung. Trường vitamin/vi chất có hậu tố đơn vị mg/ug/g trong per_100g được tính theo gram; thiếu thì ghi thiếu, không gán 0. Không tái dùng hồ sơ raw cho cooked.

Video hiện chạy hai model trên một khung hình do người dùng chọn, giữ từng box và dùng cùng Nutrition Engine. Chưa tracking/stream liên tục; không cộng khẩu phần tự động xuyên frame.
