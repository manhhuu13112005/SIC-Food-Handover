# Kiểm tra kiến trúc hai model — 16/10/2026

## Thay đổi đã triển khai

- YOLO chỉ ánh xạ nguyên liệu; danh sách huấn luyện mục tiêu gồm 20 ID đề xuất trong config/ai_pipeline.json. Bỏ nhãn YOLO ghép trạng thái của kế hoạch cũ.
- EfficientNet-B0 nạp checkpoint cục bộ, crop từng box, phân loại nhãn ingredient_state. Điểm thấp/khác nguyên liệu/thiếu model/lỗi crop/inference giữ unknown. Không suy sống/chín từ ngoại hình fresh/defect.
- Trang ảnh và video dùng chung xác nhận và Nutrition Engine. Video xử lý một khung hình được chọn, không tracking hay cộng lặp qua frame.
- Nutrition Engine hỗ trợ trường vi chất có đơn vị trong per_100g, tổng vi chất ghi coverage và complete riêng; chưa có vitamin thực tế bổ sung vào 47 hồ sơ cũ.
- Có skeleton dataset train/val/test cho YOLO và EfficientNet, data.yaml 20 lớp, manifest ví dụ và hai script huấn luyện. Dataset chưa có ảnh thật.

## Kiểm tra đã chạy

Môi trường Conda sic-food Python 3.11.15 trên máy người dùng.

1. `python -B -m unittest discover -s tests -p 'test_*.py'`: **17/17 PASS**. Bao gồm 10 test workflow cũ và 7 test mới: hợp đồng 20 lớp, crop theo box, fallback, confidence/khác nguyên liệu, vi chất thiếu, roundtrip checkpoint EfficientNet-B0 thật với weights ngẫu nhiên, đọc frame video thật được tạo tổng hợp bằng OpenCV.
2. `python -B tests/check_streamlit.py`: **PASS**. Thiếu YOLO vẫn mở app, nhập thủ công, xác nhận, tính dinh dưỡng, thay gram và loại mục nghi hỏng.
3. `python -B tests/check_image_ui.py`: **PASS**. Mock upload/inference riêng cho test; giữ hai box độc lập, reset khi đổi ảnh/lỗi/không phát hiện, giữ dự đoán gốc, loại mặc định khi EfficientNet trả nghi hỏng.
4. `python -B tests/check_video_ui.py`: **PASS**. Mock upload/inference riêng cho test; chạy trang video, đổi frame hoặc upload xóa các mục cũ.

Lần chạy trong sandbox bị chặn ghi file tạm và AppTest không hoàn tất. Đã dừng các phiên đó và chạy lại ngoài sandbox với quyền được cấp; các kết quả PASS bên trên thuộc lần chạy lại.

## Giới hạn

Weights EfficientNet dùng cho kiểm thử là ngẫu nhiên, chỉ nằm trong file tạm và được xóa sau test. Đây không phải model sản phẩm hay số đo độ chính xác. Chưa huấn luyện YOLO 20 lớp/EfficientNet trên dữ liệu thật. best.pt cũ 214 lớp được giữ nguyên. Chưa kiểm thử trình duyệt thật, tracking video dài hoặc huấn luyện dài; không công bố mAP/F1 mới.

Baseline fresh/defect có 40 lớp và chỉ phân biệt tình trạng bình thường/bất thường chung. Muốn tách héo/dập/mốc/nghi hỏng cần bộ nhãn chi tiết và checkpoint huấn luyện tương ứng. Danh sách 20 nguyên liệu đang là đề xuất cần chốt trước thu thập.
