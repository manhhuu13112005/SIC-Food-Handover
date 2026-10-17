# Kế hoạch tích hợp website — 17/10/2026

Khảo sát xác nhận Streamlit, YOLO best.pt Detection 214 lớp, 47 hồ sơ per_100g và checkpoint EfficientNet fast-track 37 lớp. Website cũ ghép YOLO và EfficientNet trên crop; parser chỉ hỗ trợ nhãn cũ và loader mặc định chưa trỏ tới checkpoint mới. Không có thư mục .git; đã sao lưu mã trước sửa.

1. Giữ YOLO ảnh/video và công thức Nutrition Engine; tách EfficientNet thành trang độc lập.
2. Sửa model_loader.py, condition_service.py, pipeline_config.py để đọc đúng metadata/state_dict/class_names và preprocessing training; giữ tương thích định dạng checkpoint cũ cho các công cụ offline.
3. Sửa main.py, page1.py, page2.py; thêm condition_page.py, app_state.py, image_service.py, ui_components.py và theme Streamlit.
4. Giữ dữ liệu dinh dưỡng/config nghiên cứu, checkpoints, notebook và datasets. Cache model, giữ input/kết quả qua thao tác, không training/download.
5. Kiểm thử syntax/import/config, unit/AppTest, model CPU thật, HTTP và trình duyệt ở desktop/mobile; sửa lỗi, ghi kết quả thực thi và giới hạn trong website_validation.md.

Kế hoạch này được thông báo trong hội thoại trước khi sửa backend và UI; mọi bước đã được triển khai. Chi tiết kiểm thử nằm trong báo cáo cuối.
