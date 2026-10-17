# Kết quả kiểm tra — 2026-10-15

## Phạm vi và môi trường

- Bản bàn giao: C:/Users/ASUS/SIC-Food, main.py tại thư mục này.
- Python thực tế: C:/MY_ MANH/APP DOWLOAD/Anaconda/envs/sic-food/python.exe, 3.11.15.
- Streamlit 1.65.0; Ultralytics 8.4.172; torch 2.14.1; torchvision 0.29.1; streamlit-option-menu 0.4.0; Pillow 12.3.0; NumPy 2.4.6; opencv-python-headless 5.0.0.93. Đây là phiên bản quan sát trong môi trường, chưa phải lockfile di động cho mọi máy.
- Không cài/nâng cấp thư viện. Không huấn luyện, không tạo hoặc thay thế best.pt.
- Không có AGENTS.md áp dụng được tìm thấy; thư mục không có Git repository. Đối chiếu trước sửa cho thấy các file chính của bản người dùng trùng bản gốc đã sao lưu.
- Backup ngay trước đồng bộ: backups/before_state_workflow_20261015_141847. sync_manifest.json trong backup liệt kê file thêm/sửa và hash để đối chiếu.

## Đã chạy

1. Kiểm tra cú pháp Python và JSON cấu hình: đạt.
2. Unit tests: 10 kiểm thử, bao gồm công thức 150 g = 1,5 lần, tên lớp cũ giữ trạng thái unknown, cooked không đồng nghĩa fresh, box riêng biệt, thiếu hồ sơ và tổng một phần, cập nhật tổng, reset trạng thái ảnh, thiếu/model lỗi, khối lượng không hợp lệ, hồ sơ cũ cần đồng ý và quy tắc rau quả.
3. AppTest chạy main.py thật với đường dẫn model không tồn tại: mở trang thành công, thêm thủ công, chọn chuối, xác nhận, nhập 150 g, chọn hồ sơ legacy và đồng ý dùng dữ liệu chưa kiểm chứng. Hiển thị 133,50 kcal theo chính dữ liệu cũ (không phải xác nhận dữ liệu đúng về dinh dưỡng). Đổi gram về 0 xóa tổng; chọn nghi hỏng tự bỏ chọn và xóa tổng.
4. AppTest trang ảnh với upload/inference mock chỉ nằm trong tests/ui_harness.py: hai đối tượng cùng loại có hai mục/ô gram; đổi ảnh xóa kết quả và input; inference lỗi/rỗng xóa kết quả cũ; nghi hỏng mặc định loại; sửa tình trạng không thay dự đoán gốc.
5. Model thật best.pt: nạp và cache thành công; có 214 lớp, tên quan sát lưu ở model_names_observed.json. Ảnh images/test image.png trả 5 box: Waffle, Pasta, Banana, Avocado, Avocado. Hai Avocado giữ ID riêng. Không suy trạng thái từ tên nguyên liệu.
6. Đối chiếu food_items.json, requirements.txt, EDA.ipynb với bản gốc: giữ nguyên.

## Giới hạn

- AppTest là kiểm thử giao diện không trình duyệt. Đổi ảnh/lỗi inference trong test tích hợp dùng fixture được khai báo rõ; kiểm thử tải file trực tiếp và bố cục trên trình duyệt chưa thực hiện.
- Lần thử trong sandbox bị chặn ở socket/thư mục tạm; sau khi được cấp quyền, các kiểm thử AppTest đã chạy thành công ngoài sandbox bằng đúng interpreter.
- Chưa kiểm thử tải model qua Internet hoặc video đầy đủ; video giữ luồng cũ với sửa đường dẫn, cache, lỗi model và dọn file tạm khi nhận diện.
- Không có bộ model trạng thái mới, không công bố độ chính xác mới. Không đo an toàn thực phẩm.
- 47 hồ sơ dinh dưỡng đều chưa kiểm chứng, chế biến unknown. Không có hồ sơ raw/cooked xác minh; những lựa chọn đó báo chưa có dữ liệu. Không tạo số liệu thay thế.

## Cấu trúc bàn giao

- main.py: đường dẫn tài nguyên theo file.
- page1.py: workflow ảnh, từng đối tượng, xác nhận, gram, chọn hồ sơ, nguồn kết quả và tổng cập nhật tức thời.
- page2.py: sửa tối thiểu để tương thích loader, đường dẫn UTF-8, thông báo lỗi và dọn video tạm khi nhận diện.
- model_loader.py: nạp lười, cache theo file, tải tùy chọn bằng file tạm, lỗi rõ ràng.
- detection_service.py, state_policy.py, nutrition_service.py: xử lý tách khỏi giao diện.
- config/: danh mục, ánh xạ dự kiến và hồ sơ kế thừa có trạng thái kiểm chứng.
- Run_SIC_Food.ipynb, README.md: chạy từ Jupyter/PowerShell bằng môi trường hiện có.
- docs/implementation_plan.md: dữ liệu, gán nhãn, chia nhóm chống rò rỉ, huấn luyện, đánh giá và tích hợp.
- tests/: logic và kiểm thử giao diện; fixture không nằm trong dữ liệu/suy luận sản phẩm.
# Ghi chú cập nhật

Kết quả kiến trúc mới dùng YOLO + EfficientNet nằm tại [two_model_validation.md](two_model_validation.md). Nội dung phía dưới là báo cáo lịch sử của kiến trúc trước, không phải mô tả hiện trạng hai model.

