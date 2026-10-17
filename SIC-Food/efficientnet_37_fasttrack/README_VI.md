# SIC-Food — EfficientNet-B0 37 lớp (Fast-track trên CPU)

Bản này huấn luyện **37 lớp đã có ảnh ứng viên**, không thu thập thêm ba lớp để đạt 40. Dùng backbone pretrained ImageNet, đóng băng backbone, trích xuất đặc trưng **một lần**, rồi huấn luyện classifier nhiều epoch. Có thể tiếp tục sau khi Jupyter/Anaconda bị ngắt bằng cache theo lô và checkpoint epoch.

## Chuẩn bị

Đặt cả hai file `.py` ở **thư mục gốc** `C:\Users\ASUS\SIC-Food`. Giữ nguyên các file bạn đã tạo:

- `datasets/efficientnet_pilot_37/train_manifest.csv` — 25.908 ảnh ứng viên thuộc 37 lớp (16.950 V3 Train + 8.157 V4 + 801 Bắp cải)
- `datasets/efficientnet_pilot_37/v3_eval_reference.csv` — chỉ sử dụng V3 `val`, không đụng đến `test`
- `datasets/efficientnet_pilot_37/class_names.json` — thứ tự 37 lớp

Dùng môi trường Anaconda đã chạy thành công PyTorch/Torchvision trong Jupyter; cài pandas, Pillow, tqdm nếu thiếu. Pretrained weights đã có trong cache.

## 1. Huấn luyện toàn bộ Train ứng viên

Trong Anaconda Prompt:

```bat
cd /d C:\Users\ASUS\SIC-Food
python train_efficientnet_37.py --epochs 15 --image-batch-size 16 --threads 4
```

Lần đầu cần xử lý đặc trưng cho ~29.122 ảnh (Train và V3 Val), vì chạy CPU nên cần đợi; thanh tiến độ cho biết số chunk hoàn thành. Nếu bị ngắt, **chạy lại đúng lệnh**: các chunk đã hoàn thiện được tái sử dụng.

Sau khi tính xong features, 15 epoch huấn luyện classifier thường nhanh hơn đáng kể so với chạy EfficientNet backbone mỗi epoch. Chạy tiếp đến 20 epoch bằng `--epochs 20` (tổng cộng 20, không phải thêm 20).

Checkpoint tự động mỗi epoch:

`models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth`

Lịch sử:

`models/efficientnet_37_fasttrack/training_history.csv`

## 2. Thử ảnh riêng (crop nguyên liệu)

```bat
python efficientnet_37_predictor.py --model models\efficientnet_37_fasttrack\efficientnet_b0_37_prototype.pth --image "C:\path\ingredient.jpg"
```

## 3. Gắn vào website **cũ**

```python
from efficientnet_37_predictor import EfficientNet37Predictor
predictor = EfficientNet37Predictor('models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth')
# crop_pil phải là đối tượng PIL.Image sau crop của YOLO (nếu YOLO detect nguyên liệu)
result = predictor.predict(crop_pil)
```

Nếu YOLO đang nhận diện **món ăn chế biến**, đầu ra món ăn **không tương đương** crop nguyên liệu phù hợp cho EfficientNet. Giữ hai luồng riêng hoặc điều chỉnh YOLO theo yêu cầu dữ liệu.

## Hạn chế đánh giá

- Đây là **prototype**, vì dữ liệu V4 và Bắp cải chưa có Validation/Test độc lập được duyệt; trong đó có lớp Cà tím hỏng chỉ 9 ảnh Train ứng viên.
- Accuracy `v3_val_26_acc_37_output` chỉ đo các **26 lớp V3**, trong khi mô hình có **37 đầu ra**. Tuyệt đối không gọi nó là `37-class accuracy`.
- V3 Test được giữ nguyên, không dùng chọn siêu tham số.
- `normal_appearance` và `visible_defect` chỉ mô tả tình trạng quan sát bằng ảnh, không chứng nhận an toàn thực phẩm.
- Chưa có đánh giá độc lập cho 11 lớp thêm vào; cần thu thập ảnh từ phiên chụp mới và đánh giá trước khi triển khai thực tế.
