# Kiểm tra nguồn State V4 và yêu cầu duyệt nhãn

Đã đọc ZIP thực tế và metadata cục bộ. Chưa phê duyệt Pork/Shrimp; không gộp dataset hoặc huấn luyện.

## Kết quả cấu trúc

| Nguồn | Ảnh | Box/bản ghi nhãn | Kết quả |
| --- | ---: | ---: | --- |
| lettuce | 2170 | 1456 | inspected |
| smartveggies | 1674 | 2112 | inspected_with_warnings |
| onion | 1616 | 1616 | inspected |
| pork | 6898 | 8496 | inspected |
| shrimp | 217 | 343 | needs_review |

SmartVeggies chỉ có train: 1.674 ảnh/annotation; YAML còn tham chiếu val/test không có trong ZIP. Chỉ xác nhận chúng vắng mặt và ghi cảnh báo, không tái dùng train làm holdout.

Shrimp dùng valid/_classes.csv: 217 ảnh, fresh bật trên cả 217, not bật trên 126. Có 91 ảnh chỉ fresh và 126 ảnh đa nhãn. Số 343 là tổng nhãn đang bật, không phải số ảnh. Fresh chưa được phê duyệt; not/Bad không ánh xạ thành hỏng.

## Tất cả nhãn nguồn

| Nguồn | Nhãn | Ảnh | Box | Trạng thái mapping |
| --- | --- | ---: | ---: | --- |
| lettuce | Healthy | 520 | 520 | approved_exact |
| lettuce | Healthyd | 2 | 2 | blocked_label |
| lettuce | Larvae | 447 | 447 | blocked_label |
| lettuce | Rotten Lettuce | 487 | 487 | approved_exact |
| smartveggies | Broccoli | 594 | 594 | blocked_label |
| smartveggies | Carrot | 165 | 165 | needs_approval |
| smartveggies | Cucumber | 198 | 198 | needs_approval |
| smartveggies | Damage Bok choy | 9 | 9 | needs_approval |
| smartveggies | Damage Chayote | 6 | 21 | needs_approval |
| smartveggies | Fresh Bok choy | 30 | 45 | needs_approval |
| smartveggies | Fresh Carrots | 18 | 42 | needs_approval |
| smartveggies | Fresh Chayote | 27 | 69 | needs_approval |
| smartveggies | Fresh Egg plant | 21 | 51 | approved_exact |
| smartveggies | Fresh Kangkong | 48 | 48 | needs_approval |
| smartveggies | Fresh Squash | 18 | 39 | needs_approval |
| smartveggies | Fresh broccoli | 15 | 33 | approved_exact |
| smartveggies | Fresh cabbage | 30 | 57 | needs_approval |
| smartveggies | Fresh corn | 6 | 18 | needs_approval |
| smartveggies | Fresh cucumber | 12 | 12 | needs_approval |
| smartveggies | Fresh gourd | 21 | 81 | needs_approval |
| smartveggies | Fresh okra | 24 | 96 | needs_approval |
| smartveggies | Fresh pepper | 3 | 3 | needs_approval |
| smartveggies | FreshBitter melon | 24 | 54 | needs_approval |
| smartveggies | Papaya | 108 | 108 | needs_approval |
| smartveggies | Potato | 135 | 135 | needs_approval |
| smartveggies | Rotten Bitter melon | 9 | 30 | needs_approval |
| smartveggies | Rotten Bok choy | 3 | 6 | needs_approval |
| smartveggies | Rotten Cabbage | 24 | 30 | needs_approval |
| smartveggies | Rotten Okra | 12 | 45 | needs_approval |
| smartveggies | Rotten Squash | 12 | 12 | needs_approval |
| smartveggies | Rotten gourd | 6 | 6 | needs_approval |
| smartveggies | RottenEgg plant | 12 | 12 | approved_exact |
| smartveggies | Tomato | 51 | 51 | needs_approval |
| smartveggies | objects | 30 | 30 | needs_approval |
| smartveggies | rotten Chayote | 9 | 12 | needs_approval |
| onion | Good Onion | 1471 | 1471 | approved_exact |
| onion | Rotten Onion | 127 | 127 | approved_exact |
| onion | onion | 18 | 18 | needs_approval |
| pork | Kasim Fresh | 989 | 1032 | needs_approval |
| pork | Kasim Spoiled | 626 | 626 | needs_approval |
| pork | Liempo Fresh | 1385 | 1408 | needs_approval |
| pork | Liempo Spoiled | 691 | 694 | needs_approval |
| pork | Not Pork Meat | 45 | 172 | blocked_label |
| pork | Pork chop Fresh | 789 | 1576 | needs_approval |
| pork | Pork chop Spoiled | 566 | 1168 | needs_approval |
| pork | Tenderloin Fresh | 1182 | 1195 | needs_approval |
| pork | Tenderloin Spoiled | 625 | 625 | needs_approval |
| shrimp | fresh | 217 | 0 | needs_review |
| shrimp | not | 126 | 0 | blocked_label |

## Pork: điểm cần người dùng duyệt

- Metadata: Meat Checker Model, phiên bản 7 “deleted half-fresh”; README ghi 6.898 ảnh, YOLOv8, không augmentation. Tham chiếu nguyên văn được lưu trong source_metadata.json.
- Đã xuất và xem 48 mẫu trong tám contact sheet: mỗi nhãn sáu ảnh phân bổ theo train/val/test. Khung đỏ và crop cho phép đối chiếu vùng thịt; xem contact_sheet_samples.csv để truy lại file/box/SHA.
- Nhiều mẫu trong các split có kiểu miếng thịt, khay/nền tương tự và màu vàng rõ. Đây là quan sát ảnh mẫu, không phải chứng minh cùng cá thể hoặc độ tươi/hư hỏng.
- Một số tên file có yellow_realistic ở cả Fresh và Spoiled. Cần xác minh có xử lý màu hay ảnh dẫn xuất hay không, dù README ghi không augmentation.
- Cần xác minh quy trình gán Fresh/Spoiled, tình trạng trước khi chụp, và mã mẫu/lô để tránh cùng nguồn xuất hiện ở nhiều split. Không xác nhận hư hỏng chỉ qua ảnh.
- Tám nhãn vẫn needs_approval; Not Pork Meat vẫn bị chặn. Chỉ sửa approved_mappings sau khi người dùng chấp thuận chính xác các nhãn.

Contact sheets: [pork_review/README.md](../datasets/reports/state_v4/pork_review/README.md).

## Phần còn thiếu

- Chưa có nhãn broccoli hỏng được xác minh. Broccoli chung chung giữ blocked_label.
- Chưa có lớp shrimp hỏng được xác minh; fresh cần làm rõ metadata đa nhãn trước khi duyệt.
- Pork có nhãn nguồn nhưng chưa có nhãn được người dùng phê duyệt.
- Không suy số nhóm độc lập từ số ảnh. Công cụ mặc định gom cùng acquisition; nếu không đủ nhóm, split_plan giữ train_candidate hoặc review.

## Kết quả prepare/audit thực tế

Đã xuất 2.701 bản ghi crop thành 2.689 file duy nhất, chỉ từ nhãn approved_exact của Lettuce, SmartVeggies và Onion. Không có ảnh Pork/Shrimp được đưa vào ứng viên.

| Lớp đích | Bản ghi crop | File duy nhất |
| --- | ---: | ---: |
| lettuce__normal_appearance | 520 | 520 |
| lettuce__visible_defect | 487 | 487 |
| eggplant__normal_appearance | 51 | 51 |
| eggplant__visible_defect | 12 | 12 |
| broccoli__normal_appearance | 33 | 33 |
| onion__normal_appearance | 1471 | 1459 |
| onion__visible_defect | 127 | 127 |

Audit đối chiếu 23.893 ảnh tham chiếu và ghi 211 liên kết SHA/pHash. Không phát hiện liên kết khác nhãn hoặc overlap tham chiếu trong ngưỡng pHash <= 4. Ba nhóm acquisition thận trọng đều có tín hiệu pHash cần xem, nên toàn bộ 2.689 ứng viên giữ review. Chưa có bằng chứng xác minh nhóm mẫu vật độc lập; không tạo Train/Val/Test. Cần duyệt audit_pairs.csv và metadata nguồn trước khi tách nhóm nhỏ hơn, không chỉ dùng số ảnh/tên split để suy tính độc lập.

Kết quả kiểm thử và bảo toàn dữ liệu: [bulk_state_dataset_validation.md](bulk_state_dataset_validation.md).
