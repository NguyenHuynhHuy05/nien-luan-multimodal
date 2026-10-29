# Bước kế tiếp: mô hình chỉ dùng ảnh

Gói bổ sung cho niên luận, chuẩn bị ngày 28/10/2026. Giải nén các mục trong ZIP vào thư mục gốc dự án hiện tại (nơi có src/, Images/, data/, requirements.txt). Gói này chỉ thêm tệp mới, không thay mã chuẩn bị dữ liệu, bảng CSV hoặc service NASA.

## Phạm vi

MobileNetV2 với trọng số IMAGENET1K_V2, thay lớp cuối thành 12 đầu ra. Mặc định đóng băng backbone và cả thống kê BatchNorm, học bộ phân loại. Đây là baseline đầu tiên, chưa kết hợp văn bản hay NASA. Có thể dùng --train-backbone cho thí nghiệm fine-tune sau khi baseline chạy ổn; khi đó cần cân nhắc learning rate nhỏ hơn.

Chỉ ảnh RGB được đưa vào mạng. Các cột nhãn, đường dẫn, group_id chỉ dùng tải ảnh/kiểm tra hoặc làm mục tiêu; không được đưa chuỗi đường dẫn hay metadata vào mạng.

Tài liệu gốc:
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Tệp được thêm

| Tệp | Vai trò |
|---|---|
| src/image_baseline/common.py | Dataset, biến đổi ảnh, mô hình, kiểm tra ánh xạ và giao nhóm, tính metrics |
| src/image_baseline/__init__.py | Định nghĩa package |
| src/train_image.py | Huấn luyện Train, chọn checkpoint bằng Validation, lưu kết quả |
| src/evaluate_image.py | Nạp checkpoint và đánh giá; mặc định Validation |
| src/test_image_baseline.py | 8 test kỹ thuật dùng ảnh tạo tạm, không phải dữ liệu nghiên cứu |
| requirements-image.txt | Thư viện huấn luyện; tham chiếu requirements.txt có sẵn |
| README_IMAGE_BASELINE.md | Hướng dẫn này |

## Dữ liệu cần có

- Images/<tên lớp>/<tên ảnh> tương ứng image_path trong CSV. image-root phải trỏ đến thư mục chứa trực tiếp các thư mục lớp.
- data/image_text/train_processed.csv
- data/image_text/val_processed.csv
- data/image_text/test_processed.csv: chỉ mở khi bạn chủ động đánh giá Test.
- data/image_text/label_mapping.json, 12 nhãn từ 0 đến 11.

Giữ nguyên split và mã nhãn hiện tại. Không dùng ImageFolder để tự sinh lại mã lớp và không tự chia lại dữ liệu.

## Cài đặt trên Windows

Đã kiểm tra code bằng Python 3.12, PyTorch 2.9.0 CPU và torchvision 0.24.0 CPU. Các lệnh dưới đây tạo môi trường riêng. Cần cài Python 3.12 trước nếu máy chưa có; kiểm tra bằng py -0p.

Trong Terminal tại thư mục gốc dự án:

```powershell
py -3.12 -m venv .venv-image
.\.venv-image\Scripts\python.exe -m pip install --upgrade pip
.\.venv-image\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-image\Scripts\python.exe -m pip install -r requirements-image.txt
```

Đây là bản CPU để bắt đầu thuận tiện. Nếu dùng NVIDIA GPU/CUDA, cài bản PyTorch phù hợp theo hướng dẫn chính thức https://pytorch.org/get-started/locally/ thay cho dòng cài CPU; không trộn torch và torchvision không tương thích.

Không cần kích hoạt PowerShell script; các lệnh gọi trực tiếp Python trong môi trường mới.

## Bước 1 — kiểm tra ảnh thật trước khi huấn luyện

```powershell
.\.venv-image\Scripts\python.exe src/train_image.py --check-data-only --verify-images
```

Lệnh kiểm tra mapping, nhãn, split, ảnh tồn tại, giải mã ảnh và giao nhau Train/Validation. Không tải trọng số và không mở CSV Test. --verify-images kiểm tra giải mã, không tính lại SHA-256 byte ảnh; sử dụng group_id đã chuẩn bị trong CSV.

Đầu ra mong đợi với bộ mẫu hiện tại: Train=9557, Validation=2046, classes=12, sau đó Data checks passed. Nếu số lượng khác, đối chiếu đúng thư mục CSV đang dùng trước khi chạy tiếp.

Nếu Images nằm ở nơi khác, thêm --image-root "D:/duong_dan/Images".

## Bước 2 — chạy thử một epoch trên ảnh thật

Lần đầu khởi tạo trọng số ImageNet cần mạng để tải và sau đó được cache.

```powershell
.\.venv-image\Scripts\python.exe src/train_image.py --epochs 1 --batch-size 16 --output-dir runs/image/first_epoch
```

Mục tiêu: xác nhận bộ nạp ảnh và quá trình huấn luyện thực tế hoạt động. Kết quả một epoch chưa phải kết quả cuối cùng của niên luận. Nếu thiếu bộ nhớ, giảm --batch-size 8. Không có thời gian chạy cố định; phụ thuộc máy và số ảnh.

## Bước 3 — huấn luyện baseline

Sau khi một epoch chạy ổn, bắt đầu một thí nghiệm riêng:

```powershell
.\.venv-image\Scripts\python.exe src/train_image.py --epochs 10 --batch-size 16 --patience 3 --output-dir runs/image/baseline_01
```

Mặc định: seed=42, learning rate=0.001, AdamW, CrossEntropyLoss, workers=0, threads=2; tự chọn CUDA nếu bản cài đặt hỗ trợ và GPU sẵn sàng, nếu không dùng CPU. Không cache cả bộ ảnh trong RAM.

- Train được lật ngang/nghiêng nhẹ, rồi biến đổi theo trọng số ImageNet.
- Validation/Test dùng resize 232, center crop 224, tensor và chuẩn hóa theo IMAGENET1K_V2.transforms(); không tăng cường ngẫu nhiên.
- Chọn best.pt theo Validation Macro F1 cao nhất; bằng điểm thì chọn Validation loss thấp hơn.
- Early stopping khi không cải thiện trong patience epoch.
- Không mở Test trong quá trình huấn luyện.
- Thư mục --output-dir phải chưa tồn tại. Chạy lại phải dùng tên mới; bỏ tham số này để chương trình tự tạo thư mục theo thời gian.
- --weights none chỉ dành kiểm tra kỹ thuật không tải mạng; không dùng kết quả đó để báo cáo như mô hình tiền huấn luyện.
- Seed hỗ trợ tái lập, không hứa kết quả bit-for-bit trên mọi phần cứng/phiên bản.

## Các tệp kết quả

| Tệp | Nội dung |
|---|---|
| config.json | Tham số, phiên bản, thứ tự lớp, SHA-256 CSV Train/Validation và mapping |
| history.csv | Loss, Accuracy, Macro F1 của Train/Validation mỗi epoch |
| best.pt | Trạng thái mô hình, thứ tự lớp, epoch và cấu hình của checkpoint được chọn |
| training_summary.json | Epoch tốt nhất, chỉ số Validation, xác nhận Test chưa được dùng |

.pt là checkpoint mô hình; khác với label_encoder.pkl là bộ mã hóa nhãn.

## Bước 4 — đánh giá checkpoint

Mặc định xem Validation trước:
```powershell
.\.venv-image\Scripts\python.exe src/evaluate_image.py --checkpoint runs/image/baseline_01/best.pt --split val
```

Sau khi đã chốt phương pháp bằng Train/Validation, mới đánh giá Test:
```powershell
.\.venv-image\Scripts\python.exe src/evaluate_image.py --checkpoint runs/image/baseline_01/best.pt --split test
```

Không chọn cấu hình theo kết quả Test. Nếu đổi CSV Train/Validation hoặc mapping sau huấn luyện, chương trình từ chối đánh giá để tránh trộn phiên bản.

Đầu ra trong evaluation_val/ hoặc evaluation_test/:
- metrics.json: Accuracy, Macro F1, Precision/Recall/F1 từng lớp và ma trận nhầm lẫn.
- confusion_matrix.csv: hàng là lớp thật, cột là lớp dự đoán.
- predictions.csv: kết quả từng ảnh theo thứ tự CSV; max_softmax_score không được coi là xác suất bệnh đã hiệu chỉnh.

Không tự ghi đè một thư mục đánh giá đã có. Dùng --output-dir khác nếu có lý do đánh giá lại.

## Kiểm tra kỹ thuật đã thực hiện

Đã chạy 8 test thành công, gồm:
1. Ảnh RGB trả tensor 3×224×224; Validation biến đổi ổn định.
2. Metadata không được đưa vào mạng.
3. Mapping sai bị từ chối.
4. Ảnh thiếu hoặc hỏng không bị bỏ qua âm thầm.
5. Đường dẫn thoát thư mục bị chặn.
6. Nhóm ảnh giao nhau giữa hai split bị chặn.
7. Một bước học cập nhật bộ phân loại, không đổi backbone và thống kê BatchNorm khi đóng băng.
8. Luồng CLI huấn luyện → lưu → nạp → đánh giá Validation/Test; huấn luyện vẫn thành công khi CSV Test không tồn tại.

```powershell
.\.venv-image\Scripts\python.exe -m unittest discover -s src -p "test_image_baseline.py" -v
```

Các test dùng ảnh tạo tạm và trọng số khởi tạo ngẫu nhiên cho kiểm tra kỹ thuật. Chưa huấn luyện hay đánh giá trên 13.650 ảnh PlantVillage vì gói dữ liệu gửi kèm không có thư mục Images. Chưa có kết quả Accuracy/F1 nghiên cứu để công bố.

## Bạn cần hiểu để trình bày

- Dataset đọc CSV, không tự chia lại tập; image_path để mở ảnh, label_id là mục tiêu.
- Tensor đầu vào có dạng batch × 3 × 224 × 224, đầu ra batch × 12 logits.
- CrossEntropyLoss nhận logits và ID nhãn; không softmax trước hàm mất mát.
- Backbone trích xuất đặc trưng; classifier ánh xạ đặc trưng thành 12 lớp.
- Frozen backbone không học trọng số; BatchNorm của backbone được giữ chế độ eval khi học head.
- optimizer cập nhật các tham số có requires_grad=True.
- model.eval() và tắt gradient được dùng khi đánh giá.
- Validation chọn mô hình; Test đánh giá sau khi chốt.
- Baseline ảnh tạo mốc cho bước thêm văn bản. Nhánh NASA vẫn độc lập.


## Kiểm toán byte ảnh và lưu phiên bản dữ liệu

Chạy riêng bước này trước huấn luyện. Nó đọc đường dẫn trong cả ba CSV Train,
Validation, Test để tính SHA-256 từ byte ảnh và kiểm tra giải mã bằng Pillow; không chạy mô hình hay tính chỉ số
trên Test. Không chuyển ảnh, sửa CSV, đổi nhãn hoặc chia lại split. Ảnh ngoài các
CSV không thuộc phạm vi kiểm toán. Hash byte không phát hiện ảnh gần trùng hoặc
ảnh đã được mã hóa lại.

PowerShell tại thư mục gốc dự án:

```powershell
$manifest = "reports/image_hashes_" + (Get-Date -Format "yyyyMMdd_HHmmss_fffffff") + ".csv"
.\.venv-image\Scripts\python.exe src/audit_image_hashes.py --output $manifest
if ($LASTEXITCODE -ne 0) { throw "Kiem toan anh that bai; dung huan luyen va ra soat thong bao." }
```

Manifest có các cột `split,image_path,sha256`, đường dẫn tương đối dưới `Images`.
Nếu phát hiện trùng byte giữa các split, công cụ vẫn lưu manifest, liệt kê mọi
đường dẫn liên quan theo hash và thoát mã 1. Không tự xử lý các ảnh trùng. Một
manifest được tạo không có nghĩa là kiểm toán đạt: phải kiểm tra mã thoát.
Nếu thiếu ảnh/CSV hoặc đầu vào không hợp lệ, công cụ dừng và không ghi manifest
hoàn chỉnh. Không ghi đè tệp đầu ra đã tồn tại.

Đối chiếu với manifest đã lưu sau này (dùng đường dẫn baseline thực tế):

```powershell
$baseline = "reports/image_hashes_BASELINE.csv"
$current = "reports/image_hashes_" + (Get-Date -Format "yyyyMMdd_HHmmss_fffffff") + ".csv"
.\.venv-image\Scripts\python.exe src/audit_image_hashes.py --compare $baseline --output $current
if ($LASTEXITCODE -ne 0) { throw "Du lieu thay doi hoac trung anh; dung va ra soat bao cao." }
```

So sánh báo `added`, `removed`, `changed` theo cặp split/đường dẫn; bất kỳ khác
biệt nào cũng trả mã 1. Lưu baseline cùng hồ sơ thí nghiệm. Kiểm toán và huấn
luyện là hai lệnh riêng: không thay đổi dữ liệu giữa hai bước. Manifest này kiểm
tra byte ảnh và vị trí split, không thay thế kiểm tra mapping/nhãn của trainer.
Trainer và evaluator hiện không tự nạp manifest; cần chạy lệnh đối chiếu trước
khi huấn luyện hoặc đánh giá để kiểm tra phiên bản ảnh.

Kiểm thử công cụ (dùng ảnh tạm, không cần PlantVillage):

```powershell
.\.venv-image\Scripts\python.exe -m unittest discover -s src -p "test_audit_image_hashes.py" -v
```
