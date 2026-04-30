# Hệ thống nhận diện xe vi phạm đèn đỏ và biển số xe

## Mô tả hệ thống

Hệ thống web sử dụng AI để tự động phát hiện xe vi phạm đèn đỏ và nhận diện biển số xe từ video giao thông.

### Tính năng chính:
- ✅ Upload video giao thông (.mp4, .avi, .mov, .mkv)
- ✅ Phát hiện đèn đỏ bằng computer vision
- ✅ **State Machine Logic**: Theo dõi hành trình xe từ vùng chờ → vùng vi phạm
- ✅ **Smart Violation Detection**: Chỉ ghi nhận xe đi từ vùng chờ qua vùng vi phạm lúc đèn đỏ
- ✅ Nhận diện xe vi phạm bằng YOLOv8 + ByteTrack
- ✅ Trích xuất biển số xe bằng EasyOCR (tối ưu cho biển Việt Nam)
- ✅ Lưu trữ vi phạm vào SQLite database với metadata đầy đủ
- ✅ Tra cứu vi phạm theo biển số xe với bộ lọc nâng cao
- ✅ Giao diện web responsive với real-time progress tracking
- ✅ **Visual State Feedback**: Màu sắc trực quan cho trạng thái xe (Xanh/Vàng/Đỏ)

### 🔄 State Machine Logic - Logic Phát hiện Vi phạm Thông minh

Hệ thống sử dụng **Máy trạng thái hữu hạn (Finite State Machine)** để theo dõi hành trình của từng xe, đảm bảo chỉ ghi nhận vi phạm khi có bằng chứng rõ ràng:

#### Các Trạng thái của Xe:
- 🟢 **NEUTRAL**: Xe chưa vào vùng quan sát
- 🟡 **IN_WAITING_ZONE**: Xe đang ở vùng chờ (chờ đèn xanh)
- 🔴 **COMMITTED_VIOLATION**: Xe đã vi phạm (đi từ vùng chờ qua vùng vi phạm lúc đèn đỏ)
- ✅ **PASSED_LEGALLY**: Xe đi qua hợp pháp (đèn xanh hoặc không vi phạm)

#### Logic Phát hiện:
1. **Theo dõi Hành trình**: Chỉ xe đi từ vùng chờ → vùng vi phạm mới được xem xét
2. **Kiểm tra Thời điểm**: Chỉ ghi nhận khi đèn đỏ tại thời điểm xe đi qua
3. **Loại bỏ False Positive**: Xe từ đường khác hoặc đã vượt lúc đèn xanh sẽ không bị ghi nhận
4. **Anti-duplication**: Mỗi xe chỉ bị ghi nhận 1 lần vi phạm

#### Ưu điểm:
- 🎯 **Độ chính xác cao**: Giảm thiểu false positive từ xe từ đường khác
- 🔍 **Logic rõ ràng**: Theo dõi được hành trình thực tế của xe
- 📊 **Dễ debug**: Màu sắc trực quan cho từng trạng thái
- ⚡ **Hiệu suất tốt**: State Machine tối ưu cho xử lý real-time

## Cài đặt

### Bước 1: Cài đặt Python và dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Cài đặt packages cơ bản
pip install Flask opencv-python numpy Pillow

# Cài đặt AI models (tùy chọn - để có độ chính xác cao hơn)
pip install ultralytics easyocr

# Hoặc cài đặt từ requirements.txt
pip install -r requirements.txt
```

### GPU Support (Khuyến nghị cho hiệu suất tốt hơn)

Hệ thống tự động phát hiện và sử dụng GPU nếu có sẵn (NVIDIA CUDA). Nếu không có, sẽ dùng CPU.

1) Kiểm tra driver và phiên bản CUDA khả dụng:
```powershell
nvidia-smi
```
Ghi chú giá trị "CUDA Version" (ví dụ 12.1, 11.8, 13.0).

2) Cài đúng PyTorch CUDA (cài torch trước, rồi mới cài requirements.txt):
```powershell
# CUDA 12.1
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Nếu driver cũ: CUDA 11.8
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# Sau đó cài phần còn lại
pip install -r requirements.txt
```

3) Xác minh GPU trong Python:
```powershell
python - << 'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY
```

### Bước 2: Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

Nếu log hiển thị "✓ GPU phát hiện ..." và không còn "Using CPU for processing" thì GPU đã hoạt động.

## Cấu trúc project

```
traffic-violation-detection/
├── app.py                 # Flask application chính
├── requirements.txt       # Dependencies
├── README.md             # Hướng dẫn này
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── processing.html
│   ├── results.html
│   └── search.html
├── uploads/              # Thư mục chứa video upload
├── processed/            # Video đã xử lý
├── violations/           # Ảnh vi phạm
└── traffic_violations.db # SQLite database
```

## Hướng dẫn sử dụng

### 1. Upload video
- Truy cập trang chủ và click "Upload Video"
- Chọn file video giao thông (tối đa 500MB)
- Click "Upload và Xử Lý"

### 2. Theo dõi quá trình xử lý
- Sau khi upload, hệ thống sẽ tự động chuyển đến trang xử lý
- Theo dõi tiến độ real-time
- Xem số lượng vi phạm được phát hiện

### 3. Xem kết quả
- Sau khi hoàn thành, xem danh sách vi phạm
- Download video đã xử lý có annotation
- Xem ảnh chụp từng vi phạm

### 4. Tra cứu vi phạm
- Vào mục "Tra cứu vi phạm"
- Nhập biển số xe (có thể nhập một phần)
- Xem lịch sử vi phạm

## Technical Details

### AI Models được sử dụng:

1. **Traffic Light Detection**: Computer vision với HSV color detection
2. **Vehicle Detection**: YOLOv8 (fallback: simple motion detection)
3. **License Plate Recognition**: YOLOv5 nano detector + YOLOv5 nano OCR, EasyOCR fallback nếu model mới không khả dụng

### Performance Optimization:

- Xử lý mỗi 5 frame thay vì tất cả để tăng tốc
- Threading để xử lý background không block UI
- SQLite database với indexing cho tra cứu nhanh
- Resize frame để giảm computational load
- Model nhận diện xe mặc định dùng `yolov8n.pt` để ưu tiên tốc độ; có thể đổi sang `yolov8m.pt` trong `config.py` nếu cần độ chính xác cao hơn
- Nhận diện biển số chỉ chạy khi xe đã được xác định vi phạm, model LPR được load một lần và có cache frame để giảm OCR lặp
- Giao diện mới hỗ trợ 3 chế độ xử lý: Fast, Balanced, Quality
- Có thể tắt ghi video output để tăng tốc khi chỉ cần ảnh/dữ liệu vi phạm
- AI models được lazy-load: mở web không chiếm GPU, chỉ load khi bắt đầu xử lý video
- **GPU Acceleration**: Tự động sử dụng GPU (CUDA) nếu có, chuyển sang CPU nếu không

### Benchmark không cần video test

```powershell
python tools\benchmark_pipeline.py --synthetic --frames 120 --size 1280x720
python tools\benchmark_pipeline.py --synthetic --frames 120 --size 1280x720 --no-models
```

### Yêu cầu video:

- **Định dạng**: MP4, AVI, MOV, MKV
- **Chất lượng**: Tối thiểu 480p, khuyến nghị 720p+
- **Nội dung**: Phải có đèn giao thông và xe cộ rõ ràng
- **Thời lượng**: 30 giây - 5 phút (tối ưu)
- **Góc quay**: Từ trên cao, nhìn rõ vạch dừng

## Troubleshooting

### Lỗi thường gặp:

1. **"Cannot open video file"**
   - Kiểm tra định dạng video có được hỗ trợ
   - Thử với video khác
   - Đảm bảo file không bị corrupt

2. **"Model initialization error"**
   - Cài đặt: `pip install ultralytics easyocr`
   - Hoặc chạy với fallback methods (vẫn hoạt động)

3. **"File quá lớn"**
   - Giới hạn hiện tại: 500MB
   - Nén video hoặc cắt ngắn thời lượng

4. **"Không phát hiện vi phạm"**
   - Kiểm tra video có xe vi phạm thực sự không
   - Đảm bảo có đèn đỏ rõ ràng trong video
   - Thử điều chỉnh violation_line_y trong code

### Performance tuning:

```python
# Trong app.py, có thể điều chỉnh:
- detection_interval: xử lý mỗi N frames (default: 5)
- violation_line_y: vị trí vạch dừng (default: 60% chiều cao)
- confidence threshold: ngưỡng tin cậy (default: 0.5)
```

## Deployment

### Development:
```bash
python app.py
```

### Production (với Gunicorn):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## API Endpoints

- `GET /` - Trang chủ
- `POST /upload` - Upload video
- `GET /process/<filename>` - Bắt đầu xử lý
- `GET /status/<job_id>` - Lấy trạng thái xử lý (JSON)
- `GET /results/<job_id>` - Xem kết quả
- `POST /search` - Tra cứu vi phạm
- `GET /download/<job_id>` - Download video đã xử lý
- `GET /violation_image/<job_id>/<violation_id>` - Xem ảnh vi phạm

## Database Schema

```sql
CREATE TABLE violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    license_plate TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    frame_number INTEGER,
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER, 
    bbox_w INTEGER,
    bbox_h INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Giới hạn hiện tại

1. **Demo mode**: Nếu không có YOLO/EasyOCR, sẽ tạo dữ liệu demo
2. **Accuracy**: Phụ thuộc vào chất lượng video và điều kiện ánh sáng
3. **Scale**: Thiết kế cho single-server, chưa optimize cho high-traffic
4. **Vietnamese plates**: Tối ưu cho biển số Việt Nam

## Mở rộng trong tương lai

- [ ] Hỗ trợ real-time streaming từ camera IP
- [ ] API REST để tích hợp với hệ thống khác
- [ ] Dashboard analytics và reporting
- [ ] Multi-language support
- [ ] Advanced AI models cho accuracy cao hơn
- [ ] Cloud deployment với auto-scaling

## Tác giả

Được phát triển bởi AI Assistant sử dụng:
- **Backend**: Flask + OpenCV + SQLite
- **AI**: YOLO + EasyOCR
- **Frontend**: Bootstrap 5 + JavaScript
- **Language**: Python 3.8+

## License

MIT License - Sử dụng tự do cho mục đích học tập và nghiên cứu.
