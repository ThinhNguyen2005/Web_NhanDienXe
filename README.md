# Hệ thống nhận diện xe vi phạm đèn đỏ và biển số xe

## Mô tả hệ thống

Hệ thống web sử dụng AI để tự động phát hiện xe vi phạm đèn đỏ và nhận diện biển số xe từ video giao thông.

### Tính năng chính:
- ✅ Upload video giao thông (.mp4, .avi, .mov, .mkv)
- ✅ Phát hiện đèn đỏ bằng computer vision
- ✅ Nhận diện xe vi phạm bằng YOLO
- ✅ Trích xuất biển số xe bằng OCR
- ✅ Lưu trữ vi phạm vào SQLite database
- ✅ Tra cứu vi phạm theo biển số xe
- ✅ Giao diện web thân thiện

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

Hệ thống tự động phát hiện và sử dụng GPU nếu có sẵn (NVIDIA với CUDA). Nếu không có GPU, sẽ tự động chuyển sang CPU.

**Để cài đặt PyTorch với GPU support:**
```bash
# Cho Windows/Linux với CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hoặc cho CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Kiểm tra GPU:**
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Bước 2: Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

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
3. **License Plate Recognition**: EasyOCR (fallback: demo plate generation)

### Performance Optimization:

- Xử lý mỗi 5 frame thay vì tất cả để tăng tốc
- Threading để xử lý background không block UI
- SQLite database với indexing cho tra cứu nhanh
- Resize frame để giảm computational load
- **GPU Acceleration**: Tự động sử dụng GPU (CUDA) nếu có, chuyển sang CPU nếu không

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
