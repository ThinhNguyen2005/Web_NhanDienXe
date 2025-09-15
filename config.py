"""
File cấu hình cho ứng dụng Flask.
Chứa các hằng số và cài đặt chung cho toàn bộ dự án.
"""
import os

# Thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Tên file cơ sở dữ liệu SQLite
DATABASE_FILE = 'traffic_violations.db'

# Thư mục để lưu trữ các file video người dùng tải lên
UPLOAD_FOLDER = 'uploads'

# Thư mục để lưu trữ các video đã qua xử lý (đã vẽ bounding box)
PROCESSED_FOLDER = 'processed'

# Thư mục để lưu trữ hình ảnh của các vi phạm
VIOLATIONS_FOLDER = 'violations'

# Thư mục cấu hình ROI
ROI_CONFIG_FOLDER = os.path.join('config', 'rois')

# Các định dạng file video được phép tải lên
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Khóa bí mật cho Flask session, cần được thay đổi trong môi trường production
SECRET_KEY = 'your-secret-key-change-this-in-production'

# Dung lượng file tối đa cho phép tải lên (500MB)
MAX_CONTENT_LENGTH = 500 * 1024 * 1024

# Cấu hình Flask
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# 1. Chọn Model YOLO:
# 'yolov8n.pt' -> Nano: Nhanh nhất, phù hợp cho GPU yếu.
# 'yolov8s.pt' -> Small: Cân bằng tốt giữa tốc độ và độ chính xác.
# 'yolov8m.pt' -> Medium: Chính xác hơn nhưng chậm hơn.
YOLO_MODEL_PATH = 'yolov8m.pt'

# 2. Chọn Thuật toán Tracking:
# 'botsort.yaml' -> Mặc định, ổn định.
# 'bytetrack.yaml' -> Thường nhanh và tốt hơn trong các cảnh đông đúc.
TRACKER_CONFIG_PATH = 'bytetrack.yaml'

# 3. Tối ưu hóa Kích thước Khung hình:
# Giảm kích thước khung hình trước khi xử lý để tăng tốc độ.
# 640 là một giá trị tốt. Tăng lên 960 nếu cần độ chính xác cao hơn.
# Đặt là None nếu muốn xử lý ở độ phân giải gốc (không khuyến khích).
PROCESSING_FRAME_WIDTH = 640


ENABLE_GPU_OPTIMIZATION = True
# Sử dụng FP16 (Half-precision) để giảm VRAM và tăng tốc.
# Tạm thời tắt FP16 để tránh dtype conflict với RTX 3050
USE_HALF_PRECISION = False
# Một xe sẽ không bị ghi nhận vi phạm lặp lại trong khoảng thời gian này.
VIOLATION_COOLDOWN_SECONDS = 30
# Hệ thống sẽ kiểm tra ROI và đèn tín hiệu mỗi K khung hình.
# Giá trị 3-5 là một sự cân bằng tốt.
CHECK_VIOLATION_INTERVAL = 3

# Số frame tối đa mà một track được phép không xuất hiện trước khi bị xoá
TRACK_TIMEOUT_FRAMES = 30 

# Số frame liên tiếp cần để xác nhận trạng thái đèn (chống nhiễu)
LIGHT_STATE_BUFFER_SIZE = 5

# Số frame ân hạn sau khi phát hiện đèn xanh (tránh false positive khi vừa chuyển đèn)
LIGHT_GREEN_GRACE_FRAMES = 12