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

# Cấu hình xử lý video
# Xử lý phát hiện mỗi N khung hình (tăng hiệu suất)
DETECTION_INTERVAL = 30