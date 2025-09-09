"""
Hệ thống nhận diện xe vi phạm đèn đỏ và biển số xe
===================================================
Tác giả: AI Assistant & Thịnh Nguyễn
Ngày tạo: 2025-09-06
Framework: Flask + OpenCV + YOLO + EasyOCR

File chính của ứng dụng Flask, chịu trách nhiệm xử lý các request từ web,
quản lý các tiến trình xử lý video và trả về kết quả cho người dùng.
"""
import os
import datetime
import logging
from threading import Thread, Lock

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

# Import các module đã được tách
import config
import database
from detector import TrafficViolationDetector
from video_processor import VideoProcessor

# Cấu hình logging để theo dõi hoạt động của ứng dụng
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# Load cấu hình từ file config.py
app.config.from_object(config)

# Tạo các thư mục cần thiết nếu chúng chưa tồn tại
for folder in [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.VIOLATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ---- Biến toàn cục để quản lý trạng thái xử lý ----
# Dùng dictionary để lưu trạng thái của nhiều tiến trình xử lý đồng thời
processing_status = {}
processing_results = {}
# Lock để đảm bảo an toàn khi nhiều luồng cùng truy cập vào các biến trên
processing_lock = Lock()
# --------------------------------------------------

# Khởi tạo đối tượng detector, chỉ một lần khi ứng dụng khởi động
detector = TrafficViolationDetector()

def allowed_file(filename):
    """Kiểm tra xem định dạng file có được phép hay không."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# ---- Định nghĩa các Route (đường dẫn) của trang web ----

@app.route('/')
def index():
    """Trang chủ của ứng dụng."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """Trang xử lý việc tải video lên."""
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('Không có file video được chọn')
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            flash('Không có file nào được chọn')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Tạo tên file an toàn và duy nhất để tránh trùng lặp
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(config.UPLOAD_FOLDER, unique_filename)
            file.save(filepath)

            flash(f'Video "{filename}" đã được tải lên thành công!')
            # Chuyển hướng đến trang bắt đầu xử lý video
            return redirect(url_for('process_video_route', filename=unique_filename))
        else:
            flash(f'Định dạng file không được hỗ trợ. Chỉ chấp nhận: {", ".join(config.ALLOWED_EXTENSIONS)}')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/process/<filename>')
def process_video_route(filename):
    """
    Bắt đầu một tiến trình xử lý video mới trong một luồng riêng.
    """
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        flash('File không tồn tại.')
        return redirect(url_for('upload_video'))

    # Tạo một ID duy nhất cho tiến trình này (job_id)
    job_id = os.path.splitext(filename)[0]

    # Khởi tạo trạng thái ban đầu cho tiến trình
    with processing_lock:
        processing_status[job_id] = {'status': 'starting', 'progress': 0}

    # Tạo và bắt đầu luồng xử lý video
    processor = VideoProcessor(filepath, detector)
    thread = Thread(target=processor.process_video, args=(job_id, processing_status, processing_results, processing_lock))
    thread.daemon = True # Luồng sẽ tự kết thúc khi chương trình chính kết thúc
    thread.start()

    # Trả về trang hiển thị tiến trình xử lý
    return render_template('processing.html', job_id=job_id, filename=filename)

@app.route('/status/<job_id>')
def get_status(job_id):
    """
    API endpoint để frontend có thể kiểm tra trạng thái xử lý của một video.
    Trả về dữ liệu dạng JSON.
    """
    with processing_lock:
        status = processing_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/results/<job_id>')
def show_results(job_id):
    """Hiển thị trang kết quả sau khi xử lý xong."""
    status = processing_status.get(job_id, {})
    results = processing_results.get(job_id, {})

    if status.get('status') != 'completed':
        flash('Quá trình xử lý video chưa hoàn tất hoặc đã xảy ra lỗi.')
        return redirect(url_for('process_video_route', filename=f"{job_id}.{list(config.ALLOWED_EXTENSIONS)[0]}"))

    return render_template('results.html', 
                         job_id=job_id, 
                         status=status, 
                         violations=results.get('violations', []))

@app.route('/search', methods=['GET', 'POST'])
def search_violations():
    """Trang tra cứu các vi phạm đã được lưu trong cơ sở dữ liệu."""
    violations = []
    search_query = ''

    if request.method == 'POST':
        search_query = request.form.get('license_plate', '').strip().upper()
        if search_query:
            # Gọi hàm tìm kiếm từ module database
            violations = database.search_violations_by_plate(search_query)
            if not violations:
                flash(f'Không tìm thấy vi phạm nào cho biển số "{search_query}"')
        else:
            flash('Vui lòng nhập biển số xe để tìm kiếm.')

    return render_template('search.html', violations=violations, search_query=search_query)

@app.route('/download/<job_id>')
def download_processed_video(job_id):
    """Cho phép người dùng tải xuống video đã được xử lý."""
    results = processing_results.get(job_id, {})
    output_video = results.get('output_video')
    
    if output_video and os.path.exists(output_video):
        return send_file(output_video, as_attachment=True)
    else:
        flash('Không tìm thấy file video đã xử lý.')
        return redirect(url_for('index'))

@app.route('/violation_image/<job_id>/<int:violation_id>')
def get_violation_image(job_id, violation_id):
    """Hiển thị hình ảnh của một vi phạm cụ thể."""
    image_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
    if os.path.exists(image_path):
        return send_file(image_path)
    return "Image not found", 404

# ---- Điểm khởi chạy của ứng dụng ----
if __name__ == '__main__':
    # Khởi tạo cơ sở dữ liệu khi bắt đầu chạy app
    database.init_database()

    logger.info("🚀 Starting Traffic Violation Detection System...")
    logger.info("📱 Access the app at: http://localhost:5000")
    
    # Chạy Flask app ở chế độ debug, cho phép đa luồng
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
