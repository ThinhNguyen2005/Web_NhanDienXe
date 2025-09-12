"""
Hệ thống nhận diện xe vi phạm đèn đỏ và biển số xe
===================================================
File chính của ứng dụng Flask.
Chịu trách nhiệm xử lý các request từ web, quản lý các tiến trình xử lý video
và trả về kết quả cho người dùng.
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
from detector_manager import TrafficViolationDetector
from video_processor import VideoProcessor
from roi_manager_enhanced import save_rois, load_rois, auto_detect_roi
import cv2
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask App
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Đăng ký blueprint cho ROI routes
# from routes.roi_routes import roi_routes
# app.register_blueprint(roi_routes)

# Tạo các thư mục cần thiết
for folder in [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.VIOLATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ---- Biến toàn cục để quản lý trạng thái và kết quả ----
processing_status = {}
processing_results = {}
processing_lock = Lock()

# ---- KHỞI TẠO MODEL AI MỘT LẦN DUY NHẤT ----
# Model sẽ được nạp vào bộ nhớ khi ứng dụng khởi động và tái sử dụng cho tất cả các request.
logger.info("Initializing AI models globally...")

# Kiểm tra và log device (GPU/CPU)
try:
    import torch
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    if torch.cuda.is_available():
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        logger.info("✓ Using CPU for processing")
except ImportError:
    logger.info("✓ PyTorch not available, using CPU fallback")

detector = TrafficViolationDetector()
logger.info("✓ Global detector initialized.")


def allowed_file(filename):
    """Kiểm tra đuôi file có được phép hay không."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# ---- Các Route của Flask ----

@app.route('/')
def index():
    """Trang chủ của ứng dụng."""
    return render_template('index.html')

@app.route('/roi_config')
def roi_config():
    """
    Trang cấu hình ROI và vạch dừng
    """
    # Lấy danh sách video đã upload
    videos = []
    
    # Từ thư mục uploads
    if os.path.exists(config.UPLOAD_FOLDER):
        for file in os.listdir(config.UPLOAD_FOLDER):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({
                    'name': file,
                    'path': os.path.join('uploads', file)
                })
    
    # Từ thư mục processed
    if os.path.exists(config.PROCESSED_FOLDER):
        for file in os.listdir(config.PROCESSED_FOLDER):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({
                    'name': f"[Processed] {file}",
                    'path': os.path.join('processed', file)
                })
    
    return render_template('roi_config.html', videos=videos)

@app.route('/api/get_video/<path:video_path>')
def get_video(video_path):
    """
    Trả về file video để xem trong trình duyệt
    """
    if '..' in video_path or video_path.startswith('/'):
        return "Access denied", 403
    
    full_path = os.path.join(config.PROJECT_ROOT, video_path)
    if not os.path.exists(full_path):
        return "File not found", 404
    
    return send_file(full_path)

@app.route('/api/save_roi', methods=['POST'])
def save_roi():
    """
    API để lưu cấu hình ROI
    """
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        camera_id = data.get('camera_id', 'default')
        waiting_zone = data.get('waiting_zone', [])
        violation_zone = data.get('violation_zone', [])
        
        save_rois(camera_id, waiting_zone, violation_zone)
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"Error saving ROI: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/load_roi/<camera_id>')
def load_roi(camera_id):
    """
    API để tải cấu hình ROI
    """
    try:
        waiting_zone, violation_zone = load_rois(camera_id)
        
        return jsonify({
            "success": True,
            "data": {
                "waiting_zone": waiting_zone,
                "violation_zone": violation_zone
            }
        })
    
    except Exception as e:
        logger.error(f"Error loading ROI: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/auto_detect_roi', methods=['POST'])
def auto_detect_roi():
    """
    API để tự động phát hiện và đề xuất vùng ROI từ frame
    """
    try:
        if 'frame' not in request.files:
            return jsonify({"success": False, "error": "No frame provided"})
        
        frame_file = request.files['frame']
        frame_bytes = frame_file.read()
        
        # Chuyển từ bytes sang numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid frame format"})
        
        # Phát hiện vùng ROI tự động
        waiting_zone, violation_zone = auto_detect_roi(frame)
        
        if waiting_zone and violation_zone:
            return jsonify({
                "success": True, 
                "waiting_zone": waiting_zone,
                "violation_zone": violation_zone
            })
        else:
            return jsonify({"success": False, "error": "ROI could not be auto-detected"})
    
    except Exception as e:
        logger.error(f"Error auto-detecting ROI: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """Trang xử lý việc tải video lên."""
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('Không có file nào trong request')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('Chưa chọn file nào để upload')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            return redirect(url_for('process_video_route', filename=filename))
        else:
            flash(f"Định dạng file không hợp lệ. Chỉ chấp nhận: {', '.join(config.ALLOWED_EXTENSIONS)}")
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

    job_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.splitext(filename)[0]}"

    with processing_lock:
        processing_status[job_id] = {'status': 'starting', 'progress': 0}

    # Truyền đối tượng `detector` đã được khởi tạo toàn cục vào VideoProcessor.
    processor = VideoProcessor(filepath, detector)
    thread = Thread(target=processor.process_video, args=(job_id, processing_status, processing_results, processing_lock))
    thread.daemon = True
    thread.start()

    return render_template('processing.html', job_id=job_id, filename=filename)

@app.route('/status/<job_id>')
def get_status(job_id):
    """API endpoint để frontend lấy thông tin tiến trình xử lý."""
    with processing_lock:
        status = processing_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/results/<job_id>')
def show_results(job_id):
    """Hiển thị trang kết quả sau khi xử lý xong hoặc từ lịch sử."""
    # Lấy trạng thái xử lý (nếu còn trong RAM)
    with processing_lock:
        status = processing_status.get(job_id, {})
        results = processing_results.get(job_id, {})

    # Nếu không còn dữ liệu trong RAM, lấy từ database
    if not status or status.get('status') != 'completed':
        # Tạo status giả từ database
        violations = database.get_violations_by_job_id(job_id)
        total_frames = 0  # Nếu muốn, có thể lưu số frame vào processed_videos
        status = {
            'status': 'completed',
            'progress': 100,
            'violations_found': len(violations),
            'output_video': database.get_output_video_by_job_id(job_id),
            'total_frames': total_frames
        }
        results = {'violations': violations}

    return render_template('results.html',
                          job_id=job_id,
                          status=status,
                          violations=results.get('violations', []))

@app.route('/search', methods=['GET', 'POST'])
def search_violations():
    """Trang tìm kiếm các vi phạm trong CSDL."""
    violations = []
    search_query = request.form.get('license_plate', '').strip()

    if request.method == 'POST' and search_query:
        try:
            violations = database.search_by_plate(search_query)
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm biển số '{search_query}': {e}")
            flash('Đã có lỗi xảy ra trong quá trình tìm kiếm.')
            
    return render_template('search.html', violations=violations, search_query=search_query)


@app.route('/download/<job_id>')
def download_processed_video(job_id):
    output_video = database.get_output_video_by_job_id(job_id)
    if output_video:
        output_video_path = os.path.join(config.PROCESSED_FOLDER, output_video)
        if os.path.exists(output_video_path):
            return send_file(output_video_path, as_attachment=True)
    flash(f'Không tìm thấy file video đã xử lý cho ID: {job_id}')
    return redirect(url_for('history'))


@app.route('/violation_image/<job_id>/<int:violation_id>')
def get_violation_image(job_id, violation_id):
    """Hiển thị ảnh của một vi phạm cụ thể."""
    image_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
    if os.path.exists(image_path):
        return send_file(image_path)
    return "Không tìm thấy ảnh", 404


@app.route('/history')
def history():
    """Trang lịch sử các video đã xử lý."""
    videos = database.get_processed_videos()
    return render_template('history.html', videos=videos)


@app.route('/delete_history/<job_id>', methods=['POST'])
def delete_history(job_id):
    """Xóa toàn bộ lịch sử vi phạm của một video."""
    success = database.delete_violations_by_job_id(job_id)
    if success:
        flash(f"Đã xóa lịch sử vi phạm cho video {job_id}.", "success")
    else:
        flash(f"Lỗi khi xóa lịch sử cho video {job_id}.", "danger")
    return redirect(url_for('history'))

# ---- Các Route cho trang Admin ----
# Các route này giờ đây gọi các hàm từ module database.py
@app.route('/admin')
def admin_dashboard():
    """Admin dashboard để xem tổng quan database."""
    try:
        stats = database.get_dashboard_stats()
        return render_template('admin.html', **stats)
    except Exception as e:
        logger.error(f"Lỗi trang admin dashboard: {e}")
        flash('Lỗi khi tải dữ liệu admin')
        return redirect(url_for('index'))

@app.route('/admin/violations')
def admin_violations():
    """Xem tất cả vi phạm với phân trang."""
    page = request.args.get('page', 1, type=int)
    try:
        violations, total, total_pages = database.get_all_violations(page=page, per_page=50)
        return render_template('admin_violations.html',
                             violations=violations,
                             page=page,
                             total_pages=total_pages,
                             total=total)
    except Exception as e:
        logger.error(f"Lỗi trang admin violations: {e}")
        flash('Lỗi khi tải danh sách vi phạm')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/query', methods=['GET', 'POST'])
def admin_query():
    """Thực thi SQL query tùy chỉnh."""
    query = request.form.get('query', 'SELECT * FROM violations LIMIT 10;').strip()
    results, column_names, error = [], [], None

    if request.method == 'POST':
        if not query:
            flash('Vui lòng nhập SQL query')
        else:
            try:
                if not query.upper().strip().startswith('SELECT'):
                    raise ValueError('Chỉ cho phép các câu lệnh SELECT để đảm bảo an toàn')
                results, column_names = database.execute_custom_query(query)
            except Exception as e:
                logger.error(f"Lỗi thực thi query từ admin: {e}")
                error = str(e)
                flash(f'Lỗi SQL: {error}')
    
    return render_template('admin_query.html', 
                           query=query,
                           results=results,
                           column_names=column_names,
                           error=error)


# ---- Điểm khởi chạy của ứng dụng ----
if __name__ == '__main__':
    database.init_database()
    logger.info("🚀 Starting Traffic Violation Detection System...")
    logger.info("📱 Access the app at: http://localhost:5000")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

