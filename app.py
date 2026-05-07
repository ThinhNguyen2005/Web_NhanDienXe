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

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, Response
from werkzeug.utils import secure_filename
from celery.result import AsyncResult

# Import các module đã được tách
import config
import database
import celery_tasks
from roi_manager_enhanced import save_rois, load_rois

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask App
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'error': 'Bad Request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not Found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal Server Error', 'message': str(error)}), 500

# Tạo các thư mục cần thiết
for folder in [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.VIOLATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ---- Biến toàn cục để quản lý trạng thái và kết quả ----
processing_status = {}
processing_results = {}
processing_lock = Lock()
# Bổ sung biến realtime để tránh lỗi NameError khi stream
realtime_processing = {}
realtime_lock = Lock()
detector = None
detector_lock = Lock()

def get_detector():
    """
    Lazy-load AI models only when video processing actually starts.
    """
    global detector
    if detector is not None:
        return detector

    with detector_lock:
        if detector is not None:
            return detector

        logger.info("Initializing AI models lazily...")
        from detector_manager import TrafficViolationDetector
        detector = TrafficViolationDetector()
        return detector

def allowed_file(filename):
    """Kiểm tra đuôi file có được phép hay không."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# ---- Các Route của Flask ----

@app.route('/')
def index():
    """Trang chủ của ứng dụng."""
    return render_template('index.html')

def list_video_files():
    videos = []
    if os.path.exists(config.UPLOAD_FOLDER):
        for file in sorted(os.listdir(config.UPLOAD_FOLDER), reverse=True):
            if file.lower().endswith(tuple(f".{ext}" for ext in config.ALLOWED_EXTENSIONS)):
                videos.append({
                    'name': file,
                    'camera_id': os.path.splitext(file)[0],
                    'path': os.path.join('uploads', file).replace('\\', '/'),
                    'source': 'uploads'
                })
    return videos

def _row_to_dict(row):
    return dict(row) if hasattr(row, 'keys') else row

def _build_results_payload(job_id):
    with processing_lock:
        status = dict(processing_status.get(job_id, {}))
        
    task_id = status.get('task_id')
    if task_id:
        result = AsyncResult(task_id, app=celery_tasks.celery_app)
        if result.ready():
            task_info = result.result if result.successful() else {}
            status.update(task_info)

    violations = database.get_violations_by_job_id(job_id)
    normalized = []
    for item in violations:
        v = _row_to_dict(item)
        violation_id = v.get('track_id') or v.get('id')
        normalized.append({
            **v,
            'image_url': url_for('get_violation_image', job_id=job_id, violation_id=violation_id),
        })

    output_video = status.get('output_video') or database.get_output_video_by_job_id(job_id)
    return {
        'job_id': job_id,
        'status': status.get('status', 'completed' if output_video else 'unknown'),
        'progress': status.get('progress', 100 if output_video else 0),
        'output_video': output_video,
        'download_url': url_for('download_processed_video', job_id=job_id) if output_video else None,
        'violations_found': status.get('violations_found', len(normalized)),
        'violations': normalized
    }

def resolve_processing_options(payload=None):
    payload = payload or {}
    mode = payload.get('mode', 'balanced')
    if mode not in config.PROCESSING_MODES:
        mode = 'balanced'
    options = dict(config.PROCESSING_MODES[mode])
    options['mode'] = mode
    return options

def start_processing_job(filepath, camera_id, options=None):
    job_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{camera_id}"
    options = options or resolve_processing_options()
    
    # Gửi task vào Celery
    task = celery_tasks.process_video_task.delay(filepath, job_id, options)
    
    with processing_lock:
        processing_status[job_id] = {
            'status': 'starting', 
            'progress': 0, 
            'options': options,
            'task_id': task.id
        }
    return job_id

@app.route('/api/upload', methods=['POST'])
def api_upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'no file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        camera_id = os.path.splitext(filename)[0]
        return jsonify({
            'success': True,
            'filename': filename,
            'camera_id': camera_id,
            'path': os.path.join('uploads', filename).replace('\\', '/'),
            'videos': list_video_files()
        })
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process/<filename>', methods=['POST'])
def api_process_video(filename):
    payload = request.get_json(silent=True) or {}
    filepath = os.path.join(config.UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'video file not found'}), 404

    camera_id = os.path.splitext(os.path.basename(filepath))[0]
    options = resolve_processing_options(payload)
    job_id = start_processing_job(filepath, camera_id, options)
    return jsonify({'success': True, 'job_id': job_id, 'status_url': url_for('get_status', job_id=job_id)})

@app.route('/status/<job_id>')
def get_status(job_id):
    """API endpoint để lấy thông tin tiến trình từ Celery."""
    with processing_lock:
        status = processing_status.get(job_id)
    
    if not status:
        return jsonify({'status': 'not_found'}), 404
        
    task_id = status.get('task_id')
    if task_id:
        result = AsyncResult(task_id, app=celery_tasks.celery_app)
        if result.state == 'PROGRESS':
            return jsonify(result.info)
        elif result.state == 'SUCCESS':
            return jsonify(result.result)
        elif result.state == 'FAILURE':
            return jsonify({'status': 'error', 'error': str(result.info)})
        elif result.state == 'PENDING':
            return jsonify({'status': 'starting', 'progress': 0})
            
    return jsonify(status)

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('Không có file nào được chọn', 'danger')
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            flash('Không có file nào được chọn', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            camera_id = os.path.splitext(filename)[0]
            waiting_zone, violation_zone = load_rois(camera_id)
            if not violation_zone or not waiting_zone:
                waiting_zone, violation_zone = load_rois("default")
            
            if not violation_zone or not waiting_zone:
                flash(f"Chưa có cài đặt ROI cho video '{filename}'. Vui lòng thiết lập trước.", "warning")
                return redirect(url_for('roi_config', video_for_setup=filepath))

            job_id = start_processing_job(filepath, camera_id)
            return render_template('processing.html', job_id=job_id, filename=filename)
            
    return render_template('upload.html')

@app.route('/results/<job_id>')
def show_results(job_id):
    data = _build_results_payload(job_id)
    return render_template('results.html', job_id=job_id, status=data, violations=data['violations'])

@app.route('/history')
def history():
    videos = database.get_processed_videos()
    return render_template('history.html', videos=videos)

@app.route('/download/<job_id>')
def download_processed_video(job_id):
    output_video = database.get_output_video_by_job_id(job_id)
    if output_video:
        path = os.path.join(config.PROCESSED_FOLDER, output_video)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
    return "File not found", 404

@app.route('/violation_image/<job_id>/<violation_id>')
def get_violation_image(job_id, violation_id):
    path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
    if os.path.exists(path):
        return send_file(path)
    return "Image not found", 404

@app.route('/api/history/<job_id>/delete', methods=['POST'])
def api_delete_history(job_id):
    database.delete_violations_by_job_id(job_id)
    return jsonify({'success': True})

@app.route('/cameras')
def cameras_page():
    cameras = database.get_all_cameras()
    return render_template('cameras.html', cameras=cameras)

@app.route('/api/cameras/add', methods=['POST'])
def api_add_camera():
    data = request.get_json()
    if database.add_camera(data['name'], data['rtsp_url']):
        return jsonify({'success': True})
    return jsonify({'success': False}), 500

@app.route('/api/cameras/delete/<int:camera_id>', methods=['POST'])
def api_delete_camera(camera_id):
    if database.delete_camera(camera_id):
        return jsonify({'success': True})
    return jsonify({'success': False}), 500

if __name__ == '__main__':
    database.init_database()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
