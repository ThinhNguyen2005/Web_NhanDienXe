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

# Import các module đã được tách
import config
import database
from roi_manager_enhanced import save_rois, load_rois

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask App
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


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
    This keeps the web UI fast and avoids occupying GPU memory just by opening the site.
    """
    global detector
    if detector is not None:
        return detector

    with detector_lock:
        if detector is not None:
            return detector

        logger.info("Initializing AI models lazily...")
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🔥 GPU detected: {device_name} (CUDA {cuda_version}, {vram_gb:.1f}GB VRAM)")
            else:
                logger.info("✓ Using CPU for processing (GPU not available)")
        except ImportError:
            logger.info("✓ PyTorch not available, using CPU fallback")

        from detector_manager import TrafficViolationDetector

        detector = TrafficViolationDetector()
        logger.info("✓ Global detector initialized.")
        return detector


def allowed_file(filename):
    """Kiểm tra đuôi file có được phép hay không."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# ---- Các Route của Flask ----

@app.route('/')
def index():
    """Trang chủ mới của ứng dụng."""
    return moda_landing()

@app.route('/moda')
def moda_landing():
    """RedLight AI Studio React interface."""
    moda_index = os.path.join(config.PROJECT_ROOT, 'static', 'moda', 'index.html')
    if not os.path.exists(moda_index):
        return "Moda landing page has not been built yet. Run: cd frontend/moda && npm install && npm run build", 503
    return send_file(moda_index)

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
    if os.path.exists(config.PROCESSED_FOLDER):
        for file in sorted(os.listdir(config.PROCESSED_FOLDER), reverse=True):
            if file.lower().endswith(tuple(f".{ext}" for ext in config.ALLOWED_EXTENSIONS)):
                videos.append({
                    'name': f"[Processed] {file}",
                    'camera_id': os.path.splitext(file)[0],
                    'path': os.path.join('processed', file).replace('\\', '/'),
                    'source': 'processed'
                })
    return videos

def _row_to_dict(row):
    return dict(row) if hasattr(row, 'keys') else row

def _build_results_payload(job_id):
    with processing_lock:
        status = dict(processing_status.get(job_id, {}))
        memory_results = processing_results.get(job_id, {})

    violations = memory_results.get('violations') if memory_results else None
    if not violations:
        violations = database.get_violations_by_job_id(job_id)

    normalized = []
    for item in violations or []:
        v = _row_to_dict(item)
        violation_id = v.get('track_id') or v.get('id')
        normalized.append({
            **v,
            'image_url': url_for('get_violation_image', job_id=job_id, violation_id=violation_id),
        })

    output_video = status.get('output_video') or database.get_output_video_by_job_id(job_id)
    return {
        'job_id': job_id,
        'status': status.get('status', 'completed' if output_video or normalized else 'unknown'),
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
    if 'write_output_video' in payload:
        options['write_output_video'] = bool(payload.get('write_output_video'))
    return options

def _run_processing_job(job_id, filepath, options=None):
    try:
        from video_processor import VideoProcessor

        processor = VideoProcessor(filepath, get_detector(), options=options)
        processor.process_video(job_id, processing_status, processing_results, processing_lock)
    except Exception as e:
        logger.error(f"Processing job {job_id} failed before processor start: {e}", exc_info=True)
        with processing_lock:
            processing_status[job_id] = {'status': 'error', 'error': str(e)}

def start_processing_job(filepath, camera_id, options=None):
    job_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{camera_id}"
    options = options or resolve_processing_options()
    with processing_lock:
        processing_status[job_id] = {'status': 'starting', 'progress': 0, 'options': options}

    thread = Thread(target=_run_processing_job, args=(job_id, filepath, options), daemon=True)
    thread.start()
    return job_id

@app.route('/api/videos')
def api_videos():
    return jsonify({'success': True, 'videos': list_video_files()})

@app.route('/api/upload', methods=['POST'])
def api_upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'missing video file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f"invalid file type. allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"}), 400

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

@app.route('/api/process/<filename>', methods=['POST'])
def api_process_video(filename):
    payload = request.get_json(silent=True) or {}
    filepath = os.path.join(config.UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'video file not found'}), 404

    camera_id = os.path.splitext(os.path.basename(filepath))[0]
    waiting_zone, violation_zone = load_rois(camera_id)
    if not violation_zone or not waiting_zone:
        waiting_zone, violation_zone = load_rois("default")
    if not violation_zone or not waiting_zone:
        return jsonify({'success': False, 'error': 'missing ROI. Save waiting and violation zones first.'}), 400

    options = resolve_processing_options(payload)
    job_id = start_processing_job(filepath, camera_id, options)
    return jsonify({'success': True, 'job_id': job_id, 'status_url': url_for('get_status', job_id=job_id)})

@app.route('/api/results/<job_id>')
def api_results(job_id):
    return jsonify({'success': True, 'data': _build_results_payload(job_id)})

@app.route('/api/search')
def api_search():
    plate = request.args.get('plate', '').strip()
    if not plate:
        return jsonify({'success': True, 'query': plate, 'violations': []})
    violations = [_row_to_dict(v) for v in database.search_by_plate(plate)]
    return jsonify({'success': True, 'query': plate, 'violations': violations})

@app.route('/api/history')
def api_history():
    return jsonify({'success': True, 'videos': database.get_processed_videos()})

@app.route('/api/history/<job_id>/delete', methods=['POST'])
def api_delete_history(job_id):
    violations_deleted = database.delete_violations_by_job_id(job_id)
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM processed_videos WHERE job_id = ?', (job_id,))
    videos_deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'violations_deleted': bool(violations_deleted), 'videos_deleted': videos_deleted})

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
    
    return send_file(full_path, as_attachment=False)

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
            video_path = os.path.join(config.UPLOAD_FOLDER, filename)
            flash(f'Video "{filename}" đã được tải lên. Vui lòng thiết lập ROI trước khi xử lý.', 'info')
            return redirect(url_for('roi_config', video_for_setup=video_path))
            # return redirect(url_for('process_video_route', filename=filename))
        else:
            flash(f"Định dạng file không hợp lệ. Chỉ chấp nhận: {', '.join(config.ALLOWED_EXTENSIONS)}")
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/process/<filename>')
def process_video_route(filename):
    """
    Kiểm tra cấu hình ROI và bắt đầu tiến trình xử lý video trong một luồng riêng.
    """
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        flash('File không tồn tại.', 'danger')
        return redirect(url_for('upload_video'))


    # Lấy camera_id từ tên file để kiểm tra xem ROI đã tồn tại chưa
    camera_id = os.path.splitext(filename)[0]
    waiting_zone, violation_zone = load_rois(camera_id)

    # Nếu không có ROI cho camera_id cụ thể, thử tải ROI "default"
    if not violation_zone or not waiting_zone:
        logger.info(f"Không tìm thấy ROI cho '{camera_id}', đang thử tải ROI 'default'...")
        waiting_zone, violation_zone = load_rois("default")

    # Nếu vẫn không có ROI nào được cấu hình, chuyển hướng người dùng đến trang thiết lập
    if not violation_zone or not waiting_zone:
        flash(f"Chưa có cài đặt ROI cho video '{filename}'. Vui lòng thiết lập trước khi xử lý.", "warning")
        # Truyền đường dẫn của video để trang roi_config có thể tự động tải nó
        video_path = os.path.join(config.UPLOAD_FOLDER, filename)
        return redirect(url_for('roi_config', video_for_setup=video_path))
    # --- KẾT THÚC LOGIC MỚI ---

    # Nếu có ROI, tiến hành xử lý như bình thường. AI model sẽ được lazy-load trong thread xử lý.
    job_id = start_processing_job(filepath, camera_id)

    return render_template('processing.html', job_id=job_id, filename=filename)

@app.route('/status/<job_id>')
def get_status(job_id):
    """API endpoint để frontend lấy thông tin tiến trình xử lý."""
    with processing_lock:
        status = processing_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/api/live_summary')
def live_summary():
    """Tóm tắt số vi phạm trong phiên live hiện tại theo video_name (query)."""
    video_name = request.args.get('video')
    if not video_name:
        return jsonify({'success': False, 'error': 'missing video'}), 400

    # Lấy processor đang chạy
    try:
        with realtime_lock:
            processor = realtime_processing.get(video_name)
        if processor is None:
            return jsonify({'success': False, 'error': 'no active live session'}), 404
        summary = processor.get_live_summary()
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        logger.error(f"live_summary error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/results/<job_id>')
def show_results(job_id):
    """Hiển thị trang kết quả sau khi xử lý xong hoặc từ lịch sử."""
    # Lấy trạng thái xử lý (nếu còn trong RAM)
    with processing_lock:
        status = processing_status.get(job_id, {})
        results = processing_results.get(job_id, {})

    # Nếu không còn dữ liệu trong RAM, hoặc status chưa completed -> lấy từ DB
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
    else:
        # Status đã completed trong RAM nhưng không có results -> fallback DB
        if not results or not results.get('violations'):
            violations = database.get_violations_by_job_id(job_id)
            status.setdefault('violations_found', len(violations))
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


@app.route('/violation_image/<job_id>/<violation_id>')
def get_violation_image(job_id, violation_id):
    """Hiển thị ảnh vi phạm với cơ chế fallback tương thích dữ liệu cũ."""
    # 1) Thử trực tiếp theo tham số hiện tại (track_id mới hoặc id cũ nếu trùng hợp)
    direct_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
    if os.path.exists(direct_path):
        return send_file(direct_path)

    # 2) Nếu tham số là ID trong DB, tra DB để map sang track_id rồi thử lại
    track_id = None
    if str(violation_id).isdigit():
        try:
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT track_id FROM violations WHERE id = ? AND job_id = ?', (int(violation_id), job_id))
            row = cursor.fetchone()
            conn.close()
            if row and row['track_id'] is not None:
                track_id = row['track_id']
        except Exception as e:
            logger.error(f"Error mapping DB id to track_id for job {job_id}: {e}")

    if track_id is not None:
        mapped_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{track_id}.jpg')
        if os.path.exists(mapped_path):
            return send_file(mapped_path)

    # 3) Fallback cuối: nếu chỉ có đúng 1 ảnh cho job_id thì trả về ảnh đó (giảm 404 dữ liệu cũ)
    try:
        import glob
        candidates = glob.glob(os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_*.jpg'))
        if len(candidates) == 1:
            return send_file(candidates[0])
    except Exception:
        pass

    return "Không tìm thấy ảnh", 404


@app.route('/history')
def history():
    """Trang lịch sử các video đã xử lý."""
    videos = database.get_processed_videos()
    return render_template('history.html', videos=videos)


@app.route('/violation_history/<job_id>')
def violation_history(job_id):
    """Trang hiển thị chi tiết vi phạm của một video cụ thể."""
    violations = database.get_violations_by_job_id(job_id)
    return render_template('results.html', 
                         job_id=job_id,
                         status={'status': 'completed', 'progress': 100},
                         violations=violations)

@app.route('/delete_history/<job_id>', methods=['POST'])
def delete_history(job_id):
    """Xóa toàn bộ lịch sử vi phạm và video đã xử lý của một job."""
    try:
        # Xóa vi phạm
        violations_deleted = database.delete_violations_by_job_id(job_id)
        
        # Xóa video đã xử lý
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM processed_videos WHERE job_id = ?', (job_id,))
        videos_deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if violations_deleted or videos_deleted:
            flash(f"Đã xóa lịch sử cho video {job_id}.", "success")
        else:
            flash(f"Không tìm thấy dữ liệu để xóa cho video {job_id}.", "warning")
    except Exception as e:
        logger.error(f"Lỗi khi xóa lịch sử cho video {job_id}: {e}")
        flash(f"Lỗi khi xóa lịch sử cho video {job_id}.", "danger")
    
    return redirect(url_for('history'))


def generate_frames(video_name):
    """ Xử lý và stream từng frame của video. """
    import cv2
    from video_processor import VideoProcessor

    # Ưu tiên lấy từ uploads, nếu không có thì thử processed
    uploads_path = os.path.join(config.UPLOAD_FOLDER, video_name)
    processed_path = os.path.join(config.PROCESSED_FOLDER, video_name)
    video_path = uploads_path if os.path.exists(uploads_path) else processed_path
    if not os.path.exists(video_path):
        return
    processor = VideoProcessor(video_path, get_detector())
    # Lưu processor để có thể lấy thống kê realtime
    with realtime_lock:
        realtime_processing[video_name] = processor

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    camera_id = os.path.splitext(video_name)[0]
    waiting_zone_pts, violation_zone_pts = load_rois(camera_id)
    if not violation_zone_pts or not waiting_zone_pts:
        waiting_zone_pts, violation_zone_pts = load_rois("default")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = processor.process_single_frame(frame, waiting_zone_pts, violation_zone_pts)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    # Kết thúc: loại bỏ processor khỏi bảng realtime
    with realtime_lock:
        try:
            if realtime_processing.get(video_name) is processor:
                del realtime_processing[video_name]
        except Exception:
            pass

@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    """ Route để trả về luồng video đã xử lý. """
    return Response(generate_frames(video_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    database.init_database()
    logger.info("🚀 Starting Traffic Violation Detection System...")
    logger.info("📱 Access the app at: http://localhost:5000")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False)
