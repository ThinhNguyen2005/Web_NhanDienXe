"""
Module quản lý tất cả các tương tác với cơ sở dữ liệu SQLite.
Bao gồm khởi tạo, lưu trữ và các hàm truy vấn cho trang admin và lịch sử.
"""
import sqlite3
import logging
import config

logger = logging.getLogger(__name__)

def get_db_connection():
    """Tạo và trả về một kết nối đến CSDL."""
    conn = sqlite3.connect(config.DATABASE_FILE)
    # Dòng row_factory này rất hữu ích, giúp bạn truy cập dữ liệu như một dictionary
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Khởi tạo CSDL và bảng nếu chưa tồn tại."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''          
            CREATE TABLE IF NOT EXISTS violations (
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
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_videos (
                job_id TEXT PRIMARY KEY,
                output_video TEXT NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_license_plate ON violations(license_plate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_id ON violations(job_id)')
        conn.commit()
        conn.close()
        logger.info("✓ Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def save_violations_to_db(job_id, violations):
    """Lưu danh sách các vi phạm vào CSDL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for v in violations:
            cursor.execute(
                '''INSERT INTO violations 
                   (job_id, license_plate, timestamp, frame_number, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (job_id, v['license_plate'], v['timestamp'], v['frame_number'], 
                 v['confidence'], v['bbox'][0], v['bbox'][1], v['bbox'][2], v['bbox'][3])
            )
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(violations)} violations for job {job_id} to database.")
    except Exception as e:
        logger.error(f"Error saving violations to DB: {e}")

def search_by_plate(plate_query):
    """Tìm kiếm vi phạm theo biển số."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM violations WHERE license_plate LIKE ? ORDER BY timestamp DESC",
        (f'%{plate_query}%',)
    )
    violations = cursor.fetchall()
    conn.close()
    return violations

# --- Các hàm cho Trang Lịch Sử ---

def save_processed_video(job_id, output_video):
    """Lưu tên file video đã xử lý vào CSDL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO processed_videos (job_id, output_video) VALUES (?, ?)',
            (job_id, output_video)
        )
        conn.commit()
        conn.close()
        logger.info(f"Saved processed video for job_id {job_id}: {output_video}")
    except Exception as e:
        logger.error(f"Error saving processed video: {e}")

def get_output_video_by_job_id(job_id):
    """Lấy tên file video đã xử lý từ CSDL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT output_video FROM processed_videos WHERE job_id = ?', (job_id,))
    row = cursor.fetchone()
    conn.close()
    return row['output_video'] if row else None

def get_violations_by_job_id(job_id):
    """Lấy danh sách vi phạm theo job_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM violations WHERE job_id = ? ORDER BY frame_number', (job_id,))
    rows = cursor.fetchall()
    conn.close()
    # Chuyển mỗi row thành dict để template truy cập bằng .timestamp
    violations = []
    for row in rows:
        violations.append(dict(row))
    return violations

def delete_violations_by_job_id(job_id):
    """Xóa tất cả các bản ghi vi phạm liên quan đến một job_id."""
    try:
        # SỬA LỖI: Sử dụng hàm get_db_connection()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM violations WHERE job_id = ?', (job_id,))
        conn.commit()
        conn.close()
        logger.info(f"Successfully deleted all violation records for job_id {job_id}.")
        return True
    except Exception as e:
        logger.error(f"Database delete error for job_id {job_id}: {e}")
        return False

# --- Các hàm cho Admin Dashboard ---

def get_dashboard_stats():
    """Lấy các thông số thống kê cho trang admin."""
    stats = {}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM violations')
        stats['total_violations'] = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(DISTINCT license_plate) as total FROM violations')
        stats['unique_plates'] = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(DISTINCT job_id) as total FROM violations')
        stats['total_jobs'] = cursor.fetchone()['total']
        
        cursor.execute('SELECT license_plate, COUNT(*) as count FROM violations GROUP BY license_plate ORDER BY count DESC LIMIT 10')
        stats['top_violators'] = cursor.fetchall()
        
        cursor.execute('SELECT * FROM violations ORDER BY created_at DESC LIMIT 20')
        stats['recent_violations'] = cursor.fetchall()
        
        conn.close()
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        # Trả về giá trị mặc định nếu có lỗi
        return {
            'total_violations': 0, 'unique_plates': 0, 'total_jobs': 0,
            'top_violators': [], 'recent_violations': []
        }
    return stats

def get_all_violations(page=1, per_page=50):
    """Lấy tất cả vi phạm với phân trang."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as total FROM violations')
    total = cursor.fetchone()['total']
    offset = (page - 1) * per_page
    cursor.execute('SELECT * FROM violations ORDER BY created_at DESC LIMIT ? OFFSET ?', (per_page, offset))
    violations = cursor.fetchall()
    conn.close()
    total_pages = (total + per_page - 1) // per_page
    return violations, total, total_pages

def execute_custom_query(query):
    """Thực thi một câu lệnh SELECT tùy chỉnh một cách an toàn."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return results, column_names

def get_processed_videos():
    """Lấy danh sách các video đã xử lý từ bảng processed_videos."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pv.job_id, pv.output_video, 
               COUNT(v.id) as violation_count,
               MIN(v.timestamp) as first_violation_time,
               MAX(v.timestamp) as last_violation_time
        FROM processed_videos pv
        LEFT JOIN violations v ON pv.job_id = v.job_id
        GROUP BY pv.job_id, pv.output_video
        ORDER BY pv.rowid DESC
    ''')
    videos = []
    for row in cursor.fetchall():
        videos.append({
            'job_id': row['job_id'],
            'video_name': row['job_id'],  # Sử dụng job_id làm tên video
            'output_video': row['output_video'],
            'processed_video_url': f"/download/{row['job_id']}",
            'violation_count': row['violation_count'] or 0,
            'timestamp': row['first_violation_time'] or 'Chưa có vi phạm',
            'violations': row['violation_count'] > 0
        })
    conn.close()
    return videos