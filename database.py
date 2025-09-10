"""
Module quản lý tất cả các tương tác với cơ sở dữ liệu SQLite.
Bao gồm khởi tạo, lưu trữ và các hàm truy vấn cho trang admin.
"""
import sqlite3
import logging
import config

logger = logging.getLogger(__name__)

def get_db_connection():
    """Tạo và trả về một kết nối đến CSDL."""
    conn = sqlite3.connect(config.DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # Giúp truy cập cột theo tên
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

# --- Các hàm cho Admin Dashboard ---

def get_dashboard_stats():
    """Lấy các thông số thống kê cho trang admin."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM violations')
    total_violations = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT license_plate) FROM violations')
    unique_plates = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT job_id) FROM violations')
    total_jobs = cursor.fetchone()[0]
    
    cursor.execute('SELECT license_plate, COUNT(*) as count FROM violations GROUP BY license_plate ORDER BY count DESC LIMIT 10')
    top_violators = cursor.fetchall()
    
    cursor.execute('SELECT * FROM violations ORDER BY created_at DESC LIMIT 20')
    recent_violations = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_violations': total_violations,
        'unique_plates': unique_plates,
        'total_jobs': total_jobs,
        'top_violators': top_violators,
        'recent_violations': recent_violations
    }

def get_all_violations(page=1, per_page=50):
    """Lấy tất cả vi phạm với phân trang."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM violations')
    total = cursor.fetchone()[0]
    
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

