"""
Module quản lý các tương tác với cơ sở dữ liệu SQLite.
Bao gồm các hàm khởi tạo, lưu và truy vấn dữ liệu vi phạm.
"""

import sqlite3
import logging
from config import DATABASE_FILE

# Thiết lập logging
logger = logging.getLogger(__name__)

def init_database():
    """
    Khởi tạo cơ sở dữ liệu và tạo bảng 'violations' nếu chưa tồn tại.
    Hàm này cũng tạo các index để tăng tốc độ truy vấn.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Tạo bảng lưu trữ thông tin vi phạm
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

        # Tạo index trên cột license_plate và job_id để tăng tốc độ tìm kiếm
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_license_plate ON violations(license_plate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_id ON violations(job_id)')

        conn.commit()
        conn.close()
        logger.info("✓ Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")

def save_violations_to_db(job_id, violations):
    """
    Lưu danh sách các vi phạm vào cơ sở dữ liệu.

    Args:
        job_id (str): ID định danh của tiến trình xử lý video.
        violations (list): Danh sách các dictionary chứa thông tin vi phạm.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        for violation in violations:
            cursor.execute(
                '''INSERT INTO violations 
                   (job_id, license_plate, timestamp, frame_number, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (job_id, violation['license_plate'], violation['timestamp'],
                 violation['frame_number'], violation['confidence'],
                 violation['bbox'][0], violation['bbox'][1],
                 violation['bbox'][2], violation['bbox'][3])
            )

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(violations)} violations for job_id {job_id} to database.")
    except sqlite3.Error as e:
        logger.error(f"Database save error for job_id {job_id}: {e}")

def search_violations_by_plate(search_query):
    """
    Tìm kiếm các vi phạm trong cơ sở dữ liệu dựa trên biển số xe.

    Args:
        search_query (str): Biển số xe cần tìm (có thể là một phần).

    Returns:
        list: Danh sách các dictionary chứa thông tin vi phạm tìm được.
    """
    violations = []
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Sử dụng LIKE để tìm kiếm gần đúng
        cursor.execute(
            '''SELECT * FROM violations 
               WHERE license_plate LIKE ? 
               ORDER BY timestamp DESC''',
            (f'%{search_query}%',)
        )
        rows = cursor.fetchall()
        
        # Chuyển đổi kết quả từ tuple sang dictionary để dễ sử dụng
        for row in rows:
            violations.append({
                'id': row[0],
                'job_id': row[1],
                'license_plate': row[2],
                'timestamp': row[3],
                'frame_number': row[4],
                'confidence': row[5],
                'created_at': row[10]
            })

        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Search error for query '{search_query}': {e}")
    
    return violations
