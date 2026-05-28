"""
Module quản lý tất cả các tương tác với cơ sở dữ liệu.
Hỗ trợ cả SQLite (cho development) và PostgreSQL (cho production).
"""
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import config
import urllib.parse as urlparse
import time

logger = logging.getLogger(__name__)

def get_db_connection():
    """Tạo và trả về một kết nối đến CSDL (Postgres hoặc SQLite)."""
    uri = config.DATABASE_URI
    
    if uri.startswith('postgres'):
        # Cấu hình cho PostgreSQL
        url = urlparse.urlparse(uri)
        conn = psycopg2.connect(
            dbname=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        return conn
    else:
        # Cấu hình cho SQLite
        db_file = uri.replace('sqlite:///', '')
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        return conn

def get_cursor(conn):
    """Trả về cursor phù hợp với loại kết nối."""
    if isinstance(conn, psycopg2.extensions.connection):
        return conn.cursor(cursor_factory=RealDictCursor)
    return conn.cursor()

def get_placeholder(conn):
    """Trả về placeholder phù hợp (? cho SQLite, %s cho Postgres)."""
    if isinstance(conn, psycopg2.extensions.connection):
        return '%s'
    return '?'

def init_database(max_retries=5, delay=5):
    """Khởi tạo CSDL và bảng nếu chưa tồn tại. Có retry cho Postgres."""
    for attempt in range(max_retries):
        try:
            conn = get_db_connection()
            cursor = get_cursor(conn)
            
            # Cú pháp tạo bảng có chút khác biệt giữa SQLite và Postgres
            is_postgres = isinstance(conn, psycopg2.extensions.connection)
            auto_inc = "SERIAL PRIMARY KEY" if is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
            
            cursor.execute(f'''          
                CREATE TABLE IF NOT EXISTS violations (
                    id {auto_inc},
                    job_id TEXT NOT NULL,
                    track_id INTEGER,
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
            
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS cameras (
                    id {auto_inc},
                    name TEXT NOT NULL,
                    rtsp_url TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_license_plate ON violations(license_plate)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_id ON violations(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON violations(created_at)')
            
            if not is_postgres:
                # Check for track_id in SQLite (migration)
                cursor.execute("PRAGMA table_info(violations)")
                cols = [row[1] for row in cursor.fetchall()]
                if 'track_id' not in cols:
                    try:
                        cursor.execute('ALTER TABLE violations ADD COLUMN track_id INTEGER')
                    except Exception:
                        pass
            
            conn.commit()
            conn.close()
            logger.info(f"✓ Database ({'PostgreSQL' if is_postgres else 'SQLite'}) initialized successfully.")
            return
        except Exception as e:
            logger.error(f"Error initializing database (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error("Failed to initialize database after maximum retries.")

def save_violations_to_db(job_id, violations):
    """Lưu danh sách các vi phạm vào CSDL."""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        p = get_placeholder(conn)
        
        for v in violations:
            cursor.execute(
                f'''INSERT INTO violations 
                   (job_id, track_id, license_plate, timestamp, frame_number, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                   VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})''',
                (job_id, v.get('track_id'), v['license_plate'], v['timestamp'], v['frame_number'], 
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
    cursor = get_cursor(conn)
    p = get_placeholder(conn)
    
    cursor.execute(
        f"SELECT * FROM violations WHERE license_plate LIKE {p} ORDER BY timestamp DESC",
        (f'%{plate_query}%',)
    )
    violations = cursor.fetchall()
    conn.close()
    return violations

def save_processed_video(job_id, output_video):
    """Lưu tên file video đã xử lý vào CSDL."""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        p = get_placeholder(conn)
        
        # Postgres sử dụng ON CONFLICT thay vì INSERT OR REPLACE
        if isinstance(conn, psycopg2.extensions.connection):
            sql = f"INSERT INTO processed_videos (job_id, output_video) VALUES ({p}, {p}) ON CONFLICT (job_id) DO UPDATE SET output_video = EXCLUDED.output_video"
        else:
            sql = f"INSERT OR REPLACE INTO processed_videos (job_id, output_video) VALUES ({p}, {p})"
            
        cursor.execute(sql, (job_id, output_video))
        conn.commit()
        conn.close()
        logger.info(f"Saved processed video for job_id {job_id}: {output_video}")
    except Exception as e:
        logger.error(f"Error saving processed video: {e}")

def get_output_video_by_job_id(job_id):
    """Lấy tên file video đã xử lý từ CSDL."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    p = get_placeholder(conn)
    cursor.execute(f'SELECT output_video FROM processed_videos WHERE job_id = {p}', (job_id,))
    row = cursor.fetchone()
    conn.close()
    return row['output_video'] if row else None

def get_violations_by_job_id(job_id):
    """Lấy danh sách vi phạm theo job_id."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    p = get_placeholder(conn)
    cursor.execute(f'SELECT * FROM violations WHERE job_id = {p} ORDER BY frame_number', (job_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_violations_by_job_id(job_id):
    """Xóa tất cả các bản ghi vi phạm liên quan đến một job_id."""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        p = get_placeholder(conn)
        cursor.execute(f'DELETE FROM violations WHERE job_id = {p}', (job_id,))
        conn.commit()
        conn.close()
        logger.info(f"Successfully deleted all violation records for job_id {job_id}.")
        return True
    except Exception as e:
        logger.error(f"Database delete error for job_id {job_id}: {e}")
        return False

def get_processed_videos():
    """Lấy danh sách các video đã xử lý từ bảng processed_videos."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    
    # SQLite sử dụng rowid, Postgres có thể sử dụng created_at hoặc tương đương nếu có
    # Để đơn giản, ta sẽ dùng ORDER BY job_id DESC hoặc tương tự
    query = '''
        SELECT pv.job_id, pv.output_video, 
               COUNT(v.id) as violation_count,
               MIN(v.timestamp) as first_violation_time,
               MAX(v.timestamp) as last_violation_time
        FROM processed_videos pv
        LEFT JOIN violations v ON pv.job_id = v.job_id
        GROUP BY pv.job_id, pv.output_video
        ORDER BY pv.job_id DESC
    '''
    cursor.execute(query)
    videos = []
    for row in cursor.fetchall():
        videos.append({
            'job_id': row['job_id'],
            'video_name': row['job_id'],
            'output_video': row['output_video'],
            'processed_video_url': f"/download/{row['job_id']}" if row['output_video'] else None,
            'violation_count': row['violation_count'] or 0,
            'timestamp': row['first_violation_time'] or 'Chưa có vi phạm',
            'violations': (row['violation_count'] or 0) > 0
        })
    conn.close()
    return videos

def get_all_cameras():
    conn = get_db_connection()
    cursor = get_cursor(conn)
    cursor.execute('SELECT * FROM cameras ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_camera(name, rtsp_url):
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        p = get_placeholder(conn)
        cursor.execute('INSERT INTO cameras (name, rtsp_url) VALUES ({p}, {p})'.format(p=p), (name, rtsp_url))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error adding camera: {e}")
        return False

def delete_camera(camera_id):
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        p = get_placeholder(conn)
        cursor.execute('DELETE FROM cameras WHERE id = {p}'.format(p=p), (camera_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting camera: {e}")
        return False
