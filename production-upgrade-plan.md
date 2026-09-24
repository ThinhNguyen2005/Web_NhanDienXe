# Kế Hoạch Nâng Cấp Hệ Thống Lên Chuẩn Production

## Goal
Khắc phục các giới hạn của bản MVP (Minimum Viable Product). Chuyển đổi kiến trúc sang hệ thống chịu tải cao, sẵn sàng triển khai thực tế với nhiều camera cùng lúc bằng cách áp dụng **PostgreSQL**, **Celery + Redis**, và hỗ trợ luồng **RTSP**.

---

## Phase 1: Chuyển Đổi Database (SQLite ➔ PostgreSQL)
*Mục tiêu: Loại bỏ tình trạng "Database Locked" khi có nhiều process cùng ghi log vi phạm.*

- [x] **Task 1.1: Cấu hình Môi trường & Thư viện**
  - **Hành động:** Thêm `psycopg2-binary` và `python-dotenv` vào `requirements.txt`.
  - **Chi tiết:** Cập nhật `config.py` để đọc biến môi trường `DATABASE_URI`. Thiết lập fallback sang SQLite nếu không có URI của PostgreSQL (để dễ dev).
- [x] **Task 1.2: Refactor `database.py`**
  - **Hành động:** Viết lại hàm `get_db_connection()`.
  - **Chi tiết:** Thay vì `sqlite3.connect()`, sử dụng `psycopg2.connect()` hoặc SQLAlchemy. Sửa đổi các câu lệnh SQL tĩnh (như `AUTOINCREMENT` đổi thành `SERIAL` trong Postgres, cú pháp tham số `?` đổi thành `%s`).
- [x] **Task 1.3: Cập nhật Docker Compose**
  - **Hành động:** Chỉnh sửa `docker-compose.yml`.
  - **Chi tiết:** Thêm service `db` sử dụng image `postgres:15-alpine`. Cấu hình volumes, environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`). Link service `web` với `db`.

---

## Phase 2: Hệ Thống Hàng Đợi (Threading ➔ Celery + Redis)
*Mục tiêu: Giảm tải cho Web Server, quản lý hàng trăm video/camera mà không bị crash.*

- [x] **Task 2.1: Cài đặt Redis & Celery**
  - **Hành động:** Thêm `celery` và `redis` vào `requirements.txt`.
  - **Chi tiết:** Cập nhật `docker-compose.yml` thêm service `redis` (image `redis:alpine`) và service `worker` (chạy lệnh khởi động celery).
- [x] **Task 2.2: Thiết lập Celery Worker**
  - **Hành động:** Tạo file mới `celery_tasks.py`.
  - **Chi tiết:** Khởi tạo instance Celery kết nối tới Redis (`CELERY_BROKER_URL`). Chuyển hàm `_run_processing_job` (từ `app.py`) thành một `@celery.task`.
- [x] **Task 2.3: Refactor luồng gọi Task trong `app.py`**
  - **Hành động:** Sửa đổi logic trong route bắt đầu xử lý video.
  - **Chi tiết:** Xóa logic `threading.Thread`. Thay vào đó, gọi `process_video_task.delay(filepath, camera_id)`. Xử lý việc lấy trạng thái (Progress) thông qua backend của Celery thay vì biến toàn cục `processing_status`.

---

## Phase 3: Hỗ Trợ Camera Real-time (RTSP Streams)
*Mục tiêu: Đọc trực tiếp từ camera IP giám sát thay vì chỉ upload video MP4.*

- [x] **Task 3.1: Nâng cấp `video_processor.py` cho Live Stream**
  - **Hành động:** Bổ sung logic xử lý stream vô tận (infinite stream).
  - **Chi tiết:** Thêm cơ chế "Frame Dropping" (bỏ qua khung hình nếu xử lý không kịp) để video không bị delay tích lũy (Lag). Bỏ qua bước tính `total_frames` nếu input là RTSP.
- [x] **Task 3.2: Giao diện Quản lý Camera (UI)**
  - **Hành động:** Thêm trang "Quản lý Camera".
  - **Chi tiết:** Tạo form nhập địa chỉ RTSP (vd: `rtsp://admin:123456@192.168.1.100/stream`). Lưu cấu hình camera này vào Database và hiển thị thành danh sách để quản lý.

---

## Done When
- [x] Hệ thống lưu trữ dữ liệu vào PostgreSQL ổn định.
- [x] Khi upload 5 video cùng lúc, Web UI không bị đơ, các video được đưa vào hàng đợi Redis và Celery Worker lần lượt xử lý.
- [x] Có thể nhập link RTSP và hệ thống chạy nhận diện thời gian thực.
