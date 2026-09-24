# Kế Hoạch Nâng Cấp Web Nhận Diện Xe (Chi tiết cho Agent)

## Goal
Nâng cấp toàn diện dự án: Cải thiện giao diện UI/UX, tối ưu hóa mã nguồn Python (sạch, OOP), và thiết lập môi trường Docker. Các task dưới đây được thiết kế đủ chi tiết để model AI (như Gemini Flash) có thể nhận và xử lý từng phần một cách độc lập.

---

## Phase 1: Refactor Code Backend (Focus: `python-patterns`, Clean Code)

- [x] **Task 1.1: Chuẩn hóa `video_processor.py`**
  - **Hành động:** Đọc kỹ file `video_processor.py`. Gói toàn bộ logic xử lý video vào một class (ví dụ: `VideoProcessor`).
  - **Chi tiết:** Thêm đầy đủ Type Hints (`typing`) cho tham số và giá trị trả về của các hàm. Quản lý tài nguyên an toàn: đảm bảo `cv2.VideoCapture` luôn được `release()` (sử dụng `try/finally` hoặc context manager).
  - **Verify:** Lệnh khởi tạo class không sinh lỗi syntax.

- [x] **Task 1.2: Refactor `app.py` và Xử lý lỗi (Error Handling)**
  - **Hành động:** Review file `app.py`. Kiểm tra các API routes nhận file upload.
  - **Chi tiết:** Đảm bảo mọi khối code gọi hàm xử lý video đều được bọc trong `try/except`. Nếu lỗi xảy ra, phải trả về response dạng JSON với status code phù hợp (ví dụ: `400 Bad Request` hoặc `500 Internal Server Error`) kèm thông báo lỗi rõ ràng. Cấu hình Flask đọc biến môi trường (Environment Variables) thay vì hardcode.
  - **Verify:** Chạy `python app.py` thành công. Gửi một file không hợp lệ sẽ nhận được JSON lỗi báo về.

---

## Phase 2: Nâng Cấp UI/UX (Focus: `frontend-design`, `ui-ux-pro-max`)

- [x] **Task 2.1: Xây dựng hệ thống CSS mới (`static/style.css`)**
  - **Hành động:** Viết lại toàn bộ `style.css`.
  - **Chi tiết:** Sử dụng CSS Variables để định nghĩa bảng màu (ưu tiên giao diện **Dark Mode** hiện đại, kết hợp màu sắc nhận diện nổi bật như Neon Green/Blue). Áp dụng hiệu ứng **Glassmorphism** (nền bán trong suốt, blur) cho các hộp thoại, form upload. Sử dụng font chữ hiện đại (như Inter hoặc Roboto).
  - **Verify:** File CSS hoàn thiện, không có lỗi cú pháp.

- [x] **Task 2.2: Nâng cấp cấu trúc HTML (`templates/index.html`)**
  - **Hành động:** Sửa đổi `index.html` để ăn khớp với CSS mới.
  - **Chi tiết:** Tổ chức lại layout bằng CSS Grid hoặc Flexbox: phân chia rõ khu vực "Upload & Video Streaming" và khu vực "Danh sách vi phạm". Thêm một thành phần "Loading Spinner" hoặc Progress Bar sẽ hiển thị (qua Javascript) khi video đang được AI xử lý.
  - **Verify:** Mở `index.html` trên trình duyệt (hoặc khởi động Flask app) hiển thị đúng cấu trúc mới đẹp mắt.

---

## Phase 3: Tối Ưu Database (Focus: `database-design`)

- [x] **Task 3.1: Thêm Index cho SQLite (`database.py`)**
  - **Hành động:** Xem xét file `database.py` (nơi tạo bảng `violations` hoặc tương đương).
  - **Chi tiết:** Thêm các truy vấn `CREATE INDEX IF NOT EXISTS` cho các cột thường xuyên được dùng để tìm kiếm hoặc lọc. Cụ thể: cột **biển số xe** (license plate) và cột **thời gian** (timestamp). Đảm bảo hàm khởi tạo DB chạy các lệnh này.
  - **Verify:** Xóa DB cũ (nếu an toàn/có bản test), chạy lại `database.py` để tạo DB mới. Dùng sqlite command line kiểm tra schema để xác nhận index đã tồn tại.

---

## Phase 4: Đóng Gói Triển Khai (Focus: `docker-expert`)

- [x] **Task 4.1: Tạo file `.dockerignore`**
  - **Hành động:** Tạo mới file `.dockerignore`.
  - **Chi tiết:** Thêm các thư mục/file không cần thiết vào: `__pycache__/`, `venv/`, `*.db` (nếu muốn DB tạo mới), các thư mục log, và video test có dung lượng lớn.

- [x] **Task 4.2: Viết `Dockerfile`**
  - **Hành động:** Tạo mới file `Dockerfile` ở thư mục gốc.
  - **Chi tiết:** Chọn base image phù hợp (`python:3.10-slim`). Cài đặt các thư viện hệ thống bắt buộc cho OpenCV: `libgl1-mesa-glx`, `libglib2.0-0`. Cài đặt Python packages qua `requirements.txt`. Expose port của Flask (ví dụ 5000) và thiết lập lệnh CMD để chạy app.
  - **Verify:** Chạy lệnh `docker build -t vehicledetect .` thành công không có lỗi tải dependencies.

---

## Done When
- [x] Hoàn thành toàn bộ 4 Phase.
- [x] Toàn bộ mã nguồn chạy trơn tru, không báo lỗi Linter/Syntax.
- [x] Giao diện Dark Mode hiển thị chính xác.
- [x] Build Docker image thành công.
