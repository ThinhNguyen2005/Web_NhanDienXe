import cv2
import json
from detector_manager import DetectorManager
from database import add_violation

# --- Tích hợp logic từ set_stop_line.py ---

# Biến toàn cục để hỗ trợ việc thiết lập vạch dừng
_points = []
_paused = False
_selected_frame = None

def _mouse_callback(event, x, y, flags, param):
    """
    Hàm xử lý sự kiện click chuột. Chỉ hoạt động khi video đang tạm dừng.
    """
    global _points, _paused
    
    # Chỉ cho phép click khi video đang dừng
    if event == cv2.EVENT_LBUTTONDOWN and _paused:
        if len(_points) < 2:
            _points.append((x, y))
            print(f"Đã chọn điểm số {len(_points)}: ({x}, {y})")
        if len(_points) == 2:
            print("Đã chọn đủ 2 điểm. Nhấn 's' để lưu, 'r' để chọn lại, hoặc SPACE để tiếp tục video.")

def setup_stop_line_interactively(video_path):
    """
    Mở một cửa sổ GUI để người dùng có thể tự vẽ vạch dừng.
    Hàm này được gọi khi 'stop_line.json' không tồn tại.
    """
    global _points, _paused, _selected_frame
    
    # Reset trạng thái mỗi khi hàm được gọi
    _points = []
    _paused = False
    _selected_frame = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video {video_path} để thiết lập vạch dừng.")
        return

    window_name = "THIET LAP VACH DUNG (STOP LINE)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print("\n--- HƯỚNG DẪN THIẾT LẬP VẠCH DỪNG ---")
    print("Do 'stop_line.json' không tồn tại, bạn cần thiết lập ngay bây giờ.")
    print("1. Nhấn 'SPACE' để DỪNG video tại khung hình bạn muốn.")
    print("2. Click chuột trái vào 2 điểm trên ảnh để vẽ vạch dừng.")
    print("3. Nhấn 's' để LƯU và tiếp tục xử lý video.")
    print("4. Nhấn 'r' để CHỌN LẠI điểm.")
    print("5. Nhấn 'q' để THOÁT (sẽ dùng vạch mặc định).")
    print("----------------------------------------\n")

    while cap.isOpened():
        if not _paused:
            ret, frame = cap.read()
            if not ret:
                print("Hết video. Không thể thiết lập vạch dừng.")
                break
            _selected_frame = frame.copy()
            display_frame = frame.copy()
            cv2.putText(display_frame, "Dang Phat - Nhan SPACE de dung", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            display_frame = _selected_frame.copy()
            cv2.putText(display_frame, "Da Dung - Chon 2 diem roi nhan 's' de luu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(_points) > 0:
                for point in _points:
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            if len(_points) == 2:
                cv2.line(display_frame, _points[0], _points[1], (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            print("Đã thoát chế độ thiết lập.")
            break
        elif key == ord(' '):
            _paused = not _paused
            if not _paused:
                _points = [] # Reset points if user resumes video
        
        if _paused:
            if key == ord('r'):
                _points = []
                print("Đã xóa điểm, vui lòng chọn lại.")
            elif key == ord('s'):
                if len(_points) == 2:
                    with open('stop_line.json', 'w') as f:
                        json.dump({'stop_line': _points}, f)
                    print(f"Thành công! Đã lưu vạch dừng vào stop_line.json: {_points}")
                    break
                else:
                    print("Vui lòng chọn đủ 2 điểm trước khi lưu.")

    cap.release()
    cv2.destroyAllWindows()

# --- Kết thúc phần tích hợp ---

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.detector_manager = DetectorManager()
        self.stop_line = self.load_stop_line()

    def load_stop_line(self):
        """
        Tải vạch dừng từ file. Nếu file không tồn tại,
        mở chế độ thiết lập thủ công.
        """
        try:
            with open('stop_line.json', 'r') as f:
                data = json.load(f)
                print("Đã tải vạch dừng từ 'stop_line.json'.")
                return data['stop_line']
        except FileNotFoundError:
            print("Cảnh báo: Không tìm thấy 'stop_line.json'.")
            setup_stop_line_interactively(self.video_path)
            
            # Sau khi thiết lập xong, thử tải lại file
            try:
                with open('stop_line.json', 'r') as f:
                    data = json.load(f)
                    print("Đã tải vạch dừng vừa được thiết lập.")
                    return data['stop_line']
            except FileNotFoundError:
                print("Lỗi: Vẫn không thể đọc file stop_line.json. Sử dụng vạch dừng mặc định.")
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return [(0, height * 2 // 3), (width, height * 2 // 3)]

    def process_video(self):
        """
        Xử lý video, phát hiện và lưu các vi phạm.
        """
        frame_nmr = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Bỏ qua một số frame để tăng tốc độ xử lý
            frame_nmr += 1
            if frame_nmr % 2 != 0:
                continue

            # TODO: Truyền self.stop_line vào detector_manager để xử lý logic vi phạm
            # Ví dụ: results = self.detector_manager.detect(frame, self.stop_line)
            results = self.detector_manager.detect(frame)
            
            # Vẽ vạch dừng lên video để trực quan
            if self.stop_line:
                cv2.line(frame, tuple(self.stop_line[0]), tuple(self.stop_line[1]), (0, 0, 255), 2)

            # TODO: Xử lý logic phát hiện vi phạm dựa trên 'results' và 'self.stop_line'
            # (Phần code này có thể đã nằm trong DetectorManager của bạn)

            # Hiển thị video (tùy chọn, có thể xóa nếu chạy trên server)
            # cv2.imshow('Video Processing', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        print("Đã xử lý xong video.")
        self.cap.release()
        cv2.destroyAllWindows()
