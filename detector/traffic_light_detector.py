"""
Module chuyên xử lý phát hiện đèn tín hiệu giao thông.
"""
import cv2
import numpy as np
try:
    from . import trafficLightColor
except ImportError:
    import trafficLightColor

from ultralytics import YOLO

import os

# Đường dẫn tới thư mục YOLO (chứa 3 file: coco.names, yolov3.cfg, yolov3.weights)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # detector/
PROJECT_DIR = os.path.dirname(BASE_DIR)                # Web_NhanDienXe/
yolo_path = os.path.join(PROJECT_DIR, "yolo-coco")

# Load class labels
labelsPath = os.path.sep.join([yolo_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Load YOLOv8 model
model = YOLO(os.path.join(PROJECT_DIR, "yolo12s.pt"))

# Hàm phát hiện đèn đỏ bằng YOLOv8
# Trả về danh sách bounding box [(x, y, w, h), ...] của đèn đỏ

def detect_red_lights(frame):
    results = model(frame)
    red_lights = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "traffic light":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0 and trafficLightColor.estimate_label(roi) == "red":
                    red_lights.append((x1, y1, x2-x1, y2-y1))
    return red_lights


def detect_stop_line(frame, roi_height_ratio=0.25, white_thresh_v=200,
                     canny_thresh1=50, canny_thresh2=150,
                     min_line_length_ratio=0.4, max_line_gap=20,
                     slope_thresh=0.2):
    """
    Detect a horizontal white stop line on the road and return its y-coordinate (int).

    Enhanced Strategy:
    - Convert to HSV and threshold bright/white pixels (low saturation, high value).
    - Apply morphological operations to clean the mask
    - Restrict search to the lower part of the frame (configurable by roi_height_ratio).
    - Run Canny edge detector and HoughLinesP to find long horizontal segments.
    - Filter lines by slope (near-horizontal) and choose the one nearest to the bottom
      (largest y) or the longest one.

    Returns:
        int or None: y-coordinate in full-frame coordinates, or None if not found.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    
    # Convert to HSV and threshold for white/bright pixels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, white_thresh_v])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Restrict to lower ROI where stop line usually appears
    roi_y = int(h * (1.0 - roi_height_ratio))
    roi = mask[roi_y:h, :]

    # Edge detection with improved parameters
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # Hough transform to detect lines
    min_line_length = int(w * min_line_length_ratio)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        return None

    candidate_lines = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        slope = abs(dy / float(dx))
        length = np.hypot(dx, dy)
        
        # More strict filtering for horizontal lines
        if slope <= slope_thresh and length >= min_line_length:
            # Check if line spans significant portion of width
            line_width = abs(x2 - x1)
            if line_width >= w * 0.3:  # Line should span at least 30% of frame width
                # map y back to full-frame coordinates (use average y of the segment)
                avg_y = int((y1 + y2) / 2) + roi_y
                candidate_lines.append((avg_y, length, line_width))

    if not candidate_lines:
        return None

    # Prefer the line closest to bottom (largest y), then by width, then by length
    candidate_lines.sort(key=lambda t: (t[0], t[2], t[1]), reverse=True)
    chosen_y = candidate_lines[0][0]

def detect_stop_line_by_traffic_light(frame, traffic_light_pos):
    """
    Phát hiện vạch dừng dựa trên vị trí đèn tín hiệu giao thông.
    Vạch dừng thường nằm ngay dưới đèn tín hiệu.
    """
    if traffic_light_pos is None or frame is None:
        return None
    
    h, w = frame.shape[:2]
    x, y, tw, th = traffic_light_pos
    
    # Vạch dừng thường nằm ở khoảng 50-150 pixels dưới đèn tín hiệu
    search_start = y + th + 20
    search_end = min(h, y + th + 150)
    
    if search_start >= search_end:
        return None
    
    roi = frame[search_start:search_end, :]
    
    # Chuyển grayscale và tìm edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Lọc contours có khả năng là vạch dừng
    stop_line_candidates = []
    for contour in contours:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(contour)
        if w_cnt > w * 0.2 and h_cnt < 10:  # Contour rộng và thấp
            center_y = search_start + y_cnt + h_cnt // 2
            stop_line_candidates.append(center_y)
    
    if stop_line_candidates:
        import statistics
        try:
            return int(statistics.mode(stop_line_candidates))
        except statistics.StatisticsError:
            return int(statistics.median(stop_line_candidates))
    
    return None

def detect_stop_line_enhanced(frame, traffic_light_pos=None, roi_top_ratio=0.5, roi_bottom_ratio=0.95, debug=False):
    """
    Hàm cải tiến để phát hiện vạch dừng, tập trung vào nửa dưới của khung hình.
    Thêm debug mode và cải thiện thuật toán.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]

    # --- THAY ĐỔI QUAN TRỌNG NHẤT ---
    # Bắt buộc vùng tìm kiếm (ROI) phải nằm ở nửa dưới của ảnh
    # Bỏ qua hoàn toàn các vùng sáng ở phía trên
    roi_y_start = int(h * roi_top_ratio)   # Bắt đầu từ 50% chiều cao
    roi_y_end = int(h * roi_bottom_ratio) # Kết thúc ở 95% chiều cao

    roi_frame = frame[roi_y_start:roi_y_end, :]
    
    # Chuyển HSV và lọc màu trắng với ngưỡng linh hoạt hơn
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # Thử nhiều ngưỡng khác nhau để tăng khả năng phát hiện
    white_thresholds = [
        (np.array([0, 0, 150]), np.array([180, 50, 255])),  # Ngưỡng mềm hơn
        (np.array([0, 0, 180]), np.array([180, 40, 255])),  # Ngưỡng gốc
        (np.array([0, 0, 200]), np.array([180, 30, 255])),  # Ngưỡng nghiêm ngặt hơn
    ]
    
    combined_mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in white_thresholds:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Áp dụng morphological operations để làm sạch mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Edge detection với Gaussian blur
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)  # Ngưỡng Canny mềm hơn
    
    # Debug: Hiển thị các bước xử lý
    if debug:
        cv2.imshow('Original ROI', roi_frame)
        cv2.imshow('Combined Mask', combined_mask)
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)
    
    # HoughLinesP với tham số linh hoạt hơn
    min_line_length = int(w * 0.2)  # Giảm yêu cầu độ dài
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,  # Giảm threshold
                            minLineLength=min_line_length, maxLineGap=50)  # Tăng maxLineGap
    
    if lines is None:
        return None

    candidate_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        slope = abs(dy / dx)
        length = np.hypot(dx, dy)
        
        # Lỏng hơn slope threshold và kiểm tra độ rộng
        if slope < 0.3 and length >= min_line_length:  # Tăng slope threshold
            line_width = abs(x2 - x1)
            if line_width >= w * 0.15:  # Giảm yêu cầu độ rộng
                # Cộng lại tọa độ y của ROI
                avg_y = int((y1 + y2) / 2) + roi_y_start
                candidate_lines.append(avg_y)
    
    # Nếu không tìm thấy đường bằng HoughLinesP, thử phương pháp contour-based
    if not candidate_lines:
        # Thử contour-based method như fallback
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        contour_candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if y > roi_frame.shape[0] * 0.3:  # Chỉ xét nửa dưới của ROI
                if cw > ch * 1.5:  # Contour rộng hơn cao
                    center_y = y + ch // 2 + roi_y_start
                    contour_candidates.append(center_y)
        
        if contour_candidates:
            import statistics
            try:
                return int(statistics.mode(contour_candidates))
            except statistics.StatisticsError:
                return int(statistics.median(contour_candidates))
        
        return None

    # Chọn đường kẻ xuất hiện nhiều nhất (ổn định nhất)
    import statistics
    try:
        return int(statistics.mode(candidate_lines))
    except statistics.StatisticsError:
        # Nếu không có mode, lấy median
        return int(statistics.median(candidate_lines))
