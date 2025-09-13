"""
Module chuyên xử lý phát hiện đèn tín hiệu giao thông.
Sử dụng YOLOv8 (qua thư viện ultralytics) để phát hiện đèn giao thông,
sau đó phân loại màu (đỏ / vàng / xanh) bằng mô-đun trafficLightColor.
Cách này tương thích tốt hơn với môi trường dự án hiện tại.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os

try:
    from . import trafficLightColor
except ImportError:
    import trafficLightColor

# Sử dụng YOLOv8n, một model nhẹ và nhanh phù hợp cho việc phát hiện đèn
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model = YOLO(os.path.join(PROJECT_DIR, "yolov8n.pt"))
# Lớp 9 trong COCO dataset là 'traffic light'
TRAFFIC_LIGHT_CLASS_ID = 9

def _classify_light_color(bgr_roi):
    """Phân loại màu đèn từ ROI BGR -> trả về 'red' | 'yellow' | 'green' | 'unknown'."""
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    try:
        # Chuẩn hóa về 32x32 và chuyển sang RGB cho bộ phân loại màu
        roi_resized = cv2.resize(bgr_roi, (32, 32), interpolation=cv2.INTER_AREA)
        rgb_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        color = trafficLightColor.estimate_label(rgb_roi)
        return color if color in ("red", "yellow", "green") else "unknown"
    except Exception:
        return "unknown"

def detect_traffic_lights_with_color(frame, conf_thresh=0.4):
    """
    Phát hiện đèn giao thông bằng YOLOv8 và phân loại màu.
    Trả về danh sách dict: { 'bbox': (x,y,w,h), 'color': 'red|yellow|green|unknown', 'conf': float }.
    """
    results = _model(frame, classes=[TRAFFIC_LIGHT_CLASS_ID], verbose=False)
    detections = []
    if results:
        for r in results:
            for box in r.boxes:
                if box.conf and box.conf[0] > conf_thresh:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    roi = frame[y1:y2, x1:x2]
                    color = _classify_light_color(roi)
                    detections.append({
                        'bbox': (x1, y1, w, h),
                        'color': color,
                        'conf': float(box.conf[0])
                    })
    return detections

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
    return chosen_y

def detect_traffic_lights(frame):
    """
    Hàm tương thích ngược, chỉ phát hiện và trả về bbox của đèn giao thông.
    """
    detections = detect_traffic_lights_with_color(frame)
    return [d['bbox'] for d in detections]

def draw_traffic_lights(frame, detections):
    """
    Vẽ khung và nhãn màu cho các đèn tín hiệu đã phát hiện.
    Tối ưu hóa: Khung đẹp hơn, màu sắc rõ ràng, nhãn thông tin đầy đủ.
    """
    # Bảng màu BGR tối ưu cho đèn giao thông
    color_map = {
        'red': (0, 0, 255),      # Đỏ đậm
        'yellow': (0, 255, 255), # Vàng sáng
        'green': (0, 255, 0),    # Xanh lá
        'unknown': (128, 128, 128) # Xám cho unknown
    }
    
    # Màu nền cho nhãn (để dễ đọc)
    label_bg_colors = {
        'red': (0, 0, 200),
        'yellow': (0, 200, 200),
        'green': (0, 200, 0),
        'unknown': (100, 100, 100)
    }
    
    for i, d in enumerate(detections):
        x, y, w, h = d['bbox']
        color_name = d.get('color', 'unknown')
        confidence = d.get('conf', 0.0)
        
        # Màu khung chính
        bgr_color = color_map.get(color_name, (128, 128, 128))
        bg_color = label_bg_colors.get(color_name, (100, 100, 100))
        
        # Vẽ khung chính với độ dày tùy theo confidence
        thickness = 3 if confidence > 0.7 else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, thickness)
        
        # Vẽ khung phụ bên trong (tạo hiệu ứng đẹp)
        inner_margin = 2
        cv2.rectangle(frame, 
                     (x + inner_margin, y + inner_margin), 
                     (x + w - inner_margin, y + h - inner_margin), 
                     bgr_color, 1)
        
        # Tạo nhãn với background
        label_text = f"🚦 {color_name.upper()}"
        confidence_text = f"{confidence:.2f}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Tính kích thước text
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale - 0.1, font_thickness - 1)
        
        # Vị trí nhãn (phía trên khung)
        label_x = x
        label_y = max(y - 10, text_h + 5)
        
        # Vẽ background cho nhãn
        padding = 5
        cv2.rectangle(frame,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + max(text_w, conf_w) + padding, label_y + padding),
                     bg_color, -1)
        
        # Vẽ text chính
        cv2.putText(frame, label_text, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # Vẽ confidence (nếu có)
        if confidence > 0:
            conf_y = label_y + conf_h + 2
            cv2.putText(frame, confidence_text, (label_x, conf_y), 
                       font, font_scale - 0.1, (200, 200, 200), font_thickness - 1)
    
    return frame

