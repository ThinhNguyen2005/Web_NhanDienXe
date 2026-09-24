"""
Module chuyên xử lý phát hiện đèn tín hiệu giao thông.
Sử dụng YOLOv8 (qua thư viện ultralytics) để phát hiện đèn giao thông,
sau đó phân loại màu (đỏ / vàng / xanh) bằng hàm estimate_label.
Cách này tương thích tốt hơn với môi trường dự án hiện tại.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import time
from threading import Lock

# Sử dụng YOLOv8n, một model nhẹ và nhanh phù hợp cho việc phát hiện đèn
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)
_model = None
_model_lock = Lock()
TRAFFIC_LIGHT_CLASS_ID = 9

def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        start = time.perf_counter()
        model_path = os.path.join(PROJECT_DIR, "yolov8n.pt")
        logger.info("Loading traffic light YOLO model lazily: %s", model_path)
        _model = YOLO(model_path)
        logger.info("Traffic light YOLO model loaded in %.2fs", time.perf_counter() - start)
        return _model


def create_feature(rgb_image):
  '''Basic brightness feature, required by Udacity'''
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) # Convert to HSV color space

  sum_brightness = np.sum(hsv[:,:,2]) # Sum the brightness values
  area = 32*32
  avg_brightness = sum_brightness / area # Find the average

  return avg_brightness

def high_saturation_pixels(rgb_image, threshold):
  '''Returns average red and green content from high saturation pixels'''
  high_sat_pixels = []
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  for i in range(32):
    for j in range(32):
      if hsv[i][j][1] > threshold:
        high_sat_pixels.append(rgb_image[i][j])

  if not high_sat_pixels:
    return highest_sat_pixel(rgb_image)

  sum_red = 0
  sum_green = 0
  for pixel in high_sat_pixels:
    sum_red += pixel[0]
    sum_green += pixel[1]

  # TODO: Use sum() instead of manually adding them up
  avg_red = sum_red / len(high_sat_pixels)
  avg_green = sum_green / len(high_sat_pixels) * 0.8 # 0.8 to favor red's chances
  return avg_red, avg_green

def highest_sat_pixel(rgb_image):
  '''Finds the higest saturation pixel, and checks if it has a higher green
  content, or a higher red content'''

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  s = hsv[:,:,1]

  x, y = (np.unravel_index(np.argmax(s), s.shape))
  if rgb_image[x, y, 0] > rgb_image[x,y, 1] * 0.9: # 0.9 to favor red's chances
    return 1, 0 # Red has a higher content
  return 0, 1



def findNonZero(rgb_image):
  rows, cols, _ = rgb_image.shape
  counter = 0

  for row in range(rows):
    for col in range(cols):
      pixel = rgb_image[row, col]
      if sum(pixel) != 0:
        counter = counter + 1

  return counter

def red_green_yellow(rgb_image):
    """
    Phân tích một ảnh RGB đã được crop của đèn giao thông và trả về màu sắc.
    Phiên bản này đã được sửa lỗi tràn số (overflow).

    Args:
        rgb_image: Ảnh crop của đèn giao thông (định dạng RGB).

    Returns:
        str: "red", "green", "yellow", hoặc "unknown".
    """
    if rgb_image is None or rgb_image.size == 0:
        return "unknown"

    # Chuyển ảnh sang không gian màu HSV để dễ dàng phân tích màu sắc
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # --- Định nghĩa các dải màu trong không gian HSV ---

    # Dải màu ĐỎ (có thể gồm 2 khoảng do màu đỏ nằm ở 2 đầu của phổ màu HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Dải màu VÀNG
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Dải màu XANH
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    # --- Tạo mask để lọc các pixel thuộc dải màu tương ứng ---
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2) # Kết hợp 2 mask đỏ

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # --- Đếm số lượng pixel khác không trong mỗi mask ---
    # Đây là cách hiệu quả và an toàn, không gây ra lỗi tràn số
    red_pixels = np.count_nonzero(mask_red)
    yellow_pixels = np.count_nonzero(mask_yellow)
    green_pixels = np.count_nonzero(mask_green)

    # Tạo một dictionary để dễ dàng tìm ra màu có nhiều pixel nhất
    colors = {
        "red": red_pixels,
        "yellow": yellow_pixels,
        "green": green_pixels
    }

    # Đặt một ngưỡng tối thiểu để tránh nhận diện nhiễu
    # Ví dụ: phải có ít nhất 5% tổng số pixel là một màu nào đó mới tính
    min_pixel_threshold = (rgb_image.shape[0] * rgb_image.shape[1]) * 0.05

    # Lọc ra các màu vượt ngưỡng
    valid_colors = {color: count for color, count in colors.items() if count > min_pixel_threshold}

    if not valid_colors:
        return "unknown"

    # Trả về màu có số lượng pixel lớn nhất
    return max(valid_colors, key=valid_colors.get)


def estimate_label(rgb_image):
    """
    Hàm chính để ước tính màu đèn. Gọi hàm red_green_yellow đã được tối ưu.
    """
    return red_green_yellow(rgb_image)

def _classify_light_color(bgr_roi):
    """Phân loại màu đèn từ ROI BGR -> trả về 'red' | 'yellow' | 'green' | 'unknown'."""
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    try:
        # Chuẩn hóa về 32x32 và chuyển sang RGB cho bộ phân loại màu
        roi_resized = cv2.resize(bgr_roi, (32, 32), interpolation=cv2.INTER_AREA)
        rgb_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        color = estimate_label(rgb_roi)
        return color if color in ("red", "yellow", "green") else "unknown"
    except Exception:
        return "unknown"

def detect_traffic_lights_with_color(frame, conf_thresh=0.4):
    """
    Phát hiện đèn giao thông bằng YOLOv8 và phân loại màu.
    Trả về danh sách dict: { 'bbox': (x,y,w,h), 'color': 'red|yellow|green|unknown', 'conf': float }.
    """
    results = get_model()(frame, classes=[TRAFFIC_LIGHT_CLASS_ID], verbose=False)
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

