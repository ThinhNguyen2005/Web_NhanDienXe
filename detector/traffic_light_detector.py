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
model = YOLO(os.path.join(PROJECT_DIR, "yolov8n.pt"))

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

    Strategy:
    - Convert to HSV and threshold bright/white pixels (low saturation, high value).
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

    # Restrict to lower ROI where stop line usually appears
    roi_y = int(h * (1.0 - roi_height_ratio))
    roi = mask[roi_y:h, :]

    # Edge detection
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
        if slope <= slope_thresh and length >= min_line_length:
            # map y back to full-frame coordinates (use average y of the segment)
            avg_y = int((y1 + y2) / 2) + roi_y
            candidate_lines.append((avg_y, length))

    if not candidate_lines:
        return None

    # Prefer the line closest to bottom (largest y). If tie, prefer longer.
    candidate_lines.sort(key=lambda t: (t[0], t[1]), reverse=True)
    chosen_y = candidate_lines[0][0]

    return int(chosen_y)
