"""
Module chuyên xử lý phát hiện đèn tín hiệu giao thông.
"""
import cv2
import numpy as np

def detect_red_lights(frame):
    """
    Phát hiện đèn tín hiệu màu đỏ trong một khung hình bằng phương pháp nhận dạng màu sắc.

    Args:
        frame (numpy.ndarray): Khung hình đầu vào (định dạng BGR).

    Returns:
        list: Danh sách các bounding box (x, y, w, h) của đèn đỏ được phát hiện.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ngưỡng màu đỏ (2 vùng: gần 0° và gần 180°)
    lower_red1 = np.array([0, 99, 99])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Tạo mask cho cả 2 vùng
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Khử nhiễu
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=50)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Lọc bỏ vùng nhỏ để tránh nhiễu
        # if (w > 5 and h > 5):
        if (w > 7 and h > 7) and (w < 50 and h < 50):
            bboxes.append((x, y, w, h))
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for (x, y, r) in circles[0, :]:
    #         # Tạo bounding box từ hình tròn
    #         bboxes.append((x-r, y-r, 2*r, 2*r))
    return bboxes


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
