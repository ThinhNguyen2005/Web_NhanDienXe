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

    # Dải màu đỏ trong không gian HSV
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_lights = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Lọc nhiễu
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.7 <= aspect_ratio <= 1.3:
                red_lights.append((x, y, w, h))

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
