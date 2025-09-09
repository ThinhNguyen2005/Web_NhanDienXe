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
