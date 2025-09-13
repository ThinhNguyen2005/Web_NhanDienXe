import numpy as np
import cv2


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

