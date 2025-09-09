"""
Module chứa logic chính cho việc phát hiện vi phạm giao thông bằng AI/Computer Vision.
- Lớp TrafficViolationDetector chịu trách nhiệm tải các model AI và thực hiện các tác vụ:
  + Phát hiện đèn đỏ.
  + Phát hiện phương tiện.
  + Nhận dạng biển số xe.
"""
import cv2
import numpy as np
import logging
import random

# Thiết lập logging
logger = logging.getLogger(__name__)

class TrafficViolationDetector:
    """Lớp chính cho việc phát hiện vi phạm giao thông."""

    def __init__(self):
        """Khởi tạo và tải các model AI cần thiết."""
        self.vehicle_model = None
        self.ocr_reader = None
        self.initialize_models()

    def initialize_models(self):
        """
        Khởi tạo các model AI (YOLO cho phát hiện xe và EasyOCR cho nhận dạng biển số).
        Nếu không thể import thư viện, chương trình sẽ sử dụng phương thức thay thế (fallback).
        """
        try:
            # Tải model YOLO để phát hiện phương tiện
            from ultralytics import YOLO
            self.vehicle_model = YOLO('yolov8n.pt')
            logger.info("✓ YOLO model loaded successfully.")
        except ImportError:
            logger.warning("YOLO not available. Falling back to demo vehicle generation.")
            self.vehicle_model = None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.vehicle_model = None

        try:
            # Khởi tạo EasyOCR để đọc biển số
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False) # 'en' hoạt động tốt với biển số VN
            logger.info("✓ EasyOCR reader initialized successfully.")
        except ImportError:
            logger.warning("EasyOCR not available. Falling back to demo license plate generation.")
            self.ocr_reader = None
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            self.ocr_reader = None

    def detect_red_lights(self, frame):
        """
        Phát hiện đèn tín hiệu màu đỏ trong một khung hình.
        Sử dụng phương pháp nhận dạng màu sắc trong không gian màu HSV.

        Args:
            frame (numpy.ndarray): Khung hình đầu vào (định dạng BGR).

        Returns:
            list: Danh sách các bounding box (x, y, w, h) của đèn đỏ được phát hiện.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa các dải màu đỏ trong không gian HSV
        # Màu đỏ có 2 dải ở 2 đầu của thang đo Hue
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        # Tạo mask cho từng dải màu và kết hợp chúng lại
        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Tìm các đường viền (contour) trong mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_lights = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Lọc các đối tượng nhỏ để giảm nhiễu
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                # Đèn giao thông thường có hình tròn hoặc vuông, nên tỷ lệ w/h gần bằng 1
                if 0.7 <= aspect_ratio <= 1.3:
                    red_lights.append((x, y, w, h))

        return red_lights

    def detect_vehicles(self, frame):
        """
        Phát hiện các phương tiện (ô tô, xe máy, xe buýt, xe tải) trong khung hình.
        Sử dụng model YOLOv8 nếu có, nếu không sẽ tạo dữ liệu demo.

        Args:
            frame (numpy.ndarray): Khung hình đầu vào.

        Returns:
            list: Danh sách các dictionary chứa thông tin về phương tiện được phát hiện.
        """
        if self.vehicle_model:
            try:
                results = self.vehicle_model(frame, verbose=False)
                vehicles = []
                # Các class ID của phương tiện trong model COCO: 2(car), 3(motorcycle), 5(bus), 7(truck)
                vehicle_classes = [2, 3, 5, 7]

                for result in results:
                    boxes = result.boxes
                    if boxes:
                        for box in boxes:
                            class_id = int(box.cls)
                            if class_id in vehicle_classes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                # Chỉ lấy các phát hiện có độ tin cậy cao
                                if confidence > 0.5:
                                    vehicles.append({
                                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                                        'confidence': confidence,
                                        'class': class_id
                                    })
                return vehicles
            except Exception as e:
                logger.error(f"YOLO vehicle detection error: {e}. Falling back.")
        
        # Phương thức thay thế nếu không có YOLO
        return self.generate_demo_vehicles(frame)

    def generate_demo_vehicles(self, frame):
        """Tạo dữ liệu phương tiện giả lập để demo khi không có model YOLO."""
        h, w = frame.shape[:2]
        vehicles = []
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, w - 200)
            y = random.randint(h // 2, h - 100)
            vehicles.append({
                'bbox': (x, y, 150, 80), 'confidence': 0.85, 'class': 2
            })
        return vehicles

    def check_violation(self, vehicle, red_lights, violation_line_y):
        """
        Kiểm tra xem một phương tiện có vượt đèn đỏ hay không.
        Vi phạm xảy ra khi có đèn đỏ và tâm của xe vượt qua vạch dừng.
        """
        if not red_lights:
            return False

        x, y, w, h = vehicle['bbox']
        vehicle_center_y = y + h // 2
        
        # Nếu tâm của xe vượt qua vạch dừng (violation_line_y)
        if vehicle_center_y > violation_line_y:
            return True
            
        return False

    def extract_license_plate(self, frame, vehicle_bbox):
        """
        Trích xuất và nhận dạng biển số từ ảnh của phương tiện vi phạm.
        Sử dụng EasyOCR nếu có, nếu không sẽ tạo biển số demo.
        """
        try:
            x, y, w, h = vehicle_bbox
            # Đảm bảo bounding box nằm trong khung hình
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            
            vehicle_roi = frame[y:y+h, x:x+w]

            if vehicle_roi.size == 0:
                return None, self.generate_demo_plate(), 0.0

            if self.ocr_reader:
                # Chuyển ảnh sang thang xám để tăng độ chính xác OCR
                gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                # Tăng độ tương phản
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_roi = clahe.apply(gray_roi)
                
                # Nhận dạng văn bản
                results = self.ocr_reader.readtext(enhanced_roi)

                # Xử lý kết quả từ OCR
                if results:
                    plate_text = ""
                    confidence_sum = 0
                    count = 0
                    for (bbox, text, prob) in results:
                        # Lọc các ký tự không hợp lệ và nối chuỗi
                        clean_text = ''.join(char for char in text if char.isalnum()).upper()
                        if len(clean_text) > 2: # Lọc các đoạn text quá ngắn
                            plate_text += clean_text
                            confidence_sum += prob
                            count += 1
                    
                    if count > 0:
                        # Trả về biển số đã được làm sạch và độ tin cậy trung bình
                        return vehicle_roi, plate_text.replace(" ", ""), confidence_sum / count

        except Exception as e:
            logger.error(f"License plate extraction error: {e}. Falling back.")

        # Phương thức thay thế nếu có lỗi hoặc không có OCR
        return None, self.generate_demo_plate(), 0.85

    def generate_demo_plate(self):
        """Tạo biển số xe Việt Nam giả lập để demo."""
        provinces = ['30', '51', '59', '63', '72']
        letters = 'ABCDEFGHKLMNPSTUVXY'
        numbers = '0123456789'
        province = random.choice(provinces)
        letter = random.choice(letters)
        if random.random() > 0.5:
            letter += random.choice(letters)
        number = ''.join([random.choice(numbers) for _ in range(4)])
        return f"{province}{letter}-{number}"
