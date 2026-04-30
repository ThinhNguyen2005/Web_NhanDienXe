"""
Module điều phối, đóng vai trò là "nhà quản lý".
Giai đoạn 3: Hoàn thiện.
"""
import logging
from detector.vehicle_detector import VehicleDetector
from detector.license_plate_detector import LicensePlateDetector
from detector import traffic_light_detector
    
logger = logging.getLogger(__name__)

class TrafficViolationDetector:
    """
    Lớp điều phối chính.
    Cung cấp các hàm dịch vụ đã được tối ưu và nâng cấp.
    """
    def __init__(self):
        """Khởi tạo tất cả các model con cần thiết."""
        logger.info("Initializing all detection modules...")
        self.vehicle_detector = VehicleDetector()
        self.lp_detector = LicensePlateDetector()
        # Alias tương thích cho các callsite cũ
        # lp_recognizer dùng cho OCR, yolo_plate_detector dùng để detect biển số (nếu có)
        self.lp_recognizer = self.lp_detector
        try:
            self.yolo_plate_detector = getattr(self.lp_detector, 'plate_detector', None)
        except Exception:
            self.yolo_plate_detector = None
        logger.info("All detection modules initialized.")

    def get_focused_traffic_light_info(self, frame):
        """
        Nhận diện màu đèn tín hiệu và trả về thông tin của đèn chính (màu và bbox).
        """
        detections = traffic_light_detector.detect_traffic_lights_with_color(frame)
        if not detections:
            return 'unknown', None # Trả về màu và bbox là None

        # Chọn đèn lớn nhất làm "đèn chính"
        main_light = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
        color = main_light.get('color', 'unknown')
        bbox = main_light.get('bbox')

        # Fallback nếu màu không xác định
        if color == 'unknown' and bbox:
            x, y, w, h = bbox
            traffic_light_crop = frame[y:y+h, x:x+w]
            if traffic_light_crop.size > 0:
                color = traffic_light_detector.estimate_label(traffic_light_crop)

        return color, bbox # Trả về cả màu và tọa độ

    def get_traffic_lights_with_color(self, frame):
        """Trả về danh sách đèn giao thông kèm màu từ YOLOv8."""
        return traffic_light_detector.detect_traffic_lights_with_color(frame)

    def extract_and_recognize_plate(self, frame, vehicle_bbox):
        """
        Cắt ROI xe rồi giao toàn bộ pipeline biển số cho LicensePlateDetector.
        Module LPR mới tự phát hiện biển số, deskew, đọc ký tự và fallback khi cần.
        """
        x_v, y_v, w_v, h_v = vehicle_bbox
        x_v, y_v = max(0, x_v), max(0, y_v)
        
        vehicle_roi = frame[y_v : y_v + h_v, x_v : x_v + w_v]

        if vehicle_roi.size == 0:
            logger.warning("Vùng ROI của xe bị rỗng.")
            return None, "NO_ROI", 0.0

        plate_text, confidence = self.lp_recognizer.recognize(vehicle_roi)
        return vehicle_roi, plate_text, confidence
