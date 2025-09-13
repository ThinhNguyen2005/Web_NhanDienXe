"""
Module điều phối, đóng vai trò là "nhà quản lý".
Giai đoạn 3: Hoàn thiện.
"""
import logging
from detector.vehicle_detector import VehicleDetector
from detector.license_plate_detector import LicensePlateDetector
from detector import trafficLightColor
from detector.traffic_light_detector import (
    detect_traffic_lights_with_color,
)

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
        logger.info("✓ All detection modules initialized.")

    def get_focused_traffic_light_info(self, frame):
        """
        Nhận diện màu đèn tín hiệu và trả về thông tin của đèn chính (màu và bbox).
        """
        detections = detect_traffic_lights_with_color(frame)
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
                color = trafficLightColor.estimate_label(traffic_light_crop)
        
        return color, bbox # Trả về cả màu và tọa độ

    def get_traffic_lights_with_color(self, frame):
        """Trả về danh sách đèn giao thông kèm màu từ YOLOv8."""
        return detect_traffic_lights_with_color(frame)

    def extract_and_recognize_plate(self, frame, vehicle_bbox):
        """Hàm chuyên biệt để trích xuất và nhận dạng biển số."""
        x, y, w, h = vehicle_bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        vehicle_roi = frame[y:y+h, x:x+w]

        if vehicle_roi.size == 0:
            logger.warning("Vùng ROI của xe bị rỗng, không thể nhận dạng biển số.")
            return None, "NO_ROI", 0.0
        
        plate_text, confidence = self.lp_detector.recognize(vehicle_roi)
        return vehicle_roi, plate_text, confidence