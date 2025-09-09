"""
Module điều phối, đóng vai trò là "nhà quản lý".
Nó import và sử dụng các detector con để xác định vi phạm.
"""
import logging
import cv2

# Import các module logic riêng lẻ từ thư mục 'detectors'
from detector.traffic_light_detector import detect_red_lights
from detector.vehicle_detector import VehicleDetector
from detector.license_plate_detector import LicensePlateDetector

logger = logging.getLogger(__name__)

class TrafficViolationDetector:
    """
    Lớp điều phối chính.
    Khởi tạo và sử dụng các detector con để thực hiện quy trình phát hiện vi phạm.
    """
    def __init__(self):
        """Khởi tạo tất cả các model con cần thiết."""
        logger.info("Initializing all detection modules...")
        self.vehicle_detector = VehicleDetector()
        self.lp_detector = LicensePlateDetector()
        logger.info("✓ All detection modules initialized.")

    def run_detection_on_frame(self, frame, violation_line_y):
        """
        Thực hiện toàn bộ quy trình phát hiện trên một khung hình.
        Đây là phương thức chính được gọi từ video_processor.

        Args:
            frame (numpy.ndarray): Khung hình cần xử lý.
            violation_line_y (int): Tọa độ y của vạch dừng.

        Returns:
            tuple: (red_lights, vehicles, violations_in_frame)
        """
        red_lights = detect_red_lights(frame)
        vehicles = self.vehicle_detector.detect(frame)
        
        violations_in_frame = []
        if red_lights: # Chỉ kiểm tra vi phạm khi có đèn đỏ
            for vehicle in vehicles:
                if self._check_violation(vehicle, violation_line_y):
                    # Trích xuất và gán thông tin biển số ngay khi phát hiện vi phạm
                    _, plate_text, plate_conf = self._extract_and_recognize_plate(frame, vehicle['bbox'])
                    vehicle['license_plate'] = plate_text
                    vehicle['license_plate_confidence'] = plate_conf
                    violations_in_frame.append(vehicle)

        return red_lights, vehicles, violations_in_frame

    def _check_violation(self, vehicle, violation_line_y):
        """Kiểm tra logic vi phạm: xe vượt vạch dừng."""
        _, y, _, h = vehicle['bbox']
        vehicle_bottom_y = y + h
        return vehicle_bottom_y > violation_line_y

    def _extract_and_recognize_plate(self, frame, vehicle_bbox):
        """Trích xuất vùng ảnh của xe và gọi module nhận dạng biển số."""
        x, y, w, h = vehicle_bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        vehicle_roi = frame[y:y+h, x:x+w]

        if vehicle_roi.size == 0:
            return None, "NO_ROI", 0.0
        
        plate_text, confidence = self.lp_detector.recognize(vehicle_roi)
        return vehicle_roi, plate_text, confidence

