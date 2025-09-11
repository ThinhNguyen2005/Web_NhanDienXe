"""
Module điều phối, đóng vai trò là "nhà quản lý".
Nó import và sử dụng các detector con để xác định vi phạm.
"""
import logging
import cv2

from detector.traffic_light_detector import detect_red_lights
from detector.vehicle_detector import VehicleDetector
from detector.license_plate_detector import LicensePlateDetector

from detector import trafficLightColor

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
        traffic_light_color = trafficLightColor.estimate_label(frame)

        violations_in_frame = []
        if red_lights: # Chỉ kiểm tra vi phạm khi có đèn đỏ
            for vehicle in vehicles:
                if self._check_violation(vehicle, violation_line_y, traffic_light_color):
                    # Trích xuất và gán thông tin biển số ngay khi phát hiện vi phạm
                    _, plate_text, plate_conf = self._extract_and_recognize_plate(frame, vehicle['bbox'])
                    vehicle['license_plate'] = plate_text
                    vehicle['license_plate_confidence'] = plate_conf
                    violations_in_frame.append(vehicle)

        return red_lights, vehicles, violations_in_frame

    # --- Compatibility wrappers for older VideoProcessor API ---
    # def detect_red_lights(self, frame):
    #     """Compatibility: detect red lights in a frame."""
    #     try:
    #         return detect_red_lights(frame)
    #     except Exception:
    #         return []

    def detect_vehicles(self, frame):
        """Compatibility: detect vehicles in a frame."""
        try:
            return self.vehicle_detector.detect(frame)
        except Exception:
            return []

    def check_violation(self, vehicle, red_lights, violation_line_y):
        """Compatibility: check if a single vehicle is violating the stop line.

        Args match older VideoProcessor usage: vehicle dict, list of red_lights, violation_line_y
        """
        if not red_lights:
            return False
        try:
            x, y, w, h = vehicle['bbox']
            vehicle_bottom_y = y + h
            return vehicle_bottom_y > violation_line_y
        except Exception:
            return False

    def extract_license_plate(self, frame, vehicle_bbox):
        """Compatibility: extract license plate image and perform OCR.

        Returns tuple: (plate_image, plate_text, confidence)
        """
        try:
            x, y, w, h = vehicle_bbox
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            vehicle_roi = frame[y:y+h, x:x+w]
            if vehicle_roi.size == 0:
                return None, 'NO_ROI', 0.0
            plate_text, confidence = self.lp_detector.recognize(vehicle_roi)
            return vehicle_roi, plate_text, confidence
        except Exception:
            return None, 'ERROR', 0.0

    def _check_violation(self, vehicle, violation_line_y, traffic_light_color):
    
        # Lấy tọa độ và chiều cao của bounding box
        _, y, _, h = vehicle['bbox']

        # Tính tọa độ y của tâm xe
        # Trong OpenCV, tọa độ y bắt đầu từ 0 ở trên cùng, nên tâm xe vượt qua vạch
        # khi tọa độ y của nó NHỎ HƠN tọa độ y của vạch.
        vehicle_center_y = y + (h / 2)

        # Kiểm tra đồng thời hai điều kiện: xe đã vượt vạch VÀ đèn đang đỏ.
        if vehicle_center_y < violation_line_y and traffic_light_color == "red":
            return True

        return False

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
