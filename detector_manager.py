# """
# Module điều phối, đóng vai trò là "nhà quản lý".
# Nó import và sử dụng các detector con để xác định vi phạm.
# """
# import logging

# from detector.traffic_light_detector import detect_red_lights
# from detector.vehicle_detector import VehicleDetector
# from detector.license_plate_detector import LicensePlateDetector

# from detector import trafficLightColor

# logger = logging.getLogger(__name__)

# class TrafficViolationDetector:
#     """
#     Lớp điều phối chính.
#     Khởi tạo và sử dụng các detector con để thực hiện quy trình phát hiện vi phạm.
#     """
#     def __init__(self):
#         """Khởi tạo tất cả các model con cần thiết."""
#         logger.info("Initializing all detection modules...")
#         self.vehicle_detector = VehicleDetector()
#         self.lp_detector = LicensePlateDetector()
#         logger.info("✓ All detection modules initialized.")

#     # def run_detection_on_frame_with_roi(self, frame, waiting_zone_pts, violation_zone_pts):
#     #     """
#     #     Thực hiện toàn bộ quy trình phát hiện trên một khung hình sử dụng ROI.
        
#     #     Args:
#     #         frame (numpy.ndarray): Khung hình cần xử lý.
#     #         waiting_zone_pts: Danh sách điểm [(x,y), ...] tạo vùng chờ
#     #         violation_zone_pts: Danh sách điểm [(x,y), ...] tạo vùng vi phạm
            
#     #     Returns:
#     #         tuple: (red_lights, vehicles, violations_in_frame)
#     #     """
#     #     from roi_manager_enhanced import check_violation_with_roi
        
#     #     # Debug log để verify ROI được truyền đúng
#     #     logger.info(f"ROI Detection: {len(waiting_zone_pts)} waiting, {len(violation_zone_pts)} violation points")
        
#     #     red_lights = detect_red_lights(frame)
#     #     vehicles = self.vehicle_detector.detect(frame)
#     #     traffic_light_color = trafficLightColor.estimate_label(frame)

#     #     violations_in_frame = []
#     #     if red_lights:  # Chỉ kiểm tra vi phạm khi có đèn đỏ
#     #         for vehicle in vehicles:
#     #             if check_violation_with_roi(vehicle['bbox'], violation_zone_pts, waiting_zone_pts, traffic_light_color):
#     #                 # Trích xuất và gán thông tin biển số ngay khi phát hiện vi phạm
#     #                 _, plate_text, plate_conf = self._extract_and_recognize_plate(frame, vehicle['bbox'])
#     #                 vehicle['license_plate'] = plate_text
#     #                 vehicle['license_plate_confidence'] = plate_conf
#     #                 violations_in_frame.append(vehicle)
#     #                 logger.info(f"VIOLATION: {plate_text} (confidence: {plate_conf:.2f})")
#     #     else:
#     #         logger.debug("No red lights detected")

#     #     return red_lights, vehicles, violations_in_frame

#     def run_tracking_and_detection_on_frame(self, frame, waiting_zone_pts, violation_zone_pts):
#         """
#         Thực hiện toàn bộ quy trình theo dõi và phát hiện trên một khung hình.
        
#         Returns:
#             tuple: (red_lights, tracked_vehicles, violations_in_frame)
#         """
#         red_lights = detect_red_lights(frame)
        
#         # Sử dụng phương thức tracking mới
#         tracked_vehicles = self.vehicle_detector.track_vehicles(frame)
        
#         traffic_light_color = trafficLightColor.estimate_label(frame)
#         violations_in_frame = []
        
#         # Chỉ kiểm tra vi phạm khi có đèn đỏ
#         if red_lights or traffic_light_color == "red":
#             from roi_manager_enhanced import check_violation_with_roi
#             for vehicle in tracked_vehicles:
#                 if check_violation_with_roi(vehicle['bbox'], violation_zone_pts, waiting_zone_pts, "red"):
#                     # Trích xuất và gán thông tin biển số ngay khi phát hiện có khả năng vi phạm
#                     _, plate_text, plate_conf = self._extract_and_recognize_plate(frame, vehicle['bbox'])
#                     vehicle['license_plate'] = plate_text
#                     vehicle['license_plate_confidence'] = plate_conf
#                     violations_in_frame.append(vehicle)
        
#         return red_lights, tracked_vehicles, violations_in_frame

# #phát hiện đèn đỏ, phương tiện và kiểm tra vi phạm
#     def run_detection_on_frame(self, frame, violation_line_y):
#         """
#         Thực hiện toàn bộ quy trình phát hiện trên một khung hình.
#         Đây là phương thức chính được gọi từ video_processor.

#         Args:
#             frame (numpy.ndarray): Khung hình cần xử lý.
#             violation_line_y (int): Tọa độ y của vạch dừng.

#         Returns:
#             tuple: (red_lights, vehicles, violations_in_frame)
#         """
#         red_lights = detect_red_lights(frame)
#         vehicles = self.vehicle_detector.detect(frame)
#         traffic_light_color = trafficLightColor.estimate_label(frame)

#         violations_in_frame = []
#         if red_lights: # Chỉ kiểm tra vi phạm khi có đèn đỏ
#             for vehicle in vehicles:
#                 if self._check_violation(vehicle, violation_line_y, traffic_light_color):
#                     # Trích xuất và gán thông tin biển số ngay khi phát hiện vi phạm
#                     _, plate_text, plate_conf = self._extract_and_recognize_plate(frame, vehicle['bbox'])
#                     vehicle['license_plate'] = plate_text
#                     vehicle['license_plate_confidence'] = plate_conf
#                     violations_in_frame.append(vehicle)

#         return red_lights, vehicles, violations_in_frame

#     # --- Compatibility wrappers for older VideoProcessor API ---
#     # def detect_red_lights(self, frame):
#     #     """Compatibility: detect red lights in a frame."""
#     #     try:
#     #         return detect_red_lights(frame)
#     #     except Exception:
#     #         return []
# ## phát hiện phương tiện trong khung hình
#     def detect_vehicles(self, frame):
#         """Tuong thich: phat hien phuong tien trong mot khung hinh."""
#         try:
#             return self.vehicle_detector.detect(frame)
#         except Exception:
#             return []
# # trính xuất và nhận dạng biển số kí tự

#     def extract_license_plate(self, frame, vehicle_bbox):
#         """Compatibility: extract license plate image and perform OCR.

#         Returns tuple: (plate_image, plate_text, confidence)
#         """
#         try:
#             x, y, w, h = vehicle_bbox
#             x, y = max(0, x), max(0, y)
#             w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
#             vehicle_roi = frame[y:y+h, x:x+w]
#             if vehicle_roi.size == 0:
#                 return None, 'NO_ROI', 0.0
#             plate_text, confidence = self.lp_detector.recognize(vehicle_roi)
#             return vehicle_roi, plate_text, confidence
#         except Exception:
#             return None, 'ERROR', 0.0

# #vùng xe vi phạm, chuỗi kí tụ biển số, độ tin cậy
#     def _extract_and_recognize_plate(self, frame, vehicle_bbox):
#         """Trích xuất vùng ảnh của xe và gọi module nhận dạng biển số."""
#         x, y, w, h = vehicle_bbox
#         x, y = max(0, x), max(0, y)
#         w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
#         vehicle_roi = frame[y:y+h, x:x+w]

#         if vehicle_roi.size == 0:
#             return None, "NO_ROI", 0.0
        
#         plate_text, confidence = self.lp_detector.recognize(vehicle_roi)
#         return vehicle_roi, plate_text, confidence





# # # kiểm tra phương tiện có vi phạm vạch dừng không
# #     def check_violation(self, vehicle, red_lights, violation_line_y):
# #         """Compatibility: check if a single vehicle is violating the stop line.

# #         Args match older VideoProcessor usage: vehicle dict, list of red_lights, violation_line_y
# #         """
# #         if not red_lights:
# #             return False
# #         try:
# #             x, y, w, h = vehicle['bbox']
# #             vehicle_bottom_y = y + h
# #             return vehicle_bottom_y > violation_line_y
# #         except Exception:
# #             return False
# #tính toán vị trí xe với vạch dừng
# #     def _check_violation(self, vehicle, violation_line_y, traffic_light_color): 
# #         # Lấy tọa độ và chiều cao của bounding box
# #         _, y, _, h = vehicle['bbox']
# #         # Tính tọa độ y của tâm xe
# #         # Trong OpenCV, tọa độ y bắt đầu từ 0 ở trên cùng, nên tâm xe vượt qua vạch
# #         # khi tọa độ y của nó NHỎ HƠN tọa độ y của vạch.
# #         vehicle_center_y = y + (h / 2)
# #         # Kiểm tra đồng thời hai điều kiện: xe đã vượt vạch VÀ đèn đang đỏ.
# #         if vehicle_center_y < violation_line_y and traffic_light_color == "red":
# #             return True
# #         return False

# detector_manager.py

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

    def get_focused_traffic_light_color(self, frame):
        """
        Nhận diện màu đèn tín hiệu một cách thông minh (ưu tiên kết quả trực tiếp từ YOLOv8 + phân loại màu).
        """
        detections = detect_traffic_lights_with_color(frame)
        if not detections:
            return 'unknown'
        
        # Chọn đèn lớn nhất làm "đèn chính"
        main_light = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
        color = main_light.get('color', 'unknown')
        
        # Fallback: nếu màu không xác định, thử crop và phân loại lại
        if color == 'unknown':
            x, y, w, h = main_light['bbox']
            traffic_light_crop = frame[y:y+h, x:x+w]
            if traffic_light_crop.size > 0:
                color = trafficLightColor.estimate_label(traffic_light_crop)

        logger.info(f"Focused traffic light check: Detected color is '{color}'.")
        return color

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