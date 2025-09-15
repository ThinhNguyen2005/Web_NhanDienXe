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
from ultralytics import YOLO # <-- IMPORT LẠI YOLO

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
        try:
            self.yolo_plate_detector = YOLO("license_plate_detector.pt")
        except Exception as e:
            logger.warning(f"Không thể tải model YOLO biển số: {e}. Sẽ fallback OCR trực tiếp.")
            self.yolo_plate_detector = None
        
        # 2. OCR (EasyOCR) để đọc ký tự trên biển số
        self.lp_recognizer = LicensePlateDetector()
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
        """
        Hàm chuyên biệt, kết hợp 2 bước:
        1. Dùng YOLO để phát hiện vị trí biển số trong ảnh xe.
        2. Dùng Tesseract để đọc ký tự từ biển số đã được cắt.
        """
        # --- BƯỚC 0: Cắt ảnh của xe ra ---
        x_v, y_v, w_v, h_v = vehicle_bbox
        x_v, y_v = max(0, x_v), max(0, y_v)
        
        vehicle_roi = frame[y_v : y_v + h_v, x_v : x_v + w_v]

        if vehicle_roi.size == 0:
            logger.warning("Vùng ROI của xe bị rỗng.")
            return None, "NO_ROI", 0.0
        
        # Fallback nếu YOLO biển số không khả dụng: OCR trực tiếp
        if getattr(self, 'yolo_plate_detector', None) is None:
            plate_text, confidence = self.lp_recognizer.recognize(vehicle_roi)
            return vehicle_roi, plate_text, confidence
        
        # --- BƯỚC 1: DÙNG YOLO ĐỂ TÌM BIỂN SỐ TRONG ẢNH XE ---
        plate_results = self.yolo_plate_detector(vehicle_roi, verbose=False)
        
        # Tìm biển số có độ tin cậy cao nhất
        best_plate_box = None
        max_conf = 0.0
        for result in plate_results:
            if result.boxes:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf > max_conf:
                        max_conf = conf
                        best_plate_box = box.xyxy.cpu().numpy()[0].astype(int)

        if best_plate_box is None:
            logger.warning(f"Không tìm thấy biển số nào trong xe.")
            return vehicle_roi, "NOT_FOUND", 0.0

        # --- BƯỚC 2: CẮT ẢNH BIỂN SỐ VÀ GỬI CHO BỘ NHẬN DẠNG KÝ TỰ ---
        # Tọa độ của biển số (so với ảnh vehicle_roi)
        x1, y1, x2, y2 = best_plate_box
        
        # Cắt chính xác ảnh biển số
        plate_crop = vehicle_roi[y1:y2, x1:x2]

        if plate_crop.size == 0:
            logger.warning("Ảnh biển số sau khi cắt bị rỗng.")
            return vehicle_roi, "CROP_FAILED", 0.0
        
        # Gửi ảnh biển số đã cắt cho bộ nhận dạng mới và mạnh mẽ
        plate_text, confidence = self.lp_recognizer.recognize(plate_crop)
        
        return vehicle_roi, plate_text, confidence