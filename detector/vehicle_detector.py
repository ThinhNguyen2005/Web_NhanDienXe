"""
Module chuyên xử lý phát hiện phương tiện giao thông sử dụng model YOLO.
"""
import logging
import random

logger = logging.getLogger(__name__)

class VehicleDetector:
    """Lớp để phát hiện xe, khởi tạo model YOLO một lần."""

    def __init__(self):
        """Khởi tạo và tải model YOLO."""
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            logger.info("✓ YOLO model loaded for vehicle detection.")
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}. Using fallback.")

    def detect(self, frame):
        """Phát hiện các phương tiện trong một khung hình."""
        if not self.model:
            return self._generate_demo_vehicles(frame)

        try:
            results = self.model(frame, verbose=False)
            vehicles = []
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

            for result in results:
                for box in result.boxes:
                    if int(box.cls) in vehicle_classes and float(box.conf[0]) > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicles.append({
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls)
                        })
            return vehicles
        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}")
            return self._generate_demo_vehicles(frame)

    def _generate_demo_vehicles(self, frame):
        """Tạo dữ liệu xe giả lập khi không có model YOLO."""
        h, w = frame.shape[:2]
        vehicles = []
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, w - 200)
            y = random.randint(h // 2, h - 100)
            vehicles.append({'bbox': (x, y, 150, 80), 'confidence': 0.85, 'class': 2})
        return vehicles
