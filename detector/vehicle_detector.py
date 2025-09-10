"""
Module chuyên xử lý phát hiện phương tiện giao thông sử dụng model YOLO.
"""
import logging


logger = logging.getLogger(__name__)

class VehicleDetector:
    """Lớp để phát hiện xe, khởi tạo model YOLO một lần."""

    def __init__(self):
        """Khởi tạo và tải model YOLO."""
        self.model = None
        try:
            from ultralytics import YOLO
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO('yolov8n.pt').to(device)
            logger.info(f"✓ YOLO model loaded for vehicle detection on {device}.")
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
