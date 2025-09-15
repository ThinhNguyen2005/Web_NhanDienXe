"""
Module chuyên xử lý phát hiện và theo dõi phương tiện giao thông.
Sửa lỗi dtype mismatch bằng cách thay đổi thứ tự fuse() và half().
"""
import logging
import config
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    Lớp để phát hiện và theo dõi xe.
    Áp dụng các cấu hình tối ưu hóa từ config.py.
    """

    def __init__(self):
        """Khởi tạo và tải model YOLO với các cờ tối ưu hóa."""
        self.model = None
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.use_half = False # Biến để kiểm soát việc sử dụng FP16

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = config.YOLO_MODEL_PATH
            
            logger.info(f"🔍 GPU Check: torch.cuda.is_available() = {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"🔥 GPU Info: {torch.cuda.get_device_name(0)} (Device {torch.cuda.current_device()})")
            
            logger.info(f"Đang tải model YOLO: '{model_path}' lên thiết bị '{device}'...")
            self.model = YOLO(model_path) # Tải model gốc trước
            
            # Chuyển model sang thiết bị trước khi fuse để tránh lỗi
            self.model.to(device)
            logger.info(f"  ✓ Model đã được chuyển lên {device.upper()}.")

            # Áp dụng các tối ưu hóa GPU trước khi fuse
            if device == 'cuda' and config.ENABLE_GPU_OPTIMIZATION:
                logger.info("Đang bật các tối ưu hóa GPU...")
                torch.backends.cudnn.benchmark = True
                
                if config.USE_HALF_PRECISION:
                    self.model.half() # Chuyển sang FP16
                    self.use_half = True
                    logger.info("  ✓ Độ chính xác bán phần (FP16) đã được bật.")
            
            # Fuse model cuối cùng (có thể bỏ qua nếu gây lỗi)
            try:
                self.model.fuse()
                logger.info("  ✓ Model layers fused.")
            except Exception as fuse_error:
                logger.warning(f"Bỏ qua fusing (không ảnh hưởng hiệu suất nhiều): {fuse_error}")
                # Không raise error, tiếp tục chạy
            
            logger.info(f"✓ Model YOLO đã được tải và cấu hình thành công.")
            logger.info(f"✓ Sử dụng {device.upper()} cho xử lý (GPU {'khả dụng' if device == 'cuda' else 'không khả dụng'})")

        except Exception as e:
            logger.error(f"LỖI: Không thể tải model YOLO. Lỗi: {e}", exc_info=True)
            raise e

    def track_vehicles(self, frame):
        """
        Phát hiện và theo dõi các phương tiện trong một khung hình.
        """
        tracked_vehicles = []
        
        # Tham số half=self.use_half vẫn rất quan trọng để xử lý frame đầu vào
        results = self.model.track(frame, 
                                   persist=True, 
                                   classes=self.vehicle_classes, 
                                   tracker=config.TRACKER_CONFIG_PATH, 
                                   verbose=False,
                                   half=self.use_half)
        
        if results[0].boxes.id is None:
            return []

        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x_center, y_center, w, h = box
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            
            tracked_vehicles.append({
                'bbox': [x, y, int(w), int(h)],
                'track_id': track_id,
                'class_id': class_id
            })
            
        return tracked_vehicles