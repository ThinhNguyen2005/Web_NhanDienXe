"""
Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
Phiên bản HOÀN CHỈNH CUỐI CÙNG: Logic tinh gọn, ổn định và hiệu quả.
"""
import os
import cv2
import datetime
import logging
from threading import Lock

from roi_manager_enhanced import load_rois, visualize_roi, check_violation_with_roi
import config
import database
from detector_manager import TrafficViolationDetector
from detector.traffic_light_detector import draw_traffic_lights

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Xử lý video với pipeline tối ưu và logic chống trùng lặp tinh gọn.
    """

    def __init__(self, video_path: str, detector: TrafficViolationDetector):
        self.video_path = video_path
        self.detector = detector
        self.violations_data = []
        # Tập hợp (set) để lưu trữ các track_id đã vi phạm -> truy xuất cực nhanh
        self.committed_track_ids = set() 
        self.tracked_vehicles_info = {} # Lưu thêm thông tin biển số để vẽ

    def _resize_frame(self, frame, target_width):
        if target_width is None:
            return frame, 1.0
        h, w, _ = frame.shape
        scale = target_width / w
        target_height = int(h * scale)
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Không thể mở file video: {self.video_path}")

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            camera_id = os.path.splitext(os.path.basename(self.video_path))[0]
            waiting_zone_pts, violation_zone_pts = load_rois(camera_id)
            if not violation_zone_pts:
                waiting_zone_pts, violation_zone_pts = load_rois("default")

            output_path = os.path.join(config.PROCESSED_FOLDER, f'processed_{job_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

            frame_count = 0
            while cap.isOpened():
                ret, frame_original = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Tầng 1: Theo dõi siêu nhẹ
                frame_processed, scale = self._resize_frame(frame_original, config.PROCESSING_FRAME_WIDTH)
                tracked_results = self.detector.vehicle_detector.track_vehicles(frame_processed)
                
                # Tầng 2: Kiểm tra vi phạm (logic tinh gọn)
                if frame_count % config.CHECK_VIOLATION_INTERVAL == 0:
                    scaled_violation_zone = [(int(p[0] * scale), int(p[1] * scale)) for p in violation_zone_pts]

                    # Chỉ kiểm tra đèn khi có xe trong khung hình
                    if tracked_results:
                        t_light_color = self.detector.get_focused_traffic_light_color(frame_processed)

                        if t_light_color == "red":
                            for vehicle_data in tracked_results:
                                track_id = vehicle_data['track_id']

                                # Bỏ qua nếu xe đã được xác nhận vi phạm
                                if track_id in self.committed_track_ids:
                                    continue

                                # Kiểm tra xe có trong vùng vi phạm không (đèn đỏ → truyền 'red')
                                is_violating = check_violation_with_roi(
                                    vehicle_data['bbox'], scaled_violation_zone, [], "red"
                                )

                                if is_violating:
                                    # Tầng 3: Xử lý vi phạm
                                    self.process_violation(vehicle_data, frame_original, scale, job_id, frame_count)

                # Vẽ kết quả
                # Vẽ kết quả (bao gồm khung đèn giao thông với màu)
                self.draw_results(tracked_results, frame_original, scale, waiting_zone_pts, violation_zone_pts, out)
                
                with processing_lock:
                    processing_status[job_id] = {'status': 'processing', 
                                                 'progress': (frame_count / self.total_frames) * 100, 
                                                 'violations_found': len(self.violations_data)
                                                 }

            cap.release()
            out.release()
            
            database.save_processed_video(job_id, os.path.basename(output_path))
            database.save_violations_to_db(job_id, self.violations_data)
            with processing_lock:
                processing_status[job_id] = {'status': 'completed', 'output_video': os.path.basename(output_path)}
                processing_results[job_id] = {
                    'violations': self.violations_data,
                    'output_video': output_path
                }
            logger.info(f"Hoàn tất xử lý job {job_id}.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng: {e}", exc_info=True)
            with processing_lock:
                processing_status[job_id] = {'status': 'error', 'error': str(e)}

    def process_violation(self, vehicle_data, frame_original, scale, job_id, frame_count):
        track_id = vehicle_data['track_id']
        logger.info(f"Xác nhận vi phạm cho Track ID {track_id}.")
        
        # Thêm ID vào bộ nhớ để chống trùng lặp
        self.committed_track_ids.add(track_id)
        
        bbox_original = [int(v / scale) for v in vehicle_data['bbox']]
        
        _, plate_text, plate_conf = self.detector.extract_and_recognize_plate(frame_original, bbox_original)
        
        # Lưu lại biển số để vẽ lên các frame sau
        self.tracked_vehicles_info[track_id] = {'license_plate': plate_text}
        
        self._handle_violation_db(track_id, job_id, frame_count, plate_text, plate_conf, bbox_original)
        self._save_violation_image(frame_original, bbox_original, job_id)

    def draw_results(self, tracked_results, frame, scale, waiting_zone, violation_zone, out_writer):
        frame_viz = visualize_roi(frame, waiting_zone, violation_zone)

        # Vẽ đèn giao thông với màu
        try:
            light_dets = self.detector.get_traffic_lights_with_color(frame)
            frame_viz = draw_traffic_lights(frame_viz, light_dets)
        except Exception as e:
            logger.warning(f"Could not draw traffic lights: {e}")

        for vehicle_data in tracked_results:
            track_id = vehicle_data['track_id']
            bbox_original = [int(v / scale) for v in vehicle_data['bbox']]
            x, y, w, h = bbox_original
            
            is_committed = track_id in self.committed_track_ids
            color = (0, 0, 255) if is_committed else (0, 255, 0)
            cv2.rectangle(frame_viz, (x, y), (x + w, y + h), color, 2)
            
            label = f"ID: {track_id}"
            # Lấy thông tin biển số đã lưu để hiển thị
            if is_committed and track_id in self.tracked_vehicles_info:
                label += f" - {self.tracked_vehicles_info[track_id]['license_plate']}"
            cv2.putText(frame_viz, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out_writer.write(frame_viz)

    def _handle_violation_db(self, track_id, job_id, frame_count, plate_text, plate_conf, bbox):
        violation_item = {
            'id': len(self.violations_data) + 1, 'track_id': track_id,
            'timestamp': datetime.datetime.now().isoformat(), 'frame_number': frame_count,
            'license_plate': plate_text, 'confidence': plate_conf, 'bbox': bbox,
        }
        self.violations_data.append(violation_item)

    def _save_violation_image(self, frame, bbox, job_id):
        x, y, w, h = bbox
        cropped_vehicle = frame[y:y+h, x:x+w]
        if cropped_vehicle.size > 0:
            img_id = len(self.violations_data)
            img_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{img_id}.jpg')
            cv2.imwrite(img_path, cropped_vehicle)