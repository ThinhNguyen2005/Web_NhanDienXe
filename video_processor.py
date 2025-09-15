"""
Module chứa lớp VideoProcessor với State Machine để phát hiện vi phạm chính xác.
Phiên bản ENHANCED: Logic máy trạng thái theo dõi hành trình xe từ vùng chờ qua vùng vi phạm.
CẢI TIẾN: Thêm logic "Trạng thái đèn bền vững" và cơ chế reset trạng thái.
"""
import os
import cv2
import datetime
import logging
from threading import Lock
import numpy as np
from collections import deque

from roi_manager_enhanced import load_rois, visualize_roi
import config
import database
from detector_manager import TrafficViolationDetector

logger = logging.getLogger(__name__)

class TrackedVehicle:
    """
    Quản lý "Máy trạng thái" và vòng đời của mỗi phương tiện.
    """
    def __init__(self, track_id, bbox, frame_count):
        self.track_id = track_id
        self.bbox = bbox
        self.license_plate = None
        self.state = 'NEUTRAL'
        self.last_seen_frame = frame_count
        # dấu vết để hợp nhất ID (anti-reid glitch)
        self.history = [bbox]

    def update(self, bbox, frame_count):
        self.bbox = bbox
        self.last_seen_frame = frame_count
        self.history.append(bbox)

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    denom = areaA + areaB - inter
    return inter/denom if denom>0 else 0.0

class VideoProcessor:
    """
    Xử lý video với logic "hành trình vi phạm" và quản lý vòng đời track.
    """
    def __init__(self, video_path: str, detector: TrafficViolationDetector):
        self.video_path = video_path
        self.detector = detector
        # Khởi tạo các biến trạng thái
        self.violations_data = []
        self.active_tracks = {}
        self.stable_light_color = 'unknown'
        self.light_color_buffer = deque(maxlen=config.LIGHT_STATE_BUFFER_SIZE)
        # Biến dành cho live stream summary
        self.live_violations = []
        self.live_violation_count = 0
        # Ân hạn khi chuyển sang đèn xanh để tránh gắn cờ sai
        self.no_violation_until_frame = 0
        self.latest_detected_light_color = 'unknown'  # Lưu màu đèn mới nhất

    def reset(self):
        """
        SỬA LỖI: Reset lại toàn bộ trạng thái của bộ xử lý để đảm bảo mỗi lần chạy là độc lập.
        """
        logger.info("Resetting VideoProcessor state for new job.")
        self.violations_data.clear()
        self.active_tracks.clear()
        self.light_color_buffer.clear()
        self.stable_light_color = 'unknown'
        self.latest_detected_light_color = 'unknown'
        self.no_violation_until_frame = 0

    def _update_stable_light_color(self, detected_color: str):
        self.light_color_buffer.append(detected_color)
        if len(self.light_color_buffer) == self.light_color_buffer.maxlen:
            first_color = self.light_color_buffer[0]
            if first_color != 'unknown' and all(color == first_color for color in self.light_color_buffer):
                if self.stable_light_color != first_color:
                    logger.info(f"Light state stabilized to: {first_color.upper()}")
                self.stable_light_color = first_color

    def _resize_frame(self, frame, target_width):
        if target_width is None:
            return frame, 1.0
        h, w, _ = frame.shape
        scale = target_width / w
        target_height = int(h * scale)
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA), scale

    def _cleanup_stale_tracks(self, current_frame_count):
        stale_ids = [track_id for track_id, vehicle in self.active_tracks.items()
                     if current_frame_count - vehicle.last_seen_frame > config.TRACK_TIMEOUT_FRAMES]
        for track_id in stale_ids:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]

    def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
        # SỬA LỖI: Gọi reset() ở đầu mỗi lần xử lý
        self.reset()
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
            if not violation_zone_pts or not waiting_zone_pts:
                logger.info(f"Loading default ROI for camera {camera_id}")
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

                frame_processed, scale = self._resize_frame(frame_original, config.PROCESSING_FRAME_WIDTH)
                tracked_results = self.detector.vehicle_detector.track_vehicles(frame_processed)
                # Hợp nhất track IDs nếu bbox trùng lớn (IoU cao) trong cùng frame
                merged = {}
                iou_thresh = 0.7
                for tr in tracked_results:
                    duplicate_of = None
                    for k, v in merged.items():
                        if _iou(tr['bbox'], v['bbox']) >= iou_thresh:
                            duplicate_of = k
                            break
                    if duplicate_of is None:
                        merged[tr['track_id']] = tr
                tracked_results = list(merged.values())

                for track_data in tracked_results:
                    track_id = track_data['track_id']
                    if track_id in self.active_tracks:
                        self.active_tracks[track_id].update(track_data['bbox'], frame_count)
                    else:
                        self.active_tracks[track_id] = TrackedVehicle(track_id, track_data['bbox'], frame_count)

                self._cleanup_stale_tracks(frame_count)

                # --- CẬP NHẬT TRẠNG THÁI ĐÈN ---
                detected_light_color, light_bbox_scaled = self.detector.get_focused_traffic_light_info(frame_processed)
                self.latest_detected_light_color = detected_light_color  # Lưu cho bước kiểm tra vi phạm
                self._update_stable_light_color(detected_light_color)
                # Kích hoạt ân hạn nếu vừa thấy đèn xanh
                if detected_light_color == 'green':
                    self.no_violation_until_frame = max(self.no_violation_until_frame, frame_count + config.LIGHT_GREEN_GRACE_FRAMES)
                
                light_bbox_original = None
                if light_bbox_scaled is not None and scale > 0:
                    light_bbox_original = [int(v / scale) for v in light_bbox_scaled]

                if frame_count % config.CHECK_VIOLATION_INTERVAL == 0 and self.active_tracks:
                    scaled_violation_zone = np.array([(int(p[0] * scale), int(p[1] * scale)) for p in violation_zone_pts], dtype=np.int32)
                    scaled_waiting_zone = np.array([(int(p[0] * scale), int(p[1] * scale)) for p in waiting_zone_pts], dtype=np.int32)

                    for vehicle in list(self.active_tracks.values()):
                        if vehicle.state in ['COMMITTED_VIOLATION', 'PASSED_LEGALLY']:
                            continue
                        
                        x, y, w, h = vehicle.bbox
                        vehicle_point = (int(x + w / 2), int(y + h))
                        is_in_waiting = cv2.pointPolygonTest(scaled_waiting_zone, vehicle_point, False) >= 0
                        is_in_violation = cv2.pointPolygonTest(scaled_violation_zone, vehicle_point, False) >= 0

                        if vehicle.state == 'NEUTRAL' and is_in_waiting:
                            vehicle.state = 'IN_WAITING_ZONE'
                        elif vehicle.state == 'IN_WAITING_ZONE' and is_in_violation:
                            # Chỉ đánh cờ khi đèn chắc chắn ĐỎ hoặc VÀNG và KHÔNG có tín hiệu XANH vừa phát hiện
                            if (self.stable_light_color in ['red', 'yellow']) \
                                and self.latest_detected_light_color != 'green' \
                                and frame_count >= self.no_violation_until_frame:
                                vehicle.state = 'COMMITTED_VIOLATION' 
                                logger.info(f"Vehicle {vehicle.track_id} COMMITTED VIOLATION (Stable light: RED)")
                                self.process_violation(vehicle, frame_original, scale, job_id, frame_count)
                            else:
                                vehicle.state = 'PASSED_LEGALLY'
                        elif vehicle.state == 'NEUTRAL' and is_in_violation:
                            vehicle.state = 'PASSED_LEGALLY'

                self.draw_results(frame_original, scale, waiting_zone_pts, violation_zone_pts, out, light_bbox_original)

                with processing_lock:
                    processing_status[job_id] = {
                        'status': 'processing',
                        'progress': (frame_count / self.total_frames) * 100,
                        'violations_found': len(self.violations_data)
                    }

            cap.release()
            out.release()
            database.save_processed_video(job_id, os.path.basename(output_path))
            database.save_violations_to_db(job_id, self.violations_data)
            with processing_lock:
                processing_status[job_id] = {
                    'status': 'completed',
                    'output_video': os.path.basename(output_path),
                    'violations_found': len(self.violations_data)
                }
                # Lưu kết quả vào bộ nhớ để trang results dùng ngay, tránh hiển thị 0
                processing_results[job_id] = {'violations': list(self.violations_data)}
            logger.info(f"Hoàn tất xử lý job {job_id}.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng: {e}", exc_info=True)
            with processing_lock:
                processing_status[job_id] = {'status': 'error', 'error': str(e)}

    def process_violation(self, vehicle, frame_original, scale, job_id, frame_count):
        bbox_original = [int(v / scale) for v in vehicle.bbox]
        _, plate_text, plate_conf = self.detector.extract_and_recognize_plate(frame_original, bbox_original)
        vehicle.license_plate = plate_text
        self._handle_violation_db(vehicle.track_id, job_id, frame_count, plate_text, plate_conf, bbox_original)
        self._save_violation_image(frame_original, bbox_original, job_id, vehicle.track_id)

    def draw_results(self, frame, scale, waiting_zone, violation_zone, out_writer, light_bbox=None):
        frame_viz = visualize_roi(frame, waiting_zone, violation_zone)
        
        light_info_map = {
            'red':    {'color': (0, 0, 255),   'label': 'DO'},
            'green':  {'color': (0, 255, 0),   'label': 'XANH'},
            'yellow': {'color': (0, 255, 255), 'label': 'VANG'}
        }
        light_info = light_info_map.get(self.stable_light_color)

        if light_bbox is not None and light_info:
            x_light, y_light, w_light, h_light = light_bbox
            color = light_info['color']
            label = light_info['label']
            cv2.rectangle(frame_viz, (x_light, y_light), (x_light + w_light, y_light + h_light), color, 2)
            cv2.putText(frame_viz, label, (x_light, y_light - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame_viz, f"Light: {self.stable_light_color.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        for vehicle in self.active_tracks.values():
            if vehicle.state == 'NEUTRAL':
                continue

            bbox_original = [int(v / scale) for v in vehicle.bbox]
            x, y, w, h = bbox_original

            state_info = {
                'IN_WAITING_ZONE':   {'color': (255, 165, 0), 'thickness': 1, 'label': 'DANG CHO'},
                'PASSED_LEGALLY':    {'color': (0, 255, 0),   'thickness': 1, 'label': 'HOP LE'},
                'COMMITTED_VIOLATION': {'color': (0, 0, 255),   'thickness': 2, 'label': 'VI PHAM'}
            }
            
            info = state_info.get(vehicle.state)
            if not info:
                continue

            color = info['color']
            thickness = info['thickness']
            label_text = info['label']

            cv2.rectangle(frame_viz, (x, y), (x + w, y + h), color, thickness)
            
            plate_text = f" - {vehicle.license_plate}" if vehicle.license_plate else ""
            full_label = f"ID:{vehicle.track_id} [{label_text}]{plate_text}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 if vehicle.state == 'COMMITTED_VIOLATION' else 0.4
            font_thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(full_label, font, font_scale, font_thickness)
            label_pos = (x, y - 5)
            
            # CẢI TIẾN: Chọn màu chữ dựa trên màu nền để dễ đọc
            text_color = (255, 255, 255) # Mặc định là màu trắng
            if label_text in ['HOP LE', 'DANG CHO']:
                text_color = (0, 0, 0) # Đổi thành màu đen cho nền sáng

            cv2.rectangle(frame_viz, (label_pos[0], label_pos[1] - text_h - baseline), (label_pos[0] + text_w, label_pos[1]), color, -1)
            cv2.putText(frame_viz, full_label, (label_pos[0], label_pos[1] - baseline // 2), font, font_scale, text_color, font_thickness)

        if out_writer is not None:
            out_writer.write(frame_viz)
        else:
            return frame_viz

    # === HÀM MỚI ĐỂ XỬ LÝ LIVE STREAM ===
    def process_single_frame(self, frame_original, waiting_zone_pts, violation_zone_pts):
        """
        Xử lý một khung hình duy nhất và trả về khung hình đã được vẽ kết quả.
        Hàm này được thiết kế để phục vụ cho việc streaming trực tiếp.
        """
        # Tăng frame count (quan trọng cho việc dọn dẹp track cũ)
        self.frame_count = getattr(self, 'frame_count', 0) + 1

        # Resize frame để xử lý
        frame_processed, scale = self._resize_frame(frame_original, config.PROCESSING_FRAME_WIDTH)

        # Theo dõi xe
        tracked_results = self.detector.vehicle_detector.track_vehicles(frame_processed)
        # Hợp nhất track IDs cho live
        merged = {}
        iou_thresh = 0.7
        for tr in tracked_results:
            duplicate_of = None
            for k, v in merged.items():
                if _iou(tr['bbox'], v['bbox']) >= iou_thresh:
                    duplicate_of = k
                    break
            if duplicate_of is None:
                merged[tr['track_id']] = tr
        tracked_results = list(merged.values())

        # Cập nhật các track
        for track_data in tracked_results:
            track_id = track_data['track_id']
            if track_id in self.active_tracks:
                self.active_tracks[track_id].update(track_data['bbox'], self.frame_count)
            else:
                self.active_tracks[track_id] = TrackedVehicle(track_id, track_data['bbox'], self.frame_count)

        self._cleanup_stale_tracks(self.frame_count)

        # Cập nhật trạng thái đèn
        detected_light_color, light_bbox_scaled = self.detector.get_focused_traffic_light_info(frame_processed)
        self.latest_detected_light_color = detected_light_color  # Lưu cho bước kiểm tra vi phạm
        self._update_stable_light_color(detected_light_color)
        # Kích hoạt ân hạn nếu vừa thấy đèn xanh
        if detected_light_color == 'green':
            self.no_violation_until_frame = max(self.no_violation_until_frame, self.frame_count + config.LIGHT_GREEN_GRACE_FRAMES)

        light_bbox_original = None
        if light_bbox_scaled is not None and scale > 0:
            light_bbox_original = [int(v / scale) for v in light_bbox_scaled]

        # Kiểm tra vi phạm
        if self.active_tracks:
            scaled_violation_zone = np.array([(int(p[0] * scale), int(p[1] * scale)) for p in violation_zone_pts], dtype=np.int32)
            scaled_waiting_zone = np.array([(int(p[0] * scale), int(p[1] * scale)) for p in waiting_zone_pts], dtype=np.int32)

            for vehicle in list(self.active_tracks.values()):
                if vehicle.state in ['COMMITTED_VIOLATION', 'PASSED_LEGALLY']:
                    continue

                x, y, w, h = vehicle.bbox
                vehicle_point = (int(x + w / 2), int(y + h))
                is_in_waiting = cv2.pointPolygonTest(scaled_waiting_zone, vehicle_point, False) >= 0
                is_in_violation = cv2.pointPolygonTest(scaled_violation_zone, vehicle_point, False) >= 0

                if vehicle.state == 'NEUTRAL' and is_in_waiting:
                    vehicle.state = 'IN_WAITING_ZONE'
                elif vehicle.state == 'IN_WAITING_ZONE' and is_in_violation:
                    if (self.stable_light_color in ['red', 'yellow']) \
                        and self.latest_detected_light_color != 'green' \
                        and self.frame_count >= self.no_violation_until_frame:
                        vehicle.state = 'COMMITTED_VIOLATION'
                        # Trong live stream, chúng ta chỉ cập nhật trạng thái và nhận dạng biển số để hiển thị
                        bbox_original = [int(v / scale) for v in vehicle.bbox]
                        _, plate_text, _ = self.detector.extract_and_recognize_plate(frame_original, bbox_original)
                        vehicle.license_plate = plate_text
                        # Cập nhật thống kê live
                        self.live_violation_count += 1
                        self.live_violations.append({
                            'track_id': vehicle.track_id,
                            'plate': plate_text or '',
                            'frame': self.frame_count
                        })
                    else:
                        vehicle.state = 'PASSED_LEGALLY'
                elif vehicle.state == 'NEUTRAL' and is_in_violation:
                    vehicle.state = 'PASSED_LEGALLY'

        # Vẽ kết quả lên frame và trả về
        # Lưu ý: Hàm draw_results của bạn phải được sửa để trả về frame_viz thay vì chỉ ghi file
        return self.draw_results(frame_original, scale, waiting_zone_pts, violation_zone_pts, out_writer=None, light_bbox=light_bbox_original)

    def get_live_summary(self):
        """Trả về thống kê live hiện tại để UI hiển thị sau khi dừng."""
        return {
            'count': int(self.live_violation_count),
            'violations': list(self.live_violations[-10:])  # chỉ trả về 10 bản ghi gần nhất
        }

    def _handle_violation_db(self, track_id, job_id, frame_count, plate_text, plate_conf, bbox):
        violation_item = {
            'id': len(self.violations_data) + 1, 'track_id': track_id,
            'timestamp': datetime.datetime.now().isoformat(), 'frame_number': frame_count,
            'license_plate': plate_text, 'confidence': plate_conf, 'bbox': bbox,
        }
        self.violations_data.append(violation_item)

    def _save_violation_image(self, frame, bbox, job_id, track_id):
        x, y, w, h = bbox
        cropped_vehicle = frame[y:y+h, x:x+w]
        if cropped_vehicle.size > 0:
            img_id = track_id
            img_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{img_id}.jpg')
            cv2.imwrite(img_path, cropped_vehicle)