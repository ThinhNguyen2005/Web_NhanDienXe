# """
# Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
# Phiên bản SIÊU CẤP: Tích hợp Trình quản lý trạng thái vi phạm và Logic tái định danh
# để loại bỏ hoàn toàn vi phạm trùng lặp.
# """
# import os
# import cv2
# import datetime
# import logging
# from threading import Lock
# from collections import deque

# from roi_manager_enhanced import load_rois, visualize_roi, check_violation_with_roi
# import config
# import database
# from detector_manager import TrafficViolationDetector
# from detector import trafficLightColor

# logger = logging.getLogger(__name__)

# def calculate_iou(boxA, boxB):
#     """Tính toán Intersection over Union (IoU) giữa hai bounding box."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]
    
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou

# class TrackedVehicle:
#     """Một lớp để quản lý trạng thái và lịch sử của mỗi phương tiện được theo dõi."""
#     def __init__(self, track_id, bbox):
#         self.track_id = track_id
#         self.bboxes = deque(maxlen=10)  # Lưu 10 vị trí cuối cùng
#         self.bboxes.append(bbox)
#         self.last_seen_frame = 0
#         self.license_plate = None
#         self.license_plate_confidence = 0.0
#         self.violation_state = 'pending' # 'pending', 'committed', 'cooldown'
#         self.last_violation_time = None

#     def update(self, bbox, frame_count):
#         self.bboxes.append(bbox)
#         self.last_seen_frame = frame_count

#     @property
#     def last_bbox(self):
#         return self.bboxes[-1]

# class VideoProcessor:
#     """Xử lý video với Trình quản lý trạng thái vi phạm."""

#     def __init__(self, video_path: str, detector: TrafficViolationDetector):
#         self.video_path = video_path
#         self.detector = detector
#         self.violations_data = []
        
#         # Cấu trúc quản lý tracking
#         self.active_tracks = {}  # Key: track_id, Value: TrackedVehicle object
#         self.lost_tracks = deque(maxlen=30) # Lưu các track đã mất trong 30 frame

#     def _resize_frame(self, frame, target_width):
#         if target_width is None:
#             return frame, 1.0
#         h, w, _ = frame.shape
#         if w == target_width:
#             return frame, 1.0
#         scale = target_width / w
#         target_height = int(h * scale)
#         return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA), scale

#     def _reidentify_tracks(self, current_tracks):
#         """Cố gắng tái định danh các track mới với các track đã mất gần đây."""
#         unmatched_current_tracks = list(current_tracks.values())
        
#         for lost_track in list(self.lost_tracks):
#             best_match = None
#             max_iou = 0.2 # Ngưỡng IoU tối thiểu
            
#             for current_track in unmatched_current_tracks:
#                 iou = calculate_iou(lost_track.last_bbox, current_track['bbox'])
#                 if iou > max_iou:
#                     max_iou = iou
#                     best_match = current_track

#             if best_match:
#                 # Tìm thấy! Khôi phục track cũ với ID mới.
#                 new_track_id = best_match['track_id']
#                 logger.info(f"Re-identified lost track (ID: {lost_track.track_id}) as new track (ID: {new_track_id}). Restoring state.")
#                 lost_track.track_id = new_track_id # Cập nhật ID mới
#                 self.active_tracks[new_track_id] = lost_track
#                 self.lost_tracks.remove(lost_track)
#                 unmatched_current_tracks.remove(best_match)

#         # Các track mới không thể tái định danh sẽ được thêm vào active_tracks
#         for track_data in unmatched_current_tracks:
#             track_id = track_data['track_id']
#             self.active_tracks[track_id] = TrackedVehicle(track_id, track_data['bbox'])
    
#     def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
#         try:
#             cap = cv2.VideoCapture(self.video_path)
#             # ... (Phần khởi tạo video, ROI, video writer giữ nguyên như trước) ...
#             self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
#             camera_id = os.path.splitext(os.path.basename(self.video_path))[0]
#             waiting_zone_pts, violation_zone_pts = load_rois(camera_id)
#             if not violation_zone_pts:
#                 waiting_zone_pts, violation_zone_pts = load_rois("default")
#             if not violation_zone_pts:
#                 raise ValueError(f"Không có cấu hình ROI cho video '{camera_id}' hoặc 'default'.")

#             output_path = os.path.join(config.PROCESSED_FOLDER, f'processed_{job_id}.mp4')
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

#             frame_count = 0
#             while cap.isOpened():
#                 ret, frame_original = cap.read()
#                 if not ret:
#                     break
#                 frame_count += 1
#                 current_time = datetime.datetime.now()

#                 frame_processed, scale = self._resize_frame(frame_original, config.PROCESSING_FRAME_WIDTH)
                
#                 # 1. Lấy kết quả tracking từ model
#                 tracked_results = self.detector.vehicle_detector.track_vehicles(frame_processed)
#                 current_frame_track_ids = {t['track_id'] for t in tracked_results}
#                 current_tracks_dict = {t['track_id']: t for t in tracked_results}
                
#                 # 2. Cập nhật và quản lý các tracks
#                 # Di chuyển các track không còn thấy vào danh sách "lost"
#                 active_ids = set(self.active_tracks.keys())
#                 lost_ids = active_ids - current_frame_track_ids
#                 for track_id in lost_ids:
#                     if (frame_count - self.active_tracks[track_id].last_seen_frame) > 10: # Mất quá 10 frame
#                          self.lost_tracks.append(self.active_tracks.pop(track_id))

#                 # Cập nhật các track đang hoạt động
#                 for track_data in tracked_results:
#                     track_id = track_data['track_id']
#                     if track_id in self.active_tracks:
#                         self.active_tracks[track_id].update(track_data['bbox'], frame_count)
                
#                 # Tái định danh và thêm các track mới
#                 new_track_ids = current_frame_track_ids - active_ids
#                 new_tracks_data = {tid: current_tracks_dict[tid] for tid in new_track_ids}
#                 if new_tracks_data:
#                     self._reidentify_tracks(new_tracks_data)

#                 # 3. Kiểm tra vi phạm cho các track đang hoạt động
#                 scaled_violation_zone = [(int(p[0] * scale), int(p[1] * scale)) for p in violation_zone_pts]
#                 t_light_color = trafficLightColor.estimate_label(frame_processed)

#                 if t_light_color == "red":
#                     for track_id, vehicle in self.active_tracks.items():
#                         if vehicle.violation_state == 'committed':
#                             continue # Đã vi phạm, bỏ qua

#                         is_violating = check_violation_with_roi(vehicle.last_bbox, scaled_violation_zone, [], "red")
                        
#                         if is_violating:
#                             # Nếu đây là lần đầu xe vi phạm, xử lý nó
#                             if vehicle.violation_state == 'pending':
#                                 logger.info(f"Track ID {track_id} committed a violation. Handling...")
#                                 # Lấy thông tin biển số chỉ một lần này
#                                 x, y, w, h = [int(v / scale) for v in vehicle.last_bbox]
#                                 _, plate_text, plate_conf = self.detector._extract_and_recognize_plate(frame_original, [x,y,w,h])
#                                 vehicle.license_plate = plate_text
#                                 vehicle.license_plate_confidence = plate_conf
#                                 vehicle.violation_state = 'committed'
#                                 vehicle.last_violation_time = current_time
                                
#                                 self._handle_violation(vehicle, frame_original, job_id, frame_count)

#                 # 4. Vẽ kết quả
#                 frame_viz = visualize_roi(frame_original, waiting_zone_pts, violation_zone_pts)
#                 for vehicle in self.active_tracks.values():
#                     x, y, w, h = [int(v / scale) for v in vehicle.last_bbox]
#                     color = (0, 0, 255) if vehicle.violation_state == 'committed' else (0, 255, 0)
#                     cv2.rectangle(frame_viz, (x, y), (x + w, y + h), color, 2)
#                     label = f"ID: {vehicle.track_id}"
#                     if vehicle.license_plate:
#                         label += f" LP: {vehicle.license_plate}"
#                     cv2.putText(frame_viz, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
#                 out.write(frame_viz)

#                 with processing_lock:
#                     processing_status[job_id] = {'status': 'processing', 'progress': (frame_count / self.total_frames) * 100}

#             # ... (Phần kết thúc xử lý và lưu DB giữ nguyên) ...
#             cap.release()
#             out.release()
            
#             output_video_filename = os.path.basename(output_path)
#             database.save_processed_video(job_id, output_video_filename)
#             database.save_violations_to_db(job_id, self.violations_data)
            
#             with processing_lock:
#                 processing_status[job_id] = {'status': 'completed', 'progress': 100, 'violations_found': len(self.violations_data), 'output_video': output_video_filename}
#                 processing_results[job_id] = {'violations': self.violations_data, 'output_video': output_path}
#             logger.info(f"Hoàn tất xử lý job {job_id}. Phát hiện {len(self.violations_data)} vi phạm.")
#         except Exception as e:
#             logger.error(f"Lỗi nghiêm trọng: {e}", exc_info=True)
#             with processing_lock:
#                 processing_status[job_id] = {'status': 'error', 'error': str(e)}

#     def _handle_violation(self, vehicle, frame, job_id, frame_count):
#         """Xử lý khi một vi phạm mới được xác nhận."""
#         violation_id = len(self.violations_data) + 1
#         x, y, w, h = [int(v) for v in vehicle.last_bbox] # Sử dụng bbox đã được scale về gốc
        
#         violation_item = {
#             'id': violation_id,
#             'track_id': vehicle.track_id,
#             'timestamp': datetime.datetime.now().isoformat(),
#             'frame_number': frame_count,
#             'license_plate': vehicle.license_plate,
#             'confidence': vehicle.license_plate_confidence,
#             'bbox': [x, y, w, h]
#         }
#         self.violations_data.append(violation_item)

#         cropped_vehicle = frame[y:y+h, x:x+w]
#         if cropped_vehicle.size > 0:
#             img_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
#             cv2.imwrite(img_path, cropped_vehicle)
# # """
# # Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
# # Phiên bản tối ưu hóa cao: resize khung hình, nhận dạng biển số có chọn lọc.
# # """
# # import os
# # import cv2
# # import datetime
# # import logging
# # from threading import Lock
# # from roi_manager_enhanced import load_rois, visualize_roi, check_violation_with_roi

# # import config
# # import database
# # from detector_manager import TrafficViolationDetector
# # from detector import trafficLightColor

# # logger = logging.getLogger(__name__)

# # class VideoProcessor:
# #     """Xử lý video với các kỹ thuật tối ưu hóa hiệu năng."""

# #     def __init__(self, video_path: str, detector: TrafficViolationDetector):
# #         self.video_path = video_path
# #         self.detector = detector
# #         self.total_frames = 0
# #         self.violations_data = []
# #         self.waiting_zone_pts = []
# #         self.violation_zone_pts = []
# #         self.violation_cooldowns = {}
# #         self.VIOLATION_COOLDOWN_SECONDS = config.VIOLATION_COOLDOWN_SECONDS

# #     def _resize_frame(self, frame, target_width):
# #         """Thay đổi kích thước khung hình trong khi vẫn giữ tỷ lệ."""
# #         if target_width is None:
# #             return frame, 1.0 # Trả về frame gốc và tỷ lệ 1.0

# #         h, w, _ = frame.shape
# #         if w == target_width:
# #             return frame, 1.0

# #         scale = target_width / w
# #         target_height = int(h * scale)
# #         resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
# #         return resized_frame, scale

# #     def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
# #         """Hàm xử lý video chính với tối ưu hóa."""
# #         try:
# #             cap = cv2.VideoCapture(self.video_path)
# #             if not cap.isOpened():
# #                 raise IOError(f"Không thể mở file video: {self.video_path}")

# #             # Lấy thông tin video gốc
# #             self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #             fps = cap.get(cv2.CAP_PROP_FPS)
# #             original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #             original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# #             # Tải ROI
# #             camera_id = os.path.splitext(os.path.basename(self.video_path))[0]
# #             self.waiting_zone_pts, self.violation_zone_pts = load_rois(camera_id)
# #             if not self.violation_zone_pts:
# #                 self.waiting_zone_pts, self.violation_zone_pts = load_rois("default")
# #             if not self.violation_zone_pts:
# #                 raise ValueError(f"Không có cấu hình ROI cho video '{camera_id}' hoặc 'default'.")

# #             output_path = os.path.join(config.PROCESSED_FOLDER, f'processed_{job_id}.mp4')
# #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #             # Video output vẫn có kích thước gốc
# #             out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

# #             frame_count = 0
# #             while cap.isOpened():
# #                 ret, frame_original = cap.read()
# #                 if not ret:
# #                     break
# #                 frame_count += 1
# #                 current_time = datetime.datetime.now()

# #                 # --- TỐI ƯU HÓA: RESIZE FRAME ---
# #                 frame_processed, scale = self._resize_frame(frame_original, config.PROCESSING_FRAME_WIDTH)

# #                 # Chạy tracking trên frame đã resize
# #                 _, tracked_vehicles, _ = self.detector.run_tracking_and_detection_on_frame(
# #                     frame_processed, self.waiting_zone_pts, self.violation_zone_pts
# #                 )

# #                 # Kiểm tra vi phạm với tọa độ đã được scale
# #                 scaled_violation_zone = [(int(p[0] * scale), int(p[1] * scale)) for p in self.violation_zone_pts]
# #                 scaled_waiting_zone = [(int(p[0] * scale), int(p[1] * scale)) for p in self.waiting_zone_pts]

# #                 t_light_color = trafficLightColor.estimate_label(frame_processed)

# #                 for vehicle in tracked_vehicles:
# #                     is_violating = check_violation_with_roi(
# #                         vehicle['bbox'], scaled_violation_zone, scaled_waiting_zone, t_light_color
# #                     )

# #                     if is_violating:
# #                         track_id = vehicle['track_id']
# #                         last_violation_time = self.violation_cooldowns.get(track_id)

# #                         if last_violation_time is None or \
# #                            (current_time - last_violation_time).total_seconds() > self.VIOLATION_COOLDOWN_SECONDS:

# #                             # --- TỐI ƯU HÓA: CHỈ NHẬN DẠNG BIỂN SỐ KHI CẦN ---
# #                             # Scale bbox về kích thước gốc để cắt ảnh
# #                             x, y, w, h = [int(v / scale) for v in vehicle['bbox']]

# #                             _, plate_text, plate_conf = self.detector._extract_and_recognize_plate(frame_original, [x, y, w, h])
# #                             vehicle['license_plate'] = plate_text
# #                             vehicle['license_plate_confidence'] = plate_conf

# #                             # Lưu bbox gốc
# #                             vehicle['bbox_original'] = [x, y, w, h]

# #                             self._handle_violation(vehicle, frame_original, job_id, frame_count)
# #                             self.violation_cooldowns[track_id] = current_time

# #                 # Vẽ kết quả lên frame gốc
# #                 frame_viz = visualize_roi(frame_original, self.waiting_zone_pts, self.violation_zone_pts)
# #                 for vehicle in tracked_vehicles:
# #                     track_id = vehicle['track_id']
# #                     # Scale lại bbox để vẽ trên frame gốc
# #                     x, y, w, h = [int(v / scale) for v in vehicle['bbox']]

# #                     is_violating_confirmed = track_id in self.violation_cooldowns
# #                     color = (0, 0, 255) if is_violating_confirmed else (0, 255, 0)
# #                     thickness = 3 if is_violating_confirmed else 2

# #                     cv2.rectangle(frame_viz, (x, y), (x + w, y + h), color, thickness)
# #                     label = f"ID: {track_id}"
# #                     cv2.putText(frame_viz, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# #                 out.write(frame_viz)

# #                 with processing_lock:
# #                     processing_status[job_id] = {'status': 'processing', 'progress': (frame_count / self.total_frames) * 100}

# #             cap.release()
# #             out.release()

# #             output_video_filename = os.path.basename(output_path)
# #             database.save_processed_video(job_id, output_video_filename)
# #             database.save_violations_to_db(job_id, self.violations_data)

# #             with processing_lock:
# #                 processing_status[job_id] = {'status': 'completed', 'progress': 100, 'violations_found': len(self.violations_data), 'output_video': output_video_filename}
# #                 processing_results[job_id] = {'violations': self.violations_data, 'output_video': output_path}
# #             logger.info(f"Hoàn tất xử lý job {job_id}. Phát hiện {len(self.violations_data)} vi phạm.")

# #         except Exception as e:
# #             logger.error(f"Lỗi nghiêm trọng: {e}", exc_info=True)
# #             with processing_lock:
# #                 processing_status[job_id] = {'status': 'error', 'error': str(e)}

# #     def _handle_violation(self, violation, frame, job_id, frame_count):
# #         """Xử lý khi một vi phạm mới được xác nhận."""
# #         violation_id = len(self.violations_data) + 1
# #         bbox_to_save = violation.get('bbox_original', violation['bbox'])

# #         violation_item = {
# #             'id': violation_id,
# #             'track_id': violation.get('track_id', 'N/A'),
# #             'timestamp': datetime.datetime.now().isoformat(),
# #             'frame_number': frame_count,
# #             'license_plate': violation.get('license_plate', 'N/A'),
# #             'confidence': violation.get('license_plate_confidence', 0.0),
# #             'bbox': bbox_to_save
# #         }
# #         self.violations_data.append(violation_item)

# #         x, y, w, h = bbox_to_save
# #         cropped_vehicle = frame[y:y+h, x:x+w]
# #         if cropped_vehicle.size > 0:
# #             img_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
# #             cv2.imwrite(img_path, cropped_vehicle)

# # ========================================================================================
# # PHIEN BAN CU - DUOC CHU THICH VA LUU LAI DE THAM KHAO
# # ========================================================================================

# """
# # --- CODE CU (VIDEO PROCESSOR V1 - SU DUNG DETECTION DON GIAN) ---
# # Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
# # Phiên bản cũ sử dụng detection đơn giản, không có tracking liên tục.

# class VideoProcessor_V1:
#     # Xử lý video trong một luồng riêng để không làm treo giao diện web.
#     # Phiên bản này sử dụng detection theo khoảng cách frame (DETECTION_INTERVAL)

#     def __init__(self, video_path: str, detector: TrafficViolationDetector):
#         self.video_path = video_path
#         self.detector = detector
#         self.total_frames = 0
#         self.violations_data = []
#         self.waiting_zone_pts = []
#         self.violation_zone_pts = []

#     def process_video_v1(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
#         # Hàm xử lý video chính, được chạy trong một luồng riêng sử dụng ROI.
#         # Chỉ chạy detection mỗi DETECTION_INTERVAL frames để tối ưu hiệu suất

#         # ... (code cũ đã được comment và lưu tại đây để tham khảo)

#         # Ưu điểm của v1:
#         # - Đơn giản, ít tài nguyên
#         # - Dễ debug và maintain
#         #
#         # Nhược điểm của v1:
#         # - Bỏ sót vi phạm giữa các detection intervals
#         # - Không theo dõi liên tục các phương tiện
#         # - Không có logic chống trùng lặp vi phạm
#         # - Độ chính xác thấp hơn v2

#     def _handle_violation_v1(self, violation, frame, job_id, frame_count):
#         # Xử lý khi một vi phạm được phát hiện.
#         # Phiên bản đơn giản, không có cooldown logic

#         # ... (code cũ đã được comment và lưu tại đây để tham khảo)
# """

# # ========================================================================================
# # SO SANH V2 vs V1:
# # ========================================================================================
# """
# VIDEO PROCESSOR V2 (HIEN TAI):
# + Sử dụng YOLOv8 tracking liên tục trên mọi frame
# + Logic cooldown 30 giây để tránh trùng lặp vi phạm
# + Theo dõi track_id của từng phương tiện
# + Visualization tốt hơn với màu sắc phân biệt
# + Độ chính xác cao hơn, ít bỏ sót vi phạm

# VIDEO PROCESSOR V1 (CU):
# - Detection chỉ chạy mỗi DETECTION_INTERVAL frames
# - Không có tracking liên tục
# - Không có logic chống trùng lặp
# - Visualization cơ bản hơn
# - Dễ bỏ sót vi phạm giữa các khoảng detection
# """
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