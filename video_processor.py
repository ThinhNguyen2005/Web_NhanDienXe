"""
Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
Điều này giúp giao diện web không bị "đóng băng" trong khi video đang được phân tích.
"""
import os
import cv2
import datetime
import logging
from threading import Lock

from config import PROCESSED_FOLDER, VIOLATIONS_FOLDER
from database import save_violations_to_db
from detector.traffic_light_detector import detect_stop_line

# Thiết lập logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Xử lý video và phát hiện vi phạm trong một luồng riêng."""

    def __init__(self, video_path, detector):
        """
        Khởi tạo VideoProcessor.

        Args:
            video_path (str): Đường dẫn đến file video cần xử lý.
            detector (TrafficViolationDetector): Đối tượng detector đã được khởi tạo.
        """
        self.video_path = video_path
        self.detector = detector
        self.violation_line_y = None
        self.total_frames = 0

    def process_video(self, job_id, processing_status, processing_results, processing_lock):
        """
        Hàm xử lý chính, chạy trong một luồng riêng.
        Đọc từng khung hình, phát hiện vi phạm và cập nhật trạng thái.
        """
        violations_data = []
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {self.video_path}")

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Thiết lập video đầu ra
            output_path = os.path.join(PROCESSED_FOLDER, f'processed_{job_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Thử tự động phát hiện vạch dừng từ một khung hình đầu tiên
            # Nếu không tìm thấy, fallback về 60% chiều cao
            ret0, first_frame = cap.read()
            if ret0:
                detected_y = detect_stop_line(first_frame)
                if detected_y is not None:
                    self.violation_line_y = detected_y
                else:
                    self.violation_line_y = int(frame_height * 0.6)
                # Rewind to start (we consumed one frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.violation_line_y = int(frame_height * 0.6)

            frame_count = 0
            violation_id_counter = 0
            tracked_vehicles = {} # Dùng để theo dõi các xe đã vi phạm, tránh ghi nhận nhiều lần

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                
                red_lights, vehicles, violations_in_frame = [], [], []

                # Xử lý mỗi 5 frame để tối ưu tốc độ
                if frame_count % 5 == 0:
                    red_lights = self.detector.detect_red_lights(frame)
                    vehicles = self.detector.detect_vehicles(frame)
                    
                    for vehicle in vehicles:
                        is_violating = self.detector.check_violation(vehicle, red_lights, self.violation_line_y)
                        
                        if is_violating:
                            violations_in_frame.append(vehicle)
                            
                            # Đơn giản hóa việc tracking: dùng tọa độ tâm xe làm ID tạm thời
                            vx, vy, vw, vh = vehicle['bbox']
                            vehicle_id = f"{int(vx + vw/2)}_{int(vy + vh/2)}"
                            
                            # Nếu xe chưa được ghi nhận vi phạm trong 100 frame gần nhất
                            if vehicle_id not in tracked_vehicles or frame_count - tracked_vehicles[vehicle_id] > 100:
                                tracked_vehicles[vehicle_id] = frame_count
                                violation_id_counter += 1

                                plate_image, plate_text, confidence = self.detector.extract_license_plate(frame, vehicle['bbox'])

                                violation_data = {
                                    'id': violation_id_counter,
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'frame_number': frame_count,
                                    'license_plate': plate_text,
                                    'confidence': confidence,
                                    'bbox': vehicle['bbox']
                                }
                                violations_data.append(violation_data)

                                # Lưu ảnh chụp màn hình vi phạm
                                img_path = os.path.join(VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id_counter}.jpg')
                                cv2.imwrite(img_path, frame)

                # Vẽ các thông tin lên video
                frame_viz = self.draw_visualizations(frame, red_lights, vehicles, violations_in_frame)
                out.write(frame_viz)

                # Cập nhật trạng thái xử lý
                with processing_lock:
                    processing_status[job_id] = {
                        'status': 'processing',
                        'progress': (frame_count / self.total_frames) * 100,
                        'violations_found': len(violations_data),
                        'current_frame': frame_count,
                        'total_frames': self.total_frames
                    }

            cap.release()
            out.release()
            
            # Lưu tất cả vi phạm vào CSDL sau khi xử lý xong video
            save_violations_to_db(job_id, violations_data)

            # Đánh dấu xử lý hoàn tất
            with processing_lock:
                processing_status[job_id]['status'] = 'completed'
                processing_status[job_id]['progress'] = 100
                processing_results[job_id] = {
                    'violations': violations_data,
                    'output_video': output_path
                }
            logger.info(f"Processing for job {job_id} completed. Found {len(violations_data)} violations.")

        except Exception as e:
            logger.error(f"Error in process_video for job {job_id}: {e}", exc_info=True)
            with processing_lock:
                processing_status[job_id] = {'status': 'error', 'error': str(e)}

    def draw_visualizations(self, frame, red_lights, vehicles, violations):
        """Vẽ các bounding box và thông tin lên khung hình."""
        frame_copy = frame.copy()
        
        # Vẽ vạch dừng
        cv2.line(frame_copy, (0, self.violation_line_y), (frame.shape[1], self.violation_line_y), (0, 255, 255), 2)
        cv2.putText(frame_copy, 'STOP LINE', (10, self.violation_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Vẽ đèn đỏ
        for (x, y, w, h) in red_lights:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Vẽ xe
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Đánh dấu xe vi phạm
        for violation in violations:
            x, y, w, h = violation['bbox']
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame_copy, 'VIOLATION!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame_copy
