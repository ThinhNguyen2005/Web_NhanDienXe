"""
Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
"""
import os
import cv2
import datetime
import logging
from threading import Lock

import config
import database
from detector_manager import TrafficViolationDetector

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Xử lý video trong một luồng riêng để không làm treo giao diện web."""

    def __init__(self, video_path: str, detector: TrafficViolationDetector):
        """
        Khởi tạo VideoProcessor.
        
        Args:
            video_path (str): Đường dẫn đến file video.
            detector (TrafficViolationDetector): Đối tượng detector đã được khởi tạo sẵn.
        """
        self.video_path = video_path
        self.detector = detector
        self.violation_line_y = None
        self.total_frames = 0
        self.violations_data = []

    def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
        """Hàm xử lý video chính, được chạy trong một luồng riêng."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Không thể mở file video: {self.video_path}")

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.violation_line_y = int(frame_height * 0.6)

            output_path = os.path.join(config.PROCESSED_FOLDER, f'processed_{job_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                red_lights, vehicles, violations_in_frame = [], [], []
                if frame_count % config.DETECTION_INTERVAL == 0:
                    red_lights, vehicles, violations_in_frame = self.detector.run_detection_on_frame(
                        frame, self.violation_line_y
                    )

                    for v in violations_in_frame:
                        self._handle_violation(v, frame, job_id, frame_count)
                
                frame_viz = self._draw_visualizations(frame, red_lights, vehicles, violations_in_frame)
                out.write(frame_viz)
                output_video_filename = os.path.basename(output_path)
                database.save_processed_video(job_id, output_video_filename)
                with processing_lock:
                    processing_status[job_id] = {
                        'status': 'processing',
                        'progress': (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0,
                        'violations_found': len(self.violations_data),
                        'current_frame': frame_count,
                        'total_frames': self.total_frames
                    }

            cap.release()
            out.release()
            
            database.save_violations_to_db(job_id, self.violations_data)
            
            with processing_lock:
                processing_status[job_id] = {
                    'status': 'completed', 'progress': 100,
                    'violations_found': len(self.violations_data),
                    'output_video': os.path.basename(output_path), 
                    'total_frames': self.total_frames
                }
                processing_results[job_id] = {
                    'violations': self.violations_data, 'output_video': output_path
                }
            logger.info(f"Processing for job {job_id} completed. Found {len(self.violations_data)} violations.")

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong quá trình xử lý video {job_id}: {e}", exc_info=True)
            with processing_lock:
                processing_status[job_id] = {'status': 'error', 'error': str(e)}
    def _handle_violation(self, violation, frame, job_id, frame_count):
        """Xử lý khi một vi phạm được phát hiện."""
        violation_id = len(self.violations_data) + 1
        
        violation_item = {
            'id': violation_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'frame_number': frame_count,
            'license_plate': violation.get('license_plate', 'N/A'),
            'confidence': violation.get('license_plate_confidence', 0.0),
            'bbox': violation['bbox']
        }
        self.violations_data.append(violation_item)

        img_path = os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{violation_id}.jpg')
        cv2.imwrite(img_path, frame)
        
    def _draw_visualizations(self, frame, red_lights, vehicles, violations_in_frame):
        """Vẽ các bounding box và thông tin lên khung hình."""
        frame_copy = frame.copy()

        if self.violation_line_y:
            cv2.line(frame_copy, (0, self.violation_line_y), (frame.shape[1], self.violation_line_y), (0, 255, 255), 2)
            cv2.putText(frame_copy, 'VACH DUNG', (10, self.violation_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for (x, y, w, h) in red_lights:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)

        for vehicle in vehicles:
            is_violating = any(v['bbox'] == vehicle['bbox'] for v in violations_in_frame)
            color = (0, 0, 255) if is_violating else (0, 255, 0)
            thickness = 3 if is_violating else 2
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
            if is_violating:
                 cv2.putText(frame_copy, f"VI PHAM: {vehicle.get('license_plate', 'N/A')}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame_copy

