"""
Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
"""
import os
import cv2
import datetime
import logging
from threading import Lock
from roi_manager_enhanced import load_rois, visualize_roi, check_violation_with_roi

# video_processor.py
from detector.traffic_light_detector import detect_red_lights
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
        self.total_frames = 0
        self.violations_data = []
        self.waiting_zone_pts = []
        self.violation_zone_pts = []

    def process_video(self, job_id: str, processing_status: dict, processing_results: dict, processing_lock: Lock):
        """Hàm xử lý video chính, được chạy trong một luồng riêng sử dụng ROI."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Không thể mở file video: {self.video_path}")

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Load ROI configuration (luôn dùng default)
            self.waiting_zone_pts, self.violation_zone_pts = load_rois("default")
            
            logger.info(f"Loaded ROI configuration:")
            logger.info(f"  - Waiting zone: {len(self.waiting_zone_pts)} points")
            logger.info(f"  - Violation zone: {len(self.violation_zone_pts)} points")
            
            # Nếu không có ROI được cấu hình, sử dụng auto-detect
            if not self.violation_zone_pts:
                logger.warning("No ROI configured, attempting auto-detection...")
                ret, first_frame = cap.read()
                if ret:
                    from roi_manager_enhanced import auto_detect_roi
                    auto_waiting, auto_violation = auto_detect_roi(first_frame)
                    if auto_violation:
                        self.waiting_zone_pts = auto_waiting
                        self.violation_zone_pts = auto_violation
                        logger.info("Auto-detected ROI zones successfully")
                    else:
                        logger.error("Could not auto-detect ROI")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            else:
                logger.info("Using saved ROI configuration")

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
                    red_lights, vehicles, violations_in_frame = self.detector.run_detection_on_frame_with_roi(
                        frame, self.waiting_zone_pts, self.violation_zone_pts
                    )

                    for v in violations_in_frame:
                        self._handle_violation(v, frame, job_id, frame_count)
                
                # Vẽ frame với ROI visualization
                frame_viz = visualize_roi(frame, self.waiting_zone_pts, self.violation_zone_pts)
                
                # Vẽ bounding boxes cho vehicles và violations
                for vehicle in vehicles:
                    is_violating = any(v['bbox'] == vehicle['bbox'] for v in violations_in_frame)
                    color = (0, 0, 255) if is_violating else (0, 255, 0)
                    thickness = 3 if is_violating else 2
                    x, y, w, h = vehicle['bbox']
                    cv2.rectangle(frame_viz, (x, y), (x+w, y+h), color, thickness)
                    if is_violating:
                        cv2.putText(frame_viz, f"VI PHAM: {vehicle.get('license_plate', 'N/A')}", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                out.write(frame_viz)
                
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
            
            # Save processed video info to database only once after processing
            output_video_filename = os.path.basename(output_path)
            database.save_processed_video(job_id, output_video_filename)
            
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

