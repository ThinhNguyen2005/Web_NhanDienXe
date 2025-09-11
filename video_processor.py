"""
Module chứa lớp VideoProcessor để xử lý video trong một luồng (thread) riêng biệt.
"""
import os
import cv2
import datetime
import logging
import imutils
from threading import Lock
# video_processor.py
from detector.traffic_light_detector import detect_stop_line, detect_stop_line_enhanced, detect_red_lights, detect_stop_line_by_traffic_light
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
            
            # Try to auto-detect stop line from multiple frames for better accuracy
            detected_lines = []
            traffic_light_pos = None
            
            # First, try to detect traffic light position from first frame
            ret, first_frame = cap.read()
            if ret:
                red_lights = detect_red_lights(first_frame)
                if red_lights:
                    # Use the first detected traffic light
                    traffic_light_pos = red_lights[0]
                    logger.info(f"✓ Detected traffic light at position: {traffic_light_pos}")
            
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Now detect stop lines using enhanced method
            for i in range(min(20, self.total_frames // 15)):  # Check up to 20 frames, spaced 15 frames apart for better coverage
                ret, frame = cap.read()
                if ret:
                    line_y = detect_stop_line_enhanced(frame, traffic_light_pos)
                    if line_y is not None:
                        detected_lines.append(line_y)
                    elif traffic_light_pos is not None:
                        # Fallback: try traffic light based detection
                        line_y = detect_stop_line_by_traffic_light(frame, traffic_light_pos)
                        if line_y is not None:
                            detected_lines.append(line_y)
                else:
                    break
            
            if detected_lines:
                # Use median to reduce outliers
                import statistics
                self.violation_line_y = int(statistics.median(detected_lines))
                logger.info(f"✓ Auto-detected stop line at y={self.violation_line_y} (from {len(detected_lines)} frames, using traffic light guidance)")
            else:
                self.violation_line_y = int(frame_height * 0.6)
                logger.info(f"⚠ Could not detect stop line from {len(detected_lines)} frames, using default at y={self.violation_line_y}")
            
            # Reset video to beginning for processing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
                
                frame_viz = frame.copy()
                
                # Vẽ vạch dừng (stop line) nếu đã phát hiện
                if self.violation_line_y is not None:
                    # Vẽ vùng màu đỏ nhạt phía trên vạch dừng (khu vực cấm)
                    overlay = frame_viz.copy()
                    cv2.rectangle(overlay, (0, 0), (frame_width, self.violation_line_y), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.1, frame_viz, 0.9, 0, frame_viz)
                    
                    # Vẽ đường ngang màu xanh lá ở vị trí vạch dừng
                    cv2.line(frame_viz, (0, self.violation_line_y), (frame_width, self.violation_line_y), (0, 255, 0), 4)
                    
                    # Thêm text label với background
                    text = "STOP LINE"
                    font_scale = 0.8
                    font_thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Vẽ background cho text
                    cv2.rectangle(frame_viz, (5, self.violation_line_y - text_height - 15), 
                                (15 + text_width, self.violation_line_y - 5), (0, 0, 0), -1)
                    
                    # Vẽ text
                    cv2.putText(frame_viz, text, (10, self.violation_line_y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                
                # Vẽ bounding boxes cho vehicles và violations
                for vehicle in vehicles:
                    is_violating = any(v['bbox'] == vehicle['bbox'] for v in violations_in_frame)
                    color = (0, 0, 255) if is_violating else (0, 255, 0)
                    thickness = 3 if is_violating else 2
                    x, y, w, h = vehicle['bbox']
                    cv2.rectangle(frame_viz, (x, y), (x+w, y+h), color, thickness)
                    if is_violating:
                        cv2.putText(frame_viz, f"VI PHAM: {vehicle.get('license_plate', 'N/A')}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
        
    def _detect_stop_line_from_first_frame(self, frame):
        """Phát hiện vạch dừng từ frame đầu tiên sử dụng thuật toán từ vachdung.py"""
        if frame is None or frame.size == 0:
            return None

        # Resize frame (tương tự vachdung.py)
        frame_resized = imutils.resize(frame, width=400)
        
        # Chuyển xám + bilateral filter
        gray_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        
        # Canny edge detection
        edged = cv2.Canny(gray_image, 30, 200)
        
        # Tìm contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sắp xếp theo diện tích và lấy top 10
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        
        # Tìm vạch dừng từ contours (logic đơn giản hóa)
        h, w = frame_resized.shape[:2]
        candidate_lines = []
        
        for contour in cnts:
            # Lấy bounding box của contour
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Chỉ xét contours ở nửa dưới của frame
            if y > h * 0.5:
                # Tính slope của bounding box (nếu width > height thì có thể là đường ngang)
                if cw > ch * 2:  # Contour rộng hơn cao nhiều lần
                    center_y = y + ch // 2
                    candidate_lines.append(center_y)
        
        if candidate_lines:
            # Chọn median để giảm outliers
            import statistics
            try:
                stop_line_y = int(statistics.median(candidate_lines))
            except statistics.StatisticsError:
                stop_line_y = int(sum(candidate_lines) / len(candidate_lines))
            
            # Scale lại về kích thước gốc
            scale_factor = frame.shape[0] / frame_resized.shape[0]
            return int(stop_line_y * scale_factor)
        
        return None

