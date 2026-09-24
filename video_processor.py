"""
Module chứa lớp VideoProcessor với State Machine để phát hiện vi phạm chính xác.
Hỗ trợ cả Video File và RTSP Stream.
"""
import os
import cv2
import datetime
import logging
import time
from threading import Lock
import numpy as np
from collections import deque
from typing import TYPE_CHECKING, List, Dict, Tuple, Any, Optional, Deque

from roi_manager_enhanced import load_rois, visualize_roi
import config
import database

if TYPE_CHECKING:
    from detector_manager import TrafficViolationDetector

logger = logging.getLogger(__name__)

class TrackedVehicle:
    def __init__(self, track_id: int, bbox: List[int], frame_count: int):
        self.track_id: int = track_id
        self.bbox: List[int] = bbox
        self.license_plate: Optional[str] = None
        self.state: str = 'NEUTRAL'
        self.last_seen_frame: int = frame_count
        self.history: List[List[int]] = [bbox]

    def update(self, bbox: List[int], frame_count: int) -> None:
        self.bbox = bbox
        self.last_seen_frame = frame_count
        self.history.append(bbox)

def _iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    denom = float(areaA + areaB - inter)
    return inter/denom if denom>0 else 0.0

class VideoProcessor:
    def __init__(self, video_path: str, detector: "TrafficViolationDetector", options: Optional[Dict[str, Any]] = None):
        self.video_path: str = video_path
        self.detector: "TrafficViolationDetector" = detector
        self.options: Dict[str, Any] = self._resolve_options(options)
        self.violations_data: List[Dict[str, Any]] = []
        self.active_tracks: Dict[int, TrackedVehicle] = {}
        self.plate_cache_by_track: Dict[int, Tuple[str, float]] = {}
        self.stable_light_color: str = 'unknown'
        self.light_color_buffer: Deque[str] = deque(maxlen=config.LIGHT_STATE_BUFFER_SIZE)
        self.live_violations: List[Dict[str, Any]] = []
        self.live_violation_count: int = 0
        self.no_violation_until_frame: int = 0
        self.latest_detected_light_color: str = 'unknown'
        self.total_frames: int = 0
        self.frame_count: int = 0
        self.is_stream: bool = video_path.startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    def _resolve_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = {
            'mode': 'balanced',
            'processing_frame_width': config.PROCESSING_FRAME_WIDTH,
            'vehicle_detection_interval': config.VEHICLE_DETECTION_INTERVAL,
            'traffic_light_interval': config.TRAFFIC_LIGHT_INTERVAL,
            'status_update_interval': config.STATUS_UPDATE_INTERVAL,
            'write_output_video': config.WRITE_OUTPUT_VIDEO,
            'output_frame_width': config.OUTPUT_FRAME_WIDTH,
        }
        if options:
            base.update({k: v for k, v in options.items() if v is not None})
        return base

    def reset(self) -> None:
        self.violations_data.clear()
        self.active_tracks.clear()
        self.light_color_buffer.clear()
        self.plate_cache_by_track.clear()
        self.stable_light_color = 'unknown'
        self.latest_detected_light_color = 'unknown'
        self.no_violation_until_frame = 0
        self.frame_count = 0

    def _update_stable_light_color(self, detected_color: str) -> None:
        self.light_color_buffer.append(detected_color)
        if len(self.light_color_buffer) == self.light_color_buffer.maxlen:
            first_color = self.light_color_buffer[0]
            if first_color != 'unknown' and all(color == first_color for color in self.light_color_buffer):
                self.stable_light_color = first_color

    def _resize_frame(self, frame: np.ndarray, target_width: Optional[int]) -> Tuple[np.ndarray, float]:
        if target_width is None: return frame, 1.0
        h, w, _ = frame.shape
        scale = float(target_width / w)
        return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale

    def process_video(self, job_id: str, processing_status: Dict[str, Any], processing_results: Dict[str, Any], processing_lock: Lock) -> None:
        self.reset()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open: {self.video_path}")

        if not self.is_stream:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        camera_id = os.path.splitext(os.path.basename(self.video_path))[0] if not self.is_stream else job_id
        waiting_zone_pts, violation_zone_pts = load_rois(camera_id)
        if not violation_zone_pts: waiting_zone_pts, violation_zone_pts = load_rois("default")

        processing_width = self.options['processing_frame_width']
        frame_scale = (processing_width / original_width) if processing_width else 1.0
        scaled_violation_zone = np.array([(int(p[0] * frame_scale), int(p[1] * frame_scale)) for p in violation_zone_pts], dtype=np.int32)
        scaled_waiting_zone = np.array([(int(p[0] * frame_scale), int(p[1] * frame_scale)) for p in waiting_zone_pts], dtype=np.int32)

        out = None
        if not self.is_stream and self.options['write_output_video']:
            output_path = os.path.join(config.PROCESSED_FOLDER, f'processed_{job_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

        timing = {'resize': 0.0, 'vehicle': 0.0, 'light': 0.0, 'logic': 0.0, 'draw': 0.0}
        
        while cap.isOpened():
            ret, frame_original = cap.read()
            if not ret: break
            self.frame_count += 1

            # Simple frame skipping for streams to stay real-time
            if self.is_stream and cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 != 0: continue

            t0 = time.perf_counter()
            frame_processed, scale = self._resize_frame(frame_original, processing_width)
            timing['resize'] += time.perf_counter() - t0

            # Vehicle Detection & Tracking
            if self.frame_count % self.options['vehicle_detection_interval'] == 0:
                t0 = time.perf_counter()
                tracked = self.detector.vehicle_detector.track_vehicles(frame_processed)
                timing['vehicle'] += time.perf_counter() - t0
                for tr in tracked:
                    tid = tr['track_id']
                    if tid in self.active_tracks: self.active_tracks[tid].update(tr['bbox'], self.frame_count)
                    else: self.active_tracks[tid] = TrackedVehicle(tid, tr['bbox'], self.frame_count)

            self._cleanup_stale_tracks(self.frame_count)

            # Traffic Light
            light_bbox_orig = None
            if self.frame_count % self.options['traffic_light_interval'] == 0:
                t0 = time.perf_counter()
                color, lbbox = self.detector.get_focused_traffic_light_info(frame_processed)
                timing['light'] += time.perf_counter() - t0
                self.latest_detected_light_color = color
                self._update_stable_light_color(color)
                if color == 'green': self.no_violation_until_frame = max(self.no_violation_until_frame, self.frame_count + config.LIGHT_GREEN_GRACE_FRAMES)
                if lbbox is not None: light_bbox_orig = [int(v / scale) for v in lbbox]

            # Violation Logic
            if self.frame_count % config.CHECK_VIOLATION_INTERVAL == 0:
                t0 = time.perf_counter()
                for v in list(self.active_tracks.values()):
                    if v.state in ['COMMITTED_VIOLATION', 'PASSED_LEGALLY']: continue
                    x, y, w, h = v.bbox
                    pt = (int(x + w / 2), int(y + h))
                    in_w = cv2.pointPolygonTest(scaled_waiting_zone, pt, False) >= 0
                    in_v = cv2.pointPolygonTest(scaled_violation_zone, pt, False) >= 0

                    if v.state == 'NEUTRAL' and in_w: v.state = 'IN_WAITING_ZONE'
                    elif v.state == 'IN_WAITING_ZONE' and in_v:
                        if self.stable_light_color in ['red', 'yellow'] and self.latest_detected_light_color != 'green' and self.frame_count >= self.no_violation_until_frame:
                            v.state = 'COMMITTED_VIOLATION'
                            self.process_violation(v, frame_original, scale, job_id, self.frame_count)
                        else: v.state = 'PASSED_LEGALLY'
                timing['logic'] += time.perf_counter() - t0

            if out: self.draw_results(frame_original, scale, waiting_zone_pts, violation_zone_pts, out, light_bbox_orig)

            if self.frame_count % self.options['status_update_interval'] == 0:
                with processing_lock:
                    progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
                    processing_status[job_id] = {'status': 'processing', 'progress': progress, 'violations': len(self.violations_data)}

        if out: out.release()
        cap.release()
        database.save_violations_to_db(job_id, self.violations_data)
        if not self.is_stream: database.save_processed_video(job_id, f'processed_{job_id}.mp4')
        
        with processing_lock:
            processing_status[job_id] = {'status': 'completed', 'violations_found': len(self.violations_data)}
            processing_results[job_id] = {'violations': self.violations_data}

    def _cleanup_stale_tracks(self, frame_count):
        stale = [tid for tid, v in self.active_tracks.items() if frame_count - v.last_seen_frame > config.TRACK_TIMEOUT_FRAMES]
        for tid in stale: del self.active_tracks[tid]

    def process_violation(self, vehicle, frame, scale, job_id, frame_count):
        bbox_orig = [int(v / scale) for v in vehicle.bbox]
        _, plate, conf = self.detector.extract_and_recognize_plate(frame, bbox_orig)
        vehicle.license_plate = plate
        violation = {'track_id': vehicle.track_id, 'timestamp': datetime.datetime.now().isoformat(), 'frame_number': frame_count, 'license_plate': plate, 'confidence': conf, 'bbox': bbox_orig}
        self.violations_data.append(violation)
        self._save_violation_image(frame, bbox_orig, job_id, vehicle.track_id)

    def _save_violation_image(self, frame, bbox, job_id, track_id):
        x, y, w, h = bbox
        crop = frame[max(0,y):y+h, max(0,x):x+w]
        if crop.size > 0: cv2.imwrite(os.path.join(config.VIOLATIONS_FOLDER, f'violation_{job_id}_{track_id}.jpg'), crop)

    def draw_results(self, frame, scale, wzone, vzone, out=None, lbbox=None):
        viz = visualize_roi(frame, wzone, vzone)
        # ... simplified drawing logic for internal use ...
        if out: out.write(viz)
        return viz
