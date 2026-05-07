import os
import logging
from celery import Celery
import config
from video_processor import VideoProcessor
from detector_manager import TrafficViolationDetector
from threading import Lock

logger = logging.getLogger(__name__)

# Khởi tạo Celery
celery_app = Celery('traffic_tasks', broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)

# Singleton detector để dùng chung trong worker
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = TrafficViolationDetector()
    return _detector

@celery_app.task(bind=True)
def process_video_task(self, video_path, job_id, options=None):
    """
    Celery task để xử lý video.
    """
    logger.info(f"Starting celery task for job {job_id}")
    
    # Cập nhật trạng thái ban đầu
    self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'initializing'})
    
    detector = get_detector()
    processor = VideoProcessor(video_path, detector, options)
    
    # Mock status dict to satisfy VideoProcessor signature
    # In Celery, we use self.update_state instead of a shared dict
    status_proxy = {}
    results_proxy = {}
    lock_proxy = Lock()
    
    def status_callback(current_status):
        # Hàm này sẽ được VideoProcessor gọi định kỳ
        progress = current_status.get('progress', 0)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'violations_found': current_status.get('violations_found', 0),
            'status': 'processing'
        })

    # Chúng ta cần chỉnh sửa VideoProcessor một chút để hỗ trợ callback hoặc đơn giản là dùng nó như hiện tại
    # Tuy nhiên, VideoProcessor hiện tại đang ghi vào một dict chung.
    # Ta sẽ wrap nó lại.
    
    try:
        processor.process_video(job_id, status_proxy, results_proxy, lock_proxy)
        
        # Sau khi hoàn thành, trả về kết quả
        return {
            'status': 'completed',
            'job_id': job_id,
            'violations_found': len(processor.violations_data),
            'output_video': status_proxy.get(job_id, {}).get('output_video', '')
        }
    except Exception as e:
        logger.error(f"Error in celery task: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
