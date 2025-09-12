"""
Module cải tiến cho việc phát hiện và quản lý ROI
"""
import os, json
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Thư mục lưu ROI và stopline
ROI_DIR = os.path.join("config", "rois")
os.makedirs(ROI_DIR, exist_ok=True)

def roi_path(camera_id):
    """Đường dẫn đến file cấu hình ROI"""
    return os.path.join(ROI_DIR, f"{camera_id}.json")

def save_rois(camera_id, waiting_pts, violation_pts):
    """
    Lưu cấu hình ROI với vùng chờ và vùng vi phạm
    
    Args:
        camera_id: ID của camera
        waiting_pts: Danh sách điểm [(x,y), ...] tạo vùng chờ
        violation_pts: Danh sách điểm [(x,y), ...] tạo vùng vi phạm
    """
    data = {
        "waiting_zone": waiting_pts,    # [(x,y), ...]
        "violation_zone": violation_pts  # [(x,y), ...]
    }
    
    with open(roi_path(camera_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã lưu ROI cho camera {camera_id}: {len(waiting_pts)} điểm vùng chờ, "
                f"{len(violation_pts)} điểm vùng vi phạm")
    return True

def load_rois(camera_id):
    """
    Đọc cấu hình ROI từ file
    
    Returns:
        tuple: (waiting_pts, violation_pts)
        - waiting_pts: Danh sách điểm [(x,y), ...] tạo vùng chờ hoặc []
        - violation_pts: Danh sách điểm [(x,y), ...] tạo vùng vi phạm hoặc []
    """
    p = roi_path(camera_id)
    if not os.path.exists(p):
        return [], []
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        w = d.get("waiting_zone") or []
        v = d.get("violation_zone") or []
        
        return w, v
    except Exception as e:
        logger.error(f"Lỗi khi đọc ROI từ {p}: {e}")
        return [], []

def point_in_polygon(point_xy, polygon_pts):
    """
    Kiểm tra điểm có nằm trong đa giác không
    
    Args:
        point_xy: Tọa độ điểm (x, y)
        polygon_pts: Danh sách điểm đa giác [(x,y), ...]
    
    Returns:
        bool: True nếu điểm nằm trong đa giác
    """
    # polygon_pts: [(x,y), ...], point: (cx, cy)
    if not polygon_pts or len(polygon_pts) < 3:
        return False
    cnt = np.array(polygon_pts, dtype=np.int32)
    res = cv2.pointPolygonTest(cnt, point_xy, False)
    return res >= 0

def check_violation_with_roi(vehicle_bbox, violation_zone_pts, waiting_zone_pts, traffic_light_color):
    """
    Kiểm tra vi phạm dựa trên ROI và đèn tín hiệu
    
    Args:
        vehicle_bbox: Tuple (x, y, w, h) của vehicle
        violation_zone_pts: Danh sách điểm [(x,y), ...] tạo vùng vi phạm
        waiting_zone_pts: Danh sách điểm [(x,y), ...] tạo vùng chờ
        traffic_light_color: Màu đèn tín hiệu ("red", "green", "yellow")
    
    Returns:
        bool: True nếu vi phạm
    """
    # Lấy tâm của bounding box xe
    x, y, w, h = vehicle_bbox
    center = (x + w // 2, y + h // 2)
    bottom_center = (x + w // 2, y + h)  # Điểm dưới cùng ở giữa xe
    
    # Nếu không có dữ liệu ROI thì không thể kiểm tra
    if not violation_zone_pts or len(violation_zone_pts) < 3:
        return False
    
    # Kiểm tra nếu xe trong vùng chờ
    if waiting_zone_pts and len(waiting_zone_pts) >= 3:
        if point_in_polygon(center, waiting_zone_pts):
            return False  # Đang chờ, không vi phạm
    
    # Kiểm tra nếu xe trong vùng vi phạm và đèn đỏ
    # Sử dụng cả điểm giữa và điểm dưới cùng để tăng độ chính xác
    if traffic_light_color == "red":
        if point_in_polygon(center, violation_zone_pts) or point_in_polygon(bottom_center, violation_zone_pts):
            return True  # Vi phạm
    
    return False

def visualize_roi(frame, waiting_pts=None, violation_pts=None):
    """
    Vẽ vùng ROI (vùng chờ và vùng vi phạm) lên frame
    
    Args:
        frame: Frame hình ảnh
        waiting_pts: Danh sách điểm [(x,y), ...] tạo vùng chờ
        violation_pts: Danh sách điểm [(x,y), ...] tạo vùng vi phạm
    
    Returns:
        frame: Frame đã vẽ
    """
    h, w = frame.shape[:2]
    frame_viz = frame.copy()
    
    # Vẽ vùng chờ (màu vàng nhạt)
    if waiting_pts and len(waiting_pts) >= 3:
        waiting_pts_arr = np.array(waiting_pts, dtype=np.int32)
        overlay = frame_viz.copy()
        cv2.fillPoly(overlay, [waiting_pts_arr], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, frame_viz, 0.7, 0, frame_viz)
        cv2.polylines(frame_viz, [waiting_pts_arr], True, (0, 255, 255), 2)
        
        # Thêm label
        cv2.putText(
            frame_viz, "VÙNG CHỜ", (waiting_pts_arr[0][0], waiting_pts_arr[0][1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Vẽ vùng vi phạm (màu đỏ nhạt)
    if violation_pts and len(violation_pts) >= 3:
        violation_pts_arr = np.array(violation_pts, dtype=np.int32)
        overlay = frame_viz.copy()
        cv2.fillPoly(overlay, [violation_pts_arr], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame_viz, 0.7, 0, frame_viz)
        cv2.polylines(frame_viz, [violation_pts_arr], True, (0, 0, 255), 2)
        
        # Thêm label
        cv2.putText(
            frame_viz, "VÙNG VI PHẠM", (violation_pts_arr[0][0], violation_pts_arr[0][1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame_viz

def auto_detect_roi(frame):
    """
    Tự động phát hiện và đề xuất vùng ROI dựa trên hình ảnh
    
    Args:
        frame: Frame hình ảnh
    
    Returns:
        tuple: (waiting_pts, violation_pts) với các điểm tự động phát hiện
              hoặc ([], []) nếu không phát hiện được
    """
    from detector.traffic_light_detector import detect_red_lights
    
    h, w = frame.shape[:2]
    waiting_pts = []
    violation_pts = []
    
    # Phát hiện đèn đỏ để tìm khu vực giao lộ
    red_lights = detect_red_lights(frame)
    
    # Nếu không tìm thấy đèn đỏ, trả về rỗng
    if not red_lights:
        return [], []
    
    # Lấy đèn đỏ đầu tiên làm tham chiếu
    traffic_light = red_lights[0]
    x, y, tw, th = traffic_light
    
    # Tính toán vùng vi phạm (khu vực giao lộ)
    # Thông thường là vùng phía trước đèn giao thông
    margin_x = w // 4  # Mở rộng ra hai bên 1/4 chiều rộng
    top_y = y + th + 20  # Bắt đầu ngay dưới đèn giao thông
    bottom_y = int(h * 0.7)  # Kết thúc ở khoảng 70% chiều cao của frame
    
    violation_pts = [
        [max(0, x - margin_x), top_y],  # Trên trái
        [min(w, x + tw + margin_x), top_y],  # Trên phải
        [min(w, x + tw + margin_x), bottom_y],  # Dưới phải
        [max(0, x - margin_x), bottom_y]  # Dưới trái
    ]
    
    # Tính toán vùng chờ (khu vực trước vạch dừng)
    waiting_top_y = bottom_y + 20  # Bắt đầu sau vùng vi phạm
    waiting_bottom_y = h - 50  # Kết thúc gần cuối frame
    
    waiting_pts = [
        [max(0, x - margin_x), waiting_top_y],  # Trên trái
        [min(w, x + tw + margin_x), waiting_top_y],  # Trên phải
        [min(w, x + tw + margin_x), waiting_bottom_y],  # Dưới phải
        [max(0, x - margin_x), waiting_bottom_y]  # Dưới trái
    ]
    
    return waiting_pts, violation_pts