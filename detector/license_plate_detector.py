# license_plate_detector.py - Enhanced with GPU Optimization

import cv2
import easyocr
import logging
import numpy as np
import re
from ultralytics import YOLO
from typing import Tuple, Optional
import torch

# Cấu hình logging
logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    Lớp chuyên trách cho việc nhận dạng biển số xe với tối ưu hóa GPU.

    Tính năng chính:
    1. Phát hiện biển số trong ảnh xe (sử dụng YOLOv8 với GPU acceleration).
    2. Nhận dạng ký tự trên biển số (sử dụng EasyOCR với GPU support).
    3. Tự động phát hiện và cấu hình GPU/CPU tối ưu.
    4. Quản lý bộ nhớ GPU thông minh để tránh memory leak.
    5. Hỗ trợ xử lý batch để tăng hiệu suất.

    Tối ưu hóa GPU:
    - YOLOv8 với FP16 precision
    - EasyOCR với GPU acceleration
    - CuDNN optimizations
    - Automatic memory management
    - Batch processing support
    """

    def __init__(self, model_path='license_plate_detector.pt'):
        """
        Khởi tạo các model cần thiết với tối ưu hóa GPU.
        - model_path: Đường dẫn đến model YOLOv8 đã được huấn luyện để phát hiện biển số.
        """
        try:
            # 1. Kiểm tra và cấu hình GPU
            self._configure_gpu_settings()

            # 2. Tải model YOLO để phát hiện biển số với tối ưu hóa GPU (nếu có)
            self.plate_detector = None
            try:
                self.plate_detector = YOLO(model_path)
                self.yolo_conf_threshold = 0.25

                # Tối ưu hóa YOLO cho GPU nếu khả dụng
                if self.gpu_available:
                    logger.info("Đang tối ưu hóa YOLO cho GPU...")
                    self.plate_detector.to(self.device)

                    # Sử dụng config để quyết định FP16
                    import config
                    if config.USE_HALF_PRECISION and hasattr(self.plate_detector.model, 'half'):
                        try:
                            self.plate_detector.model.half()
                            logger.info("  ✓ YOLO FP16 đã được kích hoạt.")
                        except Exception as e:
                            logger.info(f"  ⓘ Bỏ qua FP16 cho YOLO: {e}")
                    else:
                        logger.info("  ⓘ FP16 bị tắt trong config hoặc không hỗ trợ.")

                    if torch.cuda.is_available():
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                        logger.info("  ✓ CuDNN optimizations đã được kích hoạt.")

                logger.info(f"✓ Model phát hiện biển số '{model_path}' đã được tải thành công (Device: {self.device}).")
            except Exception as e:
                logger.warning(f"Không thể tải model YOLO biển số ('{model_path}'): {e}. Sẽ fallback OCR toàn ROI xe.")
                self.plate_detector = None

            # 3. Khởi tạo bộ đọc OCR với GPU support
            logger.info("Đang khởi tạo EasyOCR...")
            try:
                # Khởi tạo đơn giản và an toàn
                self.reader = easyocr.Reader(['en'], gpu=self.gpu_available)
                logger.info(f"✓ EasyOCR reader đã được khởi tạo thành công (GPU: {self.gpu_available}).")
            except Exception as ocr_error:
                logger.warning(f"Không thể khởi tạo EasyOCR với GPU, thử lại với CPU: {ocr_error}")
                try:
                    self.reader = easyocr.Reader(['en'], gpu=False)
                    logger.info("✓ EasyOCR đã được khởi tạo với CPU (fallback).")
                except Exception as cpu_error:
                    logger.error(f"Không thể khởi tạo EasyOCR ngay cả với CPU: {cpu_error}")
                    self.reader = None

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi khởi tạo LicensePlateDetector: {e}", exc_info=True)
            raise

    def _configure_gpu_settings(self):
        """Cấu hình các thiết lập GPU và kiểm tra tính khả dụng."""
        try:
            import torch

            # Kiểm tra GPU availability
            self.gpu_available = torch.cuda.is_available()
            self.device = 'cuda' if self.gpu_available else 'cpu'

            if self.gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.info(f"✓ GPU phát hiện: {gpu_name} ({gpu_count} GPU, {gpu_memory:.1f}GB VRAM)")

                # Set optimal GPU settings
                torch.cuda.set_device(0)  # Use first GPU
                logger.info("✓ Đã cấu hình sử dụng GPU chính.")
            else:
                logger.info("✓ Sử dụng CPU cho xử lý (GPU không khả dụng)")

        except ImportError:
            logger.warning("PyTorch không khả dụng, sử dụng CPU fallback")
            self.gpu_available = False
            self.device = 'cpu'

    def get_gpu_stats(self):
        """Lấy thông tin sử dụng GPU hiện tại."""
        if not self.gpu_available:
            return "GPU không khả dụng"

        try:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2   # MB
            return f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB"
        except Exception as e:
            return f"Không thể lấy thông tin GPU: {e}"

    def _correct_perspective(self, roi):
        """
        Cải tiến cốt lõi: Tìm contour của biển số và "làm phẳng" nó.
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4:
                        best_contour = approx
                        break
            
            if best_contour is not None:
                pts = best_contour.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(roi, M, (maxWidth, maxHeight))
                logger.debug("Perspective correction applied successfully.")
                return warped
        except Exception as e:
            logger.warning(f"Could not apply perspective correction: {e}")
        
        return roi

    def _gpu_memory_cleanup(self):
        """Dọn dẹp bộ nhớ GPU để tránh memory leak."""
        try:
            if self.gpu_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared.")
        except Exception as e:
            logger.debug(f"Không thể dọn dẹp GPU memory: {e}")

    def recognize_batch(self, vehicle_rois: list) -> list:
        """
        Nhận dạng biển số từ nhiều ảnh cùng lúc để tối ưu hiệu suất GPU.
        - vehicle_rois: Danh sách các ảnh ROI chứa phương tiện.
        Returns: Danh sách tuple (plate_text, confidence) tương ứng với từng ảnh.
        """
        if self.reader is None:
            logger.error("EasyOCR reader chưa được khởi tạo, không thể nhận dạng batch.")
            return [(None, 0.0) for _ in vehicle_rois]

        results = []

        # Xử lý từng ảnh một để tránh quá tải GPU
        for i, roi in enumerate(vehicle_rois):
            try:
                plate_text, confidence = self.recognize(roi)
                results.append((plate_text, confidence))

                # Dọn dẹp memory sau mỗi 10 ảnh
                if (i + 1) % 10 == 0:
                    self._gpu_memory_cleanup()

            except Exception as e:
                logger.error(f"Lỗi xử lý ảnh thứ {i+1}: {e}")
                results.append((None, 0.0))

        return results

    def recognize(self, vehicle_roi) -> Tuple[Optional[str], float]:
        """
        Hàm chính để thực hiện nhận dạng biển số từ một ảnh ROI của phương tiện.
        - vehicle_roi: Ảnh đã được cắt, chỉ chứa phương tiện.
        """
        if vehicle_roi is None or vehicle_roi.size == 0:
            return None, 0.0

        # Kiểm tra xem EasyOCR đã được khởi tạo thành công chưa
        if self.reader is None:
            logger.error("EasyOCR reader chưa được khởi tạo, không thể nhận dạng biển số.")
            return None, 0.0

        # BƯỚC 1: Tiền xử lý để cải thiện chất lượng ảnh
        processed_roi = self._preprocess_image(vehicle_roi)
        if len(processed_roi.shape) == 2:
            processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)
        # BƯỚC 2: Xác định vùng biển số bằng YOLO nếu có, nếu không fallback OCR toàn ROI
        best_plate_box = None
        max_conf = 0.0
        plate_crop = None

        with torch.no_grad():
            if self.plate_detector is None:
                plate_detections = None
            else:
                try:
                    device_to_use = self.device if self.gpu_available else 'cpu'
                    plate_detections = self.plate_detector(processed_roi, device=device_to_use, verbose=False, conf=self.yolo_conf_threshold)
                except RuntimeError as e:
                    if "dtype" in str(e) and "Half" in str(e):
                        # Nếu gặp lỗi dtype mismatch, thử lại với device='cpu'
                        logger.warning(f"Dtype mismatch in plate detection, falling back to CPU: {e}")
                        plate_detections = self.plate_detector(processed_roi, device='cpu', verbose=False, conf=self.yolo_conf_threshold)
                    else:
                        raise e

        # Tìm bounding box có độ tin cậy cao nhất (nếu có YOLO)
        for result in (plate_detections or []):
            if result.boxes:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf > max_conf:
                        max_conf = conf
                        best_plate_box = box.xyxy.cpu().numpy()[0].astype(int)

        if best_plate_box is None:
            logger.warning("Không tìm thấy biển số trong ảnh xe, fallback OCR toàn bộ ROI xe")
            plate_crop = processed_roi
        else:
            # Cắt ảnh chỉ chứa biển số
            x1, y1, x2, y2 = best_plate_box
            plate_crop = processed_roi[y1:y2, x1:x2]

        if plate_crop.size == 0:
            logger.warning("Ảnh biển số sau khi cắt bị rỗng")
            return None, 0.0

        # BƯỚC 3: DÙNG EASYOCR ĐỂ ĐỌC KÝ TỰ TỪ BIỂN SỐ (đa biến thể + hiệu chỉnh phối cảnh)
        try:
            # Hiệu chỉnh phối cảnh nếu có thể
            plate_crop = self._correct_perspective(plate_crop)

            # Phóng lớn để OCR tốt hơn
            ph, pw = plate_crop.shape[:2]
            if pw < 320:
                scale = 320.0 / max(1, pw)
                plate_crop = cv2.resize(plate_crop, (int(pw*scale), int(ph*scale)), interpolation=cv2.INTER_CUBIC)

            candidates = []
            # 1) Ảnh gốc (grayscale đã tiền xử lý)
            candidates.append(plate_crop)
            # 2) Adaptive threshold
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape)==3 else plate_crop
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
            candidates.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
            # 3) Invert
            inv = cv2.bitwise_not(gray)
            candidates.append(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))

            best_plate = None
            best_score = 0.0
            for cand in candidates:
                try:
                    ocr_results = self.reader.readtext(cand)
                    if not ocr_results:
                        continue
                    full_text = ""; total_confidence = 0.0
                    for (_, text, prob) in ocr_results:
                        full_text += text
                        total_confidence += float(prob)
                    cleaned = self._clean_plate_text(full_text)
                    formatted, score = self._post_process(cleaned)
                    # tăng trọng số nếu độ dài hợp lý
                    if formatted:
                        score = max(score, min(0.95, 0.6 + 0.05*len(cleaned)))
                        if score > best_score:
                            best_score = score
                            best_plate = formatted
                except Exception:
                    continue

            if best_plate:
                logger.info(f"✓ Biển số: {best_plate} (score={best_score:.2f})")
                return best_plate, best_score
            return None, 0.0

        except Exception as e:
            logger.error(f"Lỗi trong quá trình OCR: {e}")
            return None, 0.0
        finally:
            # Dọn dẹp bộ nhớ GPU sau khi xử lý
            self._gpu_memory_cleanup()

    def _preprocess_image(self, image):
        """
        Tiền xử lý ảnh để cải thiện chất lượng nhận dạng.
        """
        # Resize để có kích thước phù hợp
        h, w = image.shape[:2]
        scale = max(1, min(2, int(600 / w)))
        if scale > 1:
            image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Cải thiện độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Khử nhiễu
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised

    def _clean_plate_text(self, text: str) -> str:
        """
        Hàm hậu xử lý để làm sạch chuỗi ký tự nhận dạng được.
        - Loại bỏ các ký tự đặc biệt, chỉ giữ lại chữ cái và số.
        - Chuyển thành chữ hoa.
        """
        if not text:
            return ""
        # Chỉ giữ lại chữ cái và số
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned_text

    def _post_process(self, text: str) -> Tuple[Optional[str], float]: # <-- SỬA Ở ĐÂY
        """
        Hàm hậu xử lý TÍCH HỢP: Sửa lỗi OCR dựa trênบริบท và định dạng biển số.
        """
        if not text:
            return None, 0.0

        # 1. Chuẩn hóa đầu vào, chỉ giữ lại chữ và số
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Nếu chuỗi quá ngắn hoặc quá dài, trả về kết quả thô với điểm thấp
        if not (7 <= len(clean_text) <= 9):
            return clean_text, 0.5 

        # 2. Định nghĩa các cặp ký tự dễ nhầm lẫn
        digit_to_char_map = {'0': 'D', '8': 'B', '5': 'S', '4': 'A', '1': 'I', '2': 'Z'}
        char_to_digit_map = {'D': '0', 'B': '8', 'S': '5', 'A': '4', 'I': '1', 'L': '1', 'Z': '2', 'O': '0', 'G': '6', 'Q': '0'}

        # 3. Tách các thành phần của biển số để áp dụng quy tắc
        province_code = list(clean_text[0:2])
        series_char = list(clean_text[2:3])
        reg_number = list(clean_text[3:])

        # QUY TẮC 1: Mã tỉnh (2 ký tự đầu) phải là SỐ
        for i in range(len(province_code)):
            if province_code[i] in char_to_digit_map:
                province_code[i] = char_to_digit_map[province_code[i]]
        
        # QUY TẮC 2: Ký tự seri (vị trí thứ 3) phải là CHỮ CÁI
        if series_char and series_char[0] in digit_to_char_map:
            series_char[0] = digit_to_char_map[series_char[0]]

        # QUY TẮC 3: Số đăng ký (phần còn lại) phải là SỐ
        for i in range(len(reg_number)):
            if reg_number[i] in char_to_digit_map:
                reg_number[i] = char_to_digit_map[reg_number[i]]

        # 4. Ghép các thành phần đã sửa lại
        processed_text = "".join(province_code) + "".join(series_char) + "".join(reg_number)

        # 5. Kiểm tra và định dạng lại theo chuẩn Việt Nam
        # Biển số 5 số (VD: 51D12345)
        match_car_5 = re.match(r'([0-9]{2}[A-Z])([0-9]{5})', processed_text)
        if match_car_5:
            p1, p2 = match_car_5.groups()
            return f"{p1}-{p2[:3]}.{p2[3:]}", 0.95

        # Biển số 4 số (VD: 29H1234)
        match_car_4 = re.match(r'([0-9]{2}[A-Z])([0-9]{4})', processed_text)
        if match_car_4:
            p1, p2 = match_car_4.groups()
            return f"{p1}-{p2}", 0.90
            
        # Nếu không khớp định dạng nào, trả về kết quả đã xử lý với điểm thấp hơn
        return processed_text, 0.6
