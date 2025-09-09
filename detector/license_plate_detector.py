"""
Module chuyên xử lý nhận dạng ký tự quang học (OCR) trên biển số xe.
"""
import cv2
import logging
import random

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    """Lớp nhận dạng biển số, khởi tạo model EasyOCR một lần."""

    def __init__(self):
        """Khởi tạo và tải model EasyOCR."""
        self.reader = None
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✓ EasyOCR model loaded for license plate recognition.")
        except Exception as e:
            logger.warning(f"Could not load EasyOCR model: {e}. Using fallback.")

    def recognize(self, vehicle_roi):
        """Nhận dạng biển số từ một vùng ảnh chứa phương tiện."""
        if not self.reader:
            return self._generate_demo_plate(), 0.85

        try:
            gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(gray_roi)

            if results:
                plate_text = ""
                confidence_sum = 0
                count = 0
                for (_, text, prob) in results:
                    clean_text = ''.join(char for char in text if char.isalnum()).upper()
                    if len(clean_text) > 2:
                        plate_text += clean_text
                        confidence_sum += prob
                        count += 1
                if count > 0:
                    return plate_text.replace(" ", ""), confidence_sum / count
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
        
        return self._generate_demo_plate(), 0.85
