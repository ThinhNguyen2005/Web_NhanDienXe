import logging
import re
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from detector.lpr_yolov5 import YOLOv5LicensePlateRecognizer

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    License plate recognizer facade used by TrafficViolationDetector.

    Primary path: YOLOv5 nano plate detector + YOLOv5 nano character OCR.
    EasyOCR is kept as a lazy fallback only.
    """

    def __init__(self):
        self.yolov5_lpr: Optional[YOLOv5LicensePlateRecognizer] = None
        self.reader = None
        self.gpu_available = False
        self.device = "cpu"

        self._configure_device()
        self._init_yolov5_lpr()

    @property
    def plate_detector(self):
        """Compatibility alias for older callsites."""
        return self.yolov5_lpr.plate_detector if self.yolov5_lpr else None

    def _configure_device(self):
        try:
            import torch

            self.gpu_available = torch.cuda.is_available()
            self.device = "cuda" if self.gpu_available else "cpu"
            if self.gpu_available:
                logger.info("LPR will use GPU: %s", torch.cuda.get_device_name(0))
            else:
                logger.info("LPR will use CPU.")
        except Exception:
            self.gpu_available = False
            self.device = "cpu"
            logger.info("PyTorch is not available during LPR device check; CPU/fallback mode only.")

    def _init_yolov5_lpr(self):
        try:
            self.yolov5_lpr = YOLOv5LicensePlateRecognizer(
                detector_model_path=config.LP_DETECTOR_MODEL_PATH,
                ocr_model_path=config.LP_OCR_MODEL_PATH,
                yolov5_path=config.YOLOV5_LOCAL_PATH,
                conf_threshold=config.LPR_CONF_THRESHOLD,
                ocr_img_size=config.LPR_OCR_IMG_SIZE,
                max_candidates=getattr(config, "LPR_MAX_CANDIDATES", 2),
                cache_enabled=config.LPR_FRAME_CACHE_ENABLED,
            )
            logger.info("Primary YOLOv5 LPR pipeline initialized.")
        except Exception as exc:
            self.yolov5_lpr = None
            logger.warning("Primary YOLOv5 LPR pipeline is unavailable: %s", exc)

    def _init_easyocr_fallback(self):
        if self.reader is not None:
            return
        if not getattr(config, "LPR_ENABLE_EASYOCR_FALLBACK", True):
            logger.info("EasyOCR fallback is disabled by config.")
            return
        try:
            import easyocr

            self.reader = easyocr.Reader(["en"], gpu=self.gpu_available)
            logger.info("EasyOCR fallback initialized (GPU: %s).", self.gpu_available)
        except Exception as exc:
            self.reader = None
            logger.warning("EasyOCR fallback could not be initialized: %s", exc)

    def get_gpu_stats(self):
        if not self.gpu_available:
            return "GPU khong kha dung"
        try:
            import torch

            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            return f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB"
        except Exception as exc:
            return f"Khong the lay thong tin GPU: {exc}"

    def recognize_batch(self, vehicle_rois: list) -> list:
        return [self.recognize(roi) for roi in vehicle_rois]

    def recognize(self, vehicle_roi) -> Tuple[Optional[str], float]:
        if vehicle_roi is None or vehicle_roi.size == 0:
            return "NO_ROI", 0.0

        if self.yolov5_lpr is not None:
            try:
                result = self.yolov5_lpr.recognize(vehicle_roi)
                terminal_yolov5_states = {"NOT_FOUND", "UNKNOWN", "CROP_FAILED", "NO_ROI"}
                if result.text not in terminal_yolov5_states:
                    logger.info("YOLOv5 LPR plate: %s (%.2f)", result.text, result.confidence)
                    return result.text, result.confidence
                if result.text == "UNKNOWN":
                    logger.info("YOLOv5 LPR returned UNKNOWN; skipping EasyOCR fallback to keep processing fast.")
                    return "UNKNOWN", result.confidence
                logger.info("YOLOv5 LPR returned %s; trying fallback if enabled.", result.text)
            except Exception as exc:
                logger.warning("YOLOv5 LPR failed: %s", exc)

        fallback_text, fallback_conf = self._recognize_with_easyocr(vehicle_roi)
        if fallback_text:
            return fallback_text, fallback_conf
        return "NOT_FOUND", 0.0

    def _recognize_with_easyocr(self, vehicle_roi) -> Tuple[Optional[str], float]:
        self._init_easyocr_fallback()
        if self.reader is None:
            return None, 0.0

        try:
            processed = self._preprocess_for_easyocr(vehicle_roi)
            candidates = [processed]
            if len(processed.shape) == 2:
                candidates.append(cv2.bitwise_not(processed))

            best_plate = None
            best_score = 0.0
            for candidate in candidates:
                ocr_input = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR) if len(candidate.shape) == 2 else candidate
                ocr_results = self.reader.readtext(ocr_input)
                if not ocr_results:
                    continue
                text = "".join(item[1] for item in ocr_results)
                avg_conf = sum(float(item[2]) for item in ocr_results) / max(1, len(ocr_results))
                formatted, format_conf = self._post_process(text)
                score = max(0.0, min(0.99, avg_conf * 0.75 + format_conf * 0.25))
                if formatted and score > best_score:
                    best_plate = formatted
                    best_score = score

            if best_plate:
                logger.info("EasyOCR fallback plate: %s (%.2f)", best_plate, best_score)
                return best_plate, best_score
        except Exception as exc:
            logger.warning("EasyOCR fallback failed: %s", exc)
        return None, 0.0

    @staticmethod
    def _preprocess_for_easyocr(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < 320:
            scale = 320.0 / max(1, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(enhanced, 7, 55, 55)

    @staticmethod
    def _post_process(text: str) -> Tuple[Optional[str], float]:
        clean = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
        if not clean:
            return None, 0.0

        char_to_digit = {"D": "0", "B": "8", "S": "5", "A": "4", "I": "1", "L": "1", "Z": "2", "O": "0", "G": "6", "Q": "0"}
        digit_to_char = {"0": "D", "8": "B", "5": "S", "4": "A", "1": "I", "2": "Z"}
        if 7 <= len(clean) <= 10:
            province = [char_to_digit.get(ch, ch) for ch in clean[0:2]]
            series = [digit_to_char.get(ch, ch) for ch in clean[2:3]]
            number = [char_to_digit.get(ch, ch) for ch in clean[3:]]
            clean = "".join(province + series + number)

        match_car_5 = re.match(r"^([0-9]{2}[A-Z])([0-9]{5})$", clean)
        if match_car_5:
            p1, p2 = match_car_5.groups()
            return f"{p1}-{p2[:3]}.{p2[3:]}", 0.95

        match_car_4 = re.match(r"^([0-9]{2}[A-Z])([0-9]{4})$", clean)
        if match_car_4:
            p1, p2 = match_car_4.groups()
            return f"{p1}-{p2}", 0.90

        return clean, 0.60 if 7 <= len(clean) <= 10 else 0.35
