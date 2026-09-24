import logging
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LPRResult:
    plate_crop: Optional[np.ndarray]
    text: str
    confidence: float


class YOLOv5LicensePlateRecognizer:
    """
    Lightweight Vietnamese license plate recognizer using the two YOLOv5 nano
    models stored under models/lpr.
    """

    def __init__(
        self,
        detector_model_path: str,
        ocr_model_path: str,
        yolov5_path: str,
        conf_threshold: float = 0.45,
        ocr_img_size: int = 640,
        max_candidates: int = 2,
        cache_enabled: bool = True,
        cache_size: int = 128,
    ):
        self.detector_model_path = detector_model_path
        self.ocr_model_path = ocr_model_path
        self.yolov5_path = yolov5_path
        self.conf_threshold = conf_threshold
        self.ocr_img_size = ocr_img_size
        self.max_candidates = max(1, int(max_candidates))
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self._cache: OrderedDict[str, LPRResult] = OrderedDict()

        self.torch = None
        self.device = "cpu"
        self.plate_detector = None
        self.ocr_model = None
        self.available = False

        self._load_models()

    def _load_models(self):
        missing_paths = [
            path for path in [self.detector_model_path, self.ocr_model_path, self.yolov5_path]
            if not os.path.exists(path)
        ]
        if missing_paths:
            raise FileNotFoundError("Missing LPR resource(s): " + ", ".join(missing_paths))

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"PyTorch is required for YOLOv5 LPR: {exc}") from exc

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading YOLOv5 LPR models on %s...", self.device.upper())

        load_start = time.perf_counter()
        try:
            detector_start = time.perf_counter()
            self.plate_detector = torch.hub.load(
                self.yolov5_path,
                "custom",
                path=self.detector_model_path,
                source="local",
                force_reload=False,
            )
            logger.info("YOLOv5 plate detector loaded in %.2fs", time.perf_counter() - detector_start)
            ocr_start = time.perf_counter()
            self.ocr_model = torch.hub.load(
                self.yolov5_path,
                "custom",
                path=self.ocr_model_path,
                source="local",
                force_reload=False,
            )
            logger.info("YOLOv5 plate OCR loaded in %.2fs", time.perf_counter() - ocr_start)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YOLOv5 local loader needs an installed dependency. "
                "Run `pip install -r requirements.txt` and `pip install -r yolov5/requirements.txt`."
            ) from exc

        for model in [self.plate_detector, self.ocr_model]:
            model.conf = self.conf_threshold
            try:
                model.to(self.device)
            except Exception:
                logger.debug("YOLOv5 AutoShape model did not expose .to(); continuing.")

        self.available = True
        logger.info("YOLOv5 LPR models loaded successfully in %.2fs.", time.perf_counter() - load_start)

    def recognize(self, vehicle_roi: np.ndarray) -> LPRResult:
        if vehicle_roi is None or vehicle_roi.size == 0:
            return LPRResult(None, "NO_ROI", 0.0)

        cache_key = self._cache_key(vehicle_roi)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache.pop(cache_key)
            self._cache[cache_key] = cached
            return cached

        result = self._recognize_uncached(vehicle_roi)
        if self.cache_enabled:
            self._cache[cache_key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def _recognize_uncached(self, vehicle_roi: np.ndarray) -> LPRResult:
        detections = self._detect_rows(self.plate_detector, vehicle_roi)
        if not detections:
            return LPRResult(vehicle_roi, "NOT_FOUND", 0.0)

        best = max(detections, key=lambda row: row[4])
        x1, y1, x2, y2 = self._clip_box(best[:4], vehicle_roi.shape)
        plate_crop = vehicle_roi[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return LPRResult(vehicle_roi, "CROP_FAILED", 0.0)

        detector_conf = float(best[4])
        best_text = "UNKNOWN"
        best_conf = 0.0

        for candidate in self._plate_candidates(plate_crop):
            text, char_conf = self._read_plate(candidate)
            if text == "unknown":
                continue
            formatted, format_conf = self._normalize_plate(text)
            score = max(0.0, min(0.99, detector_conf * 0.45 + char_conf * 0.35 + format_conf * 0.20))
            if formatted and score > best_conf:
                best_text = formatted
                best_conf = score

        if best_text == "UNKNOWN":
            return LPRResult(plate_crop, "UNKNOWN", max(0.0, detector_conf * 0.4))
        return LPRResult(plate_crop, best_text, best_conf)

    def _read_plate(self, plate_img: np.ndarray) -> Tuple[str, float]:
        rows = self._detect_rows(self.ocr_model, plate_img, img_size=self.ocr_img_size)
        if len(rows) < 7 or len(rows) > 10:
            return "unknown", 0.0

        centers = []
        y_sum = 0.0
        conf_sum = 0.0
        for row in rows:
            x1, y1, x2, y2, conf, name = row
            x_c = (x1 + x2) / 2.0
            y_c = (y1 + y2) / 2.0
            y_sum += y_c
            conf_sum += float(conf)
            centers.append([x_c, y_c, str(name)])

        two_line = self._is_two_line_plate(centers)
        if two_line:
            y_mean = y_sum / len(centers)
            line_1 = [c for c in centers if c[1] <= y_mean]
            line_2 = [c for c in centers if c[1] > y_mean]
            text = "".join(c[2] for c in sorted(line_1, key=lambda c: c[0]))
            text += "".join(c[2] for c in sorted(line_2, key=lambda c: c[0]))
        else:
            text = "".join(c[2] for c in sorted(centers, key=lambda c: c[0]))

        return text, conf_sum / max(1, len(rows))

    def _detect_rows(self, model, image: np.ndarray, img_size: Optional[int] = None) -> List[list]:
        with self.torch.no_grad():
            if img_size:
                results = model(image, size=img_size)
            else:
                results = model(image)

        rows = []
        names = getattr(model, "names", {})
        try:
            tensor_rows = results.xyxy[0].detach().cpu().numpy().tolist()
            for row in tensor_rows:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                if float(conf) < self.conf_threshold:
                    continue
                name = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
                rows.append([float(x1), float(y1), float(x2), float(y2), float(conf), name])
            return rows
        except Exception:
            pass

        try:
            df_rows = results.pandas().xyxy[0].values.tolist()
            for row in df_rows:
                if float(row[4]) >= self.conf_threshold:
                    rows.append(row)
        except Exception as exc:
            logger.warning("Could not parse YOLOv5 detections: %s", exc)
        return rows

    def _plate_candidates(self, plate_crop: np.ndarray) -> Sequence[np.ndarray]:
        enhanced = self._change_contrast(plate_crop)
        candidates = [plate_crop, enhanced]
        for change_contrast in [0, 1]:
            for center_threshold in [0, 1]:
                if len(candidates) >= self.max_candidates:
                    return candidates[:self.max_candidates]
                try:
                    candidates.append(self._deskew(plate_crop, change_contrast, center_threshold))
                except Exception:
                    continue
        return candidates[:self.max_candidates]

    def _is_two_line_plate(self, centers: List[list]) -> bool:
        left = min(centers, key=lambda c: c[0])
        right = max(centers, key=lambda c: c[0])
        if math.isclose(left[0], right[0]):
            return False
        for center in centers:
            if not self._point_near_line(center[0], center[1], left[0], left[1], right[0], right[1]):
                return True
        return False

    @staticmethod
    def _point_near_line(x, y, x1, y1, x2, y2) -> bool:
        if math.isclose(x1, x2):
            return math.isclose(x, x1, abs_tol=3)
        b = y1 - (y2 - y1) * x1 / (x2 - x1)
        a = (y1 - b) / x1 if not math.isclose(x1, 0.0) else (y2 - y1) / (x2 - x1)
        y_pred = a * x + b
        return math.isclose(y_pred, y, abs_tol=3)

    @staticmethod
    def _change_contrast(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def _deskew(self, src_img: np.ndarray, change_contrast: int, center_threshold: int) -> np.ndarray:
        img = self._change_contrast(src_img) if change_contrast == 1 else src_img
        angle = self._compute_skew(img, center_threshold)
        image_center = tuple(np.array(src_img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(src_img, rot_mat, src_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    @staticmethod
    def _compute_skew(src_img: np.ndarray, center_threshold: int) -> float:
        h, w = src_img.shape[:2]
        img = cv2.medianBlur(src_img, 3)
        edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)
        if lines is None:
            return 1.0

        min_line = 100
        min_line_pos = 0
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
                if center_threshold == 1 and center_point[1] < 7:
                    continue
                if center_point[1] < min_line:
                    min_line = center_point[1]
                    min_line_pos = i

        angle = 0.0
        count = 0
        for x1, y1, x2, y2 in lines[min_line_pos]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(ang) <= 30:
                angle += ang
                count += 1
        return (angle / count) * 180 / math.pi if count else 0.0

    @staticmethod
    def _clip_box(box, shape) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _cache_key(image: np.ndarray) -> str:
        thumb = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        return str(hash(thumb.tobytes()))

    @staticmethod
    def _normalize_plate(text: str) -> Tuple[Optional[str], float]:
        clean = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
        if not clean:
            return None, 0.0

        char_to_digit = {"D": "0", "B": "8", "S": "5", "A": "4", "I": "1", "L": "1", "Z": "2", "O": "0", "G": "6", "Q": "0"}
        digit_to_char = {"0": "D", "8": "B", "5": "S", "4": "A", "1": "I", "2": "Z"}

        if 7 <= len(clean) <= 10:
            province = list(clean[0:2])
            series = list(clean[2:3])
            number = list(clean[3:])
            for i, value in enumerate(province):
                province[i] = char_to_digit.get(value, value)
            if series:
                series[0] = digit_to_char.get(series[0], series[0])
            for i, value in enumerate(number):
                number[i] = char_to_digit.get(value, value)
            clean = "".join(province + series + number)

        match_car_5 = re.match(r"^([0-9]{2}[A-Z])([0-9]{5})$", clean)
        if match_car_5:
            p1, p2 = match_car_5.groups()
            return f"{p1}-{p2[:3]}.{p2[3:]}", 0.95

        match_car_4 = re.match(r"^([0-9]{2}[A-Z])([0-9]{4})$", clean)
        if match_car_4:
            p1, p2 = match_car_4.groups()
            return f"{p1}-{p2}", 0.90

        if 7 <= len(clean) <= 10:
            return clean, 0.65
        return clean, 0.35
