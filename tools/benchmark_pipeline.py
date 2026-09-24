import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_size(value):
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must look like 1280x720") from exc


def bench_import_app():
    start = time.perf_counter()
    import app

    elapsed = time.perf_counter() - start
    return app, {
        "import_app_s": round(elapsed, 3),
        "detector_loaded_after_import": app.detector is not None,
    }


def bench_model_load(app_module):
    start = time.perf_counter()
    detector = app_module.get_detector()
    elapsed = time.perf_counter() - start
    return detector, {"model_load_s": round(elapsed, 3)}


def bench_synthetic(frames, size):
    import config
    from video_processor import VideoProcessor

    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (width // 3, height // 3), (width // 3 + 160, height // 3 + 90), (40, 40, 40), -1)
    cv2.circle(frame, (width // 10, height // 10), 18, (0, 0, 255), -1)
    processor = VideoProcessor("__synthetic__.mp4", detector=None, options={"write_output_video": False})
    waiting = [[width // 4, height // 2], [width // 2, height // 2], [width // 2, height - 80], [width // 4, height - 80]]
    violation = [[width // 2, height // 2], [width - 80, height // 2], [width - 80, height - 80], [width // 2, height - 80]]

    resize_s = 0.0
    draw_s = 0.0
    for _ in range(frames):
        start = time.perf_counter()
        processed, scale = processor._resize_frame(frame, config.PROCESSING_FRAME_WIDTH)
        resize_s += time.perf_counter() - start

        start = time.perf_counter()
        processor.draw_results(frame, scale, waiting, violation, out_writer=None)
        draw_s += time.perf_counter() - start

    total = resize_s + draw_s
    return {
        "synthetic_frames": frames,
        "synthetic_size": f"{width}x{height}",
        "resize_s": round(resize_s, 3),
        "draw_s": round(draw_s, 3),
        "synthetic_total_s": round(total, 3),
        "synthetic_fps_overhead_only": round(frames / total, 2) if total > 0 else None,
    }


def bench_lpr_image(detector, image_path):
    if not image_path:
        return {"lpr_image": None, "lpr_smoke": "skipped"}
    image = cv2.imread(image_path)
    if image is None:
        return {"lpr_image": image_path, "lpr_smoke": "image_not_readable"}
    start = time.perf_counter()
    text, confidence = detector.lp_detector.recognize(image)
    elapsed = time.perf_counter() - start
    return {
        "lpr_image": image_path,
        "lpr_text": text,
        "lpr_confidence": round(float(confidence or 0), 3),
        "lpr_s": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RedLight AI pipeline without requiring a real test video.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic frame resize/draw benchmark.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--no-models", action="store_true", help="Skip AI model loading benchmark.")
    parser.add_argument("--lpr-image", default="", help="Optional image path for LPR smoke test.")
    args = parser.parse_args()

    results = {}
    app_module, import_results = bench_import_app()
    results.update(import_results)

    detector = None
    if not args.no_models:
        detector, model_results = bench_model_load(app_module)
        results.update(model_results)

    if args.synthetic:
        results.update(bench_synthetic(args.frames, args.size))

    if detector is not None:
        results.update(bench_lpr_image(detector, args.lpr_image))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
