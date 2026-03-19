from detectors.ocr.easyocr_detector import run_ocr
from detectors.lines.hough_lines_detector import detect_lines
from utils.deskew import estimate_skew_angle_deg, rotate_bgr


def run(ctx):
    bgr_original = ctx.bgr
    bgr_used = bgr_original
    ocr_base = []
    ocr = []
    lines = []
    skew_angle = None

    try:
        ocr = run_ocr(bgr_used, fast=True)
    except Exception as exc:
        ctx.add_warning(f"OCR on original image failed: {exc}")
        ocr_base = []

    if ctx.scene.get("type") == "sketch_scan":
        try:
            lines0 = detect_lines(bgr_original, max_lines=60)
            skew_angle = estimate_skew_angle_deg(lines0)

            if skew_angle is not None and abs(skew_angle) >= 2.5:
                bgr_used = rotate_bgr(bgr_original, -skew_angle)
                ctx.debug["deskew"] = {"applied": True, "angle_deg": skew_angle}
            else:
                ctx.debug["deskew"] = {"applied": False, "angle_deg": skew_angle}

            lines = detect_lines(bgr_used, max_lines=60)

        except Exception as exc:
            ctx.add_warning(f"Sketch line detection / deskew failed: {exc}")
            ctx.debug["deskew"] = {"applied": False, "angle_deg": None}
            lines = []
            skew_angle = None

    try:
        ocr = run_ocr(bgr_used)
    except Exception as exc:
        ctx.add_warning(f"OCR on processed image failed: {exc}")
        ocr = ocr_base

    ctx.detections = {
        "ocr": ocr,
        "ocr_base": ocr_base,
        "lines": lines,
        "logo": []
    }

    ctx.debug["image_used"] = "deskewed" if bgr_used is not bgr_original else "original"
    ctx.debug["skew_angle_deg"] = skew_angle
    ctx.bgr_used = bgr_used