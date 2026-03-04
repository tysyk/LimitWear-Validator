from detectors.ocr.easyocr_detector import run_ocr
from detectors.lines.hough_lines_detector import detect_lines
from utils.deskew import estimate_skew_angle_deg, rotate_bgr

def run(ctx):
    # 1) базово OCR на оригіналі (щоб не втрачати)
    ocr_base = run_ocr(ctx.bgr)

    lines = []
    skew_angle = None
    bgr_used = ctx.bgr

    # 2) Якщо схоже на скетч — пробуємо deskew
    if ctx.scene.get("type") == "sketch_scan":
        lines0 = detect_lines(ctx.bgr, max_lines=60)
        skew_angle = estimate_skew_angle_deg(lines0)

        # якщо перекіс суттєвий — повертаємо
        if skew_angle is not None and abs(skew_angle) >= 2.5:
            bgr_used = rotate_bgr(ctx.bgr, -skew_angle)  # мінус щоб вирівняти
            ctx.debug["deskew"] = {"applied": True, "angle_deg": skew_angle}
        else:
            ctx.debug["deskew"] = {"applied": False, "angle_deg": skew_angle}

        # лінії вже рахуємо на bgr_used (вирівняному)
        lines = detect_lines(bgr_used, max_lines=60)

    # 3) OCR краще робити на bgr_used (бо після вирівнювання може зчитатися краще)
    ocr = run_ocr(bgr_used)

    ctx.detections = {
        "ocr": ocr,
        "ocr_base": ocr_base,
        "lines": lines,
        "logo": []
    }

    # збережемо, яке зображення реально аналізували для Explain (щоб рамки були консистентні)
    ctx.debug["image_used"] = "deskewed" if bgr_used is not ctx.bgr else "original"
    ctx.debug["skew_angle_deg"] = skew_angle
    ctx.bgr_used = bgr_used