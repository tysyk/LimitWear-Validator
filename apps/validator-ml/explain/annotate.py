import cv2

def clamp(v, a, b):
    return max(a, min(b, v))

def make_annotated(bgr, ctx):
    h, w = bgr.shape[:2]
    annotated = bgr.copy()

    # safe margin (4% of min side) for auto
    safe = int(min(w, h) * 0.04)
    cv2.rectangle(annotated, (safe, safe), (w - safe, h - safe), (255, 0, 0), 2)

    # OCR boxes (green)
    ocr = ctx.detections.get("ocr", [])
    for item in ocr:
        x1, y1, x2, y2 = item["bbox"]
        x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
        y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Lines (cyan-ish)
    lines = ctx.detections.get("lines", [])
    for ln in lines:
        (x1, y1) = ln["p1"]
        (x2, y2) = ln["p2"]
        x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
        y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
        cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Violations with bbox (red)
    for v in ctx.violations:
        bb = v.get("bbox")
        if bb:
            x1, y1, x2, y2 = bb
            x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
            y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # header text
    cv2.putText(
        annotated,
        f"verdict={ctx.verdict} score={ctx.score} ocr={len(ocr)} lines={len(lines)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return annotated
