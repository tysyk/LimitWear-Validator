import cv2

def run(ctx):
    bgr = ctx.bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    issues = []
    quality_score = 1.0

    if blur < 60:
        issues.append({"type": "BLUR", "message": f"Зображення розмите (blur={blur:.1f})."})
        quality_score -= 0.4

    if brightness < 60:
        issues.append({"type": "DARK", "message": f"Занадто темно (brightness={brightness:.1f})."})
        quality_score -= 0.3

    if brightness > 205:
        issues.append({"type": "OVEREXPOSED", "message": f"Занадто світло (brightness={brightness:.1f})."})
        quality_score -= 0.3

    quality_score = max(0.0, min(1.0, quality_score))

    ctx.quality = {
        "quality_score": quality_score,
        "metrics": {"blur": blur, "brightness": brightness},
        "issues": issues
    }
