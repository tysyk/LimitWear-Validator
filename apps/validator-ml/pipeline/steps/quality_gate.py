from __future__ import annotations

import cv2


MIN_WIDTH = 256
MIN_HEIGHT = 256
MIN_LAPLACIAN_VARIANCE = 40.0


def run(ctx) -> None:
    image = ctx.bgr

    if image is None:
        ctx.fail("quality_gate", "Input image is empty")
        return

    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    passed_resolution = w >= MIN_WIDTH and h >= MIN_HEIGHT
    passed_blur = blur_score >= MIN_LAPLACIAN_VARIANCE

    quality_score = 1.0
    if not passed_resolution:
        quality_score -= 0.4
    if not passed_blur:
        quality_score -= 0.3
    quality_score = max(0.0, min(1.0, quality_score))

    ctx.quality = {
        "width": w,
        "height": h,
        "passed_resolution": passed_resolution,
        "passed_blur": passed_blur,
        "blur_score": round(blur_score, 2),
        "quality_score": round(quality_score, 4),
        "min_width": MIN_WIDTH,
        "min_height": MIN_HEIGHT,
        "min_blur_score": MIN_LAPLACIAN_VARIANCE,
    }

    if not passed_resolution:
        ctx.violations.append(
            {
                "ruleId": "LOW_RESOLUTION",
                "title": "Низька роздільна здатність",
                "severity": "high",
                "message": f"Розмір зображення замалий: {w}x{h}. Мінімум {MIN_WIDTH}x{MIN_HEIGHT}.",
                "bbox": None,
                "penalty": 0,
                "meta": {
                    "width": w,
                    "height": h,
                    "minWidth": MIN_WIDTH,
                    "minHeight": MIN_HEIGHT,
                },
            }
        )
        ctx.fail("quality_gate", "Image resolution is too low", verdict="NEED_REVIEW")
        return

    if not passed_blur:
        ctx.violations.append(
            {
                "ruleId": "BLURRY_IMAGE",
                "title": "Розмите зображення",
                "severity": "medium",
                "message": f"Зображення занадто розмите (blur_score={blur_score:.2f}).",
                "bbox": None,
                "penalty": 0,
                "meta": {
                    "blurScore": round(blur_score, 2),
                    "minBlurScore": MIN_LAPLACIAN_VARIANCE,
                },
            }
        )
        ctx.fail("quality_gate", "Image is too blurry", verdict="NEED_REVIEW")
        return

    ctx.bgr_used = image
    ctx.mark_step_done("quality_gate")