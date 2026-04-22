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

    ctx.set_debug_section(
        "quality_gate",
        {
            "resolutionOk": passed_resolution,
            "blurOk": passed_blur,
            "qualityScore": round(quality_score, 4),
            "blurScore": round(blur_score, 2),
        },
    )

    ctx.add_rule_result(
        rule_id="LOW_RESOLUTION",
        passed=passed_resolution,
        severity="high",
        penalty=0 if passed_resolution else 25,
        title="Низька роздільна здатність",
        message=(
            "Роздільна здатність відповідає мінімальним вимогам."
            if passed_resolution
            else f"Розмір зображення замалий: {w}x{h}. Мінімум {MIN_WIDTH}x{MIN_HEIGHT}."
        ),
        meta={
            "width": w,
            "height": h,
            "minWidth": MIN_WIDTH,
            "minHeight": MIN_HEIGHT,
        },
    )

    ctx.add_rule_result(
        rule_id="BLURRY_IMAGE",
        passed=passed_blur,
        severity="medium",
        penalty=0 if passed_blur else 12,
        title="Розмите зображення",
        message=(
            "Різкість зображення достатня для автоматичної перевірки."
            if passed_blur
            else f"Зображення занадто розмите (blur_score={blur_score:.2f})."
        ),
        meta={
            "blurScore": round(blur_score, 2),
            "minBlurScore": MIN_LAPLACIAN_VARIANCE,
        },
    )

    if not passed_resolution:
        ctx.add_warning("Image resolution is below the recommended minimum.")

    if not passed_blur:
        ctx.add_warning("Image sharpness is low, so some detections may be less reliable.")

    ctx.bgr_used = image
    ctx.mark_step_done("quality_gate")
