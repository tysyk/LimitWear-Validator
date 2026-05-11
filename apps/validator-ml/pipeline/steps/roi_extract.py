from __future__ import annotations


def _get_input_image(ctx):
    return ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr


def _set_full_image_roi(ctx, image, *, reason: str) -> None:
    height, width = image.shape[:2]

    ctx.roi = {
        "x": 0,
        "y": 0,
        "width": width,
        "height": height,
        "used_full_image": True,
        "source": "full_image_fallback",
        "confidence": 1.0,
        "reason": reason,
    }

    ctx.bgr_used = image

    ctx.set_debug_section(
        "roi_extract",
        {
            "usedFullImage": True,
            "source": "full_image_fallback",
            "confidence": 1.0,
            "bbox": [0, 0, width, height],
            "reason": reason,
        },
    )


def run(ctx) -> None:
    image = _get_input_image(ctx)

    if image is None:
        ctx.fail("roi_extract", "No image available for ROI extraction")
        return

    # Current implementation intentionally uses the full image as ROI.
    # This keeps the pipeline stable and leaves a clean place for future
    # garment/design region extraction.
    _set_full_image_roi(
        ctx,
        image,
        reason="No dedicated garment ROI detector is enabled yet.",
    )

    ctx.mark_step_done("roi_extract")