from __future__ import annotations


def run(ctx) -> None:
    image = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

    if image is None:
        ctx.fail("roi_extract", "No image available for ROI extraction")
        return

    h, w = image.shape[:2]

    ctx.roi = {
        "x": 0,
        "y": 0,
        "width": w,
        "height": h,
        "used_full_image": True,
    }

    ctx.bgr_used = image
    ctx.mark_step_done("roi_extract")