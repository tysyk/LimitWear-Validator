import os
import cv2
from explain.annotate import make_annotated

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", ".."))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run(ctx):
    bgr_for_draw = ctx.debug.get("_bgr_used", ctx.bgr)
    bgr_for_draw = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr
    annotated = make_annotated(bgr_for_draw, ctx)
    out_path = os.path.join(RESULTS_DIR, f"{ctx.image_id}_annotated.png")
    cv2.imwrite(out_path, annotated)

    ctx.artifacts = {"annotatedPath": out_path}
