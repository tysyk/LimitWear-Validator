import os
import cv2
from explain.annotate import make_annotated

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", ".."))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(ctx):
    bgr_for_draw = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

    if bgr_for_draw is not None:
        annotated = make_annotated(bgr_for_draw, ctx)
        out_path = os.path.join(RESULTS_DIR, f"{ctx.image_id}_annotated.png")
        cv2.imwrite(out_path, annotated)
        ctx.artifacts["annotatedPath"] = out_path

    explain = [
        f"Verdict: {ctx.verdict}",
        f"Score: {ctx.score}/100",
    ]

    if ctx.scene:
        scene_type = ctx.scene.get("type", "unknown")
        confidence = ctx.scene.get("confidence")
        if confidence is not None:
            explain.append(f"Scene type: {scene_type} (confidence={confidence})")
        else:
            explain.append(f"Scene type: {scene_type}")

    if ctx.violations:
        for v in ctx.violations:
            title = v.get("title", "Violation")
            message = v.get("message", "")
            explain.append(f"{title}: {message}")
    else:
        explain.append("No violations detected.")

    if ctx.debug.get("need_review_reason"):
        explain.append(f"Manual review reason: {ctx.debug['need_review_reason']}")

    ctx.explain = explain