from __future__ import annotations

import cv2
import numpy as np

from core.config import APPAREL_CONFIDENCE_THRESHOLD


def _fallback_scene(ctx, error: Exception | None = None) -> None:
    if not isinstance(ctx.scene, dict):
        ctx.scene = {}

    ctx.scene.setdefault("type", "unknown")
    ctx.scene.setdefault("confidence", 0.0)
    ctx.scene.setdefault("type_source", "scene_type_fallback")
    ctx.scene.setdefault("is_apparel", True)
    ctx.scene.setdefault("apparel_source", "scene_fallback")
    ctx.scene.setdefault(
        "signals",
        {
            "edge_ratio": 0.0,
            "colorfulness": 0.0,
            "text_count": 0,
            "text_density": 0.0,
        },
    )

    debug_payload = {
        "type": ctx.scene.get("type"),
        "confidence": ctx.scene.get("confidence"),
        "typeSource": ctx.scene.get("type_source"),
        "signals": ctx.scene.get("signals"),
        "isApparelFromML": ctx.scene.get("apparel_source") == "ml",
        "fallback": True,
    }

    if error is not None:
        debug_payload["error"] = str(error)

    ctx.set_debug_section("scene_type", debug_payload)


def run(ctx) -> None:
    try:
        bgr = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

        if bgr is None:
            ctx.add_error("scene_type", "Input image is empty", critical=False)
            _fallback_scene(ctx)
            ctx.mark_step_done("scene_type")
            return

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = float(np.mean(edges > 0))

        b, g, r = cv2.split(bgr)

        rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        yb = np.abs(
            ((r.astype(np.int16) + g.astype(np.int16)) // 2)
            - b.astype(np.int16)
        )

        colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)))

        ocr_items = (ctx.detections or {}).get("ocr", [])
        text_count = len(ocr_items)

        total_text_len = sum(
            len(str(item.get("text", "")))
            for item in ocr_items
        )

        text_density = total_text_len / max(1, (w * h) / 10000)

        if text_count >= 10 or text_density > 20:
            scene_type = "text_heavy_cover"
            confidence = 0.85

        elif text_count >= 5 and colorfulness > 30:
            scene_type = "poster_like"
            confidence = 0.70

        elif colorfulness < 25 and edge_ratio > 0.02:
            scene_type = "sketch_scan"
            confidence = 0.70

        elif colorfulness >= 25 and text_count <= 5:
            scene_type = "apparel_candidate"
            confidence = 0.65

        else:
            scene_type = "unknown"
            confidence = 0.30

        if not isinstance(ctx.scene, dict):
            ctx.scene = {}

        apparel_ml = (ctx.ml or {}).get("apparel", {})
        ml_label = str(apparel_ml.get("label", "unknown"))

        ml_confidence = float(
            apparel_ml.get("confidence", 0.0) or 0.0
        )

        ml_reliable = bool(
            apparel_ml.get(
                "isReliable",
                ml_confidence >= APPAREL_CONFIDENCE_THRESHOLD,
            )
        )

        if ml_label == "apparel" and ml_reliable:
            scene_type = "apparel"
            confidence = ml_confidence
            ctx.scene["is_apparel"] = True
            ctx.scene["type_source"] = "ml_apparel"

        else:
            ctx.scene["type_source"] = "heuristic_metadata"

        ctx.scene["type"] = scene_type
        ctx.scene["confidence"] = round(confidence, 4)

        ctx.scene["signals"] = {
            "edge_ratio": round(edge_ratio, 4),
            "colorfulness": round(colorfulness, 2),
            "text_count": text_count,
            "text_density": round(text_density, 2),
        }

        if "is_apparel" not in ctx.scene:
            ctx.scene["is_apparel"] = scene_type not in {
                "text_heavy_cover",
                "poster_like",
            }
            ctx.scene["apparel_source"] = "scene_fallback"

        ctx.set_debug_section(
            "scene_type",
            {
                "type": scene_type,
                "confidence": round(confidence, 4),
                "typeSource": ctx.scene.get("type_source"),
                "signals": ctx.scene["signals"],
                "isApparelFromML": ctx.scene.get("apparel_source") == "ml",
                "fallback": False,
            },
        )

    except Exception as exc:
        ctx.add_error("scene_type", str(exc), critical=False)
        _fallback_scene(ctx, exc)

    ctx.mark_step_done("scene_type")