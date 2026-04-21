from __future__ import annotations

import cv2
import numpy as np


def run(ctx) -> None:
    try:
        bgr = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

        if bgr is None:
            ctx.fail("scene_type", "Input image is empty")
            return

        h, w = bgr.shape[:2]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # edges
        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = float(np.mean(edges > 0))

        # colorfulness
        b, g, r = cv2.split(bgr)
        rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        yb = np.abs(((r.astype(np.int16) + g.astype(np.int16)) // 2) - b.astype(np.int16))
        colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)))

        # OCR stats
        ocr_items = ctx.detections.get("ocr", [])
        text_count = len(ocr_items)

        total_text_len = 0
        for item in ocr_items:
            total_text_len += len(str(item.get("text", "")))

        # нормалізація
        text_density = total_text_len / max(1, (w * h) / 10000)

        # ===== класифікація =====

        # 1. текстонасичена обкладинка / книжка
        if text_count >= 10 or text_density > 20:
            scene_type = "text_heavy_cover"
            confidence = 0.85

        # 2. постер / банер
        elif text_count >= 5 and colorfulness > 30:
            scene_type = "poster_like"
            confidence = 0.7

        # 3. скан / ескіз
        elif colorfulness < 25 and edge_ratio > 0.02:
            scene_type = "sketch_scan"
            confidence = 0.7

        # 4. кандидат на дизайн одягу
        elif colorfulness >= 25 and text_count <= 5:
            scene_type = "apparel_candidate"
            confidence = 0.65

        else:
            scene_type = "unknown"
            confidence = 0.3

        is_apparel = True

        if scene_type in ["text_heavy_cover", "poster_like"]:
            is_apparel = False

        ctx.scene = {
            "type": scene_type,
            "confidence": confidence,
            "is_apparel": is_apparel,
            "signals": {
                "edge_ratio": round(edge_ratio, 4),
                "colorfulness": round(colorfulness, 2),
                "text_count": text_count,
                "text_density": round(text_density, 2),
            },
        }

        ctx.mark_step_done("scene_type")

    except Exception as e:
        ctx.add_error("scene_type", str(e), critical=False)