import cv2
import numpy as np


def run(ctx):
    bgr = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 180)
    edge_ratio = float(np.mean(edges > 0))

    b, g, r = cv2.split(bgr)
    rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    yb = np.abs(((r.astype(np.int16) + g.astype(np.int16)) // 2) - b.astype(np.int16))
    colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)))

    if colorfulness < 25 and edge_ratio > 0.02:
        scene_type = "sketch_scan"
        confidence = 0.7
    elif colorfulness >= 25:
        scene_type = "digital_or_mockup"
        confidence = 0.6
    else:
        scene_type = "unknown"
        confidence = 0.3

    ctx.scene = {
        "type": scene_type,
        "confidence": confidence,
        "signals": {
            "edge_ratio": round(edge_ratio, 4),
            "colorfulness": round(colorfulness, 2),
        }
    }