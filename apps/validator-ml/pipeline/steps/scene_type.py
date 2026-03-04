import cv2
import numpy as np

def run(ctx):
    bgr = ctx.bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Heuristic: sketch має багато “тонких ліній” + мало кольорів
    edges = cv2.Canny(gray, 60, 180)
    edge_ratio = float(np.mean(edges > 0))  # 0..1

    # colorfulness heuristic
    b, g, r = cv2.split(bgr)
    rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    yb = np.abs(((r.astype(np.int16) + g.astype(np.int16)) // 2) - b.astype(np.int16))
    colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)))

    # Якщо кольорів мало і ліній багато — схоже на скетч/скан
    if colorfulness < 25 and edge_ratio > 0.02:
        ctx.scene = {"type": "sketch_scan", "confidence": 0.7, "signals": {"edge_ratio": edge_ratio, "colorfulness": colorfulness}}
        return

    # Якщо кольорів багато — швидше digital/mockup
    if colorfulness >= 25:
        # Просте розділення mockup vs digital: mockup часто має “фотофон” (плавні градієнти/текстури)
        # Ми поки ставимо unknown між ними
        ctx.scene = {"type": "digital_or_mockup", "confidence": 0.6, "signals": {"edge_ratio": edge_ratio, "colorfulness": colorfulness}}
        return

    ctx.scene = {"type": "unknown", "confidence": 0.3, "signals": {"edge_ratio": edge_ratio, "colorfulness": colorfulness}}
