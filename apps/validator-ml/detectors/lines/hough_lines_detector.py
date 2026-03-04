import cv2
import numpy as np
import math

def detect_lines(bgr, max_lines=40):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=40,
        maxLineGap=10
    )

    out = []
    if lines is None:
        return out

    for (x1, y1, x2, y2) in lines[:, 0][:max_lines]:
        angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        length = float(math.hypot(x2 - x1, y2 - y1))
        out.append({"p1": [int(x1), int(y1)], "p2": [int(x2), int(y2)], "angle": angle, "length": length})

    return out
