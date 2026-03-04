import cv2
import numpy as np

def estimate_skew_angle_deg(lines):
    """
    lines: list of {"angle": deg, "length": float, ...} from hough detector
    returns: skew angle in degrees (float) or None
    """
    if not lines:
        return None

    angles = []
    weights = []
    for ln in lines:
        length = float(ln.get("length", 0.0))
        if length < 80:  # ігноруємо короткі
            continue
        a = float(ln.get("angle", 0.0))

        # normalize to [-90, 90]
        while a > 90: a -= 180
        while a < -90: a += 180

        angles.append(a)
        weights.append(length)

    if len(angles) < 5:
        return None

    # weighted median (стабільніше за average)
    idx = np.argsort(angles)
    ang_sorted = np.array(angles)[idx]
    w_sorted = np.array(weights)[idx]
    cum = np.cumsum(w_sorted)
    mid = cum[-1] / 2
    med_angle = float(ang_sorted[np.searchsorted(cum, mid)])

    return med_angle

def rotate_bgr(bgr, angle_deg):
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
