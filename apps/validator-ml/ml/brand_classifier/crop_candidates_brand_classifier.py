import cv2


MIN_CROP_SIZE = 24
PADDING_RATIO = 0.20


def _clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def _expand_bbox(bbox, width, height, padding_ratio=PADDING_RATIO):
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = _clip(x1 - pad_x, 0, width - 1)
    y1 = _clip(y1 - pad_y, 0, height - 1)
    x2 = _clip(x2 + pad_x, 0, width)
    y2 = _clip(y2 + pad_y, 0, height)

    return x1, y1, x2, y2


def _is_valid_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= MIN_CROP_SIZE and (y2 - y1) >= MIN_CROP_SIZE


def _normalize_bbox(item):
    bbox = item.get("bbox")

    if not bbox:
        return None

    if isinstance(bbox, dict):
        x1 = bbox.get("x1", bbox.get("x", 0))
        y1 = bbox.get("y1", bbox.get("y", 0))

        if "x2" in bbox and "y2" in bbox:
            x2 = bbox["x2"]
            y2 = bbox["y2"]
        else:
            x2 = x1 + bbox.get("w", bbox.get("width", 0))
            y2 = y1 + bbox.get("h", bbox.get("height", 0))

        return int(x1), int(y1), int(x2), int(y2)

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(map(int, bbox))

    return None


def extract_logo_crops(bgr, detections, max_crops=8):
    """
    detections — список bbox-ів від visual_logo_detector.

    Повертає:
    [
        {
            "crop": image_bgr,
            "bbox": [x1, y1, x2, y2],
            "source": "visual_logo_detector",
            "confidence": 0.73
        }
    ]
    """

    if bgr is None:
        return []

    height, width = bgr.shape[:2]
    crops = []

    if not detections:
        return crops

    sorted_detections = sorted(
        detections,
        key=lambda x: float(x.get("confidence", x.get("score", 0.0)) or 0.0),
        reverse=True,
    )

    for item in sorted_detections[:max_crops]:
        bbox = _normalize_bbox(item)

        if bbox is None:
            continue

        bbox = _expand_bbox(bbox, width, height)

        if not _is_valid_bbox(bbox):
            continue

        x1, y1, x2, y2 = bbox

        crop = bgr[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        crops.append({
            "crop": crop,
            "bbox": [x1, y1, x2, y2],
            "source": "visual_logo_detector",
            "confidence": float(item.get("confidence", item.get("score", 0.0)) or 0.0),
        })

    return crops


def fallback_chest_crops(bgr):
    """
    fallback, якщо detector не дав bbox.
    Не основний шлях, але корисно для MVP.
    """

    if bgr is None:
        return []

    h, w = bgr.shape[:2]

    crop_specs = [
        ("upper_center", 0.25, 0.05, 0.75, 0.45),
        ("center_chest", 0.25, 0.20, 0.75, 0.65),
        ("left_chest", 0.10, 0.15, 0.45, 0.55),
        ("right_chest", 0.55, 0.15, 0.90, 0.55),
        ("middle_print", 0.15, 0.30, 0.85, 0.80),

        # new tight crops
        ("tight_left_chest", 0.10, 0.08, 0.32, 0.30),
        ("tight_right_chest", 0.68, 0.08, 0.90, 0.30),
        ("ultra_tight_left_chest", 0.12, 0.10, 0.28, 0.26),
        ("ultra_tight_right_chest", 0.72, 0.10, 0.88, 0.26),
    ]

    crops = []

    for name, rx1, ry1, rx2, ry2 in crop_specs:
        x1 = int(w * rx1)
        y1 = int(h * ry1)
        x2 = int(w * rx2)
        y2 = int(h * ry2)

        crop = bgr[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        crops.append({
            "crop": crop,
            "bbox": [x1, y1, x2, y2],
            "source": f"fallback_{name}",
            "confidence": 0.0,
        })

    return crops