def check(image_info, ocr, profile):
    w, h = image_info["width"], image_info["height"]
    safe = profile["safe"]

    for item in ocr:
        x1, y1, x2, y2 = item["bbox"]
        if x1 < safe or y1 < safe or x2 > (w - safe) or y2 > (h - safe):
            return ([{
                "ruleId": "SAFE_AREA_TEXT",
                "title": "Текст у небезпечній зоні",
                "severity": "HIGH",
                "message": f"Текст '{item['text'][:30]}' заходить у safe-area ({safe}px).",
                "bbox": [x1, y1, x2, y2]
            }], 25)

    return ([], 0)
