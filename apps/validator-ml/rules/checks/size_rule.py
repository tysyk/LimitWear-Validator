def check(image_info, ocr, profile):
    w, h = image_info["width"], image_info["height"]
    violations = []
    penalty = 0

    if w != profile["width"] or h != profile["height"]:
        violations.append({
            "ruleId": "IMG_SIZE",
            "title": "Неправильний розмір",
            "severity": "HIGH",
            "message": f"Очікується {profile['width']}x{profile['height']}, отримано {w}x{h}.",
            "bbox": None
        })
        penalty = 40

    return violations, penalty
