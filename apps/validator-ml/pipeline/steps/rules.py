import statistics

def _avg_ocr_conf(ocr):
    if not ocr:
        return 0.0
    return sum(float(x.get("conf", 0.0)) for x in ocr) / len(ocr)

def _is_noise_text(t: str) -> bool:
    t = (t or "").strip()
    return len(t) <= 1

def run(ctx):
    ocr = ctx.detections.get("ocr", [])
    lines = ctx.detections.get("lines", [])

    ctx.violations = []
    ctx.score = 100

    w, h = ctx.width, ctx.height

    # --- RULE 1: Too much text (AUTO)
    block_count = len(ocr)
    word_count = sum(len((item.get("text") or "").split()) for item in ocr)

    if block_count > 8 or word_count > 35:
        ctx.violations.append({
            "ruleId": "TOO_MUCH_TEXT",
            "title": "Забагато тексту",
            "severity": "MED",
            "message": f"Блоків: {block_count}, слів: {word_count}.",
            "bbox": None
        })
        ctx.score -= 15

    # --- RULE 2: Text near edge (safe margin in %)
    safe = int(min(w, h) * 0.04)

    filtered_ocr = []
    for item in ocr:
        text = (item.get("text") or "").strip()
        conf = float(item.get("conf", 0.0))
        if conf < 0.4:
            continue
        if _is_noise_text(text):
            continue
        filtered_ocr.append(item)

    for item in filtered_ocr:
        x1, y1, x2, y2 = item["bbox"]
        if x1 < safe or y1 < safe or x2 > (w - safe) or y2 > (h - safe):
            ctx.violations.append({
                "ruleId": "TEXT_NEAR_EDGE",
                "title": "Текст біля краю",
                "severity": "MED",
                "message": f"Текст '{item['text'][:30]}' близько до краю (safe={safe}px).",
                "bbox": [x1, y1, x2, y2]
            })
            ctx.score -= 10
            break

    # --- RULE 3: Sketch line straightness
    if ctx.scene.get("type") == "sketch_scan" and len(lines) >= 5:
        angles = [l["angle"] for l in lines if l.get("length", 0) >= 60]

        if len(angles) >= 5:
            norm = []
            for a in angles:
                while a > 90: a -= 180
                while a < -90: a += 180
                norm.append(a)

            spread = statistics.pstdev(norm)
            if spread > 20:
                ctx.violations.append({
                    "ruleId": "LINES_MESSY",
                    "title": "Лінії нерівні/хаотичні",
                    "severity": "WARN",
                    "message": f"Великий розкид кутів ліній (stdev={spread:.1f}).",
                    "bbox": None
                })
                ctx.score -= 10

    # --- RULE 4: Skew angle too big
    ang = ctx.debug.get("skew_angle_deg")
    if ctx.scene.get("type") == "sketch_scan" and ang is not None:
        if abs(ang) >= 7:
            ctx.violations.append({
                "ruleId": "SCAN_SKEW_HIGH",
                "title": "Сильний перекіс скану/фото",
                "severity": "HIGH",
                "message": f"Перекіс приблизно {ang:.1f}°. Краще пересканити/перезняти рівніше.",
                "bbox": None
            })
            ctx.score -= 20
        elif abs(ang) >= 3:
            ctx.violations.append({
                "ruleId": "SCAN_SKEW",
                "title": "Є перекіс скану/фото",
                "severity": "WARN",
                "message": f"Перекіс приблизно {ang:.1f}°. Ми вирівняли, але краще робити рівно.",
                "bbox": None
            })
            ctx.score -= 5

    # --- RULE 5: Unreliable input -> signal for NEED_REVIEW
    qs = ctx.quality.get("quality_score", 1.0)
    ocr_conf = _avg_ocr_conf(ocr)
    ctx.debug["ocr_avg_conf"] = ocr_conf

    if qs < 0.7 and ocr_conf < 0.35:
        ctx.debug["need_review_reason"] = "low_quality_and_low_ocr_conf"
