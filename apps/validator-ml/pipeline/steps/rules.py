import statistics


def _avg_ocr_conf(ocr):
    if not ocr:
        return 0.0
    return sum(float(x.get("conf", 0.0)) for x in ocr) / len(ocr)


def _is_noise_text(t: str) -> bool:
    t = (t or "").strip()
    return len(t) <= 1


def _severity_rank(severity: str) -> int:
    mapping = {
        "WARN": 1,
        "MED": 2,
        "HIGH": 3,
    }
    return mapping.get(severity, 0)


def run(ctx):
    ocr = ctx.detections.get("ocr", [])
    lines = ctx.detections.get("lines", [])

    ctx.rule_results = []
    ctx.violations = []
    ctx.score = 100

    w, h = ctx.width, ctx.height

    # --- RULE 1: Too much text
    block_count = len(ocr)
    word_count = sum(len((item.get("text") or "").split()) for item in ocr)

    passed = not (block_count > 8 or word_count > 35)
    ctx.rule_results.append({
        "ruleId": "TOO_MUCH_TEXT",
        "passed": passed,
        "meta": {
            "block_count": block_count,
            "word_count": word_count,
        }
    })

    if not passed:
        ctx.violations.append({
            "ruleId": "TOO_MUCH_TEXT",
            "title": "Забагато тексту",
            "severity": "MED",
            "message": f"Блоків: {block_count}, слів: {word_count}.",
            "bbox": None
        })
        ctx.score -= 15

    # --- RULE 2: Text near edge
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

    edge_violation = None
    for item in filtered_ocr:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        if x1 < safe or y1 < safe or x2 > (w - safe) or y2 > (h - safe):
            edge_violation = {
                "ruleId": "TEXT_NEAR_EDGE",
                "title": "Текст біля краю",
                "severity": "MED",
                "message": f"Текст '{item.get('text', '')[:30]}' близько до краю (safe={safe}px).",
                "bbox": [x1, y1, x2, y2]
            }
            break

    ctx.rule_results.append({
        "ruleId": "TEXT_NEAR_EDGE",
        "passed": edge_violation is None,
        "meta": {
            "safe_margin_px": safe,
            "checked_blocks": len(filtered_ocr),
        }
    })

    if edge_violation:
        ctx.violations.append(edge_violation)
        ctx.score -= 10

    # --- RULE 3: Sketch line straightness
    messy_lines_violation = None

    if ctx.scene.get("type") == "sketch_scan" and len(lines) >= 5:
        angles = [l["angle"] for l in lines if l.get("length", 0) >= 60 and "angle" in l]

        if len(angles) >= 5:
            norm = []
            for a in angles:
                while a > 90:
                    a -= 180
                while a < -90:
                    a += 180
                norm.append(a)

            spread = statistics.pstdev(norm)
            ctx.rule_results.append({
                "ruleId": "LINES_MESSY",
                "passed": spread <= 20,
                "meta": {
                    "spread": round(spread, 2),
                    "line_count": len(angles),
                }
            })

            if spread > 20:
                messy_lines_violation = {
                    "ruleId": "LINES_MESSY",
                    "title": "Лінії нерівні/хаотичні",
                    "severity": "WARN",
                    "message": f"Великий розкид кутів ліній (stdev={spread:.1f}).",
                    "bbox": None
                }

    if messy_lines_violation:
        ctx.violations.append(messy_lines_violation)
        ctx.score -= 10

    # --- RULE 4: Skew angle too big
    ang = ctx.debug.get("skew_angle_deg")
    skew_violation = None

    if ctx.scene.get("type") == "sketch_scan" and ang is not None:
        if abs(ang) >= 7:
            skew_violation = {
                "ruleId": "SCAN_SKEW_HIGH",
                "title": "Сильний перекіс скану/фото",
                "severity": "HIGH",
                "message": f"Перекіс приблизно {ang:.1f}°. Краще пересканити/перезняти рівніше.",
                "bbox": None
            }
            ctx.rule_results.append({
                "ruleId": "SCAN_SKEW",
                "passed": False,
                "meta": {
                    "skew_angle_deg": round(float(ang), 2),
                    "level": "high",
                }
            })
            ctx.score -= 20

        elif abs(ang) >= 3:
            skew_violation = {
                "ruleId": "SCAN_SKEW",
                "title": "Є перекіс скану/фото",
                "severity": "WARN",
                "message": f"Перекіс приблизно {ang:.1f}°. Ми вирівняли, але краще робити рівно.",
                "bbox": None
            }
            ctx.rule_results.append({
                "ruleId": "SCAN_SKEW",
                "passed": False,
                "meta": {
                    "skew_angle_deg": round(float(ang), 2),
                    "level": "warn",
                }
            })
            ctx.score -= 5
        else:
            ctx.rule_results.append({
                "ruleId": "SCAN_SKEW",
                "passed": True,
                "meta": {
                    "skew_angle_deg": round(float(ang), 2),
                    "level": "ok",
                }
            })

    if skew_violation:
        ctx.violations.append(skew_violation)

    # --- RULE 5: Unreliable input -> NEED_REVIEW
    qs = float(ctx.quality.get("quality_score", 1.0))
    ocr_conf = _avg_ocr_conf(ocr)
    ctx.debug["ocr_avg_conf"] = round(ocr_conf, 4)

    need_review = qs < 0.7 and ocr_conf < 0.35

    ctx.rule_results.append({
        "ruleId": "INPUT_RELIABILITY",
        "passed": not need_review,
        "meta": {
            "quality_score": round(qs, 4),
            "ocr_avg_conf": round(ocr_conf, 4),
        }
    })

    if need_review:
        ctx.debug["need_review_reason"] = "low_quality_and_low_ocr_conf"

    # --- normalize score
    if ctx.score < 0:
        ctx.score = 0
    if ctx.score > 100:
        ctx.score = 100

    # --- final verdict
    has_high = any(v.get("severity") == "HIGH" for v in ctx.violations)
    max_severity = max((_severity_rank(v.get("severity", "")) for v in ctx.violations), default=0)

    if need_review:
        ctx.verdict = "NEED_REVIEW"
    elif has_high:
        ctx.verdict = "REJECTED"
    elif ctx.score >= 85 and max_severity <= 1:
        ctx.verdict = "APPROVED"
    elif ctx.score >= 60:
        ctx.verdict = "APPROVED_WITH_WARNINGS"
    else:
        ctx.verdict = "REJECTED"

    # --- RULE 6: Low OCR confidence
    if ocr_conf < 0.3 and len(ocr) > 0:
        ctx.rule_results.append({
            "ruleId": "LOW_OCR_CONF",
            "passed": False,
            "meta": {
                "ocr_avg_conf": round(ocr_conf, 4),
                "ocr_count": len(ocr),
            }
        })
        ctx.violations.append({
            "ruleId": "LOW_OCR_CONF",
            "title": "Погано читається текст",
            "severity": "WARN",
            "message": f"Низька впевненість OCR ({ocr_conf:.2f}). Можливо текст розмитий або неякісний.",
            "bbox": None
        })
        ctx.score -= 5
    else:
        ctx.rule_results.append({
            "ruleId": "LOW_OCR_CONF",
            "passed": True,
            "meta": {
                "ocr_avg_conf": round(ocr_conf, 4),
                "ocr_count": len(ocr),
            }
        })