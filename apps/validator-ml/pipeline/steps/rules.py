from __future__ import annotations

from typing import Any, Dict, List


SAFE_MARGIN_RATIO = 0.05
TOO_MUCH_TEXT_WORDS = 12
TOO_MUCH_TEXT_BLOCKS = 5
HIGH_SKEW_DEG = 8.0
APPAREL_SKEW_DEG = 12.0
MESSY_LINES_COUNT = 45
APPAREL_MESSY_LINES_COUNT = 65
LOGO_LIKE_REVIEW_COUNT = 3
APPAREL_LOGO_LIKE_REVIEW_COUNT = 5


def _get_bbox(item: Dict[str, Any]) -> List[int] | None:
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return bbox
    return None


def _estimate_words(ocr_items: List[Dict[str, Any]]) -> int:
    total = 0
    for item in ocr_items:
        text = str(item.get("text") or item.get("value") or "").strip()
        if text:
            total += len([word for word in text.split() if word.strip()])
    return total


def _text_near_edge(bbox: List[int], width: int, height: int, margin_ratio: float) -> bool:
    x1, y1, x2, y2 = bbox
    mx = int(width * margin_ratio)
    my = int(height * margin_ratio)
    return x1 <= mx or y1 <= my or x2 >= (width - mx) or y2 >= (height - my)


def _watermark_metrics(mark: Dict[str, Any]) -> tuple[float, float, float]:
    meta = mark.get("meta", {}) or {}
    return (
        float(mark.get("score", 0.0) or 0.0),
        float(meta.get("areaRatio", 0.0) or 0.0),
        float(meta.get("centeredness", 0.0) or 0.0),
    )


def run(ctx) -> None:
    width = int(ctx.width)
    height = int(ctx.height)

    detections = ctx.detections or {}
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})
    detectors_debug = (ctx.debug or {}).get("detectors", {}) or {}
    skew_meta = detectors_debug.get("skew", {}) or {}

    ocr_items = detections.get("ocr", []) or []
    lines = detections.get("lines") or []
    ip = detections.get("ip") or {}
    logo_like = detections.get("logoLikeMarks") or []
    visual_logo = detections.get("visualLogoMarks") or []
    qr_marks = detections.get("qrMarks") or []
    watermark_marks = detections.get("watermarkMarks") or []

    scene_type = scene.get("type")
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0)
    apparel_source = scene.get("apparel_source", "unknown")

    skew_angle = skew_meta.get("angleDeg", ctx.debug.get("skew_angle_deg"))
    skew_support = int(skew_meta.get("supportLines", 0) or 0)
    skew_confidence = float(skew_meta.get("confidence", 0.0) or 0.0)

    word_count = _estimate_words(ocr_items)
    text_blocks = len(ocr_items)

    skew_threshold = APPAREL_SKEW_DEG if is_apparel else HIGH_SKEW_DEG
    skew_min_support = 8 if is_apparel else 5
    skew_min_confidence = 0.60 if is_apparel else 0.45
    messy_lines_threshold = APPAREL_MESSY_LINES_COUNT if is_apparel else MESSY_LINES_COUNT
    logo_like_threshold = APPAREL_LOGO_LIKE_REVIEW_COUNT if is_apparel else LOGO_LIKE_REVIEW_COUNT

    rules_input = {
        "wordCount": word_count,
        "textBlocks": text_blocks,
        "lineCount": len(lines),
        "skewAngleDeg": skew_angle,
        "skewSupportLines": skew_support,
        "skewConfidence": skew_confidence,
        "logoLikeCount": len(logo_like),
        "visualLogoCount": len(visual_logo),
        "qrCount": len(qr_marks),
        "watermarkCount": len(watermark_marks),
        "ipExactHits": len(ip.get("exactHits", [])),
        "ipSuspiciousHits": len(ip.get("suspiciousHits", [])),
        "sceneType": scene_type,
        "isApparel": is_apparel,
        "apparelConfidence": round(apparel_confidence, 4),
        "apparelSource": apparel_source,
    }

    ctx.debug["rulesInput"] = rules_input
    ctx.debug["is_apparel"] = is_apparel
    ctx.set_debug_section(
        "rules",
        {
            "inputs": rules_input,
            "thresholds": {
                "safeMarginRatio": SAFE_MARGIN_RATIO,
                "skewDegrees": skew_threshold,
                "skewSupportLines": skew_min_support,
                "skewConfidence": skew_min_confidence,
                "messyLines": messy_lines_threshold,
                "logoLikeMarks": logo_like_threshold,
            },
        },
    )

    too_much_text = word_count > TOO_MUCH_TEXT_WORDS or text_blocks > TOO_MUCH_TEXT_BLOCKS
    ctx.add_rule_result(
        rule_id="TOO_MUCH_TEXT",
        passed=not too_much_text,
        severity="medium",
        penalty=20 if too_much_text else 0,
        title="Забагато тексту",
        message=(
            f"На дизайні забагато тексту: {word_count} слів, {text_blocks} текстових блоків."
            if too_much_text
            else "Кількість тексту в межах норми."
        ),
        meta={"wordCount": word_count, "textBlocks": text_blocks},
    )

    edge_violations = 0
    for item in ocr_items:
        bbox = _get_bbox(item)
        if bbox and _text_near_edge(bbox, width, height, SAFE_MARGIN_RATIO):
            edge_violations += 1
            ctx.add_rule_result(
                rule_id="TEXT_NEAR_EDGE",
                passed=False,
                severity="medium",
                penalty=8,
                title="Текст занадто близько до краю",
                message="Текстовий блок заходить у небезпечну крайову зону.",
                bbox=bbox,
                meta={"safeMarginRatio": SAFE_MARGIN_RATIO},
            )

    if edge_violations == 0:
        ctx.add_rule_result(
            rule_id="TEXT_NEAR_EDGE",
            passed=True,
            severity="low",
            penalty=0,
            title="Безпечні поля",
            message="Текст не заходить у небезпечну крайову зону.",
            meta={"safeMarginRatio": SAFE_MARGIN_RATIO},
        )

    skew_bad = (
        isinstance(skew_angle, (int, float))
        and abs(float(skew_angle)) >= skew_threshold
        and skew_support >= skew_min_support
        and skew_confidence >= skew_min_confidence
    )
    ctx.add_rule_result(
        rule_id="HIGH_SKEW",
        passed=not skew_bad,
        severity="low" if is_apparel else "medium",
        penalty=(8 if is_apparel else 15) if skew_bad else 0,
        title="Сильний перекіс",
        message=(
            f"Виявлено сильний перекіс: {float(skew_angle):.2f}°."
            if skew_bad and skew_angle is not None
            else "Критичного перекосу не виявлено."
        ),
        meta={
            "skewAngleDeg": skew_angle,
            "threshold": skew_threshold,
            "supportLines": skew_support,
            "confidence": skew_confidence,
        },
    )

    messy_lines_allowed = scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel
    messy_lines = messy_lines_allowed and len(lines) >= messy_lines_threshold
    ctx.add_rule_result(
        rule_id="MESSY_LINES",
        passed=not messy_lines,
        severity="low",
        penalty=(6 if is_apparel else 10) if messy_lines else 0,
        title="Перенавантажений ескіз",
        message=(
            f"Виявлено надто багато ліній: {len(lines)}."
            if messy_lines
            else "Кількість ліній у межах допустимого."
        ),
        meta={
            "lineCount": len(lines),
            "threshold": messy_lines_threshold,
            "skippedForScene": not messy_lines_allowed,
            "sceneType": scene_type,
        },
    )

    for hit in ip.get("exactHits", []):
        hit_type = hit.get("type", "ip")
        title_map = {
            "brand": "Виявлено бренд",
            "character": "Виявлено персонажа",
            "franchise": "Виявлено франшизу",
            "slogan": "Виявлено захищений слоган",
        }

        ctx.add_rule_result(
            rule_id=f"IP_{hit_type.upper()}_EXACT",
            passed=False,
            severity="high",
            penalty=50,
            title=title_map.get(hit_type, "Виявлено захищений контент"),
            message=f"Знайдено збіг: {hit.get('keyword')}",
            bbox=hit.get("bbox"),
            meta=hit,
        )

    for hit in ip.get("suspiciousHits", []):
        hit_type = hit.get("type", "ip")
        ctx.add_rule_result(
            rule_id=f"IP_{hit_type.upper()}_SUSPECT",
            passed=False,
            severity="medium",
            penalty=20,
            title="Підозра на захищений контент",
            message=f"Підозрілий збіг: {hit.get('keyword')} (score={hit.get('score')})",
            bbox=hit.get("bbox"),
            meta=hit,
        )

    if not ip.get("exactHits") and not ip.get("suspiciousHits"):
        ctx.add_rule_result(
            rule_id="IP_RISK_CLEAR",
            passed=True,
            severity="low",
            penalty=0,
            title="Ознак захищеного контенту не знайдено",
            message="Текстових IP-збігів не виявлено.",
        )

    logo_like_allowed = scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel
    logo_like_review = logo_like_allowed and len(logo_like) >= logo_like_threshold
    ctx.add_rule_result(
        rule_id="LOGO_LIKE_MARKS",
        passed=not logo_like_review,
        severity="low" if is_apparel else "medium",
        penalty=(8 if is_apparel else 18) if logo_like_review else 0,
        title="Підозра на візуальні емблеми",
        message=(
            f"Виявлено багато компактних контрастних емблемоподібних форм: {len(logo_like)}."
            if logo_like_review
            else "Критичної кількості емблемоподібних форм не виявлено."
        ),
        meta={
            "logoLikeCount": len(logo_like),
            "threshold": logo_like_threshold,
            "skippedForScene": not logo_like_allowed,
            "sceneType": scene_type,
        },
    )

    if scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel:
        strong_centered = []
        medium_marks = []

        for mark in visual_logo:
            score = float(mark.get("emblem_score", 0.0))
            area_ratio = float(mark.get("area_ratio", 0.0))
            center_dist = float(mark.get("center_dist", 1.0))

            if score >= 0.68 and area_ratio >= 0.02 and center_dist <= 0.38:
                strong_centered.append(mark)

            if score >= 0.62 and area_ratio >= 0.008:
                medium_marks.append(mark)

        if strong_centered:
            top = strong_centered[0]
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_CENTER",
                passed=False,
                severity="high",
                penalty=40,
                title="Великий логотип по центру",
                message="Виявлено велику емблему або логотип у центральній зоні.",
                bbox=top.get("bbox"),
                meta=top,
            )
        elif len(medium_marks) >= 3:
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_MULTIPLE",
                passed=False,
                severity="medium",
                penalty=20,
                title="Багато емблем",
                message=f"Виявлено багато підозрілих емблем: {len(medium_marks)}.",
                meta={"count": len(medium_marks), "items": medium_marks[:5]},
            )
        elif visual_logo and not (is_apparel and apparel_confidence >= 0.75):
            top = visual_logo[0]
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_SUSPECT",
                passed=False,
                severity="low",
                penalty=10,
                title="Підозра на логотип",
                message="Виявлено підозрілу емблему або логотип.",
                bbox=top.get("bbox"),
                meta=top,
            )

    qr_detected = len(qr_marks) > 0
    ctx.add_rule_result(
        rule_id="QR_DETECTED",
        passed=not qr_detected,
        severity="medium",
        penalty=25 if qr_detected else 0,
        title="Виявлено QR-код",
        message=(
            f"На зображенні виявлено {len(qr_marks)} QR-код(ів)."
            if qr_detected
            else "QR-кодів не виявлено."
        ),
        meta={"qrCount": len(qr_marks)},
    )

    watermark_allowed = scene_type != "text_heavy_cover" or is_apparel
    strong_watermark_hits = []
    apparel_strong_hits = []
    for mark in watermark_marks:
        score, area_ratio, centeredness = _watermark_metrics(mark)
        if score >= 0.82 and area_ratio >= 0.008:
            strong_watermark_hits.append(mark)
        if score >= 0.88 and area_ratio >= 0.01 and centeredness >= 0.45:
            apparel_strong_hits.append(mark)

    watermark_detected = watermark_allowed and len(strong_watermark_hits) >= 1
    if is_apparel:
        watermark_detected = watermark_allowed and (
            len(apparel_strong_hits) >= 2
            or any(
                _watermark_metrics(mark)[0] >= 0.92 and _watermark_metrics(mark)[1] >= 0.012
                for mark in watermark_marks
            )
        )

    ctx.add_rule_result(
        rule_id="WATERMARK_DETECTED",
        passed=not watermark_detected,
        severity="low" if is_apparel else "medium",
        penalty=(10 if is_apparel else 22) if watermark_detected else 0,
        title="Підозра на водяний знак",
        message=(
            f"Виявлено {len(watermark_marks)} watermark-подібну область(і)."
            if watermark_detected
            else "Водяних знаків не виявлено."
        ),
        meta={
            "watermarkCount": len(watermark_marks),
            "strongWatermarkCount": len(strong_watermark_hits),
            "apparelStrongCount": len(apparel_strong_hits),
            "skippedForScene": not watermark_allowed,
            "sceneType": scene_type,
            "apparelMode": is_apparel,
        },
    )

    non_apparel_penalty = 25 if apparel_source == "ml" and apparel_confidence >= 0.70 else 12
    ctx.add_rule_result(
        rule_id="NON_APPAREL",
        passed=is_apparel,
        severity="medium" if apparel_source == "ml" else "low",
        penalty=non_apparel_penalty if not is_apparel else 0,
        title="Не схоже на дизайн одягу",
        message=(
            "Зображення більше схоже на постер або обкладинку."
            if not is_apparel
            else "Зображення виглядає як дизайн одягу."
        ),
        meta={
            "is_apparel": is_apparel,
            "source": apparel_source,
            "confidence": round(apparel_confidence, 4),
        },
    )

    ctx.mark_step_done("rules")
