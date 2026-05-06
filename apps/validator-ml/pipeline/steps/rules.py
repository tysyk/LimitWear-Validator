from __future__ import annotations

from typing import Any, Dict, List

from core.config import (
    APPAREL_LOGO_LIKE_REVIEW_COUNT,
    APPAREL_MESSY_LINES_COUNT,
    LOGO_LIKE_REVIEW_COUNT,
    MESSY_LINES_COUNT,
    NON_APPAREL_BLOCK_CONFIDENCE,
    SAFE_MARGIN_RATIO,
    SKEW_APPAREL_ANGLE_DEG,
    SKEW_APPAREL_CONFIDENCE,
    SKEW_APPAREL_SUPPORT_LINES,
    SKEW_DOCUMENT_ANGLE_DEG,
    SKEW_DOCUMENT_CONFIDENCE,
    SKEW_DOCUMENT_SUPPORT_LINES,
    TOO_MUCH_TEXT_BLOCKS,
    TOO_MUCH_TEXT_WORDS,
    VISUAL_LOGO_CENTER_AREA_RATIO,
    VISUAL_LOGO_CENTER_DISTANCE,
    VISUAL_LOGO_CENTER_SCORE,
    VISUAL_LOGO_MEDIUM_AREA_RATIO,
    VISUAL_LOGO_MEDIUM_SCORE,
    WATERMARK_BLOCK_SCORE,
    WATERMARK_CENTEREDNESS,
    WATERMARK_STRONG_AREA_RATIO,
    WATERMARK_STRONG_SCORE,
)


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


def _confirmed_ip(ip: Dict[str, Any]) -> bool:
    return bool(ip.get("blocked") or ip.get("exactHits"))


def _strong_suspicious_ip(ip: Dict[str, Any]) -> bool:
    for hit in ip.get("suspiciousHits", []) or []:
        try:
            if float(hit.get("score", 0.0) or 0.0) >= 0.92:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_watermark_text_evidence(ocr_items: List[Dict[str, Any]]) -> bool:
    terms = ("watermark", "stock", "preview", "sample")
    for item in ocr_items:
        text = str(item.get("text") or item.get("value") or "").lower()
        if any(term in text for term in terms):
            return True
    return False


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

    skew_threshold = SKEW_APPAREL_ANGLE_DEG if is_apparel else SKEW_DOCUMENT_ANGLE_DEG
    skew_min_support = SKEW_APPAREL_SUPPORT_LINES if is_apparel else SKEW_DOCUMENT_SUPPORT_LINES
    skew_min_confidence = SKEW_APPAREL_CONFIDENCE if is_apparel else SKEW_DOCUMENT_CONFIDENCE
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
        penalty=12 if too_much_text else 0,
        title="Забагато тексту",
        message=(
            f"На дизайні забагато тексту: {word_count} слів, {text_blocks} текстових блоків."
            if too_much_text
            else "Кількість тексту в межах норми."
        ),
        meta={"wordCount": word_count, "textBlocks": text_blocks, "blocking": False},
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
                penalty=6,
                title="Текст занадто близько до краю",
                message="Текстовий блок заходить у небезпечну крайову зону.",
                bbox=bbox,
                meta={"safeMarginRatio": SAFE_MARGIN_RATIO, "blocking": False},
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
        penalty=(4 if is_apparel else 10) if skew_bad else 0,
        title="Сильний перекіс",
        message=(
            f"Виявлено помітний перекіс: {float(skew_angle):.2f}°."
            if skew_bad and skew_angle is not None
            else "Достатньо підтвердженого перекосу не виявлено."
        ),
        meta={
            "skewAngleDeg": skew_angle,
            "threshold": skew_threshold,
            "supportLines": skew_support,
            "confidence": skew_confidence,
            "blocking": False,
        },
    )

    messy_lines_allowed = scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel
    messy_lines = messy_lines_allowed and len(lines) >= messy_lines_threshold
    ctx.add_rule_result(
        rule_id="MESSY_LINES",
        passed=not messy_lines,
        severity="low",
        penalty=(4 if is_apparel else 8) if messy_lines else 0,
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
            "blocking": False,
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
            message=f"Знайдено підтверджений збіг: {hit.get('keyword')}",
            bbox=hit.get("bbox"),
            meta={**hit, "blocking": True, "riskType": "confirmed_ip"},
        )

    for hit in ip.get("suspiciousHits", []):
        hit_type = hit.get("type", "ip")
        ctx.add_rule_result(
            rule_id=f"IP_{hit_type.upper()}_SUSPECT",
            passed=False,
            severity="medium",
            penalty=16,
            title="Підозра на захищений контент",
            message=f"Підозрілий збіг: {hit.get('keyword')} (score={hit.get('score')})",
            bbox=hit.get("bbox"),
            meta={**hit, "blocking": False, "needsReview": True, "riskType": "suspected_ip"},
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

    adult_safety = detections.get("ml_adult_safety") or {}

    if adult_safety:
        label = adult_safety.get("label")
        adult_score = float(adult_safety.get("adultScore", 0.0) or 0.0)
        risk_level = adult_safety.get("riskLevel")
        is_reliable = bool(adult_safety.get("isReliable", False))

        if label == "adult_risk":
            blocking = risk_level == "block" and is_reliable

            ctx.add_rule_result(
                rule_id="ML_ADULT_RISK",
                passed=False,
                severity="high" if blocking else "medium",
                penalty=45 if blocking else 18,
                title="ML: adult/NSFW ризик",
                message=(
                    f"ML виявив високий adult/NSFW ризик (score={adult_score:.2f})."
                    if blocking
                    else f"ML виявив можливий adult/NSFW ризик (score={adult_score:.2f})."
                ),
                meta={
                    "adultScore": adult_score,
                    "riskLevel": risk_level,
                    "isReliable": is_reliable,
                    "blocking": blocking,
                    "needsReview": not blocking,
                    "source": "ml_adult_safety",
                },
            )

    logo_like_allowed = scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel
    logo_like_review = logo_like_allowed and len(logo_like) >= logo_like_threshold
    ctx.add_rule_result(
        rule_id="LOGO_LIKE_MARKS",
        passed=not logo_like_review,
        severity="low" if is_apparel else "medium",
        penalty=(4 if is_apparel else 10) if logo_like_review else 0,
        title="Емблемоподібні форми",
        message=(
            f"Виявлено кілька компактних емблемоподібних форм: {len(logo_like)}. Рекомендована перевірка."
            if logo_like_review
            else "Критичної кількості емблемоподібних форм не виявлено."
        ),
        meta={
            "logoLikeCount": len(logo_like),
            "threshold": logo_like_threshold,
            "skippedForScene": not logo_like_allowed,
            "sceneType": scene_type,
            "blocking": False,
        },
    )
    
    brand_risk = detections.get("ml_brand_risk") or {}

    if not brand_risk.get("skipped"):
        label = brand_risk.get("label")
        confidence = float(brand_risk.get("confidence", 0.0) or 0.0)
        is_reliable = bool(brand_risk.get("isReliable", False))
        risk_level = brand_risk.get("riskLevel")

        if label == "brand_logo":
            ctx.add_rule_result(
                rule_id="ML_BRAND_LOGO",
                passed=False,
                severity="medium",
                penalty=14 if confidence >= 0.85 else 8,
                title="ML: підозра на бренд-логотип",
                message=(
                    f"ML визначив брендоподібний логотип (confidence={confidence:.2f})."
                    if is_reliable
                    else f"Можлива підозра на бренд-логотип (confidence={confidence:.2f})."
                ),
                meta={
                    "confidence": confidence,
                    "isReliable": is_reliable,
                    "riskLevel": risk_level,
                    "blocking": False,
                    "needsReview": True,
                    "source": "ml_brand_risk",
                },
            )

    if scene_type not in ["text_heavy_cover", "poster_like"] or is_apparel:
        strong_centered = []
        medium_marks = []

        for mark in visual_logo:
            score = float(mark.get("emblem_score", 0.0))
            area_ratio = float(mark.get("area_ratio", 0.0))
            center_dist = float(mark.get("center_dist", 1.0))

            if (
                score >= VISUAL_LOGO_CENTER_SCORE
                and area_ratio >= VISUAL_LOGO_CENTER_AREA_RATIO
                and center_dist <= VISUAL_LOGO_CENTER_DISTANCE
            ):
                strong_centered.append(mark)

            if score >= VISUAL_LOGO_MEDIUM_SCORE and area_ratio >= VISUAL_LOGO_MEDIUM_AREA_RATIO:
                medium_marks.append(mark)

        confirmed_ip = _confirmed_ip(ip)
        strong_suspicious_ip = _strong_suspicious_ip(ip)
        logo_ml = (ctx.ml or {}).get("logo_presence", {})
        logo_ml_label = logo_ml.get("label")
        logo_ml_confidence = logo_ml.get("confidence")
        logo_ml_reliable = bool(logo_ml.get("isReliable", False))

        logo_ml_conflict = bool(
            strong_centered
            and not logo_ml_reliable
        )

        if strong_centered:
            top = strong_centered[0]
            blocking = confirmed_ip or strong_suspicious_ip
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_CENTER",
                passed=False,
                severity="high" if blocking else "medium",
                penalty=40 if blocking else 14,
                title="Емблемоподібний елемент у центрі",
                message=(
                    "Виявлено емблемоподібний елемент разом із підтвердженим IP/brand ризиком."
                    if blocking
                    else "Виявлено емблемоподібний елемент, рекомендована перевірка."
                ),
                bbox=top.get("bbox"),
                meta={
                    **top,
                    "blocking": blocking,
                    "hasConfirmedIp": confirmed_ip,
                    "hasStrongSuspiciousIp": strong_suspicious_ip,
                    "needsReview": not blocking,
                    "riskType": "visual_logo_shape",

                    "logoMlLabel": logo_ml_label,
                    "logoMlConfidence": logo_ml_confidence,
                    "logoMlReliable": logo_ml_reliable,
                    "logoConflict": not logo_ml_reliable,
                },
            )
        elif len(medium_marks) >= 3:
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_MULTIPLE",
                passed=False,
                severity="medium",
                penalty=12,
                title="Кілька емблемоподібних елементів",
                message="Виявлено кілька емблемоподібних елементів, рекомендована перевірка.",
                meta={
                    "count": len(medium_marks),
                    "items": medium_marks[:5],
                    "blocking": False,
                    "needsReview": True,
                    "riskType": "visual_logo_shape",
                },
            )
        elif visual_logo and not (is_apparel and apparel_confidence >= 0.75):
            top = visual_logo[0]
            ctx.add_rule_result(
                rule_id="VISUAL_LOGO_SUSPECT",
                passed=False,
                severity="low",
                penalty=6,
                title="Емблемоподібний елемент",
                message="Виявлено емблемоподібний елемент, рекомендована перевірка.",
                bbox=top.get("bbox"),
                meta={**top, "blocking": False, "needsReview": True, "riskType": "visual_logo_shape"},
            )

    qr_detected = len(qr_marks) > 0
    ctx.add_rule_result(
        rule_id="QR_DETECTED",
        passed=not qr_detected,
        severity="medium",
        penalty=18 if qr_detected else 0,
        title="Виявлено QR-код",
        message=(
            f"На зображенні виявлено {len(qr_marks)} QR-код(ів). Рекомендована перевірка."
            if qr_detected
            else "QR-кодів не виявлено."
        ),
        meta={"qrCount": len(qr_marks), "blocking": False, "needsReview": qr_detected},
    )

    watermark_text_evidence = _has_watermark_text_evidence(ocr_items)
    strong_watermark_hits = []
    blocking_watermark_hits = []
    for mark in watermark_marks:
        score, area_ratio, centeredness = _watermark_metrics(mark)
        if score >= WATERMARK_STRONG_SCORE and area_ratio >= WATERMARK_STRONG_AREA_RATIO:
            strong_watermark_hits.append(mark)
        if (
            score >= WATERMARK_BLOCK_SCORE
            and area_ratio >= WATERMARK_STRONG_AREA_RATIO
            and centeredness >= WATERMARK_CENTEREDNESS
            and watermark_text_evidence
        ):
            blocking_watermark_hits.append(mark)

    watermark_detected = len(strong_watermark_hits) > 0
    weak_apparel_suspicion = bool(is_apparel and watermark_marks and not strong_watermark_hits)
    watermark_blocking = bool(blocking_watermark_hits)

    ctx.add_rule_result(
        rule_id="WATERMARK_DETECTED",
        passed=not watermark_detected and not weak_apparel_suspicion,
        severity="high" if watermark_blocking else ("medium" if watermark_detected else "low"),
        penalty=30 if watermark_blocking else (8 if watermark_detected else (2 if weak_apparel_suspicion else 0)),
        title="Підозра на водяний знак",
        message=(
            "Виявлено сильні ознаки водяного знака з текстовим підтвердженням."
            if watermark_blocking
            else (
                "Виявлено watermark-подібну область, рекомендована перевірка."
                if watermark_detected
                else (
                    "Є слабка watermark-подібна ознака на apparel, але її недостатньо для блокування."
                    if weak_apparel_suspicion
                    else "Водяних знаків не виявлено."
                )
            )
        ),
        meta={
            "watermarkCount": len(watermark_marks),
            "strongWatermarkCount": len(strong_watermark_hits),
            "blockingWatermarkCount": len(blocking_watermark_hits),
            "hasTextEvidence": watermark_text_evidence,
            "sceneType": scene_type,
            "apparelMode": is_apparel,
            "blocking": watermark_blocking,
            "needsReview": watermark_detected and not watermark_blocking,
        },
    )

    non_apparel_blocking = (
        not is_apparel
        and apparel_source == "ml"
        and apparel_confidence >= NON_APPAREL_BLOCK_CONFIDENCE
    )
    non_apparel_review = not is_apparel and not non_apparel_blocking
    ctx.add_rule_result(
        rule_id="NON_APPAREL",
        passed=is_apparel,
        severity="high" if non_apparel_blocking else ("medium" if non_apparel_review else "low"),
        penalty=35 if non_apparel_blocking else (12 if non_apparel_review else 0),
        title="Не схоже на дизайн одягу",
        message=(
            "ML з високою впевненістю визначив, що зображення не є apparel-дизайном."
            if non_apparel_blocking
            else (
                "Зображення може бути non-apparel, потрібна ручна перевірка."
                if non_apparel_review
                else "Зображення виглядає як дизайн одягу."
            )
        ),
        meta={
            "is_apparel": is_apparel,
            "source": apparel_source,
            "confidence": round(apparel_confidence, 4),
            "blocking": non_apparel_blocking,
            "needsReview": non_apparel_review,
            "threshold": NON_APPAREL_BLOCK_CONFIDENCE,
        },
    )

    ctx.mark_step_done("rules")
