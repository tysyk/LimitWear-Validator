from __future__ import annotations

from typing import Any, Dict, List


SAFE_MARGIN_RATIO = 0.05
TOO_MUCH_TEXT_WORDS = 12
TOO_MUCH_TEXT_BLOCKS = 5
HIGH_SKEW_DEG = 8.0
MESSY_LINES_COUNT = 45
LOGO_LIKE_REVIEW_COUNT = 3


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
            total += len([x for x in text.split() if x.strip()])
    return total


def _text_near_edge(bbox: List[int], width: int, height: int, margin_ratio: float) -> bool:
    x1, y1, x2, y2 = bbox
    mx = int(width * margin_ratio)
    my = int(height * margin_ratio)
    return x1 <= mx or y1 <= my or x2 >= (width - mx) or y2 >= (height - my)


def run(ctx):
    width = int(ctx.width)
    height = int(ctx.height)

    ocr_items = ctx.detections.get("ocr") or []
    lines = ctx.detections.get("lines") or []
    ip = ctx.detections.get("ip") or {}
    logo_like = ctx.detections.get("logoLikeMarks") or []
    skew_angle = ctx.debug.get("skew_angle_deg")

    word_count = _estimate_words(ocr_items)
    text_blocks = len(ocr_items)

    ctx.debug["rulesInput"] = {
        "wordCount": word_count,
        "textBlocks": text_blocks,
        "lineCount": len(lines),
        "skewAngleDeg": skew_angle,
        "logoLikeCount": len(logo_like),
        "ipExactHits": len(ip.get("exactHits", [])),
        "ipSuspiciousHits": len(ip.get("suspiciousHits", [])),
    }

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

    skew_bad = isinstance(skew_angle, (int, float)) and abs(float(skew_angle)) >= HIGH_SKEW_DEG
    ctx.add_rule_result(
        rule_id="HIGH_SKEW",
        passed=not skew_bad,
        severity="medium",
        penalty=15 if skew_bad else 0,
        title="Сильний перекіс",
        message=(
            f"Виявлено сильний перекіс: {float(skew_angle):.2f}°."
            if skew_bad
            else "Критичного перекосу не виявлено."
        ),
        meta={"skewAngleDeg": skew_angle, "threshold": HIGH_SKEW_DEG},
    )

    messy_lines = len(lines) >= MESSY_LINES_COUNT
    ctx.add_rule_result(
        rule_id="MESSY_LINES",
        passed=not messy_lines,
        severity="low",
        penalty=10 if messy_lines else 0,
        title="Перенавантажений ескіз",
        message=(
            f"Виявлено надто багато ліній: {len(lines)}."
            if messy_lines
            else "Кількість ліній в межах допустимого."
        ),
        meta={"lineCount": len(lines), "threshold": MESSY_LINES_COUNT},
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

    logo_like_review = len(logo_like) >= LOGO_LIKE_REVIEW_COUNT
    ctx.add_rule_result(
        rule_id="LOGO_LIKE_MARKS",
        passed=not logo_like_review,
        severity="medium",
        penalty=18 if logo_like_review else 0,
        title="Підозра на візуальні емблеми",
        message=(
            f"Виявлено багато компактних контрастних емблемоподібних форм: {len(logo_like)}."
            if logo_like_review
            else "Критичної кількості емблемоподібних форм не виявлено."
        ),
        meta={"logoLikeCount": len(logo_like), "threshold": LOGO_LIKE_REVIEW_COUNT},
    )