from __future__ import annotations

from typing import Any, Dict, List

from core.config import (
    NON_APPAREL_BLOCK_CONFIDENCE,
    WATERMARK_BLOCK_SCORE,
    WATERMARK_CENTEREDNESS,
    WATERMARK_STRONG_AREA_RATIO,
    WATERMARK_STRONG_SCORE,
)


def _estimate_words(ocr_items: List[Dict[str, Any]]) -> int:
    total = 0
    for item in ocr_items:
        text = str(item.get("text") or item.get("value") or "").strip()
        if text:
            total += len([w for w in text.split() if w.strip()])
    return total


def _watermark_metrics(mark: Dict[str, Any]) -> tuple[float, float, float]:
    meta = mark.get("meta", {}) or {}
    return (
        float(mark.get("score", 0.0) or 0.0),
        float(meta.get("areaRatio", 0.0) or 0.0),
        float(meta.get("centeredness", 0.0) or 0.0),
    )


def _has_watermark_text_evidence(ocr_items: List[Dict[str, Any]]) -> bool:
    terms = ("watermark", "stock", "preview", "sample")
    for item in ocr_items:
        text = str(item.get("text") or item.get("value") or "").lower()
        if any(term in text for term in terms):
            return True
    return False


def _get_known_brand_signal(ctx, detections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Future-ready block for new brand ML.

    Expected future format:
    ctx.ml["brand_classifier"] = {
        "label": "nike" | "adidas" | "gucci" | "other_logo" | "no_brand",
        "confidence": 0.0-1.0,
        "isReliable": bool,
        "source": "ml_brand_classifier"
    }

    Current fallback:
    - moderation_service can detect OCR brand text.
    - old ml_brand_risk stays only as weak signal.
    """
    ml = ctx.ml or {}
    moderation = ctx.moderation or {}

    brand_classifier = ml.get("brand_classifier") or ml.get("brand_detection") or {}
    if brand_classifier:
        label = str(brand_classifier.get("label", "unknown")).lower()
        confidence = float(brand_classifier.get("confidence", 0.0) or 0.0)
        reliable = bool(brand_classifier.get("isReliable", False))

        known_brand = label not in {"", "unknown", "other_logo", "no_brand", "none"}
        if known_brand and reliable and confidence >= 0.80:
            return {
                "detected": True,
                "source": "brand_classifier",
                "label": label,
                "confidence": confidence,
                "evidence": [label],
            }

    brand_text_hits: List[str] = []
    for label in moderation.get("labels", []) or []:
        if label.get("label") == "brand_text_detected":
            brand_text_hits.extend(label.get("evidence", []) or [])

    if brand_text_hits:
        return {
            "detected": True,
            "source": "ocr_brand_text",
            "label": "brand_text",
            "confidence": 0.80,
            "evidence": sorted(set(brand_text_hits)),
        }

    return {
        "detected": False,
        "source": None,
        "label": None,
        "confidence": 0.0,
        "evidence": [],
    }


def run(ctx) -> None:
    detections = ctx.detections or {}
    scene = ctx.scene or {}
    ml = ctx.ml or {}
    moderation = ctx.moderation or {}

    ocr_items = detections.get("ocr", []) or []
    lines = detections.get("lines", []) or []
    qr_marks = detections.get("qrMarks", []) or []
    watermark_marks = detections.get("watermarkMarks", []) or []
    visual_logo_marks = detections.get("visualLogoMarks", []) or []
    ip = detections.get("ip") or {}

    apparel_ml = ml.get("apparel", {})
    logo_presence = ml.get("logo_presence", {})
    old_brand_risk = detections.get("ml_brand_risk") or ml.get("brand_risk") or {}
    adult_safety = detections.get("adultSafety") or detections.get("ml_adult_safety") or ml.get("adult_safety") or {}

    scene_type = scene.get("type")
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0)
    apparel_source = scene.get("apparel_source", "unknown")

    word_count = _estimate_words(ocr_items)
    text_blocks = len(ocr_items)

    brand_signal = _get_known_brand_signal(ctx, detections)

    ctx.debug["rulesInput"] = {
        "sceneType": scene_type,
        "isApparel": is_apparel,
        "apparelConfidence": round(apparel_confidence, 4),
        "apparelSource": apparel_source,
        "ocrWords": word_count,
        "ocrBlocks": text_blocks,
        "lineCount": len(lines),
        "qrCount": len(qr_marks),
        "watermarkCount": len(watermark_marks),
        "visualLogoCount": len(visual_logo_marks),
        "brandSignal": brand_signal,
    }

    ctx.set_debug_section(
        "rules",
        {
            "policy": "creative_apparel_marketplace",
            "principle": "creative complexity is allowed; known brands and unsafe content require review or blocking",
            "inputs": ctx.debug["rulesInput"],
        },
    )

    # 1. Text amount is NOT a problem by itself.
    ctx.add_rule_result(
        rule_id="TEXT_AMOUNT_ALLOWED",
        passed=True,
        severity="low",
        penalty=0,
        title="Кількість тексту дозволена",
        message="Кількість тексту сама по собі не є порушенням.",
        meta={
            "wordCount": word_count,
            "textBlocks": text_blocks,
            "blocking": False,
            "needsReview": False,
        },
    )

    # 2. Complex lines are allowed for apparel designs.
    ctx.add_rule_result(
        rule_id="COMPLEX_PRINT_ALLOWED",
        passed=True,
        severity="low",
        penalty=0,
        title="Складний принт дозволений",
        message="Велика кількість ліній або деталей не є порушенням для дизайну одягу.",
        meta={
            "lineCount": len(lines),
            "blocking": False,
            "needsReview": False,
        },
    )

    # 3. Confirmed IP text/risk from IP analyzer blocks.
    for hit in ip.get("exactHits", []) or []:
        hit_type = hit.get("type", "ip")
        ctx.add_rule_result(
            rule_id=f"IP_{str(hit_type).upper()}_EXACT",
            passed=False,
            severity="high",
            penalty=50,
            title="Підтверджений IP/brand ризик",
            message=f"Знайдено підтверджений збіг: {hit.get('keyword')}",
            bbox=hit.get("bbox"),
            meta={**hit, "blocking": True, "needsReview": False, "riskType": "confirmed_ip"},
        )

    # 4. Suspicious IP goes to manual review.
    for hit in ip.get("suspiciousHits", []) or []:
        ctx.add_rule_result(
            rule_id="IP_SUSPECT",
            passed=False,
            severity="medium",
            penalty=10,
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
            title="IP-ризиків не знайдено",
            message="Підтверджених IP-збігів не виявлено.",
            meta={"blocking": False, "needsReview": False},
        )

    # 5. Known brand policy.
    # Known brand does not auto-fail, because business may ask for rights/license.
    if brand_signal.get("detected"):
        ctx.add_rule_result(
            rule_id="KNOWN_BRAND_REVIEW",
            passed=False,
            severity="medium",
            penalty=10,
            title="Виявлено відомий бренд",
            message="Виявлено бренд або брендоподібний сигнал. Потрібна ручна перевірка прав на використання.",
            meta={
                "blocking": False,
                "needsReview": True,
                "riskType": "known_brand",
                **brand_signal,
            },
        )

    # 6. Old brand_risk stays weak until replaced by real known-brand classifier.
    if old_brand_risk and not old_brand_risk.get("skipped"):
        label = str(old_brand_risk.get("label", "")).lower()
        confidence = float(old_brand_risk.get("confidence", 0.0) or 0.0)
        reliable = bool(old_brand_risk.get("isReliable", False))

        logo_confidence = float(logo_presence.get("confidence", 0.0) or 0.0)
        logo_reliable = bool(logo_presence.get("isReliable", False))

        strong_unknown_logo = (
            label == "brand_logo"
            and reliable
            and confidence >= 0.97
            and logo_reliable
            and logo_confidence >= 0.95
            and not brand_signal.get("detected")
        )

        ctx.add_rule_result(
            rule_id="UNKNOWN_LOGO_SIGNAL",
            passed=not strong_unknown_logo,
            severity="low",
            penalty=2 if strong_unknown_logo else 0,
            title="Невідомий logo-like сигнал",
            message=(
                "ML бачить сильний logo-like сигнал, але конкретний бренд не підтверджено."
                if strong_unknown_logo
                else "Сильного невідомого logo-like ризику не виявлено."
            ),
            meta={
                "blocking": False,
                "needsReview": False,
                "riskType": "unknown_logo_signal",
                "label": label,
                "confidence": confidence,
                "isReliable": reliable,
                "logoConfidence": logo_confidence,
                "logoReliable": logo_reliable,
                "source": "old_ml_brand_risk",
            },
        )

    # 7. Visual logo detector is helper/debug only.
    ctx.add_rule_result(
        rule_id="VISUAL_LOGO_HELPER_SIGNAL",
        passed=True,
        severity="low",
        penalty=0,
        title="Візуальний logo-like сигнал",
        message="Візуальні емблеми використовуються як допоміжний сигнал, але не блокують дизайн без brand evidence.",
        meta={
            "count": len(visual_logo_marks),
            "blocking": False,
            "needsReview": False,
        },
    )

    # 8. QR with decoded content goes to review. Fake QR-like shapes do not matter.
    decoded_qr = [q for q in qr_marks if str(q.get("decodedText", "")).strip()]
    ctx.add_rule_result(
        rule_id="QR_CODE_DECODED",
        passed=not decoded_qr,
        severity="medium" if decoded_qr else "low",
        penalty=8 if decoded_qr else 0,
        title="QR-код",
        message=(
            "На зображенні є QR-код із decoded content. Потрібна ручна перевірка."
            if decoded_qr
            else "QR-кодів із decoded content не виявлено."
        ),
        meta={
            "qrCount": len(qr_marks),
            "decodedQrCount": len(decoded_qr),
            "blocking": False,
            "needsReview": bool(decoded_qr),
        },
    )

    # 9. Adult/NSFW ML.
    if adult_safety:
        label = str(adult_safety.get("label", "")).lower()
        adult_score = float(adult_safety.get("adultScore", 0.0) or 0.0)
        risk_level = str(adult_safety.get("riskLevel", "")).lower()
        reliable = bool(adult_safety.get("isReliable", False))

        blocking = risk_level == "block" and reliable
        review = (
            not blocking
            and (
                label in {"adult_risk", "nsfw", "sexual", "explicit", "anime_sexualized", "ecchi"}
                or adult_score >= 0.55
            )
        )

        if blocking or review:
            ctx.add_rule_result(
                rule_id="ADULT_VISUAL_RISK",
                passed=False,
                severity="high" if blocking else "medium",
                penalty=45 if blocking else 15,
                title="Adult/NSFW ризик",
                message=(
                    f"ML виявив блокуючий adult/NSFW ризик (score={adult_score:.2f})."
                    if blocking
                    else f"ML виявив можливий adult/NSFW ризик (score={adult_score:.2f})."
                ),
                meta={
                    "blocking": blocking,
                    "needsReview": review,
                    "riskType": "adult_visual",
                    "label": label,
                    "adultScore": adult_score,
                    "riskLevel": risk_level,
                    "isReliable": reliable,
                    "source": "adult_safety",
                },
            )

    # 10. Watermark policy.
    watermark_text_evidence = _has_watermark_text_evidence(ocr_items)
    strong_watermarks = []
    blocking_watermarks = []

    for mark in watermark_marks:
        score, area_ratio, centeredness = _watermark_metrics(mark)

        if score >= WATERMARK_STRONG_SCORE and area_ratio >= WATERMARK_STRONG_AREA_RATIO:
            strong_watermarks.append(mark)

        if (
            score >= WATERMARK_BLOCK_SCORE
            and area_ratio >= WATERMARK_STRONG_AREA_RATIO
            and centeredness >= WATERMARK_CENTEREDNESS
            and watermark_text_evidence
        ):
            blocking_watermarks.append(mark)

    watermark_blocking = bool(blocking_watermarks)
    watermark_review = bool(strong_watermarks) and not watermark_blocking

    ctx.add_rule_result(
        rule_id="WATERMARK_RISK",
        passed=not watermark_blocking and not watermark_review,
        severity="high" if watermark_blocking else ("medium" if watermark_review else "low"),
        penalty=30 if watermark_blocking else (8 if watermark_review else 0),
        title="Підозра на водяний знак",
        message=(
            "Виявлено сильні ознаки водяного знака з текстовим підтвердженням."
            if watermark_blocking
            else (
                "Виявлено watermark-like область. Потрібна ручна перевірка."
                if watermark_review
                else "Водяних знаків не виявлено."
            )
        ),
        meta={
            "watermarkCount": len(watermark_marks),
            "strongWatermarkCount": len(strong_watermarks),
            "blockingWatermarkCount": len(blocking_watermarks),
            "hasTextEvidence": watermark_text_evidence,
            "blocking": watermark_blocking,
            "needsReview": watermark_review,
        },
    )

    # 11. Non-apparel policy.
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
        penalty=25 if non_apparel_blocking else (4 if non_apparel_review else 0),
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
            "blocking": non_apparel_blocking,
            "needsReview": non_apparel_review,
            "isApparel": is_apparel,
            "confidence": round(apparel_confidence, 4),
            "source": apparel_source,
            "threshold": NON_APPAREL_BLOCK_CONFIDENCE,
        },
    )

    ctx.mark_step_done("rules")