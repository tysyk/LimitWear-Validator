from __future__ import annotations

from typing import Any, Dict, List

from core.config import (
    NON_APPAREL_BLOCK_CONFIDENCE,
    WATERMARK_BLOCK_SCORE,
    WATERMARK_CENTEREDNESS,
    WATERMARK_STRONG_AREA_RATIO,
    WATERMARK_STRONG_SCORE,
)
from core.messages import get_rule_message


NON_BRAND_LABELS = {
    "",
    "unknown",
    "no_brand",
    "unknown_logo",
    "none",
}


def _estimate_words(ocr_items: List[Dict[str, Any]]) -> int:
    total = 0

    for item in ocr_items:
        text = str(item.get("text") or item.get("value") or "").strip()

        if text:
            total += len([word for word in text.split() if word.strip()])

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


def _get_brand_signal(ctx) -> Dict[str, Any]:
    brand_model = (ctx.ml or {}).get("brand_crop_classifier") or {}

    raw_label = str(
        brand_model.get("raw_label")
        or brand_model.get("brand_label")
        or brand_model.get("label")
        or "unknown"
    ).lower()

    brand_label = str(
        brand_model.get("brand_label")
        or raw_label
    ).lower()

    confidence = float(brand_model.get("confidence", 0.0) or 0.0)
    is_known_brand = bool(brand_model.get("isKnownBrand", False))
    is_reliable = bool(brand_model.get("isReliable", False))

    suspected_known_brand = bool(
        brand_model.get("suspectedKnownBrand", False)
    )

    suspected_brand_label = str(
        brand_model.get("suspectedBrandLabel") or ""
    ).lower()

    suspected_brand_confidence = float(
        brand_model.get("suspectedBrandConfidence", 0.0) or 0.0
    )

    if (
        is_known_brand
        and is_reliable
        and brand_label not in NON_BRAND_LABELS
    ):
        return {
            "detected": True,
            "source": "brand_crop_classifier",
            "label": brand_label,
            "confidence": confidence,
            "bbox": brand_model.get("crop_bbox"),
            "original_bbox": brand_model.get("original_bbox"),
            "topPredictions": brand_model.get("top_predictions", []),
            "evidence": [brand_label],
            "suspected": False,
        }

    if (
        suspected_known_brand
        and suspected_brand_label
        and suspected_brand_label not in NON_BRAND_LABELS
    ):
        return {
            "detected": True,
            "source": "brand_crop_classifier_suspected",
            "label": suspected_brand_label,
            "confidence": suspected_brand_confidence,
            "bbox": brand_model.get("crop_bbox"),
            "original_bbox": brand_model.get("original_bbox"),
            "topPredictions": brand_model.get("top_predictions", []),
            "evidence": [suspected_brand_label],
            "suspected": True,
        }

    return {
        "detected": False,
        "source": "brand_crop_classifier",
        "label": brand_label,
        "confidence": confidence,
        "bbox": brand_model.get("crop_bbox"),
        "original_bbox": brand_model.get("original_bbox"),
        "topPredictions": brand_model.get("top_predictions", []),
        "evidence": [],
        "suspected": False,
    }


def _get_ocr_brand_hits(moderation: Dict[str, Any]) -> List[str]:
    brand_hits: List[str] = []

    text_signals = moderation.get("textSignals", {}) or {}
    brand_hits.extend(text_signals.get("brandHits", []) or [])

    for label in moderation.get("labels", []) or []:
        if label.get("label") == "brand_text_detected":
            brand_hits.extend(label.get("evidence", []) or [])

    return sorted(set(str(item).lower() for item in brand_hits if item))


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
    logo_candidates = detections.get("logoCandidates", []) or []
    ip = detections.get("ip") or {}

    adult_safety = (
        detections.get("adultSafety")
        or detections.get("ml_adult_safety")
        or ml.get("adult_safety")
        or {}
    )

    apparel_ml = ml.get("apparel", {}) or {}

    scene_type = scene.get("type")
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(
        scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0))
        or 0.0
    )
    apparel_source = scene.get("apparel_source", "unknown")

    word_count = _estimate_words(ocr_items)
    text_blocks = len(ocr_items)

    brand_signal = _get_brand_signal(ctx)
    ocr_brand_hits = _get_ocr_brand_hits(moderation)

    if ocr_brand_hits and not brand_signal.get("detected"):
        brand_signal = {
            "detected": True,
            "source": "ocr_brand_text",
            "label": "brand_text",
            "confidence": 0.80,
            "bbox": None,
            "original_bbox": None,
            "topPredictions": [],
            "evidence": ocr_brand_hits,
            "suspected": True,
        }

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
        "logoCandidateCount": len(logo_candidates),
        "brandSignal": brand_signal,
    }

    ctx.set_debug_section(
        "rules",
        {
            "policy": "creative_apparel_marketplace",
            "principle": (
                "creative complexity is allowed; known brands and unsafe content "
                "require review or blocking"
            ),
            "inputs": ctx.debug["rulesInput"],
        },
    )

    if brand_signal.get("detected"):
        message = get_rule_message("KNOWN_BRAND_REVIEW")

        ctx.add_rule_result(
            rule_id="KNOWN_BRAND_REVIEW",
            passed=False,
            severity="medium",
            penalty=10,
            title=message["title"],
            message=message["message"],
            bbox=brand_signal.get("bbox"),
            meta={
                "blocking": False,
                "needsReview": True,
                "riskType": "known_brand",
                **brand_signal,
            },
        )

    for hit in ip.get("exactHits", []) or []:
        hit_type = str(hit.get("type", "ip")).lower()

        if hit_type == "brand":
            message = get_rule_message("IP_BRAND_REVIEW")

            ctx.add_rule_result(
                rule_id="IP_BRAND_REVIEW",
                passed=False,
                severity="medium",
                penalty=10,
                title=message["title"],
                message=message["message"],
                bbox=hit.get("bbox"),
                meta={
                    **hit,
                    "blocking": False,
                    "needsReview": True,
                    "riskType": "brand_review",
                },
            )
        else:
            message = get_rule_message("IP_CONFIRMED")

            ctx.add_rule_result(
                rule_id=f"IP_{hit_type.upper()}_EXACT",
                passed=False,
                severity="high",
                penalty=50,
                title=message["title"],
                message=message["message"],
                bbox=hit.get("bbox"),
                meta={
                    **hit,
                    "blocking": True,
                    "needsReview": False,
                    "riskType": "confirmed_ip",
                },
            )

    for hit in ip.get("suspiciousHits", []) or []:
        message = get_rule_message("IP_SUSPECT")

        ctx.add_rule_result(
            rule_id="IP_SUSPECT",
            passed=False,
            severity="medium",
            penalty=10,
            title=message["title"],
            message=message["message"],
            bbox=hit.get("bbox"),
            meta={
                **hit,
                "blocking": False,
                "needsReview": True,
                "riskType": "suspected_ip",
            },
        )

    if not ip.get("exactHits") and not ip.get("suspiciousHits"):
        ctx.add_rule_result(
            rule_id="IP_RISK_CLEAR",
            passed=True,
            severity="low",
            penalty=0,
            title="IP-ризиків не знайдено",
            message="Підтверджених IP-збігів не виявлено.",
            meta={
                "blocking": False,
                "needsReview": False,
            },
        )

    ctx.add_rule_result(
        rule_id="VISUAL_LOGO_HELPER_SIGNAL",
        passed=True,
        severity="low",
        penalty=0,
        title="Візуальний logo-like сигнал",
        message=(
            "Візуальні емблеми використовуються як допоміжний сигнал "
            "і не блокують дизайн без підтвердження від brand classifier або OCR."
        ),
        meta={
            "visualLogoCount": len(visual_logo_marks),
            "logoCandidateCount": len(logo_candidates),
            "blocking": False,
            "needsReview": False,
        },
    )

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

    decoded_qr = [
        item for item in qr_marks
        if str(item.get("decodedText", "")).strip()
    ]

    qr_message = get_rule_message(
        "QR_CODE_REVIEW" if decoded_qr else "QR_CODE_CLEAR"
    )

    ctx.add_rule_result(
        rule_id="QR_CODE_DECODED",
        passed=not decoded_qr,
        severity="medium" if decoded_qr else "low",
        penalty=8 if decoded_qr else 0,
        title=qr_message["title"],
        message=qr_message["message"],
        meta={
            "qrCount": len(qr_marks),
            "decodedQrCount": len(decoded_qr),
            "blocking": False,
            "needsReview": bool(decoded_qr),
        },
    )

    if adult_safety:
        label = str(adult_safety.get("label", "")).lower()
        adult_score = float(adult_safety.get("adultScore", 0.0) or 0.0)
        risk_level = str(adult_safety.get("riskLevel", "")).lower()
        reliable = bool(adult_safety.get("isReliable", False))

        blocking = risk_level == "block" and reliable
        review = (
            not blocking
            and (
                label in {
                    "adult_risk",
                    "nsfw",
                    "sexual",
                    "explicit",
                    "anime_sexualized",
                    "ecchi",
                }
                or adult_score >= 0.55
            )
        )

        if blocking or review:
            message_key = (
                "ADULT_VISUAL_RISK_BLOCK"
                if blocking
                else "ADULT_VISUAL_RISK_REVIEW"
            )
            message = get_rule_message(message_key)

            ctx.add_rule_result(
                rule_id="ADULT_VISUAL_RISK",
                passed=False,
                severity="high" if blocking else "medium",
                penalty=45 if blocking else 15,
                title=message["title"],
                message=message["message"],
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

    watermark_text_evidence = _has_watermark_text_evidence(ocr_items)
    strong_watermarks = []
    blocking_watermarks = []

    for mark in watermark_marks:
        score, area_ratio, centeredness = _watermark_metrics(mark)

        if (
            score >= WATERMARK_STRONG_SCORE
            and area_ratio >= WATERMARK_STRONG_AREA_RATIO
        ):
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

    if watermark_blocking:
        watermark_message = get_rule_message("WATERMARK_RISK_BLOCK")
    elif watermark_review:
        watermark_message = get_rule_message("WATERMARK_RISK_REVIEW")
    else:
        watermark_message = get_rule_message("WATERMARK_RISK_CLEAR")

    ctx.add_rule_result(
        rule_id="WATERMARK_RISK",
        passed=not watermark_blocking and not watermark_review,
        severity="high" if watermark_blocking else ("medium" if watermark_review else "low"),
        penalty=30 if watermark_blocking else (8 if watermark_review else 0),
        title=watermark_message["title"],
        message=watermark_message["message"],
        meta={
            "watermarkCount": len(watermark_marks),
            "strongWatermarkCount": len(strong_watermarks),
            "blockingWatermarkCount": len(blocking_watermarks),
            "hasTextEvidence": watermark_text_evidence,
            "blocking": watermark_blocking,
            "needsReview": watermark_review,
        },
    )

    non_apparel_blocking = (
        not is_apparel
        and apparel_source == "ml"
        and apparel_confidence >= NON_APPAREL_BLOCK_CONFIDENCE
    )

    non_apparel_review = not is_apparel and not non_apparel_blocking

    if non_apparel_blocking:
        non_apparel_message = get_rule_message("NON_APPAREL_BLOCK")
    elif non_apparel_review:
        non_apparel_message = get_rule_message("NON_APPAREL_REVIEW")
    else:
        non_apparel_message = get_rule_message("NON_APPAREL_CLEAR")

    ctx.add_rule_result(
        rule_id="NON_APPAREL",
        passed=is_apparel,
        severity="high" if non_apparel_blocking else ("medium" if non_apparel_review else "low"),
        penalty=25 if non_apparel_blocking else (4 if non_apparel_review else 0),
        title=non_apparel_message["title"],
        message=non_apparel_message["message"],
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