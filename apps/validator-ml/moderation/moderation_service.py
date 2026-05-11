from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from core.brand_keywords import OCR_BRAND_KEYWORDS
from core.moderation_terms import (
    HATE_TERMS,
    SELF_HARM_TERMS,
    SEXUAL_BLOCK_TERMS,
    SEXUAL_REVIEW_TERMS,
    VIOLENCE_BLOCK_TERMS,
    VIOLENCE_REVIEW_TERMS,
)


BRAND_TERMS = OCR_BRAND_KEYWORDS


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_ocr_texts(ocr_items: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []

    for item in ocr_items or []:
        value = item.get("text") or item.get("value") or ""

        if not isinstance(value, str):
            continue

        value = value.strip()

        if value:
            texts.append(value)

    return texts


def _find_matches(normalized_text: str, terms: Set[str]) -> List[str]:
    hits: List[str] = []

    for term in terms:
        normalized_term = _normalize_text(term)

        if normalized_term and normalized_term in normalized_text:
            hits.append(term)

    return sorted(set(hits))


def _push_label(
    labels: List[Dict[str, Any]],
    *,
    name: str,
    score: float,
    blocked: bool,
    evidence: List[str],
    needs_review: bool = False,
) -> None:
    labels.append(
        {
            "label": name,
            "score": round(float(score), 4),
            "blocked": blocked,
            "needsReview": needs_review,
            "evidence": evidence,
        }
    )


def _collect_text_signals(merged_text: str) -> Dict[str, List[str]]:
    return {
        "sexualHits": _find_matches(merged_text, SEXUAL_BLOCK_TERMS),
        "sexualReviewHits": _find_matches(merged_text, SEXUAL_REVIEW_TERMS),
        "hateHits": _find_matches(merged_text, HATE_TERMS),
        "violenceHits": _find_matches(merged_text, VIOLENCE_BLOCK_TERMS),
        "violenceReviewHits": _find_matches(merged_text, VIOLENCE_REVIEW_TERMS),
        "selfHarmHits": _find_matches(merged_text, SELF_HARM_TERMS),
        "brandHits": _find_matches(merged_text, BRAND_TERMS),
    }


def _add_text_moderation_labels(
    *,
    labels: List[Dict[str, Any]],
    text_signals: Dict[str, List[str]],
) -> None:
    if text_signals["sexualHits"]:
        _push_label(
            labels,
            name="sexual_text",
            score=0.98,
            blocked=True,
            evidence=text_signals["sexualHits"],
        )

    if text_signals["sexualReviewHits"]:
        _push_label(
            labels,
            name="sexual_text_review",
            score=0.72,
            blocked=False,
            evidence=text_signals["sexualReviewHits"],
            needs_review=True,
        )

    if text_signals["hateHits"]:
        _push_label(
            labels,
            name="hate_symbol_or_text",
            score=0.99,
            blocked=True,
            evidence=text_signals["hateHits"],
        )

    if text_signals["violenceHits"]:
        _push_label(
            labels,
            name="graphic_or_violent_text",
            score=0.95,
            blocked=True,
            evidence=text_signals["violenceHits"],
        )

    if text_signals["violenceReviewHits"]:
        _push_label(
            labels,
            name="violent_text_review",
            score=0.70,
            blocked=False,
            evidence=text_signals["violenceReviewHits"],
            needs_review=True,
        )

    if text_signals["selfHarmHits"]:
        _push_label(
            labels,
            name="self_harm_text",
            score=0.99,
            blocked=True,
            evidence=text_signals["selfHarmHits"],
        )

    if text_signals["brandHits"]:
        _push_label(
            labels,
            name="brand_text_detected",
            score=0.80,
            blocked=False,
            evidence=text_signals["brandHits"],
            needs_review=True,
        )


def _add_visual_moderation_labels(
    *,
    labels: List[Dict[str, Any]],
    detections: Dict[str, Any],
) -> None:
    adult_ml = detections.get("adultSafety") or {}

    adult_label = str(adult_ml.get("label", "")).lower()
    adult_confidence = float(adult_ml.get("confidence", 0.0) or 0.0)

    if (
        adult_label
        in {
            "sexual",
            "nsfw",
            "explicit",
            "anime_sexualized",
            "ecchi",
        }
        and adult_confidence >= 0.55
    ):
        _push_label(
            labels,
            name="adult_visual_content",
            score=adult_confidence,
            blocked=False,
            evidence=[adult_label],
            needs_review=True,
        )


def _add_quality_moderation_labels(
    *,
    labels: List[Dict[str, Any]],
    quality: Dict[str, Any],
) -> None:
    blur_score = float(quality.get("blur_score", 0.0) or 0.0)

    if blur_score < 20:
        _push_label(
            labels,
            name="very_low_visual_reliability",
            score=0.70,
            blocked=False,
            evidence=[f"blur_score={blur_score:.2f}"],
            needs_review=True,
        )


def moderate_image_and_text(
    *,
    scene: Dict[str, Any],
    detections: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    ocr_items = detections.get("ocr") or []
    texts = _extract_ocr_texts(ocr_items)
    merged_text = _normalize_text(" ".join(texts))

    text_signals = _collect_text_signals(merged_text)

    labels: List[Dict[str, Any]] = []

    _add_text_moderation_labels(
        labels=labels,
        text_signals=text_signals,
    )

    _add_visual_moderation_labels(
        labels=labels,
        detections=detections,
    )

    _add_quality_moderation_labels(
        labels=labels,
        quality=quality,
    )

    blocked_labels = [
        label["label"]
        for label in labels
        if label.get("blocked")
    ]

    needs_review = any(
        label.get("needsReview")
        for label in labels
    )

    return {
        "ok": len(blocked_labels) == 0,
        "blocked": len(blocked_labels) > 0,
        "needsReview": needs_review and len(blocked_labels) == 0,
        "labels": labels,
        "blockedReasons": blocked_labels,
        "textSignals": text_signals,
        "sceneType": scene.get("type", "unknown"),
    }