from __future__ import annotations

import re
from typing import Any, Dict, List


SEXUAL_TERMS = {
    "sex",
    "sexy",
    "xxx",
    "porn",
    "onlyfans",
    "nude",
    "naked",
    "fetish",
    "bdsm",
    "lingerie",
    "erotic",
    "18+",
    "nsfw",
}

HATE_TERMS = {
    "nazi",
    "hitler",
    "white power",
    "kkk",
    "heil",
    "swastika",
}

VIOLENCE_TERMS = {
    "kill",
    "murder",
    "rape",
    "shoot",
    "stab",
    "bloodbath",
}

SELF_HARM_TERMS = {
    "suicide",
    "self harm",
    "cut myself",
    "kill myself",
}

BRAND_TERMS = {
    "nike",
    "adidas",
    "puma",
    "gucci",
    "prada",
    "balenciaga",
    "louis vuitton",
    "supreme",
    "chanel",
    "dior",
    "versace",
}


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


def _find_matches(normalized_text: str, terms: set[str]) -> List[str]:
    hits: List[str] = []

    for term in terms:
        if term in normalized_text:
            hits.append(term)

    return sorted(set(hits))


def moderate_image_and_text(
    *,
    scene: Dict[str, Any],
    detections: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    ocr_items = detections.get("ocr") or []
    texts = _extract_ocr_texts(ocr_items)
    merged_text = _normalize_text(" ".join(texts))

    sexual_hits = _find_matches(merged_text, SEXUAL_TERMS)
    hate_hits = _find_matches(merged_text, HATE_TERMS)
    violence_hits = _find_matches(merged_text, VIOLENCE_TERMS)
    self_harm_hits = _find_matches(merged_text, SELF_HARM_TERMS)
    brand_hits = _find_matches(merged_text, BRAND_TERMS)

    labels: List[Dict[str, Any]] = []

    def push_label(name: str, score: float, blocked: bool, evidence: List[str]) -> None:
        labels.append(
            {
                "label": name,
                "score": round(float(score), 4),
                "blocked": blocked,
                "evidence": evidence,
            }
        )

    if sexual_hits:
        push_label("sexual_text", 0.98, True, sexual_hits)

    if hate_hits:
        push_label("hate_symbol_or_text", 0.99, True, hate_hits)

    if violence_hits:
        push_label("graphic_or_violent_text", 0.95, True, violence_hits)

    if self_harm_hits:
        push_label("self_harm_text", 0.99, True, self_harm_hits)

    if brand_hits:
        push_label("brand_text_detected", 0.80, False, brand_hits)

    blur_score = float(quality.get("blur_score", 0.0) or 0.0)
    if blur_score < 20:
        push_label(
            "very_low_visual_reliability",
            0.70,
            False,
            [f"blur_score={blur_score:.2f}"],
        )

    blocked_labels = [label["label"] for label in labels if label["blocked"]]
    needs_review = any(label["label"] == "very_low_visual_reliability" for label in labels)

    return {
        "ok": len(blocked_labels) == 0,
        "blocked": len(blocked_labels) > 0,
        "needsReview": needs_review and len(blocked_labels) == 0,
        "labels": labels,
        "blockedReasons": blocked_labels,
        "textSignals": {
            "sexualHits": sexual_hits,
            "hateHits": hate_hits,
            "violenceHits": violence_hits,
            "selfHarmHits": self_harm_hits,
            "brandHits": brand_hits,
        },
        "sceneType": scene.get("type", "unknown"),
    }