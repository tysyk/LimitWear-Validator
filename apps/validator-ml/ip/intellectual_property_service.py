from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from core.brand_keywords import OCR_BRAND_KEYWORDS
from core.ip_keywords import (
    CHARACTER_KEYWORDS,
    FRANCHISE_KEYWORDS,
    SLOGAN_KEYWORDS,
)


REPLACEMENTS = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
}


def _normalize_text(text: str) -> str:
    text = text.lower().strip()

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _collect_ocr_texts(ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for item in ocr_items or []:
        raw_text = str(item.get("text") or item.get("value") or "").strip()

        if not raw_text:
            continue

        entries.append(
            {
                "text": raw_text,
                "normalized": _normalize_text(raw_text),
                "bbox": item.get("bbox"),
                "confidence": item.get("confidence"),
            }
        )

    return entries


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _deduplicate_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()

    for hit in hits:
        key = (
            hit.get("type"),
            hit.get("keyword"),
            hit.get("matchedText"),
            hit.get("matchKind"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(hit)

    return unique


def _match_keywords(
    entries: List[Dict[str, Any]],
    keywords,
    label: str,
    exact_threshold: float = 0.98,
    fuzzy_threshold: float = 0.86,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    for entry in entries:
        text = entry["normalized"]

        if not text:
            continue

        for keyword in keywords:
            normalized_keyword = _normalize_text(str(keyword))

            if not normalized_keyword:
                continue

            if normalized_keyword in text:
                hits.append(
                    {
                        "type": label,
                        "matchKind": "exact_substring",
                        "keyword": keyword,
                        "matchedText": entry["text"],
                        "bbox": entry.get("bbox"),
                        "score": 0.99,
                    }
                )
                continue

            score = _similarity(text, normalized_keyword)

            if score >= exact_threshold:
                hits.append(
                    {
                        "type": label,
                        "matchKind": "exact_fuzzy",
                        "keyword": keyword,
                        "matchedText": entry["text"],
                        "bbox": entry.get("bbox"),
                        "score": round(score, 4),
                    }
                )

            elif score >= fuzzy_threshold:
                hits.append(
                    {
                        "type": label,
                        "matchKind": "suspicious_fuzzy",
                        "keyword": keyword,
                        "matchedText": entry["text"],
                        "bbox": entry.get("bbox"),
                        "score": round(score, 4),
                    }
                )

    return _deduplicate_hits(hits)


def analyze_ip_risk(
    *,
    detections: Dict[str, Any],
) -> Dict[str, Any]:
    ocr_items = detections.get("ocr") or []
    texts = _collect_ocr_texts(ocr_items)

    brand_hits = _match_keywords(texts, OCR_BRAND_KEYWORDS, "brand")
    character_hits = _match_keywords(texts, CHARACTER_KEYWORDS, "character")
    franchise_hits = _match_keywords(texts, FRANCHISE_KEYWORDS, "franchise")
    slogan_hits = _match_keywords(texts, SLOGAN_KEYWORDS, "slogan")

    all_hits = (
        brand_hits
        + character_hits
        + franchise_hits
        + slogan_hits
    )

    exact_hits = [
        hit
        for hit in all_hits
        if hit.get("matchKind") in {"exact_substring", "exact_fuzzy"}
    ]

    suspicious_hits = [
        hit
        for hit in all_hits
        if hit.get("matchKind") == "suspicious_fuzzy"
    ]

    blocking_exact_hits = [
        hit
        for hit in exact_hits
        if hit.get("type") in {"character", "franchise"}
    ]

    review_hits = [
        hit
        for hit in exact_hits + suspicious_hits
        if hit.get("type") in {"brand", "slogan"}
    ]

    blocked = len(blocking_exact_hits) > 0
    needs_review = len(review_hits) > 0 or len(suspicious_hits) > 0

    return {
        "blocked": blocked,
        "needsReview": needs_review,
        "brandTextHits": brand_hits,
        "characterHits": character_hits,
        "franchiseHits": franchise_hits,
        "sloganHits": slogan_hits,
        "exactHits": exact_hits,
        "suspiciousHits": suspicious_hits,
        "blockingHits": blocking_exact_hits,
        "reviewHits": review_hits,
    }