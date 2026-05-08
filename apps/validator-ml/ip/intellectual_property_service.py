from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List


BRAND_KEYWORDS = [
    "nike", "adidas", "puma", "gucci", "prada", "balenciaga",
    "louis vuitton", "lv", "supreme", "chanel", "dior", "versace",
    "off-white", "new balance", "under armour", "reebok", "fila",
]

CHARACTER_KEYWORDS = [
    "mickey", "minnie", "spiderman", "spider-man", "batman", "superman",
    "naruto", "luffy", "goku", "pikachu", "pokemon", "hello kitty",
    "barbie", "elsa", "iron man", "captain america", "deadpool",
]

FRANCHISE_KEYWORDS = [
    "marvel", "dc", "disney", "pixar", "star wars", "harry potter",
    "pokemon", "dragon ball", "naruto", "one piece", "minecraft",
    "fortnite", "league of legends", "playstation", "xbox",
]

SLOGAN_KEYWORDS = [
    "just do it", "i'm lovin' it", "im lovin it", "think different",
]

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
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _collect_ocr_texts(ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in ocr_items or []:
        raw = str(item.get("text") or item.get("value") or "").strip()
        if not raw:
            continue
        out.append(
            {
                "text": raw,
                "normalized": _normalize_text(raw),
                "bbox": item.get("bbox"),
                "confidence": item.get("confidence"),
            }
        )
    return out


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _match_keywords(
    entries: List[Dict[str, Any]],
    keywords: List[str],
    label: str,
    exact_threshold: float = 0.98,
    fuzzy_threshold: float = 0.86,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    for entry in entries:
        text = entry["normalized"]

        for keyword in keywords:
            kw = _normalize_text(keyword)

            if kw in text:
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

            score = _similarity(text, kw)
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

    unique: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits:
        key = (hit["type"], hit["keyword"], hit["matchedText"], hit["matchKind"])
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique


def analyze_ip_risk(
    *,
    detections: Dict[str, Any],
) -> Dict[str, Any]:
    ocr_items = detections.get("ocr") or []
    texts = _collect_ocr_texts(ocr_items)

    brand_hits = _match_keywords(texts, BRAND_KEYWORDS, "brand")
    character_hits = _match_keywords(texts, CHARACTER_KEYWORDS, "character")
    franchise_hits = _match_keywords(texts, FRANCHISE_KEYWORDS, "franchise")
    slogan_hits = _match_keywords(texts, SLOGAN_KEYWORDS, "slogan")

    all_hits = brand_hits + character_hits + franchise_hits + slogan_hits

    exact_hits = [
        h for h in all_hits
        if h["matchKind"] in {"exact_substring", "exact_fuzzy"}
    ]
    
    suspicious_hits = [
        h for h in all_hits
        if h["matchKind"] == "suspicious_fuzzy"
    ]

    blocking_exact_hits = [
        h for h in exact_hits
        if h["type"] in {"character", "franchise"}
    ]

    review_hits = [
        h for h in exact_hits + suspicious_hits
        if h["type"] in {"brand", "slogan"}
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