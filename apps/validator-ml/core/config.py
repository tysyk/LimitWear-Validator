from __future__ import annotations


# =========================
# APPAREL ML
# =========================

APPAREL_CONFIDENCE_THRESHOLD = 0.88
APPAREL_TYPE_CONFIDENCE_THRESHOLD = 0.70
NON_APPAREL_BLOCK_CONFIDENCE = 0.95


# =========================
# QUALITY
# =========================

QUALITY_MIN_WIDTH = 256
QUALITY_MIN_HEIGHT = 256
QUALITY_MIN_LAPLACIAN_VARIANCE = 40.0
QUALITY_CRITICAL_SCORE = 0.35


# =========================
# TEXT / LAYOUT
# =========================

SAFE_MARGIN_RATIO = 0.03

TOO_MUCH_TEXT_WORDS = 50
TOO_MUCH_TEXT_BLOCKS = 20


# =========================
# SKEW / LINES
# =========================

SKEW_DOCUMENT_ANGLE_DEG = 8.0
SKEW_APPAREL_ANGLE_DEG = 18.0

SKEW_DOCUMENT_SUPPORT_LINES = 5
SKEW_APPAREL_SUPPORT_LINES = 12

SKEW_DOCUMENT_CONFIDENCE = 0.45
SKEW_APPAREL_CONFIDENCE = 0.75

MESSY_LINES_COUNT = 80
APPAREL_MESSY_LINES_COUNT = 350


# =========================
# LOGO / VISUAL MARKS
# =========================

LOGO_CONFIDENCE_THRESHOLD = 0.75

LOGO_LIKE_REVIEW_COUNT = 8
APPAREL_LOGO_LIKE_REVIEW_COUNT = 20

VISUAL_LOGO_CENTER_SCORE = 0.88
VISUAL_LOGO_CENTER_AREA_RATIO = 0.08
VISUAL_LOGO_CENTER_DISTANCE = 0.20

VISUAL_LOGO_MEDIUM_SCORE = 0.82
VISUAL_LOGO_MEDIUM_AREA_RATIO = 0.04


# =========================
# BRAND CROP CLASSIFIER
# =========================

BRAND_CONFIDENCE_THRESHOLD = 0.75
BRAND_SUSPECTED_THRESHOLD = 0.50

SAVE_DEBUG_BRAND_CROPS = False
DEBUG_BRAND_CROPS_DIR = "artifacts/debug_brand_crops"


# =========================
# CHEST LOGO FALLBACK
# =========================

CHEST_LOGO_FALLBACK_ENABLED = True
CHEST_LOGO_FALLBACK_SCORE = 0.45

# bbox format: x1, y1, x2, y2 as ratios of image width/height
CHEST_LOGO_ZONES = [
    {
        "id": "chest_candidate_1",
        "source": "chest_left_fallback",
        "bbox_ratio": [0.18, 0.12, 0.52, 0.43],
    },
    {
        "id": "chest_candidate_2",
        "source": "chest_right_fallback",
        "bbox_ratio": [0.48, 0.12, 0.82, 0.43],
    },
    {
        "id": "chest_candidate_3",
        "source": "chest_center_fallback",
        "bbox_ratio": [0.28, 0.10, 0.72, 0.48],
    },
]


# =========================
# WATERMARK
# =========================

WATERMARK_WEAK_SCORE = 0.82
WATERMARK_STRONG_SCORE = 0.90
WATERMARK_BLOCK_SCORE = 0.96

WATERMARK_STRONG_AREA_RATIO = 0.015
WATERMARK_CENTEREDNESS = 0.35


# =========================
# ARTIFACTS
# =========================

ARTIFACTS_DIR = "artifacts"
ANNOTATED_ARTIFACT_FORMAT = "jpg"