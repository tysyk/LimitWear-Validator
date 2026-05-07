from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    UNKNOWN = "UNKNOWN"
    PASS = "PASS"
    WARN = "WARN"
    NEED_REVIEW = "NEED_REVIEW"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCK = "block"
    SUSPICIOUS_LOGO = "suspicious_logo"


class SceneType(str, Enum):
    APPAREL = "apparel"
    NON_APPAREL = "non_apparel"
    POSTER_LIKE = "poster_like"
    SKETCH_SCAN = "sketch_scan"
    DIGITAL_OR_MOCKUP = "digital_or_mockup"
    UNKNOWN = "unknown"


class ApparelType(str, Enum):
    TSHIRT = "tshirt"
    HOODIE = "hoodie"
    JACKET = "jacket"
    PANTS = "pants"
    SHOES = "shoes"
    CAP = "cap"
    BAG = "bag"
    OTHER_APPAREL = "other_apparel"
    UNKNOWN = "unknown"