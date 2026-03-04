from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PipelineContext:
    image_id: str
    profile_id: str
    bgr: Any
    width: int
    height: int

    quality: Dict[str, Any] = field(default_factory=dict)
    scene: Dict[str, Any] = field(default_factory=dict)
    roi: Dict[str, Any] = field(default_factory=dict)
    moderation: Dict[str, Any] = field(default_factory=dict)
    detections: Dict[str, Any] = field(default_factory=dict)

    violations: List[Dict[str, Any]] = field(default_factory=list)
    score: int = 100
    verdict: str = "UNKNOWN"
    artifacts: Dict[str, Any] = field(default_factory=dict)

    debug: Dict[str, Any] = field(default_factory=dict)
    bgr_used: Any = None
