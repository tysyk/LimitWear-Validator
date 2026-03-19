from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PipelineContext:
    image_id: str
    profile_id: str
    bgr: Any
    width: int
    height: int

    # step outputs
    quality: Dict[str, Any] = field(default_factory=dict)
    scene: Dict[str, Any] = field(default_factory=dict)
    roi: Dict[str, Any] = field(default_factory=dict)
    moderation: Dict[str, Any] = field(default_factory=dict)
    detections: Dict[str, Any] = field(default_factory=dict)

    # rules / final result
    rule_results: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    score: int = 100
    verdict: str = "UNKNOWN"
    explain: List[str] = field(default_factory=list)

    # extra data
    artifacts: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    bgr_used: Any = None

    # runtime control
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    stop_pipeline: bool = False

    def add_error(self, step: str, message: str, critical: bool = False) -> None:
        self.errors.append({
            "step": step,
            "message": message,
            "critical": critical,
        })
        if critical:
            self.stop_pipeline = True

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def mark_step_done(self, step: str) -> None:
        self.steps_completed.append(step)

    def set_timing(self, step: str, seconds: float) -> None:
        self.timings[step] = round(seconds, 4)

    def fail(self, step: str, message: str, verdict: str = "REJECTED") -> None:
        self.add_error(step=step, message=message, critical=True)
        self.verdict = verdict

    def add_violation(
        self,
        code: str,
        message: str,
        severity: str = "medium",
        points: int = 0,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self.violations.append({
            "code": code,
            "message": message,
            "severity": severity,
            "points": points,
            "meta": meta or {},
        })

    def add_explain(self, message: str) -> None:
        self.explain.append(message)

    def to_response(self) -> Dict[str, Any]:
        return {
            "analysisId": self.image_id,
            "profileId": self.profile_id,
            "input": {
                "width": self.width,
                "height": self.height,
            },
            "quality": self.quality,
            "scene": self.scene,
            "roi": self.roi,
            "moderation": self.moderation,
            "detections": self.detections,
            "score": self.score,
            "verdict": self.verdict,
            "violations": self.violations,
            "explain": self.explain,
            "artifacts": self.artifacts,
            "debug": self.debug,
            "errors": self.errors,
            "warnings": self.warnings,
            "stepsCompleted": self.steps_completed,
            "timings": self.timings,
        }