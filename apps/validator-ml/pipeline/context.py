from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ALLOWED_VERDICTS = {"UNKNOWN", "PASS", "WARN", "FAIL", "NEED_REVIEW", "ERROR"}


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

    rule_results: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    score: int = 100
    verdict: str = "UNKNOWN"
    explain: List[str] = field(default_factory=list)

    artifacts: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    bgr_used: Any = None

    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    stop_pipeline: bool = False

    def add_error(self, step: str, message: str, critical: bool = False) -> None:
        self.errors.append(
            {
                "step": step,
                "message": message,
                "critical": critical,
            }
        )
        if critical:
            self.stop_pipeline = True

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def mark_step_done(self, step: str) -> None:
        self.steps_completed.append(step)

    def set_timing(self, step: str, seconds: float) -> None:
        self.timings[step] = round(float(seconds), 4)

    def set_verdict(self, verdict: str) -> None:
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Unsupported verdict: {verdict}")
        self.verdict = verdict

    def fail(self, step: str, message: str, verdict: str = "ERROR") -> None:
        self.add_error(step=step, message=message, critical=True)
        self.set_verdict(verdict)

    def add_rule_result(
        self,
        rule_id: str,
        passed: bool,
        severity: str = "low",
        penalty: int = 0,
        title: str = "",
        message: str = "",
        bbox: Optional[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_severity = severity.lower()

        rule = {
            "ruleId": rule_id,
            "passed": passed,
            "severity": normalized_severity,
            "penalty": max(0, int(penalty)),
            "title": title or rule_id,
            "message": message,
            "bbox": bbox,  # expected format: [x1, y1, x2, y2]
            "meta": meta or {},
        }
        self.rule_results.append(rule)

        if not passed:
            self.violations.append(
                {
                    "ruleId": rule_id,
                    "title": rule["title"],
                    "severity": normalized_severity,
                    "message": message,
                    "bbox": bbox,
                    "penalty": rule["penalty"],
                    "meta": rule["meta"],
                }
            )

    def add_explain(self, message: str) -> None:
        if message:
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
            "ruleResults": self.rule_results,
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