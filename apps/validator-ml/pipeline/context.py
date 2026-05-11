from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.messages import (
    get_summary_headline,
    get_summary_next_action,
)


ALLOWED_VERDICTS = {"UNKNOWN", "PASS", "WARN", "FAIL", "NEED_REVIEW", "ERROR"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


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
    ml: Dict[str, Any] = field(default_factory=dict)

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
        if step not in self.steps_completed:
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
            "bbox": bbox,
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

    def set_debug_section(self, section: str, payload: Dict[str, Any]) -> None:
        self.debug[section] = payload

    def merge_debug_section(self, section: str, payload: Dict[str, Any]) -> None:
        current = self.debug.get(section)

        if not isinstance(current, dict):
            current = {}

        current.update(payload)
        self.debug[section] = current

    def _finding_sort_key(self, item: Dict[str, Any]) -> tuple[Any, ...]:
        passed = bool(item.get("passed", False))
        severity = str(item.get("severity", "low")).lower()
        penalty = int(item.get("penalty", 0) or 0)
        title = str(item.get("title") or item.get("ruleId") or "")

        return (
            passed,
            SEVERITY_ORDER.get(severity, 3),
            -penalty,
            title,
        )

    def _build_apparel_signal(self) -> Dict[str, Any]:
        scene = self.scene or {}
        apparel_ml = self.ml.get("apparel", {}) if isinstance(self.ml, dict) else {}

        label = apparel_ml.get("label")

        if not label and "is_apparel" in scene:
            label = "apparel" if scene.get("is_apparel") else "non_apparel"

        confidence = apparel_ml.get(
            "confidence",
            scene.get("apparel_confidence"),
        )

        reliable = apparel_ml.get("isReliable")

        return {
            "label": label or "unknown",
            "confidence": confidence,
            "isApparel": scene.get("is_apparel"),
            "source": scene.get("apparel_source", "unknown"),
            "reliable": reliable,
        }

    def _build_apparel_type_signal(self) -> Dict[str, Any]:
        scene = self.scene or {}
        apparel_type_ml = (
            self.ml.get("apparel_type", {})
            if isinstance(self.ml, dict)
            else {}
        )

        return {
            "label": (
                apparel_type_ml.get("label")
                or scene.get("apparel_type")
                or "unknown"
            ),
            "confidence": apparel_type_ml.get(
                "confidence",
                scene.get("apparel_type_confidence"),
            ),
            "source": scene.get("apparel_type_source", "unknown"),
            "reliable": apparel_type_ml.get("isReliable"),
        }

    def _build_logo_signal(self) -> Dict[str, Any]:
        ml = self.ml.get("logo_presence", {}) if isinstance(self.ml, dict) else {}

        return {
            "label": ml.get("label"),
            "confidence": ml.get("confidence"),
            "hasLogo": ml.get("isLogo"),
            "reliable": ml.get("isReliable"),
        }

    def _build_primary_findings(
        self,
        ordered_violations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        top_findings = []

        for finding in ordered_violations[:3]:
            top_findings.append(
                {
                    "ruleId": finding.get("ruleId"),
                    "title": finding.get("title"),
                    "severity": finding.get("severity"),
                    "message": finding.get("message"),
                }
            )

        return top_findings

    def _build_summary(
        self,
        ordered_violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "headline": get_summary_headline(self.verdict),
            "decision": self.verdict,
            "score": self.score,
            "sceneType": (self.scene or {}).get("type"),
            "reviewReason": (self.debug or {}).get("need_review_reason"),
            "failReason": (self.debug or {}).get("fail_reason"),
            "apparelSignal": self._build_apparel_signal(),
            "apparelTypeSignal": self._build_apparel_type_signal(),
            "logoSignal": self._build_logo_signal(),
            "primaryFindings": self._build_primary_findings(ordered_violations),
            "nextAction": get_summary_next_action(self.verdict),
        }

    def to_response(self) -> Dict[str, Any]:
        ordered_rule_results = sorted(
            self.rule_results,
            key=self._finding_sort_key,
        )

        ordered_violations = sorted(
            self.violations,
            key=self._finding_sort_key,
        )

        return {
            "analysisId": self.image_id,
            "profileId": self.profile_id,
            "summary": self._build_summary(ordered_violations),
            "input": {
                "width": self.width,
                "height": self.height,
            },
            "quality": self.quality,
            "scene": self.scene,
            "roi": self.roi,
            "moderation": self.moderation,
            "detections": self.detections,
            "ml": self.ml,
            "ruleResults": ordered_rule_results,
            "score": self.score,
            "verdict": self.verdict,
            "violations": ordered_violations,
            "explain": self.explain,
            "artifacts": self.artifacts,
            "debug": self.debug,
            "errors": self.errors,
            "warnings": self.warnings,
            "stepsCompleted": self.steps_completed,
            "timings": self.timings,
        }