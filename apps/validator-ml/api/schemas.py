from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class AnalyzeResponse(BaseModel):
    analysisId: str
    profileId: str
    summary: Dict[str, Any]
    input: Dict[str, Any]
    quality: Dict[str, Any]
    scene: Dict[str, Any]
    roi: Dict[str, Any]
    moderation: Dict[str, Any]
    detections: Dict[str, Any]
    ml: Dict[str, Any]
    ruleResults: List[Dict[str, Any]]
    score: int
    verdict: str
    violations: List[Dict[str, Any]]
    explain: List[str]
    artifacts: Dict[str, Any]
    debug: Dict[str, Any]
    errors: List[Dict[str, Any]]
    warnings: List[str]
    stepsCompleted: List[str]
    timings: Dict[str, float]
