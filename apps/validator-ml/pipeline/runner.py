from __future__ import annotations

import time

from pipeline.steps.quality_gate import run as quality_gate
from pipeline.steps.scene_type import run as scene_type
from pipeline.steps.roi_extract import run as roi_extract
from pipeline.steps.moderation import run as moderation
from pipeline.steps.detectors import run as detectors
from pipeline.steps.rules import run as rules
from pipeline.steps.aggregate import run as aggregate
from pipeline.steps.explain import run as explain


PIPELINE_STEPS = [
    ("quality_gate", quality_gate),
    ("scene_type", scene_type),
    ("roi_extract", roi_extract),
    ("moderation", moderation),
    ("detectors", detectors),
    ("rules", rules),
    ("aggregate", aggregate),
    ("explain", explain),
]


def run_pipeline(ctx):
    for step_name, step_fn in PIPELINE_STEPS:
        if ctx.stop_pipeline and step_name not in {"aggregate", "explain"}:
            break

        started = time.perf_counter()
        try:
            step_fn(ctx)
        except Exception as exc:
            ctx.fail(step_name, f"{type(exc).__name__}: {exc}", verdict="ERROR")
        finally:
            ctx.set_timing(step_name, time.perf_counter() - started)
            ctx.mark_step_done(step_name)

    if ctx.verdict == "UNKNOWN":
        ctx.set_verdict("ERROR")
        ctx.add_error("runner", "Pipeline finished without final verdict", critical=False)

    return ctx