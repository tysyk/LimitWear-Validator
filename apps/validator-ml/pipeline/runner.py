from __future__ import annotations

import time

from pipeline.steps.quality_gate import run as quality_gate
from pipeline.steps.roi_extract import run as roi_extract
from pipeline.steps.detectors import run as detectors
from pipeline.steps.scene_type import run as scene_type
from pipeline.steps.moderation import run as moderation
from pipeline.steps.rules import run as rules
from pipeline.steps.aggregate import run as aggregate
from pipeline.steps.explain import run as explain


def _run_step(ctx, step_name: str, step_fn) -> None:
    if ctx.stop_pipeline:
        return

    started = time.perf_counter()
    try:
        step_fn(ctx)
    except Exception as e:
        ctx.add_error(step_name, str(e), critical=True)
        ctx.set_verdict("ERROR")
    finally:
        elapsed = time.perf_counter() - started
        ctx.set_timing(step_name, elapsed)


def run_pipeline(ctx):
    _run_step(ctx, "quality_gate", quality_gate)
    _run_step(ctx, "roi_extract", roi_extract)
    _run_step(ctx, "detectors", detectors)
    _run_step(ctx, "scene_type", scene_type)
    _run_step(ctx, "moderation", moderation)
    _run_step(ctx, "rules", rules)
    _run_step(ctx, "aggregate", aggregate)
    _run_step(ctx, "explain", explain)
    return ctx