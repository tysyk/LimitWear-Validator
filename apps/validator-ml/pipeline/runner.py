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
    for step_name, step_func in PIPELINE_STEPS:
        if ctx.stop_pipeline:
            break

        started_at = time.perf_counter()

        try:
            step_func(ctx)
            ctx.mark_step_done(step_name)
        except Exception as exc:
            ctx.add_error(step=step_name, message=str(exc), critical=True)
            ctx.verdict = "ERROR"
        finally:
            elapsed = time.perf_counter() - started_at
            ctx.set_timing(step_name, elapsed)

    return ctx