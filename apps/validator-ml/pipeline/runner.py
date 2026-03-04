from pipeline.steps.quality_gate import run as quality_gate
from pipeline.steps.scene_type import run as scene_type
from pipeline.steps.roi_extract import run as roi_extract
from pipeline.steps.moderation import run as moderation
from pipeline.steps.detectors import run as detectors
from pipeline.steps.rules import run as rules
from pipeline.steps.aggregate import run as aggregate
from pipeline.steps.explain import run as explain

def run_pipeline(ctx):
    quality_gate(ctx)
    scene_type(ctx)
    roi_extract(ctx)
    moderation(ctx)
    detectors(ctx)
    rules(ctx)
    aggregate(ctx)
    explain(ctx)
    return ctx
