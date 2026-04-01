from __future__ import annotations

import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline

app = FastAPI(title="LimitWear Validator")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    profile_id: str = Form("default"),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    arr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    h, w = bgr.shape[:2]

    ctx = PipelineContext(
        image_id=str(uuid.uuid4()),
        profile_id=profile_id,
        bgr=bgr,
        width=w,
        height=h,
    )

    run_pipeline(ctx)
    return ctx.to_response()