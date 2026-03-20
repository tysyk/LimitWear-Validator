from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline


class ValidateRequest(BaseModel):
    imageBase64: str
    profileId: str = "auto"
    includeDebug: bool = False
    includeArtifacts: bool = False


app = FastAPI(title="Limitwear Validator API")


@app.get("/health")
def health():
    return {"status": "ok"}


def decode_base64_image(data: str):
    try:
        image_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid imageBase64: {e}")


@app.post("/validate")
def validate(req: ValidateRequest):
    bgr = decode_base64_image(req.imageBase64)
    h, w = bgr.shape[:2]

    ctx = PipelineContext(
        image_id=str(uuid.uuid4()),
        profile_id=req.profileId or "auto",
        bgr=bgr,
        width=w,
        height=h,
    )

    ctx = run_pipeline(ctx)
    response = ctx.to_response()

    if not req.includeDebug:
        response.pop("debug", None)

    if not req.includeArtifacts:
        response.pop("artifacts", None)

    return response