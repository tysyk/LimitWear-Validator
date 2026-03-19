from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uuid

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline

app = FastAPI(title="Limitwear Validator ML", version="2.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), profileId: str = "auto"):
    if not image.content_type or not image.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file must be an image"}
        )

    content = await image.read()
    arr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file"}
        )

    h, w = bgr.shape[:2]
    image_id = str(uuid.uuid4())

    ctx = PipelineContext(
        image_id=image_id,
        profile_id=profileId,
        bgr=bgr,
        width=w,
        height=h
    )

    ctx = run_pipeline(ctx)
    return ctx.to_response()