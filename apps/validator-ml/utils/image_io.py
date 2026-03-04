import os
import uuid
import numpy as np
import cv2

def save_upload(content: bytes, upload_dir: str, filename: str | None):
    ext = os.path.splitext(filename or "")[1].lower() or ".png"
    img_id = str(uuid.uuid4())
    path = os.path.join(upload_dir, f"{img_id}{ext}")
    with open(path, "wb") as f:
        f.write(content)
    return img_id, path

def decode_image(content: bytes):
    arr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr
