import cv2
import easyocr

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["uk", "en"], gpu=False)
    return _reader

def run_ocr(bgr):
    reader = _get_reader()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = reader.readtext(rgb)

    out = []
    for bbox, text, conf in raw:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        out.append({"text": text, "conf": float(conf), "bbox": [x1, y1, x2, y2]})
    return out
