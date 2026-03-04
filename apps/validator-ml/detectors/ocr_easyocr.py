import cv2
import easyocr

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en", "uk"], gpu=False)
    return _reader

def run_ocr(bgr):
    reader = get_reader()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ocr_raw = reader.readtext(rgb)

    ocr = []
    for bbox, text, conf in ocr_raw:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        ocr.append({"text": text, "conf": float(conf), "bbox": [x1, y1, x2, y2]})
    return ocr
