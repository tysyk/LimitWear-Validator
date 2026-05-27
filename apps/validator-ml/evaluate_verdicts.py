import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline


DATASET_DIR = Path("data/evaluation")

FOLDERS = {
    "pass": "PASS",
    "need_review": "NEED_REVIEW",
    "fail": "FAIL",
}

LABELS = ["PASS", "NEED_REVIEW", "FAIL"]


def load_bgr(path: Path):
    content = path.read_bytes()
    arr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")

    return bgr


def normalize_verdict(verdict):
    text = str(verdict)

    # Якщо verdict є Enum типу Verdict.PASS
    if "." in text:
        text = text.split(".")[-1]

    return text.upper()


def main():
    y_true = []
    y_pred = []
    times_by_class = defaultdict(list)

    for folder_name, expected in FOLDERS.items():
        folder = DATASET_DIR / folder_name

        image_paths = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

        print(f"\n=== {expected}: {len(image_paths)} images ===")

        for image_path in image_paths:
            try:
                bgr = load_bgr(image_path)

                ctx = PipelineContext(
                    image_id=image_path.stem,
                    profile_id="evaluation",
                    bgr=bgr,
                    width=bgr.shape[1],
                    height=bgr.shape[0],
                )

                start = time.perf_counter()
                result_ctx = run_pipeline(ctx)
                elapsed = time.perf_counter() - start

                predicted = normalize_verdict(result_ctx.verdict)

                y_true.append(expected)
                y_pred.append(predicted)
                times_by_class[expected].append(elapsed)

                correct = "OK" if expected == predicted else "ERROR"
                print(
                    f"{correct} | {image_path.name} | "
                    f"expected={expected} | predicted={predicted} | time={elapsed:.2f}s"
                )

            except Exception as e:
                print(f"ERROR | {image_path.name} | {e}")

    print("\n=== OVERALL ACCURACY ===")
    print(f"{accuracy_score(y_true, y_pred):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print("\n=== CONFUSION MATRIX ===")
    print("labels:", LABELS)
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    print("\n=== TIME BY CLASS ===")
    for label in LABELS:
        values = times_by_class[label]
        if not values:
            continue

        print(
            f"{label}: count={len(values)}, "
            f"avg={sum(values) / len(values):.2f}s, "
            f"min={min(values):.2f}s, "
            f"max={max(values):.2f}s"
        )

    all_times = [t for values in times_by_class.values() for t in values]

    if all_times:
        print(
            f"ALL: count={len(all_times)}, "
            f"avg={sum(all_times) / len(all_times):.2f}s, "
            f"min={min(all_times):.2f}s, "
            f"max={max(all_times):.2f}s"
        )


if __name__ == "__main__":
    main()