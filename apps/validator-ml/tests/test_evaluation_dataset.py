from pathlib import Path

import cv2

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline


BASE_DIR = Path("data/evaluation")

EXPECTED = {
    "pass": "PASS",
    "need_review": "NEED_REVIEW",
    "fail": "FAIL",
}


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image


def test_evaluation_dataset():
    total = 0
    correct = 0

    for folder_name, expected_verdict in EXPECTED.items():
        folder = BASE_DIR / folder_name

        images = list(folder.glob("*"))

        print(f"\n--- {folder_name.upper()} ---")

        for image_path in images:
            image = load_image(image_path)

            height, width = image.shape[:2]

            ctx = PipelineContext(
                image_id=image_path.stem,
                profile_id="evaluation",
                bgr=image,
                width=width,
                height=height,
            )

            result = run_pipeline(ctx)

            actual = result.verdict

            is_correct = actual == expected_verdict

            if is_correct:
                correct += 1

            total += 1

            status = "OK" if is_correct else "FAIL"

            print(
                f"[{status}] "
                f"{image_path.name} | "
                f"expected={expected_verdict} "
                f"actual={actual} "
                f"score={result.score}"
            ) 

            # print("  violations:", result.violations)
            # print("  rules:", result.rule_results)
            # print("  quality:", result.quality)
            # print("  scene:", result.scene)
            # print("  ml:", result.ml)
            # print("  detections:", result.detections)
            # print("  explain:", result.explain)

    accuracy = (correct / total) * 100 if total else 0

    print("\n====================")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("====================")

    assert total > 0