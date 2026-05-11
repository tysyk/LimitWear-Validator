import sys
from pathlib import Path

import cv2

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.brand_crop_classifier.inference_brand_crop_classifier import (
    predict_single_crop,
    predict_brand_crop_classifier,
)


def test_full_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Не вдалося прочитати файл: {image_path}")
        return

    print("\n=== TEST 1: FULL IMAGE AS CROP ===")
    result = predict_single_crop(image)
    print(result)


def test_full_image_as_logo_candidate(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Не вдалося прочитати файл: {image_path}")
        return

    height, width = image.shape[:2]

    logo_candidates = [
        {
            "id": "full_image_candidate",
            "bbox": [0, 0, width, height],
            "original_bbox": [0, 0, width, height],
            "source": "manual_full_image",
            "emblem_score": 1.0,
            "crop": image,
        }
    ]

    print("\n=== TEST 2: FULL IMAGE THROUGH BRAND CLASSIFIER ===")
    result = predict_brand_crop_classifier(logo_candidates)
    print(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Приклад запуску:")
        print("python test_brand_model.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    test_full_image(image_path)
    test_full_image_as_logo_candidate(image_path)