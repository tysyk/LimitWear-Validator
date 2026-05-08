import sys
from pathlib import Path

import cv2

from ml.brand_classifier.inference_brand_classifier import (
    predict_brand_classifier,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ml/brand_classifier/test_inference.py path/to/image.jpg")
        return

    image_path = Path(sys.argv[1])

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image:", image_path)
        return

    result = predict_brand_classifier(image)

    print(result)


if __name__ == "__main__":
    main()