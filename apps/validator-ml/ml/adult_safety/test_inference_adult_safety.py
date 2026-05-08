from pathlib import Path
import sys
import cv2

from ml.adult_safety.inference_adult_safety import (
    predict_adult_safety,
)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m ml.adult_safety.test_inference path/to/image.jpg")
        return

    image_path = Path(sys.argv[1])
    bgr = cv2.imread(str(image_path))

    if bgr is None:
        print(f"Could not read image: {image_path}")
        return

    result = predict_adult_safety(bgr)
    print(result)


if __name__ == "__main__":
    main()