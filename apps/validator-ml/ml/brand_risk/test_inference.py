from pathlib import Path
import sys
import cv2

from ml.brand_risk.inference import predict_brand_risk


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m ml.brand_risk.test_inference path/to/image.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        print(f"Could not read image: {image_path}")
        return

    result = predict_brand_risk(bgr)

    print("Brand risk result:")
    print(result)


if __name__ == "__main__":
    main()