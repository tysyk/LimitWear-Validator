from pathlib import Path
import random
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "brand_classifier"
    / "raw"
)

OUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "logo_presence"
    / "raw"
    / "logo"
)

BRAND_CLASSES = [
    "nike",
    "adidas",
    "jordan",
    "gucci",
    "calvin_klein",
    "puma",
    "supreme",
    "chanel",
    "dior",
    "lv",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MIN_CROP_SIZE = 80


def save_crop(crop, path: Path):
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]

    if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
        return False

    cv2.imwrite(str(path), crop)
    return True


def make_logo_candidate_crops(image):
    h, w = image.shape[:2]

    crop_specs = [
        ("upper_center", 0.20, 0.05, 0.80, 0.45),
        ("left_chest", 0.05, 0.08, 0.50, 0.50),
        ("right_chest", 0.50, 0.08, 0.95, 0.50),
        ("center_chest", 0.20, 0.15, 0.80, 0.65),
        ("neck_area", 0.35, 0.00, 0.65, 0.22),
    ]

    crops = []

    for name, rx1, ry1, rx2, ry2 in crop_specs:
        x1 = int(w * rx1)
        y1 = int(h * ry1)
        x2 = int(w * rx2)
        y2 = int(h * ry2)

        crop = image[y1:y2, x1:x2]
        crops.append((name, crop))

    return crops


def get_images(folder: Path):
    if not folder.exists():
        return []

    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def process_brand(brand: str):
    folder = SOURCE_DIR / brand
    images = get_images(folder)

    if not images:
        print(f"[WARN] No images for {brand}")
        return

    saved = 0

    for i, image_path in enumerate(images):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        crops = make_logo_candidate_crops(image)
        random.shuffle(crops)

        for crop_name, crop in crops:
            out_path = OUT_DIR / f"brandcrop_{brand}_{i:05d}_{crop_name}.jpg"

            if save_crop(crop, out_path):
                saved += 1

    print(f"[DONE] {brand}: saved {saved} logo crops")


def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for brand in BRAND_CLASSES:
        process_brand(brand)

    print(f"\n[DONE] Total logo images now: {len(list(OUT_DIR.glob('*')))}")


if __name__ == "__main__":
    main()