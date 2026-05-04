from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "brand_risk" / "raw"

KNOWN_BRAND_DIR = DATASET_DIR / "known_brand"

MAX_IMAGES = 3000
DATASET_NAME = "axonstan/LogoDet-3K"

def ensure_dirs():
    KNOWN_BRAND_DIR.mkdir(parents=True, exist_ok=True)

def save_image(image, path: Path):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image.save(path, quality=95)

def main():
    ensure_dirs()

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")

    indexes = list(range(len(ds)))
    random.shuffle(indexes)

    saved = 0

    for idx in tqdm(indexes):
        if saved >= MAX_IMAGES:
            break

        item = ds[idx]

        image = item.get("image_path")
        if image is None:
            continue

        out_path = KNOWN_BRAND_DIR / f"known_brand_{saved:05d}.jpg"

        try:
            save_image(image, out_path)
            saved += 1
        except Exception as e:
            print(f"Skip image {idx}: {e}")

    print(f"Done. Saved {saved} images to:")
    print(KNOWN_BRAND_DIR)

if __name__ == "__main__":
    main()