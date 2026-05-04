from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "brand_risk" / "raw" / "other_logo"

MAX_IMAGES = 800
OFFSET = 5000  # щоб не брати ті самі картинки що в known_brand

def ensure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(img, path):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img.save(path, quality=95)

def main():
    ensure()

    ds = load_dataset("axonstan/LogoDet-3K", split="train")

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    saved = 0

    for i in tqdm(idxs[OFFSET:]):
        if saved >= MAX_IMAGES:
            break

        item = ds[i]
        image = item["image_path"]

        path = OUT_DIR / f"other_logo_{saved:05d}.jpg"

        try:
            save(image, path)
            saved += 1
        except:
            continue

    print(f"Saved {saved} images to {OUT_DIR}")

if __name__ == "__main__":
    main()