from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "brand_risk" / "raw" / "no_brand"

MAX_IMAGES = 800

def ensure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(img, path):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img.save(path, quality=95)

def main():
    ensure()

    ds = load_dataset("fashion_mnist", split="train")  # маленький, але як старт норм

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    saved = 0

    for i in tqdm(idxs):
        if saved >= MAX_IMAGES:
            break

        item = ds[i]
        image = item["image"]

        path = OUT_DIR / f"no_brand_{saved:05d}.jpg"

        try:
            save(image, path)
            saved += 1
        except:
            continue

    print(f"Saved {saved} images to {OUT_DIR}")

if __name__ == "__main__":
    main()