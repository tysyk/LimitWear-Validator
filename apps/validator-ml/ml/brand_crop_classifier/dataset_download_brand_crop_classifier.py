from pathlib import Path
from io import BytesIO
import random
import time

import requests
from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "datasets" / "brand_logo_raw"
IMG_SIZE = 224
IMAGES_PER_BRAND = 80
SYNTHETIC_COUNT = 300

BRANDS = {
    "nike": ["Nike logo", "Nike swoosh"],
    "adidas": ["Adidas logo", "Adidas trefoil", "Adidas three stripes"],
    "puma": ["Puma logo"],
    "gucci": ["Gucci logo", "Gucci GG logo"],
    "chanel": ["Chanel logo"],
    "supreme": ["Supreme logo"],
}

HEADERS = {
    "User-Agent": "LimitWearValidatorDatasetBuilder/1.0"
}


def ensure_dirs():
    for class_name in list(BRANDS.keys()) + ["unknown_logo", "no_brand"]:
        (RAW_DIR / class_name).mkdir(parents=True, exist_ok=True)


def search_commons(query, limit=30):
    url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": IMG_SIZE,
        "format": "json",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        print("Search failed:", query, error)
        return []

    pages = data.get("query", {}).get("pages", {})
    results = []

    for page in pages.values():
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue

        info = imageinfo[0]
        image_url = info.get("thumburl") or info.get("url")

        if image_url:
            results.append(image_url)

    return results


def save_image_from_url(url, save_path):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGBA")

        canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
        image.thumbnail((IMG_SIZE - 20, IMG_SIZE - 20))

        x = (IMG_SIZE - image.width) // 2
        y = (IMG_SIZE - image.height) // 2

        canvas.paste(image, (x, y), image)

        canvas.convert("RGB").save(save_path, quality=95)
        return True

    except Exception as error:
        print("Save failed:", url, error)
        return False


def download_brand(class_name, queries):
    target_dir = RAW_DIR / class_name
    saved = 0
    seen = set()

    print(f"\n=== {class_name} ===")

    for query in queries:
        urls = search_commons(query, limit=50)

        for url in urls:
            if saved >= IMAGES_PER_BRAND:
                break

            if url in seen:
                continue

            seen.add(url)

            save_path = target_dir / f"{class_name}_{saved + 1:04d}.jpg"

            if save_image_from_url(url, save_path):
                saved += 1
                print(f"Saved {class_name}: {saved}/{IMAGES_PER_BRAND}")

            time.sleep(0.2)

    print(f"Finished {class_name}: {saved}")


def make_no_brand_image(save_path):
    bg = random.randint(180, 245)
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (bg, bg, bg))
    draw = ImageDraw.Draw(image)

    for _ in range(random.randint(15, 60)):
        x1 = random.randint(0, IMG_SIZE)
        y1 = random.randint(0, IMG_SIZE)
        x2 = random.randint(0, IMG_SIZE)
        y2 = random.randint(0, IMG_SIZE)
        shade = random.randint(120, 240)

        draw.line(
            (x1, y1, x2, y2),
            fill=(shade, shade, shade),
            width=random.randint(1, 3),
        )

    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))
    image.save(save_path, quality=95)


def make_unknown_logo_image(save_path):
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(image)

    color = random.choice(["black", "darkblue", "darkred", "gray"])

    cx = IMG_SIZE // 2 + random.randint(-20, 20)
    cy = IMG_SIZE // 2 + random.randint(-20, 20)
    size = random.randint(45, 90)

    shape_type = random.choice(["circle", "star", "lines", "polygon", "box"])

    if shape_type == "circle":
        draw.ellipse(
            (cx - size, cy - size, cx + size, cy + size),
            outline=color,
            width=random.randint(5, 12),
        )

    elif shape_type == "box":
        draw.rectangle(
            (cx - size, cy - size, cx + size, cy + size),
            outline=color,
            width=random.randint(5, 12),
        )

    elif shape_type == "lines":
        for _ in range(random.randint(4, 9)):
            draw.line(
                (
                    cx + random.randint(-size, size),
                    cy + random.randint(-size, size),
                    cx + random.randint(-size, size),
                    cy + random.randint(-size, size),
                ),
                fill=color,
                width=random.randint(4, 10),
            )

    else:
        points = [
            (
                cx + random.randint(-size, size),
                cy + random.randint(-size, size),
            )
            for _ in range(random.randint(4, 8))
        ]
        draw.polygon(points, outline=color)

    angle = random.randint(-20, 20)
    image = image.rotate(angle, fillcolor="white")
    image.save(save_path, quality=95)


def generate_synthetic_classes():
    print("\n=== no_brand synthetic ===")
    no_brand_dir = RAW_DIR / "no_brand"
    for i in range(SYNTHETIC_COUNT):
        make_no_brand_image(no_brand_dir / f"no_brand_{i + 1:04d}.jpg")

    print("\n=== unknown_logo synthetic ===")
    unknown_dir = RAW_DIR / "unknown_logo"
    for i in range(SYNTHETIC_COUNT):
        make_unknown_logo_image(unknown_dir / f"unknown_logo_{i + 1:04d}.jpg")


def main():
    ensure_dirs()

    for class_name, queries in BRANDS.items():
        download_brand(class_name, queries)

    generate_synthetic_classes()

    print("\nDone.")
    print("Raw dataset:", RAW_DIR)


if __name__ == "__main__":
    main()