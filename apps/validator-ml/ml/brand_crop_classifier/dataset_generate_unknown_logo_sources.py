from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "datasets" / "brand_logo_sources" / "unknown_logo_raw"
OUT_DIR = PROJECT_ROOT / "datasets" / "brand_logo_sources" / "unknown_logo"

IMG_SIZE = 224
AUG_PER_IMAGE = 8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_source_images():
    return [
        path for path in SRC_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def fit_on_canvas(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")

    max_size = random.randint(120, 205)
    image.thumbnail((max_size, max_size))

    bg_color = random.choice([
        (255, 255, 255, 255),
        (245, 245, 245, 255),
        (20, 20, 20, 255),
        (35, 35, 35, 255),
    ])

    canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), bg_color)

    x = (IMG_SIZE - image.width) // 2 + random.randint(-18, 18)
    y = (IMG_SIZE - image.height) // 2 + random.randint(-18, 18)

    canvas.alpha_composite(image, (x, y))

    return canvas.convert("RGB")


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.grayscale(image).convert("RGB")

    if random.random() < 0.4:
        image = ImageOps.invert(image)

    if random.random() < 0.7:
        image = image.rotate(
            random.randint(-18, 18),
            fillcolor=random.choice(["white", "black", "gray"]),
        )

    if random.random() < 0.6:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.75, 1.45)
        )

    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.75, 1.35)
        )

    if random.random() < 0.35:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 1.0)
            )
        )

    return image


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_images = load_source_images()

    if not source_images:
        print("No source images found:", SRC_DIR)
        return

    saved = 0

    for source_path in source_images:
        try:
            source = Image.open(source_path)
        except Exception as error:
            print("Skipped:", source_path, error)
            continue

        for _ in range(AUG_PER_IMAGE):
            image = fit_on_canvas(source.copy())
            image = augment(image)

            out_path = OUT_DIR / f"unknown_logo_{saved + 1:04d}.jpg"
            image.save(out_path, quality=95)
            saved += 1

    print("Source images:", len(source_images))
    print("Generated:", saved)
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()