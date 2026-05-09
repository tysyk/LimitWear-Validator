from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = PROJECT_ROOT / "datasets" / "brand_logo_sources" / "no_brand"

IMG_SIZE = 224
COUNT = 500


def random_color(min_v=20, max_v=235):
    value = random.randint(min_v, max_v)
    return (value, value, value)


def make_background():
    base = random.randint(170, 245)
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (base, base, base))
    draw = ImageDraw.Draw(image)

    mode = random.choice(["fabric", "noise", "gradient", "plain"])

    if mode == "fabric":
        for _ in range(random.randint(80, 180)):
            x1 = random.randint(0, IMG_SIZE)
            y1 = random.randint(0, IMG_SIZE)
            x2 = x1 + random.randint(-35, 35)
            y2 = y1 + random.randint(-35, 35)
            shade = random.randint(130, 245)

            draw.line(
                (x1, y1, x2, y2),
                fill=(shade, shade, shade),
                width=random.randint(1, 2),
            )

    elif mode == "noise":
        for _ in range(random.randint(400, 900)):
            x = random.randint(0, IMG_SIZE - 1)
            y = random.randint(0, IMG_SIZE - 1)
            shade = random.randint(100, 255)
            draw.point((x, y), fill=(shade, shade, shade))

    elif mode == "gradient":
        for y in range(IMG_SIZE):
            shade = base + int((y / IMG_SIZE) * random.randint(-35, 35))
            shade = max(0, min(255, shade))
            draw.line((0, y, IMG_SIZE, y), fill=(shade, shade, shade))

    return image


def add_non_logo_graphics(image):
    draw = ImageDraw.Draw(image)

    graphic_type = random.choice([
        "none",
        "simple_lines",
        "corner_shape",
        "abstract_blob",
        "geometric",
        "scratches",
    ])

    color = random_color(20, 120)

    if graphic_type == "none":
        return image

    if graphic_type == "simple_lines":
        for _ in range(random.randint(1, 5)):
            draw.line(
                (
                    random.randint(0, IMG_SIZE),
                    random.randint(0, IMG_SIZE),
                    random.randint(0, IMG_SIZE),
                    random.randint(0, IMG_SIZE),
                ),
                fill=color,
                width=random.randint(2, 8),
            )

    elif graphic_type == "corner_shape":
        x = random.choice([0, IMG_SIZE - random.randint(25, 80)])
        y = random.choice([0, IMG_SIZE - random.randint(25, 80)])
        w = random.randint(20, 70)
        h = random.randint(20, 70)

        draw.rectangle(
            (x, y, min(IMG_SIZE, x + w), min(IMG_SIZE, y + h)),
            fill=color,
        )

    elif graphic_type == "abstract_blob":
        points = [
            (
                random.randint(20, IMG_SIZE - 20),
                random.randint(20, IMG_SIZE - 20),
            )
            for _ in range(random.randint(5, 10))
        ]
        draw.polygon(points, fill=color)

    elif graphic_type == "geometric":
        for _ in range(random.randint(1, 4)):
            x1 = random.randint(0, IMG_SIZE - 40)
            y1 = random.randint(0, IMG_SIZE - 40)
            x2 = x1 + random.randint(20, 90)
            y2 = y1 + random.randint(20, 90)

            if random.random() < 0.5:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=random.randint(3, 8))
            else:
                draw.ellipse((x1, y1, x2, y2), outline=color, width=random.randint(3, 8))

    elif graphic_type == "scratches":
        for _ in range(random.randint(8, 25)):
            x = random.randint(0, IMG_SIZE)
            y = random.randint(0, IMG_SIZE)
            draw.line(
                (
                    x,
                    y,
                    x + random.randint(-40, 40),
                    y + random.randint(-40, 40),
                ),
                fill=color,
                width=1,
            )

    return image


def augment(image):
    if random.random() < 0.4:
        image = image.rotate(
            random.randint(-20, 20),
            fillcolor=(255, 255, 255),
        )

    if random.random() < 0.5:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 1.2),
            )
        )

    return image


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(COUNT):
        image = make_background()
        image = add_non_logo_graphics(image)
        image = augment(image)

        out_path = OUT_DIR / f"no_brand_{i + 1:04d}.jpg"
        image.save(out_path, quality=95)

    print("Generated:", COUNT)
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()