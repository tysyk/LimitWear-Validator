from pathlib import Path
import random
import shutil

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "datasets" / "brand_logo_sources"
OUT_DIR = PROJECT_ROOT / "datasets" / "brand_crop_classifier"

IMG_SIZE = 224

AUG_TRAIN = 25
AUG_VAL = 5
AUG_TEST = 5

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def reset_output():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def fit_on_canvas(src_path: Path):
    image = Image.open(src_path).convert("RGBA")
    max_size = random.randint(95, 210)
    image.thumbnail((max_size, max_size))

    bg = random.choice([
        (255, 255, 255, 255),
        (245, 245, 245, 255),
        (230, 230, 230, 255),
        (35, 35, 35, 255),
        (0, 0, 0, 255),
    ])

    canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), bg)

    x = (IMG_SIZE - image.width) // 2 + random.randint(-24, 24)
    y = (IMG_SIZE - image.height) // 2 + random.randint(-24, 24)

    canvas.alpha_composite(image, (x, y))
    return canvas.convert("RGB")


def augment(image: Image.Image):
    if random.random() < 0.45:
        image = ImageOps.grayscale(image).convert("RGB")

    if random.random() < 0.25:
        image = ImageOps.invert(image)

    if random.random() < 0.75:
        image = image.rotate(
            random.randint(-18, 18),
            fillcolor=random.choice(["white", "black", "gray"]),
        )

    if random.random() < 0.65:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.6))

    if random.random() < 0.55:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.35))

    if random.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    return image


def save_augmented(src_path: Path, dst_path: Path):
    image = fit_on_canvas(src_path)
    image = augment(image)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst_path, quality=95)


def split_files(files):
    random.shuffle(files)

    if len(files) < 3:
        return {
            "train": files,
            "val": files,
            "test": files,
        }

    test_count = max(1, int(len(files) * 0.15))
    val_count = max(1, int(len(files) * 0.15))

    return {
        "test": files[:test_count],
        "val": files[test_count:test_count + val_count],
        "train": files[test_count + val_count:],
    }


def main():
    reset_output()

    classes = sorted([
        path.name
        for path in SRC_DIR.iterdir()
        if path.is_dir() and not path.name.endswith("_raw")
    ])

    for class_name in classes:
        source_files = [
            p for p in (SRC_DIR / class_name).rglob("*")
            if p.suffix.lower() in EXTS
        ]

        files = [p for p in source_files if is_valid_image(p)]

        print(f"\n{class_name}: source={len(source_files)}, valid={len(files)}")

        splits = split_files(files)

        for split, split_files_list in splits.items():
            aug_count = AUG_TRAIN if split == "train" else AUG_VAL if split == "val" else AUG_TEST
            saved = 0

            for src_index, file in enumerate(split_files_list):
                for aug_index in range(aug_count):
                    target = (
                        OUT_DIR
                        / split
                        / class_name
                        / f"{class_name}_{src_index + 1:04d}_{aug_index + 1:03d}.jpg"
                    )

                    try:
                        save_augmented(file, target)
                        saved += 1
                    except Exception as error:
                        print("Skipped:", file, error)

            print(f"{class_name} {split}: files={len(split_files_list)}, saved={saved}")

    print("\nDone:", OUT_DIR)


if __name__ == "__main__":
    main()