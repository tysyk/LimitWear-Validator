from pathlib import Path
import random
import shutil


BASE = Path(__file__).resolve().parents[1] / "datasets" / "apparel_type"

VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = [
    "tshirt",
    "hoodie",
    "pants",
    "jacket",
    "cap",
    "shoes",
    "bag",
    "other_apparel",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)


def get_images(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def move_files(files, target_folder: Path):
    target_folder.mkdir(parents=True, exist_ok=True)

    for file in files:
        target_path = target_folder / file.name

        if target_path.exists():
            target_path = target_folder / f"{file.stem}_{random.randint(100000, 999999)}{file.suffix}"

        shutil.move(str(file), str(target_path))


def split_class(cls: str):
    train_folder = BASE / "train" / cls
    val_folder = BASE / "val" / cls
    test_folder = BASE / "test" / cls

    images = get_images(train_folder)
    random.shuffle(images)

    total = len(images)
    val_count = int(total * VAL_RATIO)
    test_count = int(total * TEST_RATIO)

    val_files = images[:val_count]
    test_files = images[val_count:val_count + test_count]

    move_files(val_files, val_folder)
    move_files(test_files, test_folder)

    print(f"\nClass: {cls}")
    print(f"Total before split: {total}")
    print(f"Moved to val: {len(val_files)}")
    print(f"Moved to test: {len(test_files)}")
    print(f"Train left: {len(get_images(train_folder))}")
    print(f"Val now: {len(get_images(val_folder))}")
    print(f"Test now: {len(get_images(test_folder))}")


def main():
    print("Dataset base:", BASE)

    for cls in CLASSES:
        split_class(cls)

    print("\nDONE")


if __name__ == "__main__":
    main()