from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "datasets" / "brand_risk" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "brand_risk" / "processed"

SOURCE_TO_TARGET = {
    "known_brand": "brand_logo",
    "other_logo": "brand_logo",
    "no_brand": "no_brand",
}

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_images(folder: Path):
    if not folder.exists():
        return []

    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def clear_processed():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in SPLITS.keys():
        for class_name in ["brand_logo", "no_brand"]:
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_files(files, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        shutil.copy2(file, target_dir / file.name)


def split_files(files):
    random.shuffle(files)

    total = len(files)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def main():
    clear_processed()

    grouped = {
        "brand_logo": [],
        "no_brand": [],
    }

    for source_class, target_class in SOURCE_TO_TARGET.items():
        source_dir = RAW_DIR / source_class
        images = get_images(source_dir)
        grouped[target_class].extend(images)

        print(f"{source_class} -> {target_class}: {len(images)} images")

    print()

    for target_class, files in grouped.items():
        splits = split_files(files)

        for split_name, split_files_list in splits.items():
            target_dir = PROCESSED_DIR / split_name / target_class
            copy_files(split_files_list, target_dir)

        print(f"{target_class}:")
        print(f"  train: {len(splits['train'])}")
        print(f"  val:   {len(splits['val'])}")
        print(f"  test:  {len(splits['test'])}")

    print("\nDone. Processed dataset created at:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()