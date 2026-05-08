from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "brand_classifier"
    / "raw"
)

OUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "brand_classifier"
    / "processed"
)

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def clear_output():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)


def get_images(folder: Path):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_images(images):
    total = len(images)

    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def copy_files(split_name, label, files):
    target_dir = OUT_DIR / split_name / label
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def process_class(class_dir: Path):
    label = class_dir.name

    images = get_images(class_dir)

    if len(images) == 0:
        print(f"[WARN] {label}: no images found")
        return

    random.shuffle(images)

    splits = split_images(images)

    for split_name, files in splits.items():
        copy_files(split_name, label, files)

    print(
        f"[DONE] {label} | "
        f"total={len(images)} | "
        f"train={len(splits['train'])} | "
        f"val={len(splits['val'])} | "
        f"test={len(splits['test'])}"
    )


def main():
    random.seed(42)

    clear_output()

    class_dirs = [
        d for d in RAW_DIR.iterdir()
        if d.is_dir()
    ]

    for class_dir in class_dirs:
        process_class(class_dir)

    print("\nDataset prepared successfully.")


if __name__ == "__main__":
    main()