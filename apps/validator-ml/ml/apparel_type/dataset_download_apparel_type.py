from pathlib import Path
from icrawler.builtin import BingImageCrawler
import shutil
import uuid


BASE = Path(__file__).resolve().parents[1] / "datasets" / "apparel_type" / "train"
TEMP = Path(__file__).resolve().parents[1] / "datasets" / "_tmp_apparel_type"

TARGET_PER_CLASS = 400
BATCH_SIZE = 100

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


KEYWORDS = {
    "tshirt": [
        "t shirt product photo",
        "blank t shirt mockup",
        "oversized tshirt fashion",
        "t shirt on person",
    ],
    "hoodie": [
        "hoodie product photo",
        "hoodie mockup front",
        "oversized hoodie fashion",
        "hoodie on person",
    ],
    "pants": [
        "pants product photo",
        "jeans product photo",
        "cargo pants fashion",
        "trousers product photo",
    ],
    "jacket": [
        "jacket product photo",
        "streetwear jacket",
        "denim jacket product photo",
        "bomber jacket fashion",
    ],
    "cap": [
        "cap product photo",
        "baseball cap mockup",
        "streetwear cap",
        "blank cap product photo",
    ],
    "shoes": [
        "sneakers product photo",
        "shoes product photo",
        "streetwear sneakers",
        "running shoes product photo",
    ],
    "bag": [
        "bag product photo",
        "tote bag mockup",
        "backpack product photo",
        "streetwear bag",
    ],
    "other_apparel": [
        "fashion accessories product photo",
        "scarf product photo",
        "beanie product photo",
        "socks product photo",
        "gloves product photo",
    ],
}


def count_images(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def move_tmp_images(tmp_dir: Path, target_dir: Path, prefix: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    for img in tmp_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue

        new_name = f"{prefix}_{uuid.uuid4().hex[:12]}{img.suffix.lower()}"
        shutil.move(str(img), str(target_dir / new_name))
        moved += 1

    return moved


def download_keyword(keyword: str, target_dir: Path, class_name: str) -> int:
    safe_keyword = keyword.replace(" ", "_").replace("/", "_")
    tmp_dir = TEMP / class_name / safe_keyword

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading [{class_name}]: {keyword}")
    crawler = BingImageCrawler(storage={"root_dir": str(tmp_dir)})
    crawler.crawl(keyword=keyword, max_num=BATCH_SIZE)

    moved = move_tmp_images(tmp_dir, target_dir, class_name)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Moved: {moved}")
    return moved


def fill_class(class_name: str, keywords: list[str]) -> None:
    target_dir = BASE / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    keyword_index = 0
    empty_rounds = 0

    while count_images(target_dir) < TARGET_PER_CLASS:
        current = count_images(target_dir)
        keyword = keywords[keyword_index % len(keywords)]

        print("\n==============================")
        print(f"Class: {class_name}")
        print(f"Current: {current}/{TARGET_PER_CLASS}")
        print(f"Keyword: {keyword}")
        print("==============================")

        moved = download_keyword(keyword, target_dir, class_name)

        if moved == 0:
            empty_rounds += 1
        else:
            empty_rounds = 0

        keyword_index += 1

        if empty_rounds >= len(keywords):
            print(f"No new images for {class_name}. Stopping.")
            break

    print(f"Finished {class_name}: {count_images(target_dir)} images")


def main():
    print("Dataset base:", BASE)

    for class_name, keywords in KEYWORDS.items():
        fill_class(class_name, keywords)

    shutil.rmtree(TEMP, ignore_errors=True)

    print("\nDONE")
    for class_name in KEYWORDS:
        print(class_name, ":", count_images(BASE / class_name))


if __name__ == "__main__":
    main()