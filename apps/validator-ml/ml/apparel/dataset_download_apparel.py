from pathlib import Path
from icrawler.builtin import BingImageCrawler
import shutil
import uuid

BASE = Path(__file__).resolve().parents[1] / "datasets" / "apparel" / "train"
TEMP = Path(__file__).resolve().parents[1] / "datasets" / "_tmp_downloads"

TARGET_APPAREL = 600
TARGET_NON_APPAREL = 600
BATCH_SIZE = 100

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

APPAREL_KEYWORDS = [
    "t shirt mockup",
    "blank tshirt product photo",
    "hoodie mockup front",
    "hoodie clothing",
    "streetwear outfit",
    "fashion model clothing",
    "clothing flat lay",
    "pants clothing product photo",
    "jacket clothing product photo",
    "cap clothing product photo",
    "sneakers product photo",
    "bag fashion product photo",
]

NON_APPAREL_KEYWORDS = [
    "office workspace",
    "food photography",
    "car photography",
    "room interior",
    "landscape nature",
    "random objects table",
    "computer screenshot ui",
    "document scan paper",
    "kitchen interior photo",
    "city street photo",
    "poster graphic design",
    "phone screenshot",
]


def count_images(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


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


def download_keyword(keyword: str, target_dir: Path, prefix: str, max_num: int) -> int:
    safe_keyword = keyword.replace(" ", "_").replace("/", "_")
    tmp_dir = TEMP / prefix / safe_keyword

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading keyword: {keyword}")
    print(f"Temp folder: {tmp_dir}")

    crawler = BingImageCrawler(storage={"root_dir": str(tmp_dir)})
    crawler.crawl(keyword=keyword, max_num=max_num)

    moved = move_tmp_images(tmp_dir, target_dir, prefix)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Moved new files: {moved}")
    return moved


def fill_class(target_dir: Path, keywords: list[str], target: int, prefix: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    keyword_index = 0
    empty_rounds = 0

    while count_images(target_dir) < target:
        current = count_images(target_dir)
        keyword = keywords[keyword_index % len(keywords)]

        print("\n==============================")
        print(f"Class: {prefix}")
        print(f"Current: {current}/{target}")
        print(f"Keyword: {keyword}")
        print("==============================")

        moved = download_keyword(keyword, target_dir, prefix, BATCH_SIZE)

        if moved == 0:
            empty_rounds += 1
        else:
            empty_rounds = 0

        keyword_index += 1

        if empty_rounds >= len(keywords):
            print(f"No new images after full keyword cycle for {prefix}. Stopping.")
            break

    print(f"\nFinished {prefix}: {count_images(target_dir)} images")


def main():
    apparel_dir = BASE / "apparel"
    non_apparel_dir = BASE / "non_apparel"

    print("Dataset train base:", BASE)

    fill_class(apparel_dir, APPAREL_KEYWORDS, TARGET_APPAREL, "apparel")
    fill_class(non_apparel_dir, NON_APPAREL_KEYWORDS, TARGET_NON_APPAREL, "non_apparel")

    shutil.rmtree(TEMP, ignore_errors=True)

    print("\nDONE")
    print("Apparel:", count_images(apparel_dir))
    print("Non-apparel:", count_images(non_apparel_dir))


if __name__ == "__main__":
    main()