from pathlib import Path
from icrawler.builtin import BingImageCrawler

OUT_DIR = Path("ml/ datasets/brand_classifier/raw")

BRANDS = {
    "nike": "nike t shirt logo",
    "adidas": "adidas t shirt logo",
    "jordan": "jordan t shirt logo",
    "gucci": "gucci t shirt logo",
    "calvin_klein": "calvin klein t shirt logo",
    "puma": "puma t shirt logo",
    "supreme": "supreme box logo t shirt",
    "chanel": "chanel t shirt logo",
    "dior": "dior t shirt logo",
    "lv": "louis vuitton t shirt logo",
    "no_brand": "plain blank t shirt no logo",
}

IMAGES_PER_CLASS = 250


def download_class(label, keyword):
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*")))

    if existing >= IMAGES_PER_CLASS:
        print(f"[SKIP] {label}: {existing} images")
        return

    print(f"[INFO] Downloading {label}: {keyword}")

    crawler = BingImageCrawler(
        storage={"root_dir": str(out_dir)}
    )

    crawler.crawl(
        keyword=keyword,
        max_num=IMAGES_PER_CLASS - existing,
        file_idx_offset=existing,
    )

    total = len(list(out_dir.glob("*")))
    print(f"[DONE] {label}: {total}/{IMAGES_PER_CLASS}")


def main():
    for label, keyword in BRANDS.items():
        download_class(label, keyword)


if __name__ == "__main__":
    main()